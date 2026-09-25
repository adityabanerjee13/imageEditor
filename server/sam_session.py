"""Interactive SAM 3: run the image encoder once per uploaded image, decode each box prompt against the cached embeddings.

`Sam3TrackerModel.get_image_embeddings(pixel_values)` is the expensive part (~1-2 s on the iGPU); `forward(image_embeddings=...,
input_boxes=...)` is only the prompt encoder + mask decoder (tens of ms).  Embeddings are kept on the CPU (a few tens of MB
per image), the model itself is dropped from the accelerator while a heavy job runs and reloaded lazily afterwards.
All methods must be called from the GPU worker thread."""
import time
from collections import OrderedDict

import numpy as np
import torch

from object_edit.common import DEV, WEIGHTS, empty_cache
from object_edit.segment import clean_mask


class SamSession:
    def __init__(self, max_images=4):
        self.proc = self.model = None
        self.cache: OrderedDict[str, tuple[list, tuple[int, int]]] = OrderedDict()   # image_id -> (embeddings on CPU, (H, W))
        self.max_images = max_images
        self.load_s = None

    # ---- residency
    def load(self):
        if self.model is None:
            from transformers import Sam3TrackerModel, Sam3TrackerProcessor
            t = time.perf_counter()
            self.proc = Sam3TrackerProcessor.from_pretrained(str(WEIGHTS / "sam3"))
            self.model = Sam3TrackerModel.from_pretrained(str(WEIGHTS / "sam3")).to(DEV).eval()
            self.load_s = round(time.perf_counter() - t, 1)
            print(f"[sam] loaded in {self.load_s} s")

    def unload(self):
        if self.model is not None:
            self.model = None
            empty_cache()
            print("[sam] unloaded")

    # ---- encode once
    def encode(self, image_id, pil):
        if image_id in self.cache:
            self.cache.move_to_end(image_id)
            return
        self.load()
        t = time.perf_counter()
        inp = self.proc(images=pil, return_tensors="pt")
        with torch.no_grad():
            emb = self.model.get_image_embeddings(inp["pixel_values"].to(DEV))
        H, W = (int(v) for v in inp["original_sizes"][0])
        self.cache[image_id] = ([e.cpu() for e in emb], (H, W))
        while len(self.cache) > self.max_images:
            self.cache.popitem(last=False)
        print(f"[sam] encoded {image_id} ({W}x{H}) in {time.perf_counter() - t:.2f} s")

    def has(self, image_id):
        return image_id in self.cache

    # ---- decode per box
    def propose(self, image_id, box, min_frac=0.01):
        """Up to 3 proposals for one box prompt, cleaned (closed, component under the box centre, holes filled), sorted by
        IoU; the sub-part levels often come back (near-)empty for a box prompt, so proposals under `min_frac` of the box
        area are dropped (at least one is kept).  Returns [(mask bool HxW, iou float)], decode time in ms."""
        emb, (H, W) = self.cache[image_id]
        self.load()
        x0, y0, x1, y1 = box
        t = time.perf_counter()
        inp = self.proc(original_sizes=[[H, W]], input_boxes=[[[x0, y0, x1, y1]]], return_tensors="pt")
        with torch.no_grad():
            out = self.model(image_embeddings=[e.to(DEV) for e in emb], input_boxes=inp["input_boxes"].to(DEV), multimask_output=True)
        masks = self.proc.post_process_masks(out.pred_masks.cpu(), inp["original_sizes"])[0][0].numpy() > 0    # (3, H, W)
        iou = out.iou_scores.flatten().cpu().numpy()
        ms = (time.perf_counter() - t) * 1000
        centre = (min(max((x0 + x1) // 2, 0), W - 1), min(max((y0 + y1) // 2, 0), H - 1))
        min_px = min_frac * (x1 - x0) * (y1 - y0)
        cleaned = [(clean_mask(masks[k], centre) if masks[k].any() else masks[k], float(iou[k])) for k in np.argsort(-iou)]
        props = [(m, s) for m, s in cleaned if m.sum() >= min_px]
        if not props:                                       # keep the largest so the user always sees something
            props = [max(cleaned, key=lambda ms_: ms_[0].sum())]
        return props, ms
