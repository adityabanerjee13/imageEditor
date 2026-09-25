"""OmniPaint object insertion (FLUX.1-dev + insertion LoRA).  Two conditions: the scene with the target box blacked
out, and the subject (the object cut out with its SAM mask on white, padded square, 512x512).  The model generates the
object *to fill the black box*, so the box is the user's target bounding box (move.dst_box: position AND size of the
inserted object; the MoGe depth scale is not used for this backend).  Without a target box (point-based JSON runs) the
axis-aligned bbox of the depth-scaled paste footprint + `grow` px is used instead.  A rectangle, not the silhouette:
tight silhouettes of a few latent patches are ignored and nothing is placed.  Identity is approximate (a generative
re-render, not a copy).

- mode="window" (default): the same ~3x-object square windows FreeFine uses, resized to 512, so the object spans ~1/3 of
  the canvas whatever the photo size.  ~12 min per object at 28 steps on the iGPU.
- mode="full": the whole frame downscaled to `res` on the long side (1024).  Whole-scene context; the object must still
  cover a reasonable number of latent patches (>= ~8x8 on the canvas, i.e. >= ~130 px of a 1024 canvas) or it is
  ignored.  ~30+ min per object at 1024 / 28 steps on the iGPU.
What is composited back is the target edit region (see the class docstring)."""
import cv2
import numpy as np
from PIL import Image

from object_edit import flux_omnipaint as fx
from object_edit.common import RES, Stage, bbox, composite_back, edit_region, crop, square_window, to_512
from object_edit.insertion.base import Inserter
from object_edit.insertion.paste import coarse_paste, composite, save_coarse


def subject_on_white(original, mask, size=RES):
    """Tight crop of the object on a white canvas, centred in a square, resized to `size` (what OmniPaint's CarveKit
    step would produce, but from our mask)."""
    x0, y0, x1, y1 = bbox(mask)
    crop_ = original[y0:y1 + 1, x0:x1 + 1]
    a = mask[y0:y1 + 1, x0:x1 + 1][..., None]
    obj = (crop_ * a + 255 * (1 - a)).astype(np.uint8)
    side = max(obj.shape[:2])
    canvas = Image.new("RGB", (side, side), (255, 255, 255))
    canvas.paste(Image.fromarray(obj), ((side - obj.shape[1]) // 2, (side - obj.shape[0]) // 2))
    return canvas.resize((size, size), Image.LANCZOS)


def rect_of(mask, grow):
    x0, y0, x1, y1 = bbox(mask)
    r = np.zeros_like(mask)
    r[max(0, y0 - grow):y1 + grow + 1, max(0, x0 - grow):x1 + grow + 1] = True
    return r


def box_mask(box, shape):
    """bool mask of an (x0, y0, x1, y1) box, clipped to `shape`."""
    H, W = shape
    x0, y0, x1, y1 = [int(round(v)) for v in box]
    r = np.zeros((H, W), bool)
    r[max(0, y0):min(H, y1), max(0, x0):min(W, x1)] = True
    return r


def target_window(move, c, W, H, factor=3.0):
    """Window mode with a user target box: a square window of max(512, factor x box size) around the box centre (the
    default destination window is centred on the depth-scaled footprint and may not contain a larger box)."""
    x0, y0, x1, y1 = move.dst_box
    return square_window(((x0 + x1) / 2, (y0 + y1) / 2), max(RES, factor * max(x1 - x0, y1 - y0)), W, H)


class OmniPaintInserter(Inserter):
    """Target edit region: the blacked-out rectangle is always where the object goes; `region` decides how much of the
    re-rendered canvas is kept - "mask": the rectangle (+3 px), "dilated"/"box": a band around it where OmniPaint's
    cast shadow and lighting land, "full": the whole window / frame (feathered border)."""
    name = "omnipaint"
    shared_model = "flux"
    generative = True

    def __init__(self, steps=28, grow=8, mode="window", res=1024, region="mask", margin=0.35):
        super().__init__(region, margin)
        self.steps, self.grow, self.mode, self.res = steps, grow, mode, res

    def load(self, log=None):
        fx.get(log)

    def unload(self):
        fx.release()

    def insert(self, original, background, move, *, seed, log, out_dir=None, idx=0, total=1):
        c = coarse_paste(original, background, move)           # windows + affine paste; also the full-frame footprint below
        save_coarse(c, out_dir, idx)
        if not c.tgt.any():
            print(f"[object {idx}] target lies outside the frame - skipped"); return background

        subject = subject_on_white(original, move.mask)
        if out_dir is not None:
            subject.save(out_dir / f"obj{idx}_subject_512.png")
        if self.mode == "full":
            return self._full(original, background, move, c, subject, seed, log, out_dir, idx, total)
        # ---- window mode: the target box (or, without one, the footprint bbox) blacked out on the 512 destination canvas
        if move.dst_box is not None:
            H, W = background.shape[:2]
            c.dwin = target_window(move, c, W, H)
            c.dst512 = to_512(crop(background, c.dwin))
            f = RES / c.dwin[2]
            rect = box_mask([(move.dst_box[0] - c.dwin[0]) * f, (move.dst_box[1] - c.dwin[1]) * f,
                             (move.dst_box[2] - c.dwin[0]) * f, (move.dst_box[3] - c.dwin[1]) * f], (RES, RES))
            c.tgt = rect
        else:
            rect = rect_of(c.tgt, self.grow)
        with Stage(log, "regeneration", f"OmniPaint insert object {idx + 1}/{total} (window {c.side}px, {self.steps} steps, region {self.region})") as st:
            bg512 = Image.fromarray(c.dst512)
            masked = Image.composite(Image.new("RGB", bg512.size, (0, 0, 0)), bg512, Image.fromarray(rect.astype(np.uint8) * 255))
            if out_dir is not None:
                masked.save(out_dir / f"obj{idx}_condition_512.png")          # the scene condition exactly as the model sees it
            out = fx.run("insertion", [fx.condition("insertion", masked), fx.condition("insertion", subject, position_delta=(0, -32))],
                         seed, steps=self.steps, log=log)
            result = np.asarray(out.convert("RGB")).astype(np.uint8)
        st.row.update(source_window=list(c.swin), dest_window=list(c.dwin), scale=round(move.scale, 3), target="box" if move.dst_box else "footprint")
        if out_dir is not None:
            Image.fromarray(result).save(out_dir / f"obj{idx}_result_512.png")
        c.tgt = rect                       # "mask" region = the rectangle (+3 px); wider regions keep more of the re-render
        reg = None if self.region == "mask" else edit_region(rect, self.region, self.margin)
        return composite(background, c, result, grow_kernel=3, region=reg)

    def _full(self, original, background, move, c, subject, seed, log, out_dir, idx, total):
        """Whole frame at `res`: the target footprint is the paste footprint mapped from the destination window back to
        frame coordinates."""
        H, W = background.shape[:2]
        dx0, dy0, ds = c.dwin
        tgt_full = np.zeros((H, W), bool)
        tgt_full[dy0:dy0 + ds, dx0:dx0 + ds] = cv2.resize(c.tgt.astype(np.uint8), (ds, ds), interpolation=cv2.INTER_NEAREST) > 0
        rect = box_mask(move.dst_box, (H, W)) if move.dst_box is not None else rect_of(tgt_full, int(round(self.grow * ds / RES)))
        cw, ch = fx.canvas_size(W, H, self.res)
        with Stage(log, "regeneration", f"OmniPaint insert object {idx + 1}/{total}, full frame {cw}x{ch} ({self.steps} steps, region {self.region})") as st:
            bg = Image.fromarray(background).resize((cw, ch), Image.LANCZOS)
            mask = Image.fromarray(rect.astype(np.uint8) * 255).resize((cw, ch), Image.NEAREST)
            masked = Image.composite(Image.new("RGB", (cw, ch), (0, 0, 0)), bg, mask)
            if out_dir is not None:
                masked.save(out_dir / f"obj{idx}_condition_full.png")         # the scene condition exactly as the model sees it
            out = fx.run("insertion", [fx.condition("insertion", masked), fx.condition("insertion", subject, position_delta=(0, -32))],
                         seed, steps=self.steps, log=log, size=(cw, ch))
            result = cv2.resize(np.asarray(out.convert("RGB")), (W, H), interpolation=cv2.INTER_LANCZOS4)
        st.row.update(dest_window=list(c.dwin), scale=round(move.scale, 3), canvas=[cw, ch])
        if out_dir is not None:
            out.save(out_dir / f"obj{idx}_result_full.png")
        reg = edit_region(rect, self.region, self.margin)
        if reg.all():                      # whole frame: feather at the border
            reg = cv2.erode(reg.astype(np.uint8), np.ones((9, 9), np.uint8)) > 0
        return np.asarray(composite_back(background, result, reg, feather_px=4))
