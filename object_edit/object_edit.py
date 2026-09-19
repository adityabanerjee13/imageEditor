"""Object move: shift the object(s) under `initial` (x, y) so that point lands on `final`, with FreeFine.

    python object_edit/object_edit.py --coords input/object_move.json
    python object_edit/object_edit.py --coords input/object_move_scene.json [--scale 1.0] [--start-step 35]

Inputs are only the image and point pairs - no object name or caption is given to any model.
input/*.json: {"scene": ..., "moves": [{"initial": {x, y}, "final": {x, y}}, ...]}
(a single {"initial", "final"} pair at the top level is also accepted).

Stages (all objects are processed together: one removal pass, one regeneration pass)
  1. object masks      SAM 3 tracker, one positive click per object at `initial`; of the three proposals
                       (sub-part / part / whole) the largest one under 30 % of the frame is kept and
                       reduced to the component containing the click
  2. mask dilation     each mask grown by --dilate-radius (6) px: covers SAM edge error and the contact
                       halo; the union of the dilated masks is the removal hole
                       (the object pixels themselves are copied with the tight masks)
  3. perspective scale MoGe-2 metric depth, per object: scale = z(initial) / z(final)  (--scale overrides)
  4. ROI               square window covering every object at its source and destination, resized to 512
                       (FreeFine's native size); only this window is edited
  5. object removal    --removal lama (default): LaMa big-lama at native resolution, all holes at once
                       with Geomagical feature refinement on by default (--no-lama-refine to skip);
                       --removal freefine: FreeFine background generation ("empty scene", GeoBench settings)
                       Original pixels are kept outside the holes either way
  6. coarse edit       affine copy of every object to its destination (translate + scale), in order
  7. refine            FreeFine detail-preserving regeneration (guidance_text "", DDIM inversion with the
                       original as reference, regeneration from --start-step), run once per object on the
                       shared coarse image so each target attends to its own source; other targets held
  8. composite         holes = removal fill (native res), target regions = regeneration; the rest of the
                       full-resolution image is untouched

Output: outputs/object_edit/<scene>/output.jpg (+ mask_object.png (all objects), mask_dilated.png, background_roi.png, background_512.png,
coarse_input_512.png, mask_target_512.png, result_512.png, task_points.png, metrics.json)

Weights (weights/): sam3, moge-2-vitl-normal, stable-diffusion-v1-5 (fp16), big-lama.  Code: repos/FreeFine, repos/MoGe, repos/lama.
"""
import argparse
import json
import random
import sys
import time
import types
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
WEIGHTS = ROOT / "weights"
REPOS = ROOT / "repos"
OUT_ROOT = ROOT / "outputs" / "object_edit"
sys.path.insert(0, str(REPOS / "MoGe"))
sys.path.insert(0, str(REPOS / "FreeFine"))
# FreeFine imports rembg only for an optional matting helper that is never called -> stub it
sys.modules.setdefault("rembg", types.SimpleNamespace(remove=lambda *a, **k: (_ for _ in ()).throw(RuntimeError("rembg stub"))))

from diffusers import DDIMScheduler                                                    # noqa: E402
from src.demo.model import FreeFinePipeline                                             # noqa: E402
from src.utils.attention import (Attention_Modulator, register_attention_control,       # noqa: E402
                                 register_attention_control_4bggen)
from src.utils.vis_utils import re_edit_2d                                              # noqa: E402

DEV = torch.device("cuda" if torch.cuda.is_available() else "xpu" if torch.xpu.is_available() else "cpu")
RES = 512


# ============================================================================ instrumentation
def sync():
    if DEV.type == "cuda":
        torch.cuda.synchronize()
    elif DEV.type == "xpu":
        torch.xpu.synchronize()


def empty_cache():
    if DEV.type == "cuda":
        torch.cuda.empty_cache()
    elif DEV.type == "xpu":
        torch.xpu.empty_cache()


def peak_mem_gb():
    if DEV.type == "cuda":
        return torch.cuda.max_memory_allocated() / 1e9
    if DEV.type == "xpu":
        return torch.xpu.max_memory_allocated() / 1e9
    return 0.0


def reset_peak():
    if DEV.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    elif DEV.type == "xpu":
        torch.xpu.reset_peak_memory_stats()


class Stage:
    """Times a stage (device-synced) and records peak accelerator memory into `log`."""

    def __init__(self, log, name, model=""):
        self.log, self.row = log, {"stage": name, "model": model}

    def __enter__(self):
        sync(); reset_peak(); self.t = time.perf_counter()
        return self

    def __exit__(self, *_):
        sync()
        self.row["time_s"] = round(time.perf_counter() - self.t, 2)
        self.row["gpu_peak_gb"] = round(peak_mem_gb(), 2)
        self.log.append(self.row)
        print(f"[{self.row['stage']}] {self.row['time_s']} s, peak {self.row['gpu_peak_gb']} GB  {self.row['model']}")


def timed(fn):
    t = time.perf_counter(); r = fn(); return r, round(time.perf_counter() - t, 2)


# ============================================================================ task
class Move:
    """One object: click point, destination point, masks and scale (all in the task's pixel frame)."""

    def __init__(self, src, dst, mask=None, mask_dilated=None, scale=1.0):
        self.src, self.dst, self.mask, self.mask_dilated, self.scale = src, dst, mask, mask_dilated, scale

    @property
    def shift(self):
        return self.dst[0] - self.src[0], self.dst[1] - self.src[1]


class MoveTask:
    def __init__(self, scene_path, image, moves):
        self.scene_path, self.image, self.moves = scene_path, image, moves
        self.meta = {}

    @property
    def size(self):
        return self.image.size            # (W, H)

    @property
    def mask(self):                        # union of tight masks
        return np.any([m.mask for m in self.moves], axis=0)

    @property
    def mask_dilated(self):                # union of dilated masks = removal hole
        return np.any([m.mask_dilated for m in self.moves], axis=0)


def load_task(scene, coords_json):
    c = json.loads(Path(coords_json).read_text())
    scene = Path(scene) if scene else ROOT / c["scene"]
    pairs = c.get("moves") or [{"initial": c["initial"], "final": c["final"]}]
    moves = [Move((int(m["initial"]["x"]), int(m["initial"]["y"])), (int(m["final"]["x"]), int(m["final"]["y"]))) for m in pairs]
    return MoveTask(scene, Image.open(scene).convert("RGB"), moves)


# ============================================================================ 1. object mask (SAM 3, one click)
def segment_objects(task, log, max_frac=0.3):
    from transformers import Sam3TrackerModel, Sam3TrackerProcessor
    W, H = task.size
    (proc, model), load_s = timed(lambda: (Sam3TrackerProcessor.from_pretrained(str(WEIGHTS / "sam3")),
                                           Sam3TrackerModel.from_pretrained(str(WEIGHTS / "sam3")).to(DEV).eval()))
    with Stage(log, "segmentation", f"SAM 3 point prompt x{len(task.moves)}") as st:
        st.row["objects"] = []
        for mv in task.moves:
            x, y = mv.src
            inp = proc(images=task.image, input_points=[[[[x, y]]]], input_labels=[[[1]]], return_tensors="pt").to(DEV)
            with torch.no_grad():
                o = model(**inp, multimask_output=True)
            masks = proc.post_process_masks(o.pred_masks.cpu(), inp["original_sizes"])[0][0].numpy() > 0    # (3, H, W)
            iou = o.iou_scores.flatten().cpu().numpy()
            areas = masks.reshape(3, -1).sum(1)
            ok = np.where(areas < max_frac * H * W)[0]             # largest plausible proposal = whole object
            best = int(ok[areas[ok].argmax()]) if len(ok) else int(iou.argmax())
            mv.mask = clean_mask(masks[best], (x, y))
            st.row["objects"].append({"src": list(mv.src), "iou": np.round(iou, 3).tolist(), "areas": areas.tolist(), "picked": best})
    st.row["load_s"] = load_s
    del model; empty_cache()


def clean_mask(m, point, close_px=7):
    """Morphological close, keep the component containing `point` (else the largest), fill holes."""
    k = np.ones((close_px, close_px), np.uint8)
    m8 = cv2.morphologyEx(m.astype(np.uint8), cv2.MORPH_CLOSE, k)
    n, lab = cv2.connectedComponents(m8)
    if n > 1:
        x, y = point
        keep = lab[y, x] if lab[y, x] > 0 else 1 + np.bincount(lab[lab > 0]).argmax()
        m8 = (lab == keep).astype(np.uint8)
    flood = m8.copy(); h, w = flood.shape
    cv2.floodFill(flood, np.zeros((h + 2, w + 2), np.uint8), (0, 0), 1)
    return (m8 | (1 - flood)).astype(bool)


# ============================================================================ 2. mask dilation
def dilate_object_masks(task, radius, log):
    """Removal hole = each object mask grown by `radius` px (kernel 2r+1), unioned."""
    with Stage(log, "mask_dilation", f"radius {radius} px") as st:
        k = np.ones((2 * radius + 1, 2 * radius + 1), np.uint8)
        for mv in task.moves:
            mv.mask_dilated = cv2.dilate(mv.mask.astype(np.uint8), k) > 0
    st.row.update(mask_px=int(task.mask.sum()), dilated_px=int(task.mask_dilated.sum()))


# ============================================================================ 3. perspective scale (MoGe-2)
def estimate_perspective_scale(task, log):
    """Size ratio for a rigid object whose anchor moves from src to dst: z_src / z_dst (metric depth)."""
    from moge.model.v2 import MoGeModel
    model, load_s = timed(lambda: MoGeModel.from_pretrained(str(WEIGHTS / "moge-2-vitl-normal" / "model.pt")).to(DEV).eval())
    x = torch.tensor(np.asarray(task.image) / 255.0, dtype=torch.float32, device=DEV).permute(2, 0, 1)
    with Stage(log, "geometry", "MoGe-2 ViT-L") as st:
        with torch.no_grad():
            o = model.infer(x, use_fp16=False)          # fp16 autocast is broken on XPU for this model
    st.row["load_s"] = load_s
    z = o["points"][..., 2].cpu().numpy(); valid = o["mask"].cpu().numpy().astype(bool)
    del model; empty_cache()

    def zat(p, r=6):
        x0, y0 = p
        win, vw = z[max(0, y0 - r):y0 + r + 1, max(0, x0 - r):x0 + r + 1], valid[max(0, y0 - r):y0 + r + 1, max(0, x0 - r):x0 + r + 1]
        return float(np.median(win[vw])) if vw.any() else float(np.median(win))
    st.row["objects"] = []
    for mv in task.moves:
        z_src, z_dst = zat(mv.src), zat(mv.dst)
        mv.scale = float(np.clip(z_src / max(z_dst, 1e-3), 0.25, 4.0))
        st.row["objects"].append({"z_src": round(z_src, 3), "z_dst": round(z_dst, 3), "scale": round(mv.scale, 3)})
        print(f"[geometry] {mv.src} -> {mv.dst}: z_src={z_src:.2f} m z_dst={z_dst:.2f} m -> perspective scale {mv.scale:.2f}")


# ============================================================================ 3. ROI helpers
def roi_box(task, margin=0.25, min_size=RES):
    """Square crop (x0, y0, side) covering every object at its source and at its destination."""
    W, H = task.size
    bx0, bx1, by0, by1 = W, 0, H, 0
    for mv in task.moves:
        ys, xs = np.where(mv.mask)
        x0, x1, y0, y1 = xs.min(), xs.max(), ys.min(), ys.max()
        sx, sy = mv.src; dx, dy = mv.shift; scale = mv.scale
        tx0, tx1 = sx + (x0 - sx) * scale + dx, sx + (x1 - sx) * scale + dx
        ty0, ty1 = sy + (y0 - sy) * scale + dy, sy + (y1 - sy) * scale + dy
        bx0, bx1, by0, by1 = min(bx0, x0, tx0), max(bx1, x1, tx1), min(by0, y0, ty0), max(by1, y1, ty1)
    side = int(np.ceil(max(max(bx1 - bx0, by1 - by0) * (1 + 2 * margin), min_size)))
    side = min(side, W, H)
    cx, cy = (bx0 + bx1) / 2, (by0 + by1) / 2
    return int(np.clip(round(cx - side / 2), 0, W - side)), int(np.clip(round(cy - side / 2), 0, H - side)), side


def crop_task(task, box):
    x0, y0, s = box
    moves = [Move((m.src[0] - x0, m.src[1] - y0), (m.dst[0] - x0, m.dst[1] - y0), m.mask[y0:y0 + s, x0:x0 + s].copy(),
                  m.mask_dilated[y0:y0 + s, x0:x0 + s].copy(), m.scale) for m in task.moves]
    t = MoveTask(task.scene_path, task.image.crop((x0, y0, x0 + s, y0 + s)), moves)
    t.meta = dict(task.meta, crop=box)
    return t


def to_512(img):
    return cv2.resize(np.asarray(img), (RES, RES), interpolation=cv2.INTER_LANCZOS4)


def mask_to_512(m):
    return cv2.resize(m.astype(np.uint8), (RES, RES), interpolation=cv2.INTER_NEAREST)


def composite_back(original, edited, region, feather_px=3):
    """Paste `edited` onto `original` inside `region` through a feathered edge."""
    o, e = np.asarray(original).astype(np.float32), np.asarray(edited).astype(np.float32)
    a = region.astype(np.float32)
    if feather_px:
        a = cv2.GaussianBlur(a, (0, 0), feather_px)
    return Image.fromarray(np.clip(o * (1 - a[..., None]) + e * a[..., None], 0, 255).astype(np.uint8))


def draw_points(img, moves, r=10):
    a = np.array(img).copy()
    for mv in moves:
        cv2.arrowedLine(a, mv.src, mv.dst, (255, 220, 0), 3, tipLength=0.05)
        cv2.circle(a, mv.src, r, (255, 0, 0), 3); cv2.circle(a, mv.dst, r, (0, 255, 0), 3)
    return Image.fromarray(a)


# ============================================================================ 4a. LaMa removal
_lama = None


def load_lama():
    global _lama
    if _lama is None:
        sys.path.insert(0, str(REPOS / "lama"))
        # saicinpainting.training.data.aug needs an albumentations<1.0 API; the predict path never uses it
        aug = types.ModuleType("saicinpainting.training.data.aug"); aug.IAAAffine2 = aug.IAAPerspective2 = object
        sys.modules.setdefault("saicinpainting.training.data.aug", aug)
        import yaml
        from omegaconf import OmegaConf
        import saicinpainting.training.trainers as trainers
        root = WEIGHTS / "big-lama" / "big-lama"
        cfg = OmegaConf.create(yaml.safe_load((root / "config.yaml").read_text()))
        cfg.training_model.predict_only = True
        cfg.visualizer.kind = "noop"
        # the Lightning checkpoint pickles callback objects -> torch>=2.6 needs weights_only=False
        state = torch.load(str(root / "models" / "best.ckpt"), map_location="cpu", weights_only=False)
        model = trainers.make_training_model(cfg)
        model.load_state_dict({k: v for k, v in state["state_dict"].items() if k.startswith("generator.")}, strict=False)
        model.freeze()
        _lama = model.to(DEV)
    return _lama


def lama_inpaint(img_rgb, hole, refine=False, n_iters=15):
    """img_rgb uint8 HxWx3, hole bool HxW (True = fill); whole crop, padded to a multiple of 8.
    refine=True runs LaMa's feature refinement (Geomagical, arXiv 2206.13644; repos/lama-with-refiner ==
    upstream `refine=True`): an image pyramid where the generator's inner features are optimised for
    `n_iters` Adam steps per scale so the full-res fill agrees with the low-res prediction."""
    model = load_lama()
    H, W = hole.shape
    ph, pw = (8 - H % 8) % 8, (8 - W % 8) % 8
    img = torch.from_numpy(np.pad(img_rgb, ((0, ph), (0, pw), (0, 0)), mode="reflect")).float().permute(2, 0, 1)[None] / 255.0
    m = torch.from_numpy(np.pad(hole.astype(np.float32), ((0, ph), (0, pw)), mode="constant"))[None, None]
    if refine:
        import saicinpainting.evaluation.refinement as R

        class _TorchOnDev:                      # the refiner builds "cuda:<id>" device strings
            def __getattr__(self, k): return getattr(torch, k)
            def device(self, d): return torch.device(str(DEV)) if str(d).startswith("cuda") else torch.device(d)
        R.torch = _TorchOnDev()
        out = R.refine_predict({"image": img, "mask": m, "unpad_to_size": [torch.tensor([H]), torch.tensor([W])]}, model,
                               gpu_ids="0,", modulo=8, n_iters=n_iters, lr=0.002, min_side=512, max_scales=3, px_budget=1800000)
        out = out[0].permute(1, 2, 0).detach().cpu().numpy()[:H, :W]
    else:
        with torch.no_grad():
            out = model({"image": img.to(DEV), "mask": m.to(DEV)})["inpainted"][0].permute(1, 2, 0).cpu().numpy()[:H, :W]
    return np.clip(out * 255, 0, 255).astype(np.uint8)


# ============================================================================ 4b-6. FreeFine
def load_pipeline():
    model = FreeFinePipeline.from_pretrained(str(WEIGHTS / "stable-diffusion-v1-5"), torch_dtype=torch.float16,
                                             variant="fp16", safety_checker=None).to(DEV)
    model.scheduler = DDIMScheduler.from_config(model.scheduler.config)

    # upstream hard-codes .cuda() for the text encoder; route through the pipeline device instead
    @torch.no_grad()
    def get_text_embeddings(self, prompt):
        ti = self.tokenizer(prompt, padding="max_length", max_length=77, return_tensors="pt")
        return self.text_encoder(ti.input_ids.to(self.device))[0]
    model.get_text_embeddings = types.MethodType(get_text_embeddings, model)
    return model


def install_controller(model, controller, for_background):
    model.controller = controller
    (register_attention_control_4bggen if for_background else register_attention_control)(model, controller)
    model.modify_unet_forward()
    model.enable_attention_slicing()          # xformers is unavailable on XPU; the hooks replace the processors anyway


def generate_background(model, ori_img, dil_mask, prompt, seed):
    """Object removal: FreeFine background generation inside the (already dilated) hole mask (settings
    of evaluation/FreeFine/freefine_batch_infer_bggen_2d.py), original pixels kept outside the hole."""
    install_controller(model, Attention_Modulator(), for_background=True)
    bg = model.FreeFine_background_generation(ori_img, dil_mask, prompt, guidance_scale=7.5, eta=1.0, end_step=35,
                                              num_step=50, end_scale=0.5, start_step=1, share_attn=True, method_type="tca",
                                              local_text_edit=True, local_perturbation=True, verbose=False, seed=seed,
                                              return_intermediates=False, latent_blended=False)
    m = dil_mask.astype(np.float32)
    return (ori_img * (1 - m) + bg * m).astype(np.uint8)


def regenerate(model, ori_img, ori_mask, coarse, target_mask, args, seed, cons_area=None):
    """Refine one pasted object (settings of evaluation/FreeFine/freefine_batch_infer_2d.py, guidance_text "")."""
    install_controller(model, Attention_Modulator(start_layer=10), for_background=False)
    out = model.FreeFine_generation(ori_img=ori_img, ori_mask=ori_mask, coarse_input=coarse, target_mask=target_mask,
                                    guidance_text=args.prompt, guidance_scale=7.5, eta=1.0, end_scale=args.end_scale,
                                    end_step=50, num_step=50, start_step=args.start_step, seed=seed, draw_mask=None,
                                    return_intermediates=False, use_auto_draw=True, reduce_inp_artifacts=True,
                                    cons_area=target_mask if cons_area is None else cons_area)
    return np.asarray(out).astype(np.uint8)


# ============================================================================ pipeline
def run(full_task, args):
    log = []
    od = OUT_ROOT / full_task.scene_path.stem
    od.mkdir(parents=True, exist_ok=True)
    segment_objects(full_task, log)
    dilate_object_masks(full_task, args.dilate_radius, log)
    if args.scale == "auto":
        estimate_perspective_scale(full_task, log)
    else:
        for mv in full_task.moves:
            mv.scale = float(args.scale)
    Image.fromarray(full_task.mask.astype(np.uint8) * 255).save(od / "mask_object.png")
    Image.fromarray(full_task.mask_dilated.astype(np.uint8) * 255).save(od / "mask_dilated.png")
    draw_points(full_task.image, full_task.moves).save(od / "task_points.png")

    box = roi_box(full_task) if args.roi else (0, 0, min(full_task.size))
    task = crop_task(full_task, box)
    W, H = task.size
    print(f"[roi] box x={box[0]} y={box[1]} side={box[2]} -> {RES}px (factor {RES / box[2]:.2f})")
    ori_img = to_512(task.image)
    ori_mask = np.repeat(mask_to_512(task.mask)[..., None], 3, -1)      # union of tight masks, 3-channel {0,1}

    seed = args.seed if args.seed >= 0 else random.randint(0, 2 ** 31)

    def load_sd():
        m, load_s = timed(load_pipeline)
        log.append({"stage": "load", "model": "SD-1.5 fp16 (FreeFine)", "time_s": load_s})
        return m

    # ---- 5. removal: all holes at once (SD-1.5 is loaded afterwards so the refiner's backward pass
    # does not share the accelerator with it)
    hole = task.mask_dilated                                     # union of dilated masks, ROI coordinates
    dil_mask = np.repeat(mask_to_512(hole)[..., None], 3, -1)
    if args.removal == "lama":
        with Stage(log, "background_generation", f"LaMa big-lama (native {W}px, {len(task.moves)} holes{', refined' if args.lama_refine else ''})"):
            bg_native = lama_inpaint(np.asarray(task.image), hole, refine=args.lama_refine)
        Image.fromarray(bg_native).save(od / "background_roi.png")      # native-res ROI crop
        bg = to_512(bg_native)
        model = load_sd()
    else:
        model = load_sd()
        with Stage(log, "background_generation", "FreeFine bg-gen 50 steps"):
            bg = generate_background(model, ori_img, dil_mask, args.bg_prompt, seed)
    Image.fromarray(bg).save(od / "background_512.png")

    # ---- 6. coarse edit: paste every object onto the background (translate + scale about its click point)
    coarse, tgts, srcs = bg, [], []
    for mv in task.moves:
        m512 = np.repeat(mask_to_512(mv.mask)[..., None], 3, -1)
        dx512, dy512 = mv.shift[0] * RES / W, mv.shift[1] * RES / H
        ys, xs = np.where(m512[..., 0])
        cx, cy = (xs.min() + xs.max()) / 2, (ys.min() + ys.max()) / 2   # re_edit_2d scales about the mask centre
        sx, sy = mv.src[0] * RES / W, mv.src[1] * RES / H
        dx512 += (mv.scale - 1) * (cx - sx); dy512 += (mv.scale - 1) * (cy - sy)
        coarse, t, _ = re_edit_2d(ori_img, m512, [dx512, dy512, 0.0, mv.scale, mv.scale], coarse)
        tgts.append(((t[..., 0] if t.ndim == 3 else t) > 0)); srcs.append(m512)
    tgt = np.any(tgts, axis=0)
    Image.fromarray(coarse).save(od / "coarse_input_512.png")
    Image.fromarray(tgt.astype(np.uint8) * 255).save(od / "mask_target_512.png")

    # ---- 7. regeneration, one object at a time on the shared coarse image: each target region attends
    # only to its own source object; the union of all target regions is constrained (cons_area) so the
    # other objects' pasted pixels stay put.  (FreeFine's multi-object composition API is stale in the
    # released code - wrapper/attention batching mismatch - so the single-object path is used per object.)
    result = coarse
    for i, (mv, m512, t) in enumerate(zip(task.moves, srcs, tgts)):
        with Stage(log, "regeneration", f"FreeFine gen object {i + 1}/{len(task.moves)} start_step={args.start_step}"):
            result = regenerate(model, ori_img, m512, result, t.astype(np.uint8) * 255, args, seed,
                                cons_area=tgt.astype(np.uint8) * 255)
    Image.fromarray(result).save(od / "result_512.png")
    del model; empty_cache()

    # ---- 8. composite at full resolution: holes keep the removal fill (native res for LaMa); only the
    # target regions come from the 512 regeneration (--composite-hole: holes from the regeneration too,
    # FreeFine's own behaviour, which tends to hallucinate objects there)
    up = cv2.resize(result, (W, H), interpolation=cv2.INTER_LANCZOS4)
    tgt_full = cv2.resize(tgt.astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST) > 0
    region = tgt_full | (hole if args.composite_hole else False)
    region = cv2.dilate(region.astype(np.uint8), np.ones((15, 15), np.uint8)) > 0
    base = Image.fromarray(bg_native) if args.removal == "lama" else composite_back(task.image, cv2.resize(bg, (W, H), interpolation=cv2.INTER_LANCZOS4), hole, feather_px=3)
    final_crop = composite_back(base, up, region, feather_px=4)
    x0, y0, s = box
    final = np.asarray(full_task.image).copy()
    final[y0:y0 + s, x0:x0 + s] = np.asarray(final_crop)
    Image.fromarray(final).save(od / "output.jpg", quality=95)
    (od / "metrics.json").write_text(json.dumps({"device": str(DEV), "stages": log, "seed": seed, "roi_box": list(box),
                                                 "moves": [{"src": list(m.src), "dst": list(m.dst), "scale": round(m.scale, 3)} for m in full_task.moves],
                                                 "removal": args.removal, "start_step": args.start_step, "end_scale": args.end_scale}, indent=2))
    print("saved", od / "output.jpg")


def parse():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--coords", default=str(ROOT / "input" / "object_move.json"))
    ap.add_argument("--scene", default=None, help="override the scene path in the coords file")
    ap.add_argument("--scale", default="auto", help="object scale at the destination: number or 'auto' (MoGe-2 depth ratio)")
    ap.add_argument("--prompt", default="", help='refine-stage guidance text (FreeFine benchmark: "")')
    ap.add_argument("--bg-prompt", default="empty scene", help="background-generation text (FreeFine benchmark value)")
    ap.add_argument("--removal", default="lama", choices=["lama", "freefine"], help="object-removal backend")
    ap.add_argument("--no-lama-refine", dest="lama_refine", action="store_false",
                    help="disable LaMa feature refinement (on by default: multi-scale, 15 Adam steps/scale, ~+45 s)")
    ap.add_argument("--dilate-radius", type=int, default=6, help="stage 2: object mask grown by this many native px (6)")
    ap.add_argument("--start-step", type=int, default=15, help="benchmark uses 35 (keeps more pasted pixels); 15 regenerates more")
    ap.add_argument("--end-scale", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--composite-hole", action="store_true", help="take the source hole from the 512 regeneration instead of the removal backend")
    ap.add_argument("--no-roi", dest="roi", action="store_false", help="edit the whole frame squashed to 512x512")
    return ap.parse_args()


if __name__ == "__main__":
    args = parse()
    run(load_task(args.scene, args.coords), args)
