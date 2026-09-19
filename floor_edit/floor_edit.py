"""floor_edit — replace the floor in a room photo with a tiled texture, keeping the scene's lighting.

    python floor_edit/floor_edit.py --scene input/scene.jpeg --pattern input/pattern.png
    python floor_edit/floor_edit.py --scene input/black_tile_flooring.png --seg sam3 --illum heuristic
    python floor_edit/floor_edit.py --scene input/scene.jpeg --tta 3        # 3 augmented ensemble passes + majority vote

Pipeline
  1. floor segmentation      --seg   ensemble (SegFormer-B5 + UPerNet ConvNeXt-L, ADE20K)  [default] | sam3 (text prompt)
                              --tta N runs the ensemble N times on randomly augmented copies (mirror, noise, brightness /
                              contrast jitter) and takes a per-pixel majority vote (N=1: single pass, default)
  2. metric geometry          MoGe-2 ViT-L: point map + intrinsics
  3. illumination map         --illum rgbx (RGB->X diffuse irradiance, linear-in / gamma-decoded) [default] | heuristic
  4. render                   least-squares floor plane -> per-pixel metric floor coords -> texture tiled at --tile-m,
                              shaded by the illumination map (thin old grout lines / specks filtered out), composited
Output: outputs/floor_edit/<scene>/<pattern>/<seg>/output.jpg  (+ mask_floor.png, illumination.png, metrics.json;
        <seg> = ensemble | ensemble_tta<N> | sam3; with --tta also mask_vote_<i>.png per pass)

Weights (weights/): segformer-b5-ade, upernet-convnext-l, sam3, moge-2-vitl-normal, rgb-to-x.
Code deps (repos/): MoGe (moge.model.v2), rgbx (rgb2x diffusers pipeline).
"""
import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
WEIGHTS = ROOT / "weights"
REPOS = ROOT / "repos"
sys.path.insert(0, str(REPOS / "MoGe"))
sys.path.insert(0, str(REPOS))

DEV = torch.device("cuda" if torch.cuda.is_available() else "xpu" if torch.xpu.is_available() else "cpu")
FLOOR_CLASSES = {"floor", "rug"}          # ADE20K names counted as floor


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


class Stage:
    """Times a stage (device-synced) and records peak accelerator memory."""

    def __init__(self, log, name, model):
        self.log, self.row = log, dict(stage=name, model=model)

    def __enter__(self):
        sync()
        empty_cache()
        if DEV.type in ("cuda", "xpu"):
            getattr(torch, DEV.type).reset_peak_memory_stats()
        self.t0 = time.perf_counter()
        return self

    def __exit__(self, *exc):
        sync()
        self.row["gen_s"] = round(time.perf_counter() - self.t0, 3)
        self.row["gpu_peak_gb"] = round(getattr(torch, DEV.type).max_memory_allocated() / 1e9, 3) if DEV.type in ("cuda", "xpu") else 0.0
        self.log.append(self.row)
        print(f"  {self.row['stage']:<22s} {self.row['model']:<36s} {self.row['gen_s']:7.2f} s  {self.row['gpu_peak_gb']:5.2f} GB")


def timed(fn):
    t0 = time.perf_counter()
    r = fn()
    return r, round(time.perf_counter() - t0, 2)


# ============================================================================ 1. segmentation
def augment(image, rng, noise_std=0.02, jitter=0.08):
    """Random test-time augmentation. Returns (augmented PIL image, mirrored: bool, description)."""
    x = np.asarray(image).astype(np.float32) / 255.0
    desc = []
    mirrored = bool(rng.random() < 0.5)
    if mirrored:
        x = x[:, ::-1]
        desc.append("mirror")
    b = float(rng.uniform(-jitter, jitter))
    c = float(rng.uniform(1 - jitter, 1 + jitter))
    x = (x - 0.5) * c + 0.5 + b
    desc += [f"brightness{b:+.3f}", f"contrast x{c:.3f}"]
    sd = float(rng.uniform(0.0, noise_std))
    x = x + rng.normal(0.0, sd, x.shape).astype(np.float32)
    desc.append(f"noise sd={sd:.3f}")
    return Image.fromarray((np.clip(x, 0, 1) * 255).astype(np.uint8)), mirrored, ", ".join(desc)


def seg_ensemble(image, log, size_hw=(1024, 576), tta=1, seed=0):
    """SegFormer-B5 + UPerNet ConvNeXt-L mean-softmax ensemble. tta>1: that many passes on augmented copies, majority vote."""
    from transformers import (AutoImageProcessor, SegformerForSemanticSegmentation, SegformerImageProcessor,
                              UperNetForSemanticSegmentation)
    W, H = image.size
    if W > H:                                   # keep the model's 16:9-ish inference aspect for landscape inputs
        size_hw = (size_hw[1], size_hw[0])
    rng = np.random.default_rng(seed)

    def load():
        sp = SegformerImageProcessor.from_pretrained(str(WEIGHTS / "segformer-b5-ade"))
        sm = SegformerForSemanticSegmentation.from_pretrained(str(WEIGHTS / "segformer-b5-ade")).to(DEV).eval()
        up = AutoImageProcessor.from_pretrained(str(WEIGHTS / "upernet-convnext-l"))
        um = UperNetForSemanticSegmentation.from_pretrained(str(WEIGHTS / "upernet-convnext-l")).to(DEV).eval()
        return sp, sm, up, um
    (sp, sm, up, um), load_s = timed(load)
    names = {i: n.split(",")[0].strip() for i, n in sm.config.id2label.items()}
    fidx = [i for i, n in names.items() if n in FLOOR_CLASSES]

    def run(model, proc, img):
        inp = proc(images=img, return_tensors="pt", size={"height": size_hw[0], "width": size_hw[1]}).to(DEV)
        with torch.no_grad():
            logits = model(**inp).logits
        return torch.nn.functional.interpolate(logits, size=(H, W), mode="bilinear", align_corners=False)[0].softmax(0)

    votes, augs = [], []
    label = "SegFormer-B5 + UPerNet ConvNeXt-L" + (f", {tta}x TTA + majority vote" if tta > 1 else "")
    with Stage(log, "segmentation", label) as st:
        for _ in range(tta):
            img_i, mirrored, desc = (image, False, "none") if tta == 1 else augment(image, rng)
            seg = ((run(sm, sp, img_i) + run(um, up, img_i)) / 2).argmax(0).cpu().numpy()
            floor_i = np.isin(seg, fidx)
            votes.append(floor_i[:, ::-1] if mirrored else floor_i)
            augs.append(desc)
        votes = np.stack(votes)
        floor = votes.sum(0) >= int(np.ceil(tta / 2))            # simple majority (tta=1: the single pass)
    st.row["load_s"] = load_s
    if tta > 1:
        st.row["tta"] = tta
        st.row["augmentations"] = augs
        st.row["vote_iou_vs_majority"] = [round(float((v & floor).sum() / max((v | floor).sum(), 1)), 4) for v in votes]
        st.row["unanimous_pct"] = round(float(((votes.sum(0) == 0) | (votes.sum(0) == tta)).mean() * 100), 2)
    floor = cv2.morphologyEx(floor.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8)).astype(bool)
    n, lab, stats, _ = cv2.connectedComponentsWithStats(floor.astype(np.uint8), 8)
    floor = np.isin(lab, [i for i in range(1, n) if stats[i, cv2.CC_STAT_AREA] > 0.002 * H * W])
    del sm, um
    empty_cache()
    return floor, votes


def seg_sam3(image, log, threshold=0.4):
    from transformers import Sam3Model, Sam3Processor
    W, H = image.size
    (proc, model), load_s = timed(lambda: (Sam3Processor.from_pretrained(str(WEIGHTS / "sam3")),
                                           Sam3Model.from_pretrained(str(WEIGHTS / "sam3"), dtype=torch.float32).to(DEV).eval()))
    with Stage(log, "segmentation", "SAM 3 (text prompt 'floor')") as st:
        inp = proc(images=image, text="floor", return_tensors="pt").to(DEV)
        with torch.no_grad():
            o = model(**inp)
        res = proc.post_process_instance_segmentation(o, threshold=threshold, mask_threshold=0.5, target_sizes=[(H, W)])[0]
        m = res["masks"].cpu().numpy().astype(bool) if len(res["masks"]) else np.zeros((0, H, W), bool)
    st.row["load_s"] = load_s
    st.row["scores"] = np.round(res["scores"].cpu().numpy(), 3).tolist() if len(res["scores"]) else []
    del model
    empty_cache()
    floor = m.any(0) if len(m) else np.zeros((H, W), bool)
    return floor, floor[None]


SEGMENTERS = {"ensemble": seg_ensemble, "sam3": seg_sam3}


# ============================================================================ 2. geometry
def geometry(image, log):
    from moge.model.v2 import MoGeModel
    model, load_s = timed(lambda: MoGeModel.from_pretrained(str(WEIGHTS / "moge-2-vitl-normal" / "model.pt")).to(DEV).eval())
    x = torch.tensor(np.asarray(image) / 255.0, dtype=torch.float32, device=DEV).permute(2, 0, 1)
    with Stage(log, "geometry", "MoGe-2 ViT-L") as st:
        with torch.no_grad():
            o = model.infer(x, use_fp16=False)          # fp16 autocast is broken on XPU for this model
    st.row["load_s"] = load_s
    geo = dict(points=o["points"].cpu().numpy(), mask=o["mask"].cpu().numpy().astype(bool), intrinsics=o["intrinsics"].cpu().numpy())
    del model
    empty_cache()
    return geo


# ============================================================================ 3. illumination
def illum_heuristic(image, floor, log, ref_pct=90.0):
    """Luminance / unshadowed floor level. Thin-structure filtering happens in render()."""
    with Stage(log, "illumination", "heuristic (luminance ratio)"):
        lum = (np.asarray(image).astype(np.float32) / 255.0) @ np.array([0.299, 0.587, 0.114], np.float32)
        illum = np.clip(lum / max(np.percentile(lum[floor], ref_pct), 1e-3), 0.0, 1.15)
    return illum


def illum_rgbx(image, floor, log, steps=10, short=768):
    """RGB->X 'Irradiance (diffuse lighting)' (Zeng et al. 2024): linear-RGB input, gamma-decoded output."""
    from diffusers import DDIMScheduler
    from rgbx.rgb2x.pipeline_rgb2x import StableDiffusionAOVMatEstPipeline
    W, H = image.size
    sc = short / min(W, H)
    w, h = (int(W * sc) // 8) * 8, (int(H * sc) // 8) * 8
    photo = torch.from_numpy(np.asarray(image.resize((w, h), Image.LANCZOS)).astype(np.float32) / 255.0).permute(2, 0, 1) ** 2.2

    def load():
        pipe = StableDiffusionAOVMatEstPipeline.from_pretrained(str(WEIGHTS / "rgb-to-x"), torch_dtype=torch.float16).to(DEV)
        pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config, rescale_betas_zero_snr=True, timestep_spacing="trailing")
        pipe.set_progress_bar_config(disable=True)
        return pipe
    pipe, load_s = timed(load)
    with Stage(log, "illumination", f"RGB->X irradiance ({steps} steps)") as st:
        g = torch.Generator(device=DEV).manual_seed(0)
        irr = pipe(prompt="Irradiance (diffuse lighting)", photo=photo.to(DEV, torch.float16), num_inference_steps=steps,
                   generator=g, required_aovs=["irradiance"], output_type="pt").images[0][0].float().cpu()
        irr = torch.nn.functional.interpolate(irr[None], size=(H, W), mode="bicubic", align_corners=False)[0].clamp(0, 1)
        illum = (irr.permute(1, 2, 0).numpy() @ np.array([0.299, 0.587, 0.114], np.float32)) ** 2.2
    st.row["load_s"] = load_s
    del pipe
    empty_cache()
    return illum


ILLUMINATORS = {"rgbx": illum_rgbx, "heuristic": illum_heuristic}


# ============================================================================ 4. render
def fit_plane(pts):
    c = pts.mean(0)
    _, _, vt = np.linalg.svd(pts - c, full_matrices=False)
    n = vt[2]
    if n[1] > 0:                                  # OpenCV camera (y down): floor normal points up (-y)
        n = -n
    u = np.array([1.0, 0.0, 0.0]) - n * n[0]
    u /= np.linalg.norm(u)
    return c, n, u, np.cross(n, u)


def render(image, floor, geo, illum, tex, log, tile_m=0.6, line_px=21, blur=2.0, ref_pct=90.0):
    scene = np.asarray(image).astype(np.float32) / 255.0
    H, W = scene.shape[:2]
    K = geo["intrinsics"].copy()
    K[0] *= W
    K[1] *= H
    with Stage(log, "render", "plane fit + tiling + shading (CPU)") as st:
        # floor plane (2 rounds of inlier re-fit)
        pts = geo["points"][floor & geo["mask"]]
        c, n, u, v = fit_plane(pts)
        for _ in range(2):
            d = np.abs((pts - c) @ n)
            c, n, u, v = fit_plane(pts[d < np.percentile(d, 80)])
        resid_cm = float(np.median(np.abs((pts - c) @ n)) * 100)
        # per-pixel ray / plane intersection -> metric floor coordinates
        ys, xs = np.mgrid[0:H, 0:W]
        rays = np.stack([(xs - K[0, 2]) / K[0, 0], (ys - K[1, 2]) / K[1, 1], np.ones_like(xs, np.float64)], -1)
        denom = rays @ n
        tval = (c @ n) / np.where(np.abs(denom) < 1e-6, 1e-6, denom)
        rel = rays * tval[..., None] - c
        uu, vv = rel @ u, rel @ v
        # tiled texture
        th, tw = tex.shape[:2]
        flat = cv2.remap(tex, ((uu / tile_m) % 1.0 * (tw - 1)).astype(np.float32), ((vv / tile_m) % 1.0 * (th - 1)).astype(np.float32),
                         cv2.INTER_LINEAR, borderMode=cv2.BORDER_WRAP)
        # illumination: normalise on the floor, remove thin structures (old grout lines / specks), light blur
        shade = np.clip(illum / max(np.percentile(illum[floor], ref_pct), 1e-3), 0.0, 1.15).astype(np.float32)
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (line_px, line_px))
        shade = cv2.morphologyEx(cv2.morphologyEx(shade, cv2.MORPH_CLOSE, k), cv2.MORPH_OPEN, k)
        shade = cv2.GaussianBlur(shade, (0, 0), blur)
        # composite
        soft = cv2.GaussianBlur((floor & (tval > 0)).astype(np.float32), (0, 0), 1.0)[..., None]
        comp = scene * (1 - soft) + flat * shade[..., None] * soft
    st.row.update(plane_residual_cm=round(resid_cm, 2), tile_m=tile_m)
    return (comp.clip(0, 1) * 255).astype(np.uint8), (shade.clip(0, 1) * 255).astype(np.uint8)


# ============================================================================ main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scene", required=True)
    ap.add_argument("--pattern", default=str(ROOT / "input" / "pattern.png"))
    ap.add_argument("--seg", default="ensemble", choices=SEGMENTERS)
    ap.add_argument("--tta", type=int, default=1, help="ensemble only: number of augmented passes for a majority vote (1 = off)")
    ap.add_argument("--seed", type=int, default=0, help="seed for the TTA augmentations")
    ap.add_argument("--illum", default="rgbx", choices=ILLUMINATORS)
    ap.add_argument("--tile-m", type=float, default=0.6, help="physical size of one texture repeat on the floor (m)")
    ap.add_argument("--rgbx-steps", type=int, default=10)
    ap.add_argument("--line-px", type=int, default=21, help="structures thinner than this in the illumination are treated as texture")
    ap.add_argument("--out-root", default=str(ROOT / "outputs" / "floor_edit"))
    args = ap.parse_args()

    scene_p, pattern_p = Path(args.scene), Path(args.pattern)
    seg_name = args.seg + (f"_tta{args.tta}" if args.seg == "ensemble" and args.tta > 1 else "")
    out = Path(args.out_root) / scene_p.stem / pattern_p.stem / seg_name
    out.mkdir(parents=True, exist_ok=True)
    image = Image.open(scene_p).convert("RGB")
    tex = np.asarray(Image.open(pattern_p).convert("RGB")).astype(np.float32) / 255.0
    log = []
    print(f"floor_edit: {scene_p.name} + {pattern_p.name}  seg={seg_name} illum={args.illum}  device={DEV}")

    floor, votes = (seg_ensemble(image, log, tta=args.tta, seed=args.seed) if args.seg == "ensemble" else seg_sam3(image, log))
    if len(votes) > 1:
        print(f"    augmentations: {log[-1]['augmentations']}")
        print(f"    vote IoU vs majority: {log[-1]['vote_iou_vs_majority']}   unanimous pixels: {log[-1]['unanimous_pct']}%")
    geo = geometry(image, log)
    illum = (illum_rgbx(image, floor, log, steps=args.rgbx_steps) if args.illum == "rgbx" else illum_heuristic(image, floor, log))
    result, shade = render(image, floor, geo, illum, tex, log, tile_m=args.tile_m, line_px=args.line_px)

    Image.fromarray(result).save(out / "output.jpg", quality=95)
    Image.fromarray((floor * 255).astype(np.uint8)).save(out / "mask_floor.png")
    if len(votes) > 1:
        for i, v in enumerate(votes):
            Image.fromarray((v * 255).astype(np.uint8)).save(out / f"mask_vote_{i}.png")
    Image.fromarray(shade).save(out / "illumination.png")
    meta = dict(scene=str(scene_p), pattern=str(pattern_p), seg=seg_name, tta=args.tta, seed=args.seed, illum=args.illum, tile_m=args.tile_m,
                device=str(DEV), floor_coverage_pct=round(float(floor.mean() * 100), 1),
                total_gen_s=round(sum(r["gen_s"] for r in log), 2), stages=log)
    json.dump(meta, open(out / "metrics.json", "w"), indent=2)
    print(f"  total {meta['total_gen_s']:.2f} s (excl. model loads)  ->  {out / 'output.jpg'}")


if __name__ == "__main__":
    main()
