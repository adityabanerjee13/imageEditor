"""floor_edit — replace the floor in a room photo with a tiled texture, keeping the scene's lighting.

    python floor_edit/floor_edit.py --scene input/scene.jpeg --pattern input/pattern.png
    python floor_edit/floor_edit.py --scene input/black_tile_flooring.png --seg sam3 --illum heuristic
    python floor_edit/floor_edit.py --scene input/scene.jpeg --tta 3        # 3 augmented ensemble passes + majority vote

Pipeline
  1. floor segmentation      --seg   ensemble (SegFormer-B5 + UPerNet ConvNeXt-L, ADE20K)  [default] | sam3 (text prompt)
                              --tta N runs the ensemble N times on randomly augmented copies (mirror, noise, brightness /
                              contrast jitter) and takes a per-pixel majority vote (N=1: single pass, default)
  2. metric geometry          MoGe-2 ViT-L: point map + intrinsics
  3. irradiance field         --illum rgbx (RGB->X diffuse irradiance) [default] | heuristic
  4. render                   least-squares floor plane -> per-pixel metric floor coords -> texture tiled at --tile-m,
                              shaded by the irradiance field (old grout lines / specks filtered out), composited
Output: outputs/floor_edit/<scene>/<pattern>/<seg>/<material>/output.jpg  (+ mask_floor.png, irradiance.png, metrics.json;
        <seg> = ensemble | ensemble_tta<N> | sam3; with --tta also mask_vote_<i>.png per pass)

Weights (weights/): segformer-b5-ade, upernet-convnext-l, sam3, moge-2-vitl-normal, rgb-to-x.
Code deps (repos/): MoGe (moge.model.v2), rgbx (rgb2x diffusers pipeline).
"""
import argparse
import hashlib
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


class StageLog(list):
    """A stage list that notifies `on_append(row)` whenever a stage finishes (used by the server for progress)."""

    def __init__(self, on_append=None):
        super().__init__()
        self.on_append = on_append

    def append(self, row):
        super().append(row)
        if self.on_append:
            self.on_append(row)


def timed(fn):
    t0 = time.perf_counter()
    r = fn()
    return r, round(time.perf_counter() - t0, 2)


# ============================================================================ radiometry
# Gamma 2.2 (not the piecewise sRGB curve) in both directions, matching rgb2x's own convention
# on its input and on its gamma-corrected AOVs (repos/rgbx/rgb2x/load_image.py:88).
GAMMA = 2.2
LUMA = np.array([0.2126, 0.7152, 0.0722], np.float32)      # Rec.709 luminance of *linear* RGB


def to_linear(x):
    return np.clip(x, 0.0, None).astype(np.float32) ** GAMMA


def to_srgb(x):
    return np.clip(x, 0.0, None).astype(np.float32) ** (1.0 / GAMMA)


def luminance(x):
    """Linear luminance of an (H, W, 3) linear-RGB array."""
    return x @ LUMA


def shoulder(x, knee=0.8):
    """Reinhard-style rolloff above `knee`: sun patches compress toward 1 instead of clipping flat."""
    return np.where(x < knee, x, knee + (1.0 - knee) * (1.0 - np.exp(-(x - knee) / max(1.0 - knee, 1e-6))))


def guided_filter(guide, src, r, eps):
    """He et al. edge-preserving filter (cv2.ximgproc is absent from plain opencv-python).

    Smooths `src` while keeping the discontinuities of `guide`, so RGB->X noise goes but shadow
    boundaries stay sharp -- unlike the fixed-radius morphology it replaces, which also ate the
    narrow shadows under chair legs.
    """
    if src.ndim == 3:
        return np.stack([guided_filter(guide, src[..., i], r, eps) for i in range(src.shape[2])], -1)
    k = (2 * int(r) + 1, 2 * int(r) + 1)
    box = lambda x: cv2.boxFilter(x.astype(np.float32), -1, k, normalize=True, borderType=cv2.BORDER_REFLECT)
    mg, ms = box(guide), box(src)
    a = (box(guide * src) - mg * ms) / (box(guide * guide) - mg * mg + eps)
    return box(a) * guide + box(ms - a * mg)


# ============================================================================ materials
# One BRDF, two parameter sets:  L = (1-F)*rho*E_n + F*L_env,  F = Schlick(ior), roughness-capped.
# `legacy` is undocumented: it only reproduces the pre-PBR output so ablations/*.csv stay comparable.
# Both presets share f0: a dielectric interface reflects ~4% at normal incidence whatever its
# roughness, so the finish cannot change head-on reflectance -- only how wide the lobe is, and hence
# how fast reflectance climbs toward grazing. Giving glossy a *second* interface (a clearcoat over an
# already-specular base) would double-count that surface and make it visibly brighter underfoot.
# `coat` stays available via --coat for genuinely two-layer materials (waxed wood over open grain).
MATERIALS = {
    "smooth-matte":  dict(albedo=0.35, roughness=0.75, ior=1.50, coat=0.0, coat_roughness=0.03, delight=0.8),
    "smooth-glossy": dict(albedo=0.35, roughness=0.08, ior=1.50, coat=0.0, coat_roughness=0.03, delight=0.8),
    "legacy": None,
}


def resolve_material(name, **overrides):
    """Preset + explicit overrides (None keeps the preset value). Returns None for `legacy`."""
    if MATERIALS[name] is None:
        return None
    m = dict(MATERIALS[name], name=name)
    m.update({k: v for k, v in overrides.items() if v is not None})
    m["f0"] = ((m["ior"] - 1.0) / (m["ior"] + 1.0)) ** 2
    return m


def fresnel(cos_v, f0, roughness):
    """Schlick, capped by roughness (Lagarde's Fresnel-roughness form for a constant environment).

    A smooth surface reaches ~1 at grazing incidence; a rough one tops out near `1 - roughness`,
    because its wide lobe loses energy to microfacet masking. This is what makes smooth-matte read
    as a soft haze toward the horizon and smooth-glossy as a hard brightening, from the same f0.
    """
    return f0 + (max(1.0 - roughness, f0) - f0) * (1.0 - cos_v) ** 5


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


def segment(image, log, seg="ensemble", tta=1, seed=0, cache=True):
    """Floor mask, cached per (scene, segmenter, tta, seed)."""
    path = _cache_path("seg", _scene_key(image, seg, tta, seed))
    if cache and path.exists():
        with Stage(log, "segmentation", f"{seg} (cached)") as st:
            z = np.load(path)
            floor, votes = z["floor"].astype(bool), [v.astype(bool) for v in z["votes"]]
        st.row["cached"] = True
        return floor, votes
    floor, votes = (seg_ensemble(image, log, tta=tta, seed=seed) if seg == "ensemble"
                    else seg_sam3(image, log))
    if cache:
        np.savez_compressed(path, floor=floor, votes=np.asarray(votes))
    return floor, votes


# ============================================================================ 2. geometry
def geometry(image, floor, log, cache=True):
    """MoGe-2 point map + intrinsics, then the least-squares floor plane (2 inlier re-fits).

    The plane is fitted here rather than in render() because the irradiance stage needs it for
    ambient occlusion, and stage order is segmentation -> geometry -> irradiance -> render.
    """
    path = _cache_path("moge", _scene_key(image))
    cached = cache and path.exists()
    if cached:
        with Stage(log, "geometry", "MoGe-2 ViT-L (cached)") as st:
            z = np.load(path)
            geo = dict(points=z["points"], mask=z["mask"].astype(bool), intrinsics=z["intrinsics"])
            W, H = image.size
    else:
        from moge.model.v2 import MoGeModel
        model, load_s = timed(lambda: MoGeModel.from_pretrained(str(WEIGHTS / "moge-2-vitl-normal" / "model.pt")).to(DEV).eval())
        x = torch.tensor(np.asarray(image) / 255.0, dtype=torch.float32, device=DEV).permute(2, 0, 1)
        with Stage(log, "geometry", "MoGe-2 ViT-L") as st:
            with torch.no_grad():
                o = model.infer(x, use_fp16=False)      # fp16 autocast is broken on XPU for this model
            geo = dict(points=o["points"].cpu().numpy().astype(np.float32),
                       mask=o["mask"].cpu().numpy().astype(bool),
                       intrinsics=o["intrinsics"].cpu().numpy())
            W, H = image.size
        st.row["load_s"] = load_s
        del model
        empty_cache()
        if cache:
            np.savez_compressed(path, **geo)

    # The plane depends on the floor mask as well as the point map, and an SVD over the masked
    # points is a fraction of a second, so it is refitted rather than cached.
    K = geo["intrinsics"].copy()
    K[0] *= W
    K[1] *= H
    pts = geo["points"][floor & geo["mask"]]
    c, n, u, v = fit_plane(pts)
    for _ in range(2):
        d = np.abs((pts - c) @ n)
        c, n, u, v = fit_plane(pts[d < np.percentile(d, 80)])
    geo["K"], geo["plane"] = K, (c, n, u, v)
    st.row["plane_residual_cm"] = round(float(np.median(np.abs((pts - c) @ n)) * 100), 2)
    return geo


# ============================================================================ 3. irradiance
# Stage A -- recover the light field E the *old* floor received, so the new one can be given the
# same one:  A.1 recovery (E = L_old / rho_old) -> A.2 edge-aware refinement -> A.3 ambient/direct
# split with chromaticity and occlusion -> A.4 anchoring.  A.2-A.4 live in shading_field(), called
# from render(); the illuminators below only do A.1 and return *linear, per-channel, un-normalised*
# irradiance.  (The old code returned a normalised scalar from one illuminator and an un-normalised
# one from the other, then normalised again in render().)
NTSC = np.array([0.299, 0.587, 0.114], np.float32)     # luma weights of the pre-PBR path, kept for `legacy`


def _cache_path(kind, key):
    """Scene-keyed disk cache. Everything before the render depends only on the input photo (and,
    for the mask, on the segmenter), so switching pattern, material or tile size should cost the
    render alone -- not another 20 s of SegFormer + MoGe + diffusion."""
    d = ROOT / "data" / "cache" / "floor_edit"
    d.mkdir(parents=True, exist_ok=True)
    return d / f"{kind}_{key}.npz"


def _scene_key(image, *parts):
    h = hashlib.sha1(np.asarray(image).tobytes()).hexdigest()[:16]
    return "_".join([h] + [str(x) for x in parts])


def illum_heuristic(image, floor, log, **kw):
    """Linear scene radiance, assuming a *uniform* old floor albedo.

    It cannot separate reflectance from light, so a dark old floor reads as deep shadow -- which is
    exactly what turned the black-tile scene black. Kept as the no-model fallback; --illum rgbx
    divides the albedo out instead. Callers pass grout_px for this path, since there is no albedo
    estimate to cancel the old tile joints.
    """
    with Stage(log, "irradiance", "heuristic (linear radiance)"):
        srgb = np.asarray(image).astype(np.float32) / 255.0
        E = to_linear(srgb)
        lum = srgb @ NTSC
        legacy = np.clip(lum / max(np.percentile(lum[floor], 90.0), 1e-3), 0.0, 1.15)
    return dict(E=E, albedo=None, legacy=legacy, refine=True)


def illum_rgbx(image, floor, log, steps=10, short=768, cache=True, divide=False):
    """RGB->X (Zeng et al. 2024): the irradiance AOV as the light field, plus albedo for reference.

    Two AOVs means two DDIM loops -- they are different prompts, so neither can be derived from the
    other -- but the pair depends only on the scene, so it is cached to disk at the resolution the
    model actually produced. Re-running the same room with another pattern or material then skips
    the diffusion entirely.

    The irradiance AOV contains no reflectance by construction, so it carries no grout lines and
    needs no morphological filtering. `divide` instead estimates E = L_old / rho_old, which is
    sharper in principle but fragile: see the note below.
    """
    W, H = image.size
    sc = short / min(W, H)
    w, h = (int(W * sc) // 8) * 8, (int(H * sc) // 8) * 8
    path = _cache_path("aov2", _scene_key(image, steps, short))
    if cache and path.exists():
        with Stage(log, "irradiance", "RGB->X irradiance + albedo (cached)") as st:
            z = np.load(path)
            irr, alb = z["irr"], z["alb"]
        st.row["cached"] = True
    else:
        from diffusers import DDIMScheduler
        from rgbx.rgb2x.pipeline_rgb2x import StableDiffusionAOVMatEstPipeline
        photo = torch.from_numpy(np.asarray(image.resize((w, h), Image.LANCZOS)).astype(np.float32) / 255.0).permute(2, 0, 1) ** GAMMA

        def load():
            pipe = StableDiffusionAOVMatEstPipeline.from_pretrained(str(WEIGHTS / "rgb-to-x"), torch_dtype=torch.float16).to(DEV)
            pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config, rescale_betas_zero_snr=True, timestep_spacing="trailing")
            pipe.set_progress_bar_config(disable=True)
            return pipe
        pipe, load_s = timed(load)

        def aov(prompt, name):
            g = torch.Generator(device=DEV).manual_seed(0)
            x = pipe(prompt=prompt, photo=photo.to(DEV, torch.float16), num_inference_steps=steps,
                     generator=g, required_aovs=[name], output_type="pt").images[0][0].float().cpu()
            return x.permute(1, 2, 0).numpy().astype(np.float32)   # native (h, w, 3), still gamma-encoded

        with Stage(log, "irradiance", f"RGB->X irradiance + albedo ({steps} steps x2)") as st:
            irr = aov("Irradiance (diffuse lighting)", "irradiance")
            alb = aov("Albedo (diffuse basecolor)", "albedo")
        st.row["load_s"] = load_s
        del pipe
        empty_cache()
        if cache:
            np.savez_compressed(path, irr=irr, alb=alb)

    def upsample(x, clamp01):
        t = torch.from_numpy(np.ascontiguousarray(x)).permute(2, 0, 1)[None]
        t = torch.nn.functional.interpolate(t, size=(H, W), mode="bicubic", align_corners=False)[0]
        return (t.clamp(0, 1) if clamp01 else t.clamp_min(0)).permute(1, 2, 0).numpy()

    # Both AOVs are gamma-corrected on the way out (pipeline_rgb2x.py:802) -- decode to linear.
    E = irr.astype(np.float32) ** GAMMA
    if divide:
        # E = L_old / rho_old keeps sharper cast shadows, but it must be done at the AOVs' own
        # resolution. At full resolution the photo carries detail the upsampled albedo has lost, so
        # the old grout lines no longer cancel (they survive as a residual and show through the new
        # texture) and at every object boundary the blurred albedo collapses while the photo is
        # still bright floor, ringing the furniture with a bright halo. It is also ill-conditioned
        # on a dark floor: on the kitchen scene the old tile's albedo has a median of 0.026, and
        # dividing by that amplifies noise into a 100:1 "irradiance" range across a flat floor.
        alb_lin = alb.astype(np.float32) ** GAMMA
        small = to_linear(np.asarray(image.resize((w, h), Image.LANCZOS)).astype(np.float32) / 255.0)
        weak = (luminance(alb_lin) < 0.05)[..., None]
        E = np.where(weak, E, small / np.maximum(alb_lin, 0.05))
    # The AOV is produced at `short` px, so upsampled it is already band-limited and needs no
    # denoising. Refining it would be worse than useless: the guided filter's guide is the scene
    # luminance, which still contains the *old* floor, so any structure it transfers is old tile
    # joints reappearing through the new texture.
    return dict(E=upsample(E, False), albedo=upsample(alb, True) ** GAMMA, refine=divide,
                legacy=((upsample(irr, True) @ NTSC) ** GAMMA))


ILLUMINATORS = {"rgbx": illum_rgbx, "heuristic": illum_heuristic}


def floor_ao(geo, floor, max_m=0.30, min_h=0.02, dirs=8, steps=12, min_px=1.5, scale=2):
    """Wrapper: run the march on a `scale`-subsampled frame and bilinearly upsample the result.

    The march costs dirs*steps random gathers per candidate pixel, which is ~7 s at 900x1600 and by
    far the most expensive thing in the render. V_amb multiplies the *ambient* term, which is smooth
    by construction, so half resolution changes it by well under a JPEG quantisation step while
    costing a quarter as much. Small frames (under 512 px) run at full resolution.
    """
    H, W = floor.shape
    if scale > 1 and min(H, W) >= 512:
        sl = slice(None, None, scale)
        K = geo["K"].copy()
        K[0, 0] /= scale; K[0, 2] /= scale
        K[1, 1] /= scale; K[1, 2] /= scale
        small = dict(points=geo["points"][sl, sl], mask=geo["mask"][sl, sl], K=K, plane=geo["plane"])
        v = _floor_ao_march(small, floor[sl, sl], max_m, min_h, dirs, steps, min_px)
        return cv2.resize(v, (W, H), interpolation=cv2.INTER_LINEAR)
    return _floor_ao_march(geo, floor, max_m, min_h, dirs, steps, min_px)


def _floor_ao_march(geo, floor, max_m=0.30, min_h=0.02, dirs=8, steps=12, min_px=1.5):
    """Horizon-based ambient occlusion on the floor plane, from the MoGe-2 point map.

    For every floor pixel we step `dirs` azimuths outward *in metres on the plane*, project each
    sample back through K, and read whatever 3-D point sits there; anything standing above the plane
    raises the horizon in that azimuth and removes part of the ambient hemisphere.

    Radii stop at `max_m` (30 cm) on purpose: E already contains the scene's real cast shadows, so
    this must only *add* the tight contacts a 768 px irradiance map cannot resolve, not re-darken
    shadows we already have. Returns V_amb in [0,1], which multiplies the ambient term only.

    Two things stop it over-darkening. The elevation is measured against the sample's *true*
    horizontal distance from the pixel, not the step length -- a point read off a distant wall is
    far away, and treating it as 30 cm away would put the horizon near vertical. And far from the
    camera a 30 cm step is sub-pixel, so the sample would be the source pixel itself and depth noise
    alone would read as an occluder; samples closer than `min_px` in the image are skipped.
    """
    c, n, u, v = (np.asarray(a, np.float32) for a in geo["plane"])
    K, P = geo["K"], geo["points"]
    H, W = floor.shape
    pts, valid = P.reshape(-1, 3).astype(np.float32), geo["mask"].ravel()
    V = np.ones(H * W, np.float32)
    if not floor.any():
        return V.reshape(H, W)
    # Only floor within reach of something that stands above the plane can be occluded at all; the
    # open floor is V=1 and would otherwise cost dirs*steps gathers per pixel for nothing.
    occ = ((pts @ n - c @ n) > min_h).reshape(H, W) & geo["mask"]
    near_z = max(float(np.percentile(P[..., 2][floor], 2)), 1e-3)
    rad_px = int(np.clip(max_m * K[0, 0] / near_z, 3, 200))
    cand = floor & (cv2.dilate(occ.astype(np.uint8), cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (2 * rad_px + 1, 2 * rad_px + 1))) > 0)
    idx = np.flatnonzero(cand.ravel())
    if idx.size == 0:
        return V.reshape(H, W)

    p = pts[idx]
    pz = np.maximum(p[:, 2], 1e-6)
    px0, py0 = p[:, 0] / pz * K[0, 0] + K[0, 2], p[:, 1] / pz * K[1, 1] + K[1, 2]
    tmax = np.zeros((dirs, idx.size), np.float32)
    for j in range(dirs):
        phi = 2.0 * np.pi * j / dirs
        d = np.cos(phi) * u + np.sin(phi) * v
        for rr in np.linspace(max_m / steps, max_m, steps):
            q = p + d * rr
            z = np.maximum(q[:, 2], 1e-6)
            fx, fy = q[:, 0] / z * K[0, 0] + K[0, 2], q[:, 1] / z * K[1, 1] + K[1, 2]
            xs, ys = np.rint(fx).astype(np.int32), np.rint(fy).astype(np.int32)
            ok = (np.hypot(fx - px0, fy - py0) > min_px) & (xs >= 0) & (xs < W) & (ys >= 0) & (ys < H)
            f = np.clip(ys, 0, H - 1) * W + np.clip(xs, 0, W - 1)
            rel = pts[f] - p
            h = rel @ n                                              # height above this floor pixel
            horiz = np.maximum(np.linalg.norm(rel - h[:, None] * n, axis=1), 1e-3)
            tmax[j] = np.maximum(tmax[j], np.where(ok & valid[f] & (h > min_h), h / horiz, 0.0))
    V[idx] = 1.0 - np.mean(tmax / np.sqrt(1.0 + tmax * tmax), axis=0)   # mean sin(horizon elevation)
    return V.reshape(H, W)


def shading_field(E, scene_lin, floor, geo, ambient=0.0, ao=True, ao_strength=1.0, grout_px=0,
                  refine=True, grout_rel=0.02):
    """Stages A.2-A.4: refine E, split it into ambient + direct, recolour each, anchor it.

    Returns E_n whose luminance averages 1 over the floor, so the rendered floor's mean radiance is
    its albedo -- the new floor receives exactly the irradiance the old one did.

    `ambient` is the fraction of floor irradiance that survives full occlusion. It defaults to 0
    (all light treated as direct), which gives deep, contrasty shadows. Passing None estimates it
    from the scene's own deepest shadow instead, on the reasoning that a matte floor in shadow still
    sees the ceiling and wall hemisphere; that is physically truer but visibly lifts the blacks.
    """
    H, W = floor.shape
    # A.2 -- edge-aware refinement. Replaces the 21 px close->open, which was resolution-dependent
    # and removed genuine narrow shadows along with the old grout lines.
    E = E.astype(np.float32)
    if refine:
        guide = luminance(scene_lin).astype(np.float32)
        E = guided_filter(guide, E, max(3, int(round(0.02 * min(H, W)))), 1e-3)
    if grout_px:
        # RGB->X's irradiance AOV is NOT albedo-free in practice: it bakes the old floor's tile
        # joints into the "lighting", and since the shading is the only route old-floor content has
        # into the composite, those joints show straight through the new texture. Closing removes
        # thin *dark* structures and opening thin *bright* ones, but applying the result everywhere
        # also rounds off broad shadows -- so substitute only where the morphology actually moved
        # the value, leaving cast shadows bit-identical.
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (grout_px, grout_px))
        flat = cv2.morphologyEx(cv2.morphologyEx(E, cv2.MORPH_CLOSE, k), cv2.MORPH_OPEN, k)
        thin = np.abs(flat - E) > grout_rel * np.maximum(E, 1e-3)
        E = np.where(thin, flat, E)
    E = np.maximum(E, 1e-5)

    # A.4 -- anchor on the mean, not p90: a mostly-shadowed floor puts p90 inside shadow (whole
    # floor too bright), a sunlit patch puts it in the highlight (whole floor too dark).
    ref = float(luminance(E)[floor].mean())
    E_n = E / max(ref, 1e-5)
    Y = luminance(E_n)
    yf = Y[floor]

    # A.3 -- split into an ambient floor and a direct term. When estimated (ambient=None), the
    # deepest shadow is taken as the point where the direct term is ~0, so whatever is left there
    # *is* the ambient level.
    a = float(ambient) if ambient is not None else float(
        np.clip(np.percentile(yf, 3) / max(np.percentile(yf, 95), 1e-5), 0.05, 0.6))
    S = np.clip((Y - a) / max(1.0 - a, 1e-5), 0.0, None)

    # chromaticity of each term, from the darkest / brightest floor quartiles: shadows carry the
    # cool bounced ambient, lit areas the warm lamp. Normalised so luminance is unchanged.
    C = E_n / np.maximum(Y, 1e-5)[..., None]
    lo, hi = np.percentile(yf, 25), np.percentile(yf, 75)

    def chroma(sel):
        if sel.sum() < 50:
            return np.ones(3, np.float32)
        m = C[sel].mean(0).astype(np.float32)
        return m / max(float(m @ LUMA), 1e-5)
    C_amb, C_dir = chroma(floor & (Y <= lo)), chroma(floor & (Y >= hi))

    V = np.ones((H, W), np.float32)
    if ao:
        V = 1.0 - ao_strength * (1.0 - floor_ao(geo, floor))
    amb, dir_ = a * V, (1.0 - a) * S
    if a <= 1e-6:
        # With no ambient term there is nothing for occlusion to remove, so the contact shadows
        # would silently vanish. Apply them to the direct term instead: at radii under 30 cm this
        # is still adding detail the 768 px irradiance map cannot resolve, not re-darkening the
        # cast shadows E already carries.
        dir_ = dir_ * V
    out = amb[..., None] * C_amb + dir_[..., None] * C_dir
    info = dict(grout_filtered_pct=round(float(thin[floor].mean() * 100), 2) if grout_px else 0.0,
                ambient_fraction=round(a, 4), ref_irradiance=round(ref, 5),
                ao_min=round(float(V[floor].min()), 3),
                amb_chroma=[round(float(x), 3) for x in C_amb],
                dir_chroma=[round(float(x), 3) for x in C_dir])
    return out.astype(np.float32), V, info


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


def floor_rays(geo, H, W):
    """Per-pixel ray / plane intersection -> metric floor coordinates (u, v) and the ray depth."""
    K, (c, n, u, v) = geo["K"], geo["plane"]
    ys, xs = np.mgrid[0:H, 0:W]
    rays = np.stack([(xs - K[0, 2]) / K[0, 0], (ys - K[1, 2]) / K[1, 1], np.ones_like(xs, np.float64)], -1)
    denom = rays @ n
    tval = (c @ n) / np.where(np.abs(denom) < 1e-6, 1e-6, denom)
    rel = rays * tval[..., None] - c
    return rays, tval, rel @ u, rel @ v


# ---------------------------------------------------------------- stage B: material
def _blur_lowfreq(x, sigma):
    """Wide Gaussian via downsample -> blur -> upsample. A sigma of ~0.15x the side means a ~900 px
    kernel, which costs 1.8 s at full resolution; the pyramid is ~25x faster and differs by <0.01,
    which is irrelevant for a low-frequency estimate that is renormalised straight afterwards."""
    sc = max(1, int(sigma / 4))
    if sc == 1:
        return cv2.GaussianBlur(x, (0, 0), sigma)
    small = cv2.resize(x, (max(x.shape[1] // sc, 1), max(x.shape[0] // sc, 1)), interpolation=cv2.INTER_AREA)
    return cv2.resize(cv2.GaussianBlur(small, (0, 0), sigma / sc), (x.shape[1], x.shape[0]),
                      interpolation=cv2.INTER_LINEAR)


M_PER_1000PX = 5.0            # how much floor 1000 px of pattern covers, when nothing says otherwise


def default_tile_m(tex, m_per_1000px=M_PER_1000PX):
    """Repeat width (m) for a pattern whose physical size nobody told us.

    A pattern PNG carries no scale, so the default reads one off its pixel count: at the default
    `m_per_1000px` a 1000 px wide image becomes a 5 m repeat and a 500 px one 2.5 m. Bigger scans
    are assumed to cover more floor rather than to be the same slab at higher resolution, which is
    what a texture library actually does. Set `m_per_1000px` (CLI `--m-per-1000px`, UI "1000 px =")
    to say what the scale really is, or `--tile-m` to give the repeat width outright.
    """
    return tex.shape[1] / 1000.0 * m_per_1000px


def tile_extent(tex, tile_m=None):
    """Metric footprint (along u, along v) of one texture repeat on the floor.

    `tile_m` sizes the repeat along the texture's width; the other axis follows the image's own
    aspect ratio so the texels stay square on the floor. Mapping both axes to `tile_m` squeezes any
    non-square pattern into a square footprint, which visibly distorts veining in a 2:1 slab.
    `None` falls back to `default_tile_m` at its own default scale, so the height and width both
    come from the image; callers that let the user set the scale resolve `tile_m` before this.
    """
    th, tw = tex.shape[:2]
    if tile_m is None:
        tile_m = default_tile_m(tex)
    return tile_m, tile_m * (th / tw)


def tile_uv(tex, uu, vv, tile_m=None, offset=(0.5, 0.5)):
    """Floor coordinates (m) -> texture sample coordinates (px). Layout only: this is where the
    grid sits and how big it is, and it has nothing to do with what the surface is made of.

    (uu, vv) are measured from the fitted plane's centroid, which sits near the middle of the
    visible floor -- so without a phase shift `0 % 1 == 0` puts a tile *corner* there and a joint
    runs through the centre of frame, cutting the nearest slab in half. The default half-tile
    offset centres a whole slab on the floor centroid instead.
    """
    th, tw = tex.shape[:2]
    su, sv = tile_extent(tex, tile_m)
    return (((uu / su + offset[0]) % 1.0 * (tw - 1)).astype(np.float32),
            ((vv / sv + offset[1]) % 1.0 * (th - 1)).astype(np.float32))


def texture_albedo(tex, material):
    """Texture photo -> reflectance. Material only: no geometry, no tiling.

    A pattern PNG is a photograph: it carries its own baked lighting and its own exposure.
    De-lighting divides out its low frequencies so only reflectance variation is left, then the mean
    is set to the material's albedo -- so floor lightness is a stated physical property rather than
    whatever the source image happened to be exposed at, and the scene's shading is not multiplied
    by the texture's own shading.
    """
    t = to_linear(tex)
    d = material["delight"]
    if d > 0:
        low = _blur_lowfreq(t, 0.15 * min(t.shape[:2]))
        flat = t / np.maximum(low, 1e-4) * float(luminance(low).mean())
        t = (1.0 - d) * t + d * flat
    # Clipping to a physical reflectance range pulls the mean back down, so re-solve for the scale
    # a few times; a very contrasty texture at a high target albedo may not reach it exactly, which
    # is why the achieved value is reported in metrics.json rather than assumed.
    for _ in range(4):
        t = np.clip(t * (material["albedo"] / max(float(luminance(t).mean()), 1e-4)), 0.01, 0.95)
    return t.astype(np.float32)


def planar_reflection(scene_lin, geo, floor, roughness, min_h=0.02):
    """What the floor reflects, as a per-pixel radiance map.

    The floor is a known plane and MoGe gives us the whole point cloud, so the reflection is exact
    rather than a screen-space guess: mirror every above-floor point about the plane and re-project
    it through the same camera. That is what a viewer sees in a mirror floor, with the perspective
    stretch toward the horizon falling out of the projection for free -- no ray march, no depth-test
    bias, no step count.

    Returns (radiance, confidence). Confidence is 0 where nothing in frame reflects to that pixel
    (the mirrored ray would leave the image), so the caller can fade to a constant environment.
    Like any screen-space method this can only reflect geometry the photo actually contains.
    """
    c, n, u, v = geo["plane"]
    K, P = geo["K"], geo["points"]
    H, W = floor.shape
    h = (P - c) @ n                                  # height above the floor plane
    sel = geo["mask"] & ~floor & (h > min_h)         # only things standing on the floor reflect
    if sel.sum() < 100:
        return None, None

    pts, col = P[sel], scene_lin[sel]
    mirrored = pts - 2.0 * h[sel][:, None] * n       # reflect the geometry, not the camera
    z = mirrored[:, 2]
    keep = z > 1e-3
    mirrored, col, z = mirrored[keep], col[keep], z[keep]
    x = np.rint(mirrored[:, 0] / z * K[0, 0] + K[0, 2]).astype(np.int32)
    y = np.rint(mirrored[:, 1] / z * K[1, 1] + K[1, 2]).astype(np.int32)
    inb = (x >= 0) & (x < W) & (y >= 0) & (y < H)
    x, y, col, z = x[inb], y[inb], col[inb], z[inb]
    if x.size == 0:
        return None, None

    # Z-buffer by scattering far-to-near, so the nearest mirrored surface is written last and wins.
    flat = (y * W + x)[np.argsort(-z)]
    refl = np.zeros((H * W, 3), np.float32)
    hit = np.zeros(H * W, np.float32)
    refl[flat] = col[np.argsort(-z)]
    hit[flat] = 1.0
    refl, hit = refl.reshape(H, W, 3), hit.reshape(H, W)

    # The scatter is sparse, and a real surface blurs its reflection by its own roughness. One
    # normalised convolution does both: it fills the gaps between scattered points and applies the
    # roughness blur. A rough floor blurs so wide that the result approaches the average of the
    # room -- which is exactly the constant environment this replaces, so both materials can use it.
    sigma = float(np.clip(roughness * 0.05 * min(H, W), 2.0, 80.0))
    wb = cv2.GaussianBlur(hit, (0, 0), sigma)
    rb = cv2.GaussianBlur(refl, (0, 0), sigma)
    filled = rb / np.maximum(wb, 1e-4)[..., None]
    ref = max(float(np.percentile(wb[wb > 0], 75)), 1e-4)
    return filled.astype(np.float32), np.clip(wb / ref, 0.0, 1.0).astype(np.float32)


def environment_colour(scene_lin, geo, floor, min_h=0.3):
    """Mean linear radiance of everything standing above the floor plane: a single-value stand-in for
    the upper hemisphere the specular lobe reflects. Screen-space reflections would replace this with
    a real per-pixel lookup along the mirror ray."""
    c, n = geo["plane"][0], geo["plane"][1]
    above = geo["mask"] & ~floor & (((geo["points"] - c) @ n) > min_h)
    src = scene_lin[above] if above.sum() > 100 else scene_lin[~floor]
    return src.mean(0).astype(np.float32) if src.size else np.full(3, 0.1, np.float32)


# ---------------------------------------------------------------- stage C: render
def render(image, floor, geo, ill, tex, material, log, tile_m=None, ambient=0.0, ao=True,
           ao_strength=1.0, grout_px=0, knee=0.8, tile_offset=(0.5, 0.5), ssr=True):
    """L = (1-F)*rho*E_n + F*L_env, plus an optional clearcoat lobe, composited in linear light.

    Everything here happens on linear radiance and is encoded exactly once at the end. The pre-PBR
    path multiplied a linear shading ratio onto an sRGB-encoded texture, which applied the shadow
    factor twice over and made every shadow about 2x too dark.
    """
    scene_lin = to_linear(np.asarray(image).astype(np.float32) / 255.0)
    H, W = scene_lin.shape[:2]
    n = geo["plane"][1]
    with Stage(log, "render", f"{material['name']} BRDF + tiling (CPU)") as st:
        E_n, V, info = shading_field(ill["E"], scene_lin, floor, geo, ambient=ambient, ao=ao,
                                     ao_strength=ao_strength, grout_px=grout_px,
                                     refine=ill.get("refine", True))
        rays, tval, uu, vv = floor_rays(geo, H, W)
        rho = cv2.remap(texture_albedo(tex, material), *tile_uv(tex, uu, vv, tile_m, tile_offset),
                        cv2.INTER_LINEAR, borderMode=cv2.BORDER_WRAP)
        # Fresnel against the view direction. A matte floor is view-independent in its diffuse term,
        # but no real floor is a pure Lambertian: reflectance climbs from f0 toward 1 at grazing
        # incidence, which is why floors brighten toward the horizon.
        cos_v = np.abs((rays / np.linalg.norm(rays, axis=-1, keepdims=True)) @ n).astype(np.float32)
        env = environment_colour(scene_lin, geo, floor)
        if ssr:
            refl, conf = planar_reflection(scene_lin, geo, floor, material["roughness"])
            env = env if refl is None else conf[..., None] * refl + (1.0 - conf[..., None]) * env
        F = fresnel(cos_v, material["f0"], material["roughness"])[..., None]
        L = (1.0 - F) * rho * E_n + F * env          # (1-F) keeps the far floor from gaining energy
        if material["coat"] > 0:                     # a sealant layer over the base material
            Fc = material["coat"] * fresnel(cos_v, 0.04, material["coat_roughness"])[..., None]
            L = Fc * env + (1.0 - Fc) * L
        L = shoulder(L, knee)                        # compress highlights instead of clipping them
        soft = cv2.GaussianBlur((floor & (tval > 0)).astype(np.float32), (0, 0), 1.0)[..., None]
        comp = to_srgb(np.clip(scene_lin * (1.0 - soft) + L * soft, 0.0, 1.0))
        fmax = float(fresnel(np.float32(0.0), material["f0"], material["roughness"]))
        energy = (1.0 - fmax) * material["albedo"] + fmax        # a passive surface reflects <= 1
    st.row.update(tile_m=tile_m, tile_uv_m=[round(x, 4) for x in tile_extent(tex, tile_m)],
                  tile_offset=list(tile_offset), ssr=bool(ssr),
                  material=material["name"], albedo=material["albedo"],
                  albedo_achieved=round(float(luminance(rho[floor]).mean()), 4),
                  roughness=material["roughness"], ior=material["ior"], coat=material["coat"],
                  energy_max=round(float(energy), 4), **info)
    if energy > 1.0:
        print(f"    WARNING: energy_max {energy:.3f} > 1 - albedo {material['albedo']} is too high for ior {material['ior']}")
    shade = to_srgb(np.clip(luminance(E_n) / 1.5, 0.0, 1.0))     # display-encoded view of E_n
    return (comp * 255).astype(np.uint8), (shade * 255).astype(np.uint8), V


def render_legacy(image, floor, geo, illum, tex, log, tile_m=None, line_px=21, blur=2.0, ref_pct=90.0):
    """The pre-PBR path, kept only so ablations/*.csv stay comparable. Do not extend it.

    Known wrong: `shade` is a linear ratio multiplied onto an sRGB texture (double gamma), it can
    reach 0 (no ambient floor), and the morphological open/close erases narrow real shadows along
    with the old grout lines.
    """
    scene = np.asarray(image).astype(np.float32) / 255.0
    H, W = scene.shape[:2]
    with Stage(log, "render", "legacy (pre-PBR) shading") as st:
        _, tval, uu, vv = floor_rays(geo, H, W)
        th, tw = tex.shape[:2]      # square mapping on both axes, as the pre-PBR path did
        flat = cv2.remap(tex, ((uu / tile_m) % 1.0 * (tw - 1)).astype(np.float32), ((vv / tile_m) % 1.0 * (th - 1)).astype(np.float32),
                         cv2.INTER_LINEAR, borderMode=cv2.BORDER_WRAP)
        shade = np.clip(illum / max(np.percentile(illum[floor], ref_pct), 1e-3), 0.0, 1.15).astype(np.float32)
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (line_px, line_px))
        shade = cv2.morphologyEx(cv2.morphologyEx(shade, cv2.MORPH_CLOSE, k), cv2.MORPH_OPEN, k)
        shade = cv2.GaussianBlur(shade, (0, 0), blur)
        soft = cv2.GaussianBlur((floor & (tval > 0)).astype(np.float32), (0, 0), 1.0)[..., None]
        comp = scene * (1 - soft) + flat * shade[..., None] * soft
    st.row.update(tile_m=tile_m, material="legacy")
    return (comp.clip(0, 1) * 255).astype(np.uint8), (shade.clip(0, 1) * 255).astype(np.uint8), None


# ============================================================================ main
def run_floor_edit(scene, pattern, out_dir, seg="ensemble", tta=1, seed=0, illum="rgbx", tile_m=None,
                   m_per_1000px=M_PER_1000PX, rgbx_steps=10,
                   line_px=21, on_stage=None, preclean=False, material="smooth-matte", albedo=None, roughness=None,
                   ior=None, coat=None, delight=None, ambient=0.0, ao=True, ao_strength=1.0, cache=True,
                   albedo_divide=False, tile_offset=(0.5, 0.5), ssr=True):
    """Run the pipeline on `scene` with texture `pattern`; writes output.jpg, mask_floor.png, irradiance.png,
    metrics.json to `out_dir` and returns the metrics dict. `on_stage(row)` is called after every finished stage
    (server progress). Every shading argument is defaulted, so existing callers need no change."""
    scene_p, pattern_p, out = Path(scene), Path(pattern), Path(out_dir)
    seg_name = seg + (f"_tta{tta}" if seg == "ensemble" and tta > 1 else "")
    out.mkdir(parents=True, exist_ok=True)
    image = Image.open(scene_p).convert("RGB")
    tex = np.asarray(Image.open(pattern_p).convert("RGB")).astype(np.float32) / 255.0
    if tile_m is None:
        tile_m = default_tile_m(tex, m_per_1000px)
    mat = resolve_material(material, albedo=albedo, roughness=roughness, ior=ior, coat=coat, delight=delight)
    log = StageLog(on_stage)
    if preclean:                                  # DDRM identity pre-clean (the web UI does this on upload)
        sys.path.insert(0, str(ROOT))
        from object_edit.preclean import preclean as _pc
        image = Image.fromarray(_pc(np.asarray(image), log)); image.save(out / "input_preclean.png")
    print(f"floor_edit: {scene_p.name} + {pattern_p.name}  seg={seg_name} illum={illum} material={material}  device={DEV}")

    floor, votes = segment(image, log, seg=seg, tta=tta, seed=seed, cache=cache)
    if len(votes) > 1:
        print(f"    augmentations: {log[-1]['augmentations']}")
        print(f"    vote IoU vs majority: {log[-1]['vote_iou_vs_majority']}   unanimous pixels: {log[-1]['unanimous_pct']}%")
    geo = geometry(image, floor, log, cache=cache)
    ill = (illum_rgbx(image, floor, log, steps=rgbx_steps, cache=cache, divide=albedo_divide) if illum == "rgbx"
           else illum_heuristic(image, floor, log))
    if mat is None:
        result, shade, ao_map = render_legacy(image, floor, geo, ill["legacy"], tex, log, tile_m=tile_m, line_px=line_px)
    else:
        result, shade, ao_map = render(image, floor, geo, ill, tex, mat, log, tile_m=tile_m, ambient=ambient,
                                       ao=ao, ao_strength=ao_strength, grout_px=line_px,
                                       tile_offset=tuple(tile_offset), ssr=ssr)

    Image.fromarray(result).save(out / "output.jpg", quality=95)
    Image.fromarray((floor * 255).astype(np.uint8)).save(out / "mask_floor.png")
    if len(votes) > 1:
        for i, v in enumerate(votes):
            Image.fromarray((v * 255).astype(np.uint8)).save(out / f"mask_vote_{i}.png")
    Image.fromarray(shade).save(out / "irradiance.png")      # normalised irradiance E_n, display-encoded
    if ao_map is not None and ao:
        Image.fromarray((np.clip(ao_map, 0, 1) * 255).astype(np.uint8)).save(out / "ao.png")
    if ill["albedo"] is not None:
        Image.fromarray((to_srgb(ill["albedo"]) * 255).astype(np.uint8)).save(out / "albedo_old.png")
    meta = dict(scene=str(scene_p), pattern=str(pattern_p), seg=seg_name, tta=tta, seed=seed, illum=illum,
                tile_m=tile_m, m_per_1000px=m_per_1000px, material=material, material_params=mat, device=str(DEV),
                floor_coverage_pct=round(float(floor.mean() * 100), 1),
                total_gen_s=round(sum(r["gen_s"] for r in log), 2), stages=list(log))
    json.dump(meta, open(out / "metrics.json", "w"), indent=2)
    print(f"  total {meta['total_gen_s']:.2f} s (excl. model loads)  ->  {out / 'output.jpg'}")
    return meta


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scene", required=True)
    ap.add_argument("--pattern", default=str(ROOT / "input" / "pattern.png"))
    ap.add_argument("--seg", default="ensemble", choices=SEGMENTERS)
    ap.add_argument("--tta", type=int, default=1, help="ensemble only: number of augmented passes for a majority vote (1 = off)")
    ap.add_argument("--seed", type=int, default=0, help="seed for the TTA augmentations")
    ap.add_argument("--illum", default="rgbx", choices=ILLUMINATORS)
    ap.add_argument("--tile-m", type=float, default=None,
                    help="physical width of one texture repeat on the floor (m); the depth follows the "
                         "pattern's own aspect ratio. Default: derived from --m-per-1000px")
    ap.add_argument("--m-per-1000px", type=float, default=M_PER_1000PX, metavar="M",
                    help=f"how much floor 1000 px of the pattern covers, in metres (default {M_PER_1000PX:g}); a "
                         "1000 px pattern then tiles at that width and a 2000 px one at twice it. "
                         "Ignored when --tile-m is given")
    ap.add_argument("--rgbx-steps", type=int, default=10)
    ap.add_argument("--tile-offset", type=float, nargs=2, default=[0.5, 0.5], metavar=("U", "V"),
                    help="phase of the tile grid in tile units (default 0.5 0.5: a whole slab centred "
                         "on the floor, rather than a joint running through the middle of frame)")
    ap.add_argument("--out-root", default=str(ROOT / "outputs" / "floor_edit"))
    ap.add_argument("--preclean", action="store_true", help="DDRM identity pre-clean of the input first (what the web UI does on upload)")
    g = ap.add_argument_group("material")
    g.add_argument("--material", default="smooth-matte", choices=list(MATERIALS),
                   help="surface finish preset ('legacy' reproduces the pre-PBR shading)")
    g.add_argument("--albedo", type=float, help="mean diffuse reflectance; OVERRIDES the texture's own mean "
                                                "(0.04 black slate .. 0.35 mid tile .. 0.6 light marble)")
    g.add_argument("--roughness", type=float, help="0 = mirror, 1 = fully diffuse; sets the grazing sheen's cap")
    g.add_argument("--ior", type=float, help="index of refraction -> f0 (1.5 polyurethane, 1.55 epoxy)")
    g.add_argument("--coat", type=float, help="clearcoat weight, a second smoother lobe over the base (sealant)")
    g.add_argument("--delight", type=float, help="0 = use the pattern photo raw, 1 = strip its baked lighting")
    g.add_argument("--no-ssr", dest="ssr", action="store_false",
                   help="reflect a single averaged room colour instead of the actual room geometry")
    g2 = ap.add_argument_group("light field")
    g2.add_argument("--ambient", type=float, default=0.0,
                    help="fraction of floor irradiance surviving full occlusion (default 0: all light "
                         "is treated as direct, so shadows fall toward black)")
    g2.add_argument("--ambient-auto", dest="ambient", action="store_const", const=None,
                    help="estimate the ambient fraction from the scene's own deepest shadow instead")
    g2.add_argument("--no-ao", dest="ao", action="store_false", help="skip depth-based contact shadows")
    g2.add_argument("--ao-strength", type=float, default=1.0)
    g2.add_argument("--no-cache", dest="cache", action="store_false",
                    help="recompute the cached per-scene stages (mask, point map, RGB->X AOVs)")
    g2.add_argument("--albedo-divide", action="store_true",
                    help="estimate the light field as L_old/rho_old instead of taking the irradiance AOV; "
                         "sharper cast shadows, but unstable on dark floors")
    g2.add_argument("--line-px", type=int, default=21,
                    help="width of the thin dark/bright structures removed from the light field "
                         "(the old floor's tile joints); 0 disables the filter")
    args = ap.parse_args()
    seg_name = args.seg + (f"_tta{args.tta}" if args.seg == "ensemble" and args.tta > 1 else "")
    out = Path(args.out_root) / Path(args.scene).stem / Path(args.pattern).stem / seg_name / args.material
    run_floor_edit(args.scene, args.pattern, out, seg=args.seg, tta=args.tta, seed=args.seed, illum=args.illum,
                   tile_m=args.tile_m, m_per_1000px=args.m_per_1000px, rgbx_steps=args.rgbx_steps, line_px=args.line_px, preclean=args.preclean,
                   material=args.material, albedo=args.albedo, roughness=args.roughness, ior=args.ior,
                   coat=args.coat, delight=args.delight, ambient=args.ambient, ao=args.ao,
                   ao_strength=args.ao_strength, cache=args.cache, albedo_divide=args.albedo_divide,
                   tile_offset=tuple(args.tile_offset), ssr=args.ssr)


if __name__ == "__main__":
    main()
