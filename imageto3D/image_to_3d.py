"""Image -> 3D proxy, using the DIRECT repo's TRELLIS pipeline as an API.

DIRECT (ICML 2026, `repos/DIRECT`) does pose-controllable object insertion by first reconstructing a
3D proxy of the object and then rendering it as geometric guidance for a FLUX inpainting model.  This
module reuses only that first half - the image -> 3D reconstruction - exactly as DIRECT calls it:

    demo/demo.py                      TrellisImageTo3DPipeline.from_pretrained(...)
                                      .run(img, seed=..., formats=["gaussian"])
    preprocess/preprocess_example.py  .get_slat(img, mask_image=..., apply_rmbg=False)
                                      .decode_slat(slat, ["gaussian"])

Both go through `trellis.pipelines.TrellisImageTo3DPipeline` (vendored at
`repos/DIRECT/third_party/trellis`), which is what this module imports and drives.  The stages are
DINOv2 conditioning -> sparse-structure flow (16^3 latent -> 64^3 occupancy) -> structured-latent
flow -> per-format decoders (Gaussian splats / FlexiCubes mesh).

    python -m object_edit.image_to_3d --image data/jobs/c127fa4b044d/obj0_subject_512.png

Outputs (under `outputs/image_to_3d/<image stem>/`): `input_processed.png`, `gaussian.ply`,
`mesh.glb`, optionally `slat.pt`, and `metrics.json` with per-stage time / GPU peak.
"""
import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from imageto3D import trellis_compat
from object_edit.common import DEV, REPOS, ROOT, WEIGHTS, Stage, StageLog, empty_cache, timed

# TRELLIS is vendored inside DIRECT; `direct` itself sits one level up (unused here, but keeping it
# importable leaves room for anyone extending this towards DIRECT's insertion half).
for _p in (REPOS / "DIRECT", REPOS / "DIRECT" / "third_party"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

OUT_ROOT = ROOT / "outputs" / "image_to_3d"
# Local weights dir, else the HF id that DIRECT's demo hardcodes.
TRELLIS_LOCAL = WEIGHTS / "trellis-image-large"
TRELLIS_HUB = "microsoft/TRELLIS-image-large"

FORMATS = ("gaussian", "mesh", "radiance_field")
# Y-up export basis, matching trellis' Gaussian.save_ply default and utils/postprocessing_utils.to_glb.
YUP = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]], dtype=np.float32)


def configure_backends(attn="sdpa", spconv_algo="native"):
    """Pick trellis' compute backends.  Must run before `import trellis` - both `modules/attention`
    and `modules/sparse` latch their backend from the environment at import time.

    Dense attention accepts 'xformers' | 'flash_attn' | 'sdpa' | 'naive'; 'sdpa' is the only one of
    those that is not CUDA-only, so it is the default here.  Sparse attention
    (`modules/sparse/__init__.py`) only accepts 'xformers' | 'flash_attn' and has no such fallback,
    so it keeps its 'xformers' default and `trellis_compat` supplies a pure-PyTorch stand-in for it,
    along with one for spconv.
    """
    os.environ.setdefault("ATTN_BACKEND", attn)
    os.environ.setdefault("SPCONV_ALGO", spconv_algo)
    if attn not in ("xformers", "flash_attn"):
        trellis_compat.install()
    _stub_optional_deps()


def _stub_optional_deps():
    """Stub two heavy imports that trellis pulls in at module scope but never reaches on this path
    (the same trick sd15.py uses for FreeFine's rembg):

      rembg   - `trellis_image_to_3d` only calls it on the `apply_rmbg=False` matting fallback.
      open3d  - imported solely by `TrellisTextTo3DPipeline`, which `pipelines/__init__` loads and
                this module never constructs.

    Both raise if something actually calls them, rather than failing silently.
    """
    import importlib.util
    import types

    class _Missing:
        """Resolves any attribute (so `o3d.geometry.TriangleMesh` works as a type annotation - the
        text pipeline evaluates that at class-definition time) but raises as soon as it is called."""

        def __init__(self, name):
            self._name = name

        def __getattr__(self, name):
            # Dunders must raise AttributeError: anything that walks sys.modules probes __file__ and
            # friends (torch.library's fake-kernel registration does, via inspect).
            if name.startswith("__"):
                raise AttributeError(name)
            return _Missing(f"{self._name}.{name}")

        def __call__(self, *_a, **_k):
            raise RuntimeError(f"{self._name} is not installed (stubbed by object_edit.image_to_3d)")

    class _StubModule(types.ModuleType):
        def __init__(self, name):
            super().__init__(name)
            # A module in sys.modules with __spec__ None makes importlib.util.find_spec(name) raise
            # ValueError - see trellis_compat._module.
            self.__spec__ = importlib.util.spec_from_loader(name, loader=None)

        def __getattr__(self, name):
            if name.startswith("__"):
                raise AttributeError(name)
            return _Missing(f"{self.__name__}.{name}")

    sys.modules.setdefault("rembg", _StubModule("rembg"))
    if "open3d" not in sys.modules:
        o3d = _StubModule("open3d")
        o3d.__file__ = "<stubbed by object_edit.image_to_3d>"
        sys.modules["open3d"] = o3d


def load_pipeline(path=None, fp32=False):
    """DIRECT's loader: `TrellisImageTo3DPipeline.from_pretrained(...)`, then move to the device.

    `path` defaults to `weights/trellis-image-large` when present, else the HF repo id (which the
    pipeline downloads on first use - ~3.3 GB, two checkpoints of which are over this project's 1 GB
    per-file cap).  DINOv2, the image conditioner, is pulled separately through `torch.hub`.
    """
    trellis_compat.redirect_cuda(DEV)
    from trellis.pipelines import TrellisImageTo3DPipeline

    src = str(path) if path else (str(TRELLIS_LOCAL) if (TRELLIS_LOCAL / "pipeline.json").exists() else TRELLIS_HUB)
    pipeline, load_s = timed(lambda: TrellisImageTo3DPipeline.from_pretrained(src))
    pipeline.to(DEV)
    if fp32:
        # The released checkpoints are fp16; XPU autocast is unreliable for some ops (the same
        # reason floor_edit pins MoGe-2 to fp32), so allow forcing everything to fp32.
        # `convert_to_fp32()` only re-casts the parameters - the models also carry `self.dtype`,
        # set once from `use_fp16`, and cast their activations with it (`h.type(self.dtype)`).
        # Without resetting it too, activations stay Half against Float weights and every matmul
        # raises "mat1 and mat2 must have the same dtype".
        for m in pipeline.models.values():
            if hasattr(m, "convert_to_fp32"):
                m.convert_to_fp32()
            if hasattr(m, "dtype"):
                m.dtype = torch.float32
            if hasattr(m, "use_fp16"):
                m.use_fp16 = False
            m.float()
    pipeline.set_progress_bar_config(disable=False)
    return pipeline, load_s, src


def subject_from_mask(image, mask, margin=0.15):
    """Scene + a SAM mask -> the selected object alone, cropped to its bbox, as RGBA.

    This is what the web UI feeds the pipeline: the mask becomes the alpha channel, so
    `preprocess_image` takes its has-alpha branch and no matting model (RMBG-2.0, gated here) runs.
    """
    rgb = np.array(image.convert("RGB"))
    m = np.asarray(mask.convert("L")) > 127 if isinstance(mask, Image.Image) else np.asarray(mask) > 0
    if not m.any():
        raise ValueError("mask is empty")
    ys, xs = np.where(m)
    y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    pad = int(round(margin * max(y1 - y0, x1 - x0)))
    y0, y1 = max(0, y0 - pad), min(rgb.shape[0], y1 + pad)
    x0, x1 = max(0, x0 - pad), min(rgb.shape[1], x1 + pad)
    rgba = np.dstack([rgb[y0:y1, x0:x1], np.where(m[y0:y1, x0:x1], 255, 0).astype(np.uint8)])
    return Image.fromarray(rgba, "RGBA")


def white_to_alpha(image, tol=12):
    """Alpha from a white backdrop, for subject crops that are already cut out.

    `object_edit`'s `obj<i>_subject_512.png` (OmniPaint's insertion condition) is the object on
    white, so no matting model is needed.  Thresholding white outright would punch holes in pale
    parts of the object, so this keeps only the white that is connected to the border.
    """
    import cv2

    rgb = np.array(image.convert("RGB"))
    white = np.all(rgb >= 255 - tol, axis=2).astype(np.uint8)
    n, labels = cv2.connectedComponents(white, connectivity=4)
    edge = np.concatenate([labels[0], labels[-1], labels[:, 0], labels[:, -1]])
    bg = np.isin(labels, [l for l in np.unique(edge) if l != 0])
    alpha = np.where(bg, 0, 255).astype(np.uint8)
    # Close pinholes left by near-white speckle inside the object.
    alpha = cv2.morphologyEx(alpha, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    out = image.convert("RGB")
    out.putalpha(Image.fromarray(alpha))
    return out


def _has_alpha(image):
    return image.mode == "RGBA" and not np.all(np.array(image)[:, :, 3] == 255)


def _border_is_white(image, tol=12, frac=0.9):
    rgb = np.array(image.convert("RGB"))
    edge = np.concatenate([rgb[0], rgb[-1], rgb[:, 0], rgb[:, -1]])
    return float(np.all(edge >= 255 - tol, axis=1).mean()) >= frac


def choose_matte(image, matte):
    """Resolve `--matte auto`: use an existing alpha, else a white backdrop, else RMBG-2.0."""
    if matte != "auto":
        return matte
    if _has_alpha(image):
        return "none"
    return "white" if _border_is_white(image) else "rmbg"


def preprocess(pipeline, image, mask=None, matte="auto", log=None):
    """DIRECT's two preprocessing routes, both taken from the pipeline itself.

    With a mask -> `preprocess_image_with_mask` (what `preprocess_example.py` uses via `get_slat`).
    Without -> `preprocess_image`, which takes the image's alpha when it has one and otherwise mattes
    with RMBG-2.0 (demo.py) or rembg/u2net.  Either way the object is cropped to 1.2x its bbox and
    resized to 518 on black.

    `matte='white'` adds a route DIRECT does not need: it supplies the alpha itself from a white
    backdrop, so `preprocess_image` takes its has-alpha branch and no matting model is loaded.
    RMBG-2.0 is a gated repo, so that route needs an authorised HF token.
    """
    from torchvision import transforms

    resolved = "mask" if mask is not None else choose_matte(image, matte)
    with Stage(log if log is not None else [], "preprocess", f"matte={resolved}"):
        if mask is not None:
            to_t = transforms.Compose([transforms.ToTensor(), transforms.Normalize([0.5] * 3, [0.5] * 3)])
            img_t = to_t(image.convert("RGB")).to(DEV)
            mask_t = (transforms.ToTensor()(mask.convert("L")).to(DEV) > 0.5).float()
            out = pipeline.preprocess_image_with_mask(img_t, mask_t, apply_rmbg=False, apply_rembg=False)
            return (out[0] if isinstance(out, tuple) else out), resolved
        if resolved == "white":
            image = white_to_alpha(image)
        elif resolved == "rmbg":
            pipeline.init_rmbg_model()
        return pipeline.preprocess_image(image, apply_rmbg=(resolved == "rmbg")), resolved


def save_mesh(mesh, path):
    """FlexiCubes output -> vertex-coloured GLB.

    `trellis.utils.postprocessing_utils.to_glb` bakes a real texture but needs nvdiffrast (CUDA
    only), so this writes the per-vertex colours the mesh decoder already produces
    (`MeshExtractResult.vertex_attrs`) instead.
    """
    import trimesh

    v = mesh.vertices.detach().cpu().numpy().astype(np.float32) @ YUP
    f = mesh.faces.detach().cpu().numpy()
    colors = None
    if mesh.vertex_attrs is not None:
        c = mesh.vertex_attrs.detach().cpu().numpy()
        colors = (np.clip(c[:, :3], 0, 1) * 255).astype(np.uint8)
    trimesh.Trimesh(vertices=v, faces=f, vertex_colors=colors, process=False).export(path)
    return {"vertices": int(v.shape[0]), "faces": int(f.shape[0])}


SH_C0 = 0.28209479177387814      # DC term of the SH basis, the 3DGS colour convention


def gaussian_ply_to_points(src, dst, min_opacity=0.1):
    """3D-Gaussian-Splatting PLY -> plain coloured point cloud, for ordinary 3D viewers.

    `Gaussian.save_ply` writes the splat format: colour as SH DC coefficients (`f_dc_*`) and
    `inverse_sigmoid` opacity, with no `red`/`green`/`blue`.  Open3D and friends load it as bare
    XYZ and show it colourless, so this decodes DC -> RGB and drops near-transparent splats.
    """
    from plyfile import PlyData, PlyElement

    v = PlyData.read(str(src))["vertex"].data
    alpha = 1.0 / (1.0 + np.exp(-v["opacity"].astype(np.float32)))
    keep = alpha >= min_opacity
    rgb = np.stack([0.5 + SH_C0 * v[f"f_dc_{i}"].astype(np.float32) for i in range(3)], 1)
    rgb = (np.clip(rgb[keep], 0, 1) * 255).astype(np.uint8)

    out = np.empty(int(keep.sum()), dtype=[("x", "f4"), ("y", "f4"), ("z", "f4"),
                                           ("red", "u1"), ("green", "u1"), ("blue", "u1")])
    for i, a in enumerate("xyz"):
        out[a] = v[a][keep]
    for i, a in enumerate(("red", "green", "blue")):
        out[a] = rgb[:, i]
    PlyData([PlyElement.describe(out, "vertex")]).write(str(dst))
    return {"points": int(keep.sum()), "dropped": int((~keep).sum())}


def vertex_normals(mesh):
    """Area-weighted vertex normals from the FlexiCubes result's face normals."""
    # float64 throughout: FlexiCubes leaves a few sliver faces whose float32 cross products underflow
    # when normalised, which would leave those vertices with non-unit normals.
    v = mesh.vertices.detach().cpu().numpy().astype(np.float64)
    f = mesh.faces.detach().cpu().numpy()
    fn = np.cross(v[f[:, 1]] - v[f[:, 0]], v[f[:, 2]] - v[f[:, 0]])
    vn = np.zeros_like(v)
    for i in range(3):
        np.add.at(vn, f[:, i], fn)
    norm = np.linalg.norm(vn, axis=1, keepdims=True)
    return np.divide(vn, norm, out=np.zeros_like(vn), where=norm > 0).astype(np.float32)


def render_preview(points, colors, path, normals=None, views=4, res=384, margin=0.06):
    """Turntable preview strip: `views` azimuths of a coloured point set, z-buffered.

    TRELLIS' own renderers need diff-gaussian-rasterization / nvdiffrast (both CUDA only), so this
    is a small painter's-algorithm splat - enough to see what was reconstructed.  Fed either the
    mesh vertices (pass `normals` for diffuse shading, which is what makes the shape readable) or
    the Gaussian centres.
    """
    p = np.asarray(points, dtype=np.float32)
    c = np.asarray(colors, dtype=np.float32)[:, :3]
    p = p - (p.min(0) + p.max(0)) / 2
    p /= max(np.abs(p).max(), 1e-6)
    n = None if normals is None else np.asarray(normals, dtype=np.float32)

    light = np.array([0.3, 0.5, 0.8], dtype=np.float32)
    light /= np.linalg.norm(light)
    tiles = []
    for i in range(views):
        a = 2 * np.pi * i / views
        rot = np.array([[np.cos(a), 0, np.sin(a)],
                        [0, 1, 0],
                        [-np.sin(a), 0, np.cos(a)]], dtype=np.float32)
        q = p @ rot.T
        rgb = c
        if n is not None:
            # Two-sided: FlexiCubes leaves ~half the vertex normals pointing inwards, so a one-sided
            # lambert term would drop half the surface to ambient and the preview would read black.
            lam = np.abs((n @ rot.T) @ light)[:, None]
            rgb = np.clip(c * (0.55 + 0.75 * lam), 0, 255)
        s = (res / 2) * (1 - margin)
        x = np.clip((q[:, 0] * s + res / 2).astype(int), 0, res - 1)
        y = np.clip((-q[:, 1] * s + res / 2).astype(int), 0, res - 1)
        img = np.full((res, res, 3), 255, np.uint8)
        order = np.argsort(-q[:, 2])          # far first, so near points overwrite
        img[y[order], x[order]] = rgb[order].astype(np.uint8)
        tiles.append(img)
    Image.fromarray(np.concatenate(tiles, axis=1)).save(path)
    return path


def image_to_3d(image_path, out_dir=None, mask_path=None, seed=42, formats=("gaussian", "mesh"),
                matte="auto", fp32=False, ss_steps=None, slat_steps=None, cfg=None,
                save_slat=False, preview=True, viewer_ply=True, weights=None, on_stage=None):
    """Reconstruct a 3D proxy of the object in `image_path` and write it to `out_dir`."""
    image_path = Path(image_path)
    out_dir = Path(out_dir) if out_dir else OUT_ROOT / image_path.stem
    out_dir.mkdir(parents=True, exist_ok=True)
    log = StageLog(on_stage)

    image = Image.open(image_path)
    mask = Image.open(mask_path) if mask_path else None
    pipeline, load_s, src = load_pipeline(weights, fp32)

    proc, resolved_matte = preprocess(pipeline, image, mask, matte, log)
    proc.save(out_dir / "input_processed.png")

    # Sampler overrides - the same knobs DIRECT's demo exposes.
    ss_params = {k: v for k, v in (("steps", ss_steps), ("cfg_strength", cfg)) if v is not None}
    slat_params = {k: v for k, v in (("steps", slat_steps), ("cfg_strength", cfg)) if v is not None}

    # Same call sequence as pipeline.run(), unrolled so each stage can be timed separately.
    with Stage(log, "reconstruct", f"TRELLIS-image-large {'+'.join(formats)}") as st:
        with torch.no_grad():
            cond = pipeline.get_cond([proc])
            torch.manual_seed(seed)
            coords = pipeline.sample_sparse_structure(cond, 1, ss_params)
            slat = pipeline.sample_slat(cond, coords, slat_params)
            outputs = pipeline.decode_slat(slat, list(formats))
    st.row["load_s"] = load_s
    st.row["voxels"] = int(coords.shape[0])

    written = {}
    if save_slat:
        torch.save({"feats": slat.feats.detach().cpu(), "coords": slat.coords.detach().cpu()},
                   out_dir / "slat.pt")
        written["slat"] = "slat.pt"
    if "gaussian" in outputs:
        outputs["gaussian"][0].save_ply(str(out_dir / "gaussian.ply"))
        written["gaussian"] = "gaussian.ply"
        if viewer_ply:
            written["gaussian_points"] = gaussian_ply_to_points(
                out_dir / "gaussian.ply", out_dir / "gaussian_points.ply")
    if "mesh" in outputs:
        written["mesh"] = save_mesh(outputs["mesh"][0], str(out_dir / "mesh.glb"))
        if viewer_ply:
            # Open3D imports GLB through ASSIMP and often drops vertex colours; PLY keeps them.
            save_mesh(outputs["mesh"][0], str(out_dir / "mesh.ply"))
            written["mesh_ply"] = "mesh.ply"
    if preview:
        m = outputs.get("mesh", [None])[0]
        if m is not None and m.vertex_attrs is not None:
            pts = m.vertices.detach().cpu().numpy() @ YUP
            cols = (np.clip(m.vertex_attrs[:, :3].detach().cpu().numpy(), 0, 1) * 255).astype(np.uint8)
            render_preview(pts, cols, out_dir / "preview.png", normals=vertex_normals(m) @ YUP)
            written["preview"] = "preview.png"

    metrics = {"image": str(image_path), "mask": str(mask_path) if mask_path else None, "weights": src,
               "device": str(DEV), "seed": seed, "formats": list(formats), "fp32": fp32,
               "matte": resolved_matte, "attn_backend": os.environ.get("ATTN_BACKEND"),
               "voxels": int(coords.shape[0]), "outputs": written, "stages": list(log)}
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    del pipeline
    empty_cache()
    print(f"[image_to_3d] wrote {out_dir}")
    return out_dir, metrics


def main():
    ap = argparse.ArgumentParser(description="Image -> 3D via DIRECT's TRELLIS pipeline.")
    ap.add_argument("--image", default="data/jobs/c127fa4b044d/obj0_subject_512.png")
    ap.add_argument("--mask", default=None, help="binary object mask; uses DIRECT's masked preprocessing path")
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--weights", default=None, help="local TRELLIS dir or HF id (default: weights/trellis-image-large)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--formats", nargs="+", default=["gaussian", "mesh"], choices=FORMATS)
    ap.add_argument("--matte", default="auto", choices=["auto", "white", "rmbg", "none"],
                    help="how to get the object's alpha: auto (existing alpha, else white backdrop, "
                         "else RMBG-2.0), white (cut out a white background), rmbg (RMBG-2.0, gated "
                         "on HF), none (image already has alpha)")
    ap.add_argument("--fp32", action="store_true", help="force fp32 (the checkpoints ship fp16)")
    ap.add_argument("--ss-steps", type=int, default=None, help="sparse-structure sampler steps")
    ap.add_argument("--slat-steps", type=int, default=None, help="structured-latent sampler steps")
    ap.add_argument("--cfg", type=float, default=None, help="classifier-free guidance strength")
    ap.add_argument("--save-slat", action="store_true", help="also dump the sparse latent (DIRECT's slat.pt)")
    ap.add_argument("--no-preview", action="store_true", help="skip the turntable preview strip")
    ap.add_argument("--no-viewer-ply", action="store_true",
                    help="skip mesh.ply / gaussian_points.ply (the Open3D-friendly copies)")
    ap.add_argument("--attn", default="sdpa", choices=["sdpa", "naive", "xformers", "flash_attn"])
    a = ap.parse_args()

    configure_backends(a.attn)
    image_to_3d(a.image, a.out_dir, a.mask, a.seed, tuple(a.formats), a.matte, a.fp32,
                a.ss_steps, a.slat_steps, a.cfg, a.save_slat, not a.no_preview,
                not a.no_viewer_ply, a.weights)


if __name__ == "__main__":
    main()
