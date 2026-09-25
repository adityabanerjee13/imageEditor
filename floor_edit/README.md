# floor_edit

Replace the floor in a room photo with a tiled texture — in correct perspective, at real-world scale, with the
photo's own shadows and lighting kept — without a generative image editor. One script, four stages, ~10 s per image
on an Intel Arc iGPU (~2 s on a re-run, see [Caching](#caching)).

| input scene | pattern | output |
|---|---|---|
| `input/scene.jpeg` | `input/pattern.png` | `outputs/floor_edit/scene/pattern/ensemble/smooth-matte/output.jpg` |

## Quick start

```bash
# defaults: --seg ensemble --illum rgbx --material smooth-matte
python floor_edit/floor_edit.py --scene input/scene.jpeg --pattern input/pattern.png

# polished floor, large-format slabs
python floor_edit/floor_edit.py --scene input/black_tile_flooring.png --material smooth-glossy --tile-m 3.6

# ensemble with test-time augmentation: 3 augmented passes, per-pixel majority vote
python floor_edit/floor_edit.py --scene input/scene.jpeg --tta 3
```

Results go to `outputs/floor_edit/<scene-stem>/<pattern-stem>/<seg>/<material>/`
(`<seg>` = `ensemble`, `ensemble_tta<N>` or `sam3`):

| file | content |
|---|---|
| `output.jpg` | the edited photo (scene resolution) |
| `mask_floor.png` | floor mask used for compositing |
| `irradiance.png` | the normalised irradiance field `E_n` applied to the texture (display-encoded) |
| `ao.png` | ambient-occlusion / contact-shadow term, with `--no-ao` off |
| `albedo_old.png` | RGB→X's estimate of the *old* floor's reflectance (diagnostic) |
| `metrics.json` | per-stage time, model load time, peak memory, floor coverage, plane residual, resolved material parameters |
| `mask_vote_<i>.png` | with `--tta N`: the floor mask of each augmented pass |

## Materials

Two finishes, one BRDF. Both are the same dielectric — a smooth surface reflects ~4 % at normal incidence
*whatever its roughness*, so the finish cannot change how bright the floor is underfoot, only how fast reflectance
climbs toward grazing and how wide the reflection lobe is.

| | `smooth-matte` | `smooth-glossy` |
|---|---|---|
| mean albedo `ρ̄` | 0.35 | 0.35 — polishing changes finish, not colour |
| roughness `α` | 0.75 | 0.08 |
| IOR → `f0` | 1.50 → 0.040 | 1.50 → 0.040 |
| grazing reflectance | 0.25 (soft haze) | 0.92 (mirrors the room) |

`--material legacy` is undocumented and reproduces the pre-PBR shading byte-for-byte, so the CSVs in `../ablations/`
stay comparable. Do not extend it.

## Options

| flag | default | meaning |
|---|---|---|
| `--scene` | required | room photo (jpg/png/webp) |
| `--pattern` | `input/pattern.png` | texture image; one full image = one tile |
| `--seg` | `ensemble` | floor segmenter: `ensemble` (SegFormer-B5 + UPerNet ConvNeXt-L) or `sam3` (text prompt "floor") |
| `--tta` | `1` | ensemble only: randomly augmented passes to majority-vote over (1 = single pass) |
| `--seed` | `0` | seed for the TTA augmentations |
| `--illum` | `rgbx` | irradiance estimator: `rgbx` (RGB→X AOV) or `heuristic` (linear radiance) |
| `--material` | `smooth-matte` | surface finish preset (see above) |
| `--albedo` | preset | mean diffuse reflectance; **overrides** the texture's own mean (0.04 slate … 0.6 light marble) |
| `--roughness` `--ior` `--coat` | preset | individual BRDF overrides |
| `--delight` | `0.8` | 0 = use the pattern photo raw, 1 = strip its baked-in lighting |
| `--no-ssr` | off | reflect one averaged room colour instead of the actual room geometry |
| `--tile-m` | `0.6` | physical **width** of one repeat, in metres; the depth follows the pattern's aspect ratio |
| `--tile-offset` | `0.5 0.5` | phase of the tile grid, in tile units |
| `--ambient` | `0.0` | fraction of floor irradiance surviving full occlusion; `--ambient-auto` estimates it from the scene |
| `--no-ao` | off | skip depth-based contact shadows |
| `--ao-strength` | `1.0` | scales the contact shadows |
| `--line-px` | `21` | width of the thin dark/bright structures removed from the irradiance field (old grout lines); `0` disables |
| `--albedo-divide` | off | estimate irradiance as `L_old / ρ_old` instead of taking the AOV (sharper, unstable on dark floors) |
| `--no-cache` | off | recompute the cached per-scene stages |
| `--rgbx-steps` | `10` | DDIM steps for RGB→X (2-10 is enough; it is a zero-SNR model) |
| `--out-root` | `outputs/floor_edit` | output root |

## How it works

```
photo ──► 1. segmentation ──► floor mask ─────────────────────────────────┐
      ├─► 2. geometry (MoGe-2) ──► 3-D points + intrinsics ──► plane fit ─┤
      └─► 3. irradiance ──► light field E ────────────────────────────────┤
                                                                          ▼
pattern ──► albedo ρ ──────────────────────────► 4. render: L = (1-F)·ρ·E_n + F·L_env ──► output.jpg
```

Everything is done on **linear radiance** and encoded to sRGB exactly once, at the end.

1. **Floor segmentation.** ADE20K semantic models; floor = classes {floor, rug}.
   *ensemble*: mean of SegFormer-B5 and UPerNet ConvNeXt-L softmaxes at 1024×576, then a 5×5 close and removal of
   components < 0.2 % of the image. Chosen because it has the cleanest boundaries around chair legs / furniture bases
   (boundary-F 0.81 vs 0.25 for SAM 3, measured against geometry-derived pseudo ground truth).
   *sam3*: SAM 3 with the concept prompt "floor"; instances scoring > 0.4 are unioned. Text-promptable, but slower,
   heavier (5.2 GB) and looser at edges.
   *--tta N* (ensemble only): N randomly augmented copies — horizontal mirror (p = 0.5), brightness ±0.08,
   contrast ×0.92-1.08, Gaussian noise σ ≤ 0.02 — combined by per-pixel majority vote. On the test scenes the passes
   agree at IoU > 0.99; it is there for harder scenes.
2. **Geometry.** MoGe-2 ViT-L predicts a metric point map and the camera intrinsics. The floor plane is fitted to the
   masked points by least squares with two inlier re-fits (residual ≈ 0.2 cm on the test scenes). The plane is fitted
   here, not in the render, because the irradiance stage needs it for ambient occlusion.
3. **Irradiance.** The light field `E` the *old* floor received, so the new one can be given the same.
   *rgbx*: RGB→X (Zeng et al. 2024) "Irradiance (diffuse lighting)" AOV, fed linear RGB at a 768 px short side and
   gamma-decoded on the way out. Note this AOV is **not** albedo-free in practice — it bakes the old floor's tile
   joints into the "lighting" — so thin dark/bright structures are filtered out (`--line-px`), substituting only where
   the morphology actually moves the value so broad cast shadows stay untouched.
   *heuristic*: linear scene radiance, assuming a uniform old-floor albedo. Reads a dark floor as deep shadow.
   The field is then split into ambient + direct (`--ambient`), each recoloured with the chromaticity of the shadowed
   and lit floor quartiles, multiplied by a horizon-based ambient occlusion term, and anchored so its mean over the
   floor is 1 — so the rendered floor's mean radiance equals its albedo.
4. **Render.** Camera rays are intersected with the plane to get metric floor coordinates (u, v); the pattern is
   sampled at `(u mod tile_u, v mod tile_v)`, giving correct foreshortening and scale. Shading is
   `L = (1−F)·ρ·E_n + F·L_env` with Schlick Fresnel against the view angle; `(1−F)` on the diffuse term is what stops
   the far floor gaining energy. Highlights are rolled off with a shoulder rather than clipped, and the result is
   composited through a 1 px-softened floor mask, so nothing outside the mask changes.

### Reflections

The floor is a known plane and MoGe gives the whole point cloud, so the reflection is exact rather than a
screen-space guess: every above-floor point is **mirrored about the plane and re-projected through the same camera**,
z-buffered far-to-near. The perspective stretch toward the horizon falls out of the projection — no ray march, no step
count, no depth bias. One normalised convolution then fills the gaps between scattered points *and* applies the
roughness blur; at `roughness 0.75` that blur is wide enough to converge on the room average, which is why both
materials use the same path. Like any screen-space method it can only reflect geometry the photo contains: where a
mirrored ray leaves the frame, confidence falls to zero and it fades to a single averaged room colour.

### Caching

Every neural stage depends only on the input photo, so all three are cached to `data/cache/floor_edit/`:

| cache | keyed on |
|---|---|
| floor mask | scene + segmenter + tta + seed |
| MoGe-2 point map | scene |
| RGB→X AOVs | scene + steps + resolution |

Switching pattern, material, tile size or finish therefore costs the render alone. The floor plane is *not* cached —
it depends on both the point map and the mask, and the SVD is a fraction of a second. `--no-cache` forces recomputation.

## Performance (Intel Arc 140T iGPU, 16 GB shared, PyTorch XPU)

| stage | model | cold | cached |
|---|---|---|---|
| segmentation | ensemble | ~4 s | 0.03 s |
| segmentation | SAM 3 | ~8 s | 0.03 s |
| geometry | MoGe-2 ViT-L | ~4 s | 0.09 s |
| irradiance | RGB→X, 10 steps ×2 AOVs | 25-43 s | 0.07 s |
| irradiance | heuristic | < 0.1 s | — |
| render | CPU | 1-6 s | — |

≈ 10 s for a new scene and ≈ 2 s for any re-run, excluding model loads (3-4 s each) and the 5-15 s kernel warm-up the
first call in a process pays. Render cost is dominated by ambient occlusion (`--no-ao` removes it). On a CUDA GPU the
same script runs unchanged (`DEV` picks cuda > xpu > cpu).

## Requirements

- Python ≥ 3.10; `torch` (tested 2.13 XPU), `transformers` ≥ 5, `diffusers` ≥ 0.30, `opencv-python(-headless)`,
  `numpy`, `pillow`, `huggingface_hub`, `utils3d_moge` (for MoGe: `pip install "utils3d_moge @ git+https://github.com/EasternJournalist/utils3d-moge.git@62f09d5"`).
  `cv2.ximgproc` is *not* required: the guided filter is implemented on `cv2.boxFilter`.
- Code dependencies in `repos/`: `MoGe` (microsoft/MoGe, `moge.model.v2`) and `rgbx` (zheng95z/rgbx, the rgb2x diffusers pipeline).
- Weights in `weights/` (download with `huggingface_hub.snapshot_download(..., local_dir=...)`):

| folder | HF repo | size |
|---|---|---|
| `segformer-b5-ade` | `nvidia/segformer-b5-finetuned-ade-640-640` | 0.34 GB |
| `upernet-convnext-l` | `openmmlab/upernet-convnext-large` | 0.94 GB |
| `sam3` | `facebook/sam3` (gated; `jetjodh/sam3` mirror) | 3.4 GB |
| `moge-2-vitl-normal` | `Ruicheng/moge-2-vitl-normal` (`model.pt`) | 1.3 GB |
| `rgb-to-x` | `zheng95z/rgb-to-x` (safetensors only) | 4.9 GB |

## Tips and limits

- **Tile size.** `--tile-m` is the real-world *width* of one repeat; the depth follows the image's aspect ratio, so a
  2:1 slab stays 2:1 on the floor. Large-format slabs are 1.2-2.4 m; beyond that a low-resolution pattern visibly
  softens, and a higher-resolution source is the fix rather than a larger `--tile-m`.
- **Tile phase.** `(u, v)` are measured from the plane centroid, so without the default half-tile `--tile-offset` a
  joint runs through the middle of frame and cuts the nearest slab in half.
- **Floor lightness** is set by `--albedo`, not by the pattern image's exposure. Nothing real sits below 0.02 or
  above 0.9.
- **Textured old floors.** If remnants show through, raise `--line-px`; if a thin shadow edge gets softened, lower it.
  Check `irradiance.png` — if the old joints are visible *there*, that is the route they are taking.
- **Dark floors.** `--albedo-divide` is ill-conditioned when the old floor is near-black (the kitchen scene's old tile
  has a median albedo of 0.026); the default AOV path is stable there.
- **Non-planar floors / multiple levels** are not handled: one plane is fitted to the whole mask.
- SAM 3 and RGB→X exceed the project's usual 1 GB-per-file download limit; they were fetched with explicit approval.

## Provenance

Chosen after ablations over 10 editing/segmentation/geometry models (MatSwap, ControlTile, LBM, MARBLE, IntrinsicEdit,
FLUX Kontext, Qwen-Image-Edit, Insert Anything, SAM 2.1/3, Mask2Former, OneFormer, SegFormer, UPerNet, Intrinsic v1/v2.1,
RGB→X). The result CSVs are in `../ablations/`; the geometric render + irradiance field won on speed (~20 s vs
2.5-25 min for the diffusion editors), product fidelity (CLIP similarity to the pattern 0.80 vs 0.55-0.69) and zero
change outside the mask.
