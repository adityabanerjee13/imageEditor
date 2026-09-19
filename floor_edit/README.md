# floor_edit

Replace the floor in a room photo with a tiled texture — in correct perspective, at real-world scale, with the
photo's own shadows and lighting kept — without a generative image editor. One script, four stages, ~20 s per image
on an Intel Arc iGPU.

| input scene | pattern | output |
|---|---|---|
| `input/scene.jpeg` | `input/pattern.png` | `outputs/floor_edit/scene/pattern/ensemble/output.jpg` |

## Quick start

```bash
# defaults: --seg ensemble --illum rgbx
python floor_edit/floor_edit.py --scene input/scene.jpeg --pattern input/pattern.png

# SAM 3 segmentation, heuristic illumination, 45 cm tiles
python floor_edit/floor_edit.py --scene input/black_tile_flooring.png --seg sam3 --illum heuristic --tile-m 0.45

# ensemble with test-time augmentation: 3 augmented passes, per-pixel majority vote
python floor_edit/floor_edit.py --scene input/scene.jpeg --tta 3
```

Results go to `outputs/floor_edit/<scene-stem>/<pattern-stem>/<seg>/` (`<seg>` = `ensemble`, `ensemble_tta<N>` or `sam3`):

| file | content |
|---|---|
| `output.jpg` | the edited photo (scene resolution) |
| `mask_floor.png` | floor mask used for compositing |
| `illumination.png` | the shading map applied to the texture (after normalisation + line filtering) |
| `metrics.json` | per-stage generation time, model load time, peak accelerator memory, floor coverage, plane-fit residual (+ TTA agreement stats) |
| `mask_vote_<i>.png` | with `--tta N`: the floor mask of each augmented pass |

## Options

| flag | default | meaning |
|---|---|---|
| `--scene` | required | room photo (jpg/png) |
| `--pattern` | `input/pattern.png` | texture image; one full image = one tile |
| `--seg` | `ensemble` | floor segmenter: `ensemble` (SegFormer-B5 + UPerNet ConvNeXt-L) or `sam3` (text prompt "floor") |
| `--tta` | `1` | ensemble only: number of randomly augmented passes to majority-vote over (1 = single pass) |
| `--seed` | `0` | seed for the TTA augmentations |
| `--illum` | `rgbx` | illumination model: `rgbx` (RGB->X diffuse irradiance) or `heuristic` (luminance ratio) |
| `--tile-m` | `0.6` | physical edge length of one texture repeat on the floor, in metres |
| `--rgbx-steps` | `10` | DDIM steps for RGB->X (2-10 is enough; it is a zero-SNR model) |
| `--line-px` | `21` | structures thinner than this in the illumination map are treated as old floor texture and removed |
| `--out-root` | `outputs/floor_edit` | output root |

## How it works

```
photo ──► 1. segmentation ──► floor mask ─────────────────────────────────┐
      ├─► 2. geometry (MoGe-2) ──► 3-D points + intrinsics ──► plane fit ─┤
      └─► 3. illumination ──► shading map ────────────────────────────────┤
                                                                          ▼
pattern ──────────────────────────────────────► 4. render: tile on plane × shading, composite ──► output.jpg
```

1. **Floor segmentation.** ADE20K semantic models; floor = classes {floor, rug}.
   *ensemble*: mean of SegFormer-B5 and UPerNet ConvNeXt-L softmaxes at 1024x576, then a 5x5 close and removal of
   components < 0.2 % of the image. Chosen because it has the cleanest boundaries around chair legs / furniture bases
   (boundary-F 0.81 vs 0.25 for SAM 3, measured against geometry-derived pseudo ground truth).
   *sam3*: SAM 3 with the concept prompt "floor"; instances scoring > 0.4 are unioned. Text-promptable, but slower,
   heavier (5.2 GB) and looser at edges.
   *--tta N* (ensemble only): the ensemble is run N times on randomly augmented copies of the photo — horizontal mirror
   (p = 0.5), brightness ±0.08, contrast ×0.92-1.08, Gaussian pixel noise σ ≤ 0.02 — masks are un-mirrored and combined
   by per-pixel majority vote (≥ ⌈N/2⌉). On the test scenes the passes agree at IoU > 0.99 and the vote changes < 0.3 %
   of pixels vs a single pass, at ~N× the segmentation time; it is there for harder scenes.
2. **Geometry.** MoGe-2 ViT-L predicts a metric point map (x, y, z per pixel) and the camera intrinsics. The floor
   plane is fitted to the masked points by least squares with two inlier re-fits (residual ≈ 0.2 cm on the test scenes).
3. **Illumination.** A per-pixel brightness map of the floor's lighting, *without* its own texture.
   *rgbx*: RGB->X (Zeng et al. 2024) "Irradiance (diffuse lighting)" channel. Fed **linear RGB** (sRGB^2.2) at a
   768 px short side and the output is **gamma-decoded** (^2.2) — with sRGB in/out the map is almost flat and shadows vanish.
   *heuristic*: luminance divided by the 90th-percentile floor luminance. Fine on uniform floors; on floors whose
   albedo varies tile to tile (dark slate) it mistakes albedo for lighting.
4. **Render.** Every pixel's camera ray is intersected with the plane to get metric floor coordinates (u, v); the
   pattern is sampled at `(u mod tile_m, v mod tile_m)`, which gives correct foreshortening and scale. The illumination
   map is normalised on the floor, passed through a grayscale close→open with a `line_px` ellipse (drops old grout
   lines and specks, keeps cast shadows and light fall-off, which are wider) and a 2 px blur, then multiplied into the
   texture. The result is composited through a 1 px-softened floor mask, so nothing outside the mask changes.

## Performance (Intel Arc 140T iGPU, 16 GB shared, PyTorch XPU)

| stage | model | time | peak memory | weights |
|---|---|---|---|---|
| segmentation | ensemble | ~4 s (×N with `--tta N`) | 4.2 GB | 0.34 + 0.94 GB |
| segmentation | SAM 3 | ~8 s | 5.2 GB | 3.4 GB |
| geometry | MoGe-2 ViT-L | ~4 s | 2.6-2.8 GB | 1.3 GB |
| illumination | RGB->X, 10 steps | 9-14 s | 3.7-4.6 GB | 4.9 GB |
| illumination | heuristic | < 0.1 s | — | — |
| render | CPU | 0.3-0.5 s | — | — |

≈ 17-26 s per image with the defaults, excluding model loads (3-4 s each) and the 5-15 s kernel warm-up the first
call in a process pays. On a CUDA GPU the same script runs unchanged (`DEV` picks cuda > xpu > cpu).

## Requirements

- Python ≥ 3.10; `torch` (tested 2.13 XPU), `transformers` ≥ 5, `diffusers` ≥ 0.30, `opencv-python(-headless)`,
  `numpy`, `pillow`, `huggingface_hub`, `utils3d_moge` (for MoGe: `pip install "utils3d_moge @ git+https://github.com/EasternJournalist/utils3d-moge.git@62f09d5"`).
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

- **Tile size.** `--tile-m` is the real-world size of one repeat of `pattern.png`; if the pattern image itself contains
  several tiles, set `--tile-m` to the physical width of the whole image.
- **Textured old floors.** If remnants of the old floor show through, raise `--line-px` (wider grout) or switch to
  `--illum rgbx`; if a thin shadow edge gets softened, lower it.
- **Dark floors.** The heuristic illumination reads dark tiles as shadow — use `rgbx` (default) there.
- **Non-planar floors / multiple levels** are not handled: one plane is fitted to the whole mask.
- **Reflections and gloss** are not synthesised; the render keeps diffuse lighting only. A relighting or blending model
  (e.g. LBM, FLUX Kontext) can be run on `output.jpg` afterwards if a glossier look is wanted.
- SAM 3 and RGB->X exceed the project's usual 1 GB-per-file download limit; they were fetched with explicit approval.

## Provenance

Chosen after ablations over 10 editing/segmentation/geometry models (MatSwap, ControlTile, LBM, MARBLE, IntrinsicEdit,
FLUX Kontext, Qwen-Image-Edit, Insert Anything, SAM 2.1/3, Mask2Former, OneFormer, SegFormer, UPerNet, Intrinsic v1/v2.1,
RGB->X). The result CSVs are in `../ablations/`; the geometric render + illumination map won on speed (~20 s vs
2.5-25 min for the diffusion editors), product fidelity (CLIP similarity to the pattern 0.80 vs 0.55-0.69) and zero
change outside the mask.
