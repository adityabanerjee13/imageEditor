# imageEditor — floor re-tiling pipeline

Replace the floor in a room photo with a tiled texture while keeping the scene's lighting and shadows.

## Run
```
python floor_edit/floor_edit.py --scene input/scene.jpeg --pattern input/pattern.png            # defaults: --seg ensemble --illum rgbx
python floor_edit/floor_edit.py --scene input/black_tile_flooring.png --seg sam3 --illum heuristic
```
Output: `outputs/floor_edit/<scene>/<pattern>/<seg>/output.jpg` (+ `mask_floor.png`, `illumination.png`, `metrics.json` with per-stage time / GPU peak).
Options: `--seg {ensemble,sam3}`, `--illum {rgbx,heuristic}`, `--tile-m` (texture repeat in metres, 0.6), `--rgbx-steps` (10), `--line-px` (21).

## Layout
- `floor_edit/floor_edit.py` — the whole pipeline in one file (segmentation -> MoGe-2 geometry -> illumination -> plane-fit render).
- `weights/` — `segformer-b5-ade`, `upernet-convnext-l` (ensemble), `sam3`, `moge-2-vitl-normal`, `rgb-to-x` (~11 GB total).
- `repos/MoGe` (`moge.model.v2`), `repos/rgbx` (rgb2x diffusers pipeline) — code dependencies only.
- `ablations/` — result CSVs of all earlier experiments (model survey timings/quality, segmenter comparison, pipeline-v1 stage metrics, illumination-model timings). Everything else from those runs was deleted.
- `input/` — `scene.jpeg` (bedroom 900x1600), `black_tile_flooring.png` (kitchen 1080x1080), `pattern.png` (marble).

## Pipeline
1. **Segmentation** — default: mean-softmax ensemble of SegFormer-B5 + UPerNet ConvNeXt-L (ADE20K, inference 1024x576), floor = {floor, rug}, then 5x5 close + small-component removal. Alternative: SAM 3 with text prompt "floor" (instances > 0.4 unioned). Ensemble has the cleanest boundaries (boundary-F 0.81 vs 0.25 for SAM 3 on the bedroom); SAM 3 is slower and heavier.
2. **Geometry** — MoGe-2 ViT-L point map + intrinsics (fp32; fp16 autocast is broken on XPU).
3. **Illumination** — default RGB->X "Irradiance (diffuse lighting)": **linear RGB in (sRGB^2.2), 10 DDIM steps at 768 px short side, output gamma-decoded (^2.2)** — without this treatment the map is nearly flat. Heuristic = luminance / 90th-percentile floor level.
4. **Render** — least-squares floor plane on the mask's 3-D points (2 inlier re-fits, residual ~0.2 cm), ray/plane intersection -> metric (u,v), texture tiled at `tile_m`, illumination normalised on the floor then grayscale close->open (`line_px`) to drop old grout lines / specks while keeping cast shadows, 2 px blur, composited through a 1 px soft mask.

## Environment
Windows 11, Python 3.13, PyTorch 2.13 XPU (Intel Arc 140T iGPU, 16 GB shared), no CUDA. Typical run: ensemble 4 s, MoGe-2 4 s, RGB->X 9-14 s (incl. first-call warm-up), render 0.3-0.5 s -> ~17-26 s per image excluding model loads.
Constraints: per-file download cap 1 GB (exceptions were granted for SAM 3 3.4 GB and RGB->X 4.9 GB); `facebook/sam3` is gated on this account -> weights came from the `jetjodh/sam3` mirror.

## Gotchas
- Keep GPU runs sequential; the first model call in a process pays 5-15 s of kernel warm-up.
- `weights/*/.cache` may hold stale `.incomplete` downloads from interrupted fetches — delete them.
- Do not spawn subagents for this project.

# object_edit — object move pipeline (FreeFine only; consolidated 2026-09-19)
Move the object under `initial` (x,y) so that point lands on `final`. Inputs: image + two points only — no object name is given to any model (user requirement; FreeFine's GeoBench scripts use `guidance_text=""` / `"empty scene"`).
```
python object_edit/object_edit.py --coords input/object_move.json          # kitchen: potted plant (282,440) -> (918,442)
python object_edit/object_edit.py --coords input/object_move_scene.json    # bedroom: chair (170,713) -> (730,714)
```
Output: `outputs/object_edit/<scene>/output.jpg` + intermediates + `metrics.json`. Single file `object_edit/object_edit.py` (docstring documents the stages).
- Stages: SAM 3 one-click mask (largest proposal < 30 % of frame) -> MoGe-2 perspective scale z_src/z_dst (`--scale` overrides) -> square ROI covering both positions at 512 px -> object removal `--removal lama` (default; big-lama at native res, hole = mask dilated by 6 px, ~5 s) or `freefine` (bg-gen, `"empty scene"`, ~90 s) -> affine paste -> FreeFine regeneration (`start_step` 15; benchmark 35) -> only hole+target composited back.
- Weights: sam3, moge-2-vitl-normal (shared with floor_edit), stable-diffusion-v1-5 fp16 (2 GB), big-lama (410 MB). Code deps: repos/FreeFine (rembg stubbed; one `.cuda()` patched at runtime), repos/MoGe, repos/lama (its `data.aug` module stubbed; ckpt loaded with weights_only=False). ~1.7 min per image on the Arc iGPU with LaMa removal (regen ~85 s, peak 7.1 GB).
- Known limits: one click selects one SAM object (the towel on the chair and cast shadows are not in the mask); no shadow synthesis.
- Removed on consolidation (2026-09-19): Depth-Aware-Editing, DragFlow, SH-GAN repos/venvs and their weights (AnyDoor, SD-2.1, FLUX Q8, T5-XXL); LaMa was re-added the same day as the default removal. Findings: Depth-Aware-Editing worked (AnyDoor placement, ~15 min, hallucinates small items); DragFlow (FLUX.1-dev) OOMs on the 16 GB iGPU; SH-GAN weights are dead links.
- Gated on this account: `black-forest-labs/FLUX.1-*`, `stabilityai/stable-diffusion-2-*` (mirrors exist: `second-state/FLUX.1-dev-GGUF`, `sd2-community/...`).
- iGPU "device" memory is carved from system RAM: never keep two large models resident.
