# imageEditor

Two local, non-generative-first editing pipelines for room photos, built for a 16 GB Intel Arc iGPU (PyTorch XPU; runs
unchanged on CUDA or CPU):

| pipeline | what it does | script | time on the iGPU |
|---|---|---|---|
| **floor_edit** | replace the floor with a tiled texture in true perspective and scale, keeping the photo's shadows and lighting | `floor_edit/floor_edit.py` | ~20 s |
| **object_edit** | move the object under a click point to another point, with perspective rescale and background fill | `object_edit/object_edit.py` | ~1.7 min |

Nothing outside the edited region is touched, and no object names or captions are given to any model.

## Quickstart

```bash
# 1. PyTorch for your accelerator (pick one), then the rest
pip install torch torchvision --index-url https://download.pytorch.org/whl/xpu     # Intel XPU
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128   # CUDA
pip install -r requirements.txt

# 2. code dependencies (repos/) + model weights (weights/, ~13 GB from Hugging Face)
python setup.py                    # or --pipeline floor | object, --check to see what is present

# 3. run
python floor_edit/floor_edit.py --scene input/scene.jpeg --pattern input/pattern.png
python object_edit/object_edit.py --coords input/object_move.json
```

Outputs land in `outputs/floor_edit/<scene>/<pattern>/<seg>/output.jpg` and `outputs/object_edit/<scene>/output.jpg`,
each with the intermediate masks / maps and a `metrics.json` (per-stage time, peak accelerator memory).

`setup.py` is a bootstrap script, not a setuptools file: it clones `MoGe`, `rgbx`, `FreeFine`, `lama` at pinned
commits and downloads the weights below with `huggingface_hub`. It skips anything already present, so it can be re-run
after an interrupted download. `facebook/sam3` is gated, so the `jetjodh/sam3` mirror is used; set `HF_TOKEN` if a
repo asks for it.

| weights folder | source | used by | size |
|---|---|---|---|
| `segformer-b5-ade` | `nvidia/segformer-b5-finetuned-ade-640-640` | floor | 0.3 GB |
| `upernet-convnext-l` | `openmmlab/upernet-convnext-large` | floor | 0.9 GB |
| `sam3` | `jetjodh/sam3` (mirror of `facebook/sam3`) | both | 3.4 GB |
| `moge-2-vitl-normal` | `Ruicheng/moge-2-vitl-normal` | both | 1.3 GB |
| `rgb-to-x` | `zheng95z/rgb-to-x` | floor | 4.9 GB |
| `stable-diffusion-v1-5` | `stable-diffusion-v1-5/stable-diffusion-v1-5` (fp16) | object | 2.0 GB |
| `big-lama` | `smartywu/big-lama` | object | 0.4 GB |

## floor_edit

```
photo ──► 1. floor mask ─────────────────────────────────┐
      ├─► 2. MoGe-2 point map + intrinsics ─► plane fit ─┤
      └─► 3. illumination map ───────────────────────────┤
                                                         ▼
pattern ─────────────────────► 4. tile on the plane × shading, composite ──► output.jpg
```

1. **Segmentation** — mean-softmax ensemble of SegFormer-B5 + UPerNet ConvNeXt-L (ADE20K classes floor + rug), or SAM 3
   with the text prompt "floor" (`--seg sam3`). Optional test-time augmentation with majority vote (`--tta N`).
2. **Geometry** — MoGe-2 ViT-L metric point map and camera intrinsics; a least-squares plane is fitted to the floor points.
3. **Illumination** — RGB→X "diffuse irradiance" (linear-RGB in, gamma-decoded out) or a luminance heuristic (`--illum`).
   Thin structures (old grout lines, specks) are filtered out so only real shadows and light fall-off remain.
4. **Render** — each pixel's ray is intersected with the plane to get metric floor coordinates, the pattern is tiled at
   `--tile-m` metres per repeat, multiplied by the shading map and composited through a soft floor mask.

Details, options, timings and limits: [floor_edit/README.md](floor_edit/README.md).

## object_edit

Input is a JSON with the scene and one or more `{"initial": {x, y}, "final": {x, y}}` pairs (see `input/object_move*.json`).

1. **Mask** — SAM 3 tracker, one positive click at `initial`; the largest proposal under 30 % of the frame is kept.
2. **Scale** — MoGe-2 depth ratio `z(initial) / z(final)` gives the perspective rescale (`--scale` overrides).
3. **ROI** — a square window covering source and destination is edited at 512 px; the rest of the frame is untouched.
4. **Removal** — LaMa fills the dilated source hole at native resolution (`--removal freefine` uses FreeFine's
   background generation instead).
5. **Paste + refine** — affine copy of the object to its destination, then FreeFine detail-preserving regeneration
   (empty guidance text, DDIM inversion against the original, `--start-step`).
6. **Composite** — hole from the removal, target region from the regeneration, everything else original.

Known limits: one click selects one SAM object (attached items and cast shadows are not carried along); no shadow synthesis.

## Layout

```
floor_edit/floor_edit.py    whole floor pipeline (docstring documents stages and flags)
object_edit/object_edit.py  whole object-move pipeline
input/                      test scenes, marble pattern, object-move JSONs
ablations/                  CSVs from the model survey and stage ablations that led to these choices
setup.py, requirements.txt  environment bootstrap
repos/, weights/, outputs/  created by setup.py / the scripts; git-ignored
```

## Notes

- Tested on Windows 11, Python 3.13, PyTorch 2.13 XPU (Intel Arc 140T, 16 GB shared memory). The device is chosen
  automatically (cuda > xpu > cpu). On XPU, MoGe-2 runs in fp32 (fp16 autocast is broken there).
- iGPU memory is carved from system RAM: run GPU jobs sequentially and never keep two large models resident. The first
  model call in a process pays 5-15 s of kernel warm-up.
- The choice of every model comes from the experiments in `ablations/`: a 10-model survey of generative editors
  (MatSwap, ControlTile, LBM, FLUX Kontext, Qwen-Image-Edit, …) was dropped in favour of the geometric render because
  it is 5-70× faster, keeps the pattern faithful and changes nothing outside the mask.
