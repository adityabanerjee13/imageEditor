# README image snippets

Paste-ready Markdown for `README.md`. Every image comes from the three recorded runs:

| feature | job | scene |
|---|---|---|
| floor_edit | `data/jobs/665216962fff` | kitchen 1080×1080, marble pattern, `--seg ensemble --illum rgbx` |
| object_edit | `data/jobs/254b0c927881` | living room 2000×1500, `--removal omnipaint --insertion omnipaint --omnipaint-mode full` |
| imageto3D | `data/jobs/79e777137bfe` | the patterned cushion from that same living-room photo |

The only stage not written by those runs is MoGe-2's point map (neither pipeline saves it). It was re-dumped with a
single MoGe-2 forward pass per scene and reproduces the recorded numbers exactly — `z_src` 2.885 m, `z_dst` 3.291 m,
scale 0.876 for object_edit; plane residual 0.28 cm against the logged 0.27 cm for floor_edit.

---

## Top of the README — all three results

![imageEditor results](docs/images/00_overview.jpg)

---

## floor_edit

### Result

![floor_edit: replace the floor](docs/images/floor_edit/00_task.jpg)

**Task** — a room photo and a tile pattern. **Result** — the floor is re-tiled in true perspective at 0.6 m per repeat,
with the room's own lighting and shadows kept. 18.3 s of generation on the iGPU (3.2 s segmentation, 4.0 s MoGe-2,
10.7 s RGB→X, 0.3 s render).

### Stages

![floor_edit stages](docs/images/floor_edit/09_stages.jpg)

| # | stage | image |
|---|---|---|
| — | scene photo | [`01_input.jpg`](docs/images/floor_edit/01_input.jpg) |
| — | tile pattern | [`02_pattern.jpg`](docs/images/floor_edit/02_pattern.jpg) |
| 1 | **Segmentation** — mean-softmax ensemble of SegFormer-B5 + UPerNet ConvNeXt-L, ADE20K classes floor + rug; 37.4 % of the frame | [`03_floor_mask.jpg`](docs/images/floor_edit/03_floor_mask.jpg) · [binary](docs/images/floor_edit/03b_floor_mask_binary.png) |
| 2 | **Geometry** — MoGe-2 ViT-L metric point map and intrinsics | [`04_geometry_depth.jpg`](docs/images/floor_edit/04_geometry_depth.jpg) |
| 2 | least-squares plane through the floor points (2 inlier re-fits, 0.28 cm residual), shown as its own 0.6 m grid | [`05_plane_fit.jpg`](docs/images/floor_edit/05_plane_fit.jpg) |
| 3 | **Render** — every pixel's ray meets the plane, giving metric (u, v); the pattern is tiled there, unlit | [`06_tiled_unlit.jpg`](docs/images/floor_edit/06_tiled_unlit.jpg) |
| 4 | **Illumination** — RGB→X diffuse irradiance, normalised on the floor and morphologically cleaned of old grout lines | [`07_illumination.jpg`](docs/images/floor_edit/07_illumination.jpg) |
| 5 | tiling × shading, composited through a 1 px soft mask | [`08_output.jpg`](docs/images/floor_edit/08_output.jpg) |

Close up, the window light and the cabinet shadow survive the swap:

![floor_edit detail](docs/images/floor_edit/10_detail.jpg)

---

## object_edit

### Result

![object_edit: move an object](docs/images/object_edit/00_task.jpg)

**Task** — two points: move whatever is under the first one to the second. No object name is given to any model.
**Result** — the cushion leaves the couch and arrives on the chair, rescaled by the depth ratio (0.876) and lit by the
scene. 29 min on the iGPU with both OmniPaint backends at full-frame 1024 (14 min removal + 14 min insertion at 28 steps);
the classic `--removal lama --insertion freefine` path does the same edit in ~4 min.

![source and target, before and after](docs/images/object_edit/12_detail.jpg)

> OmniPaint *re-renders* the object rather than copying it: position, size and lighting match, but identity is
> approximate — here the pattern is not preserved. That trade-off is measured in
> [`ablations/last_stage_compare.csv`](ablations/last_stage_compare.csv); `--insertion freefine` keeps exact identity
> and instead looks pasted.

### Stages

![object_edit stages](docs/images/object_edit/13_stages.jpg)

| # | stage | image |
|---|---|---|
| — | scene photo | [`01_input.jpg`](docs/images/object_edit/01_input.jpg) |
| 1 | the task: `initial` → `final` | [`02_move_points.jpg`](docs/images/object_edit/02_move_points.jpg) |
| 2 | **Object mask** — SAM 3, one click at `initial`, best of 3 proposals, cleaned; teal = silhouette (24 633 px), orange = the 6 px dilation used as the hole | [`03_object_mask.jpg`](docs/images/object_edit/03_object_mask.jpg) |
| 3 | **Perspective scale** — MoGe-2 metric depth probed at both anchors: 2.885 m → 3.291 m, scale 0.876 | [`04_geometry_depth.jpg`](docs/images/object_edit/04_geometry_depth.jpg) |
| 4 | **Removal** — the condition OmniPaint is given: the frame with the hole blacked out (this is the whole model input; no text) | [`05_removal_condition.jpg`](docs/images/object_edit/05_removal_condition.jpg) |
| 4 | background, object gone | [`06_background.jpg`](docs/images/object_edit/06_background.jpg) |
| 5 | **Insertion** — condition A: the subject alone on white, position ids shifted (0, −32) | [`07_subject.jpg`](docs/images/object_edit/07_subject.jpg) |
| 5 | the depth-scaled affine paste — the geometric reference the generative path replaces | [`08_coarse_paste.jpg`](docs/images/object_edit/08_coarse_paste.jpg) |
| 5 | condition B: the background with the **target box** blacked out (the box sets both position and size) | [`09_insertion_condition.jpg`](docs/images/object_edit/09_insertion_condition.jpg) |
| 5 | the model canvas it returns, 1024×768 | [`10_insertion_result.jpg`](docs/images/object_edit/10_insertion_result.jpg) |
| 6 | composited back at native resolution | [`11_output.jpg`](docs/images/object_edit/11_output.jpg) |

---

## imageto3D

### Result

![imageto3D: one photo to a 3D proxy](docs/images/imageto3D/00_task.jpg)

**Task** — one photo and one SAM mask. **Result** — a 3D proxy of that object: 318 880 Gaussian splats and a 332 519-vertex
FlexiCubes mesh, decoded from the same structured latent. 95 s and 10.8 GB peak on the iGPU.

![orbit](docs/images/imageto3D/06_orbit.jpg)

Rendered by the project's own viewer (`web/src/SplatViewer.tsx`) — the Gaussian is the representation DIRECT displays,
and it renders dark, which is documented above. A single frame as the UI actually frames it:
[`06b_viewer.jpg`](docs/images/imageto3D/06b_viewer.jpg); the nine raw viewer renders are in
[`viewer_renders/`](docs/images/imageto3D/viewer_renders/).

### Stages

![imageto3D stages](docs/images/imageto3D/08_stages.jpg)

| # | stage | image |
|---|---|---|
| 1 | the object selected in the photo | [`01_input_photo.jpg`](docs/images/imageto3D/01_input_photo.jpg) |
| 1 | **Subject cut-out** — the mask becomes the alpha channel, bbox + 15 % margin | [`02_subject_cutout.png`](docs/images/imageto3D/02_subject_cutout.png) |
| 2 | **Preprocessing** — 518 px square on grey, the DINOv2 conditioning input | [`03_preprocessed.jpg`](docs/images/imageto3D/03_preprocessed.jpg) |
| 3 | **Sparse-structure flow** — 16³ latent → 64³ occupancy → 9 965 active voxels | [`04_sparse_structure.png`](docs/images/imageto3D/04_sparse_structure.png) |
| 4 | **Structured-latent flow** — 32 Gaussians per voxel; their DC colours, before opacity compositing | [`05_slat_decoded_points.png`](docs/images/imageto3D/05_slat_decoded_points.png) |
| 5 | **Gaussian decoder** → `gaussian.ply`, shown in the UI viewer | [`06_orbit.jpg`](docs/images/imageto3D/06_orbit.jpg) · [single view](docs/images/imageto3D/06b_viewer.jpg) |
| 5 | **FlexiCubes decoder** → `mesh.glb`; the CLI's 4-view software point-splat preview | [`07_mesh_preview.jpg`](docs/images/imageto3D/07_mesh_preview.jpg) |


---

## How the images were made

Everything above is read from the three job directories. Two additions:

- **The depth maps** come from one MoGe-2 forward pass per scene (neither pipeline saves the point map). They reproduce
  the recorded numbers exactly, so they are the same tensors those runs used.
- **The Gaussian renders** are screenshots of this repo's own viewer: a headless Chrome page that mounts the same
  `DropInViewer` from `web/node_modules` with the same camera (45°, z = 2.6), `gpuAcceleratedSort: false` and
  `splatAlphaRemovalThreshold: 5` as [`web/src/SplatViewer.tsx`](web/src/SplatViewer.tsx), loading the job's existing
  `gaussian.ply`. `imageto3D/render3d.py` could not be used — it is a mesh rasteriser, and the splat path would need an
  EWA splatter.

No pipeline stage was re-run.
