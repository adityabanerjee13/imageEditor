"""Object move: shift the object(s) under `initial` (x, y) so that point lands on `final`.

    python object_edit/object_edit.py --coords input/object_move.json
    python object_edit/object_edit.py --coords input/object_move_scene.json [--scale 1.0] [--start-step 35]
    python object_edit/object_edit.py --coords input/object_move.json --removal lama-plain --insertion paste   # fast, no SD

Inputs are only the image and point pairs - no object name or caption is given to any model.
input/*.json: {"scene": ..., "moves": [{"initial": {x, y}, "final": {x, y}}, ...]}
(a single {"initial", "final"} pair at the top level is also accepted; per move "src_box"/"dst_box" [x0,y0,x1,y1] may
replace the points - their centres are used - and "mask" (binary PNG path) supplies the object mask, skipping SAM).

Architecture - four blocks, the last two are swappable modules (see object_edit/removal, object_edit/insertion):
  1. object masks      segment.py   SAM 3 tracker, one positive click per object at `initial`; of the three proposals
                                    (sub-part / part / whole) the best-IoU one is taken, upgraded to a larger proposal
                                    only when that is < 4x its area, then reduced to the component containing the click.
                                    Each mask is grown by --dilate-radius (6) px: the union is the removal hole
  2. perspective scale geometry.py  MoGe-2 metric depth, per object: scale = z(initial) / z(final)  (--scale overrides)
  3. object removal    removal/     --removal lama (default: big-lama + Geomagical feature refinement, native res, all
                                    holes at once) | lama-plain (no refinement, ~5 s) | freefine (SD-1.5 bg-gen,
                                    "empty scene", 512 window) | omnipaint (FLUX.1-dev Q8 + removal LoRA, one 512 window
                                    per hole, ~9 min each).  Original pixels are kept outside the holes
  4. object insertion  insertion/   --insertion freefine (default: affine paste into a 512 window ~3x the object, then
                                    FreeFine detail-preserving regeneration from --start-step with the source window as
                                    reference, feathered composite of the target region) | paste (affine paste only) |
                                    omnipaint (FLUX.1-dev Q8 + insertion LoRA: subject cut-out + rectangular target in
                                    the same window, generative re-render, ~12 min per object).
                                    Objects are inserted in order; later ones see earlier insertions.

Edit regions (user choice, --src-region / --dst-region, each mask | dilated | box | full; --region-margin): how far around
the object the remover / inserter may repaint.  "mask" is the cut-paste baseline (nothing outside the silhouette changes,
old shadows stay); "dilated" and "box" open a band around it so the generative backends can remove the cast shadow at the
source and render one at the target; "full" keeps everything the model renders (OmniPaint / FreeFine only).

Working resolution (--sr-factor 2|3|4): the frame is block-averaged before removal / insertion, which then run on the small
frame, and the result is super-resolved with DDRM (deepinv, DRUNet denoiser, --sr-steps DDIM timesteps; object_edit/
resolution.py); only the pixels the edit changed are taken from the super-resolved image, everything else stays native.

Output: <out>/output.jpg (+ mask_object.png, mask_dilated.png, region_source.png, background.png, task_points.png,
[with --sr-factor: input_lowres.png, output_lowres.png, output_sr_full.png]
per object: obj<i>_source_512.png, obj<i>_coarse_512.png, obj<i>_target_512.png, obj<i>_result_512.png; metrics.json)
Default <out> = outputs/object_edit/<scene>.

Weights (weights/): sam3, moge-2-vitl-normal, stable-diffusion-v1-5 (fp16), big-lama; omnipaint backends: flux1-dev-gguf,
flux-vae, omnipaint (LoRAs + prompt embeddings).  Code: repos/FreeFine, repos/MoGe, repos/lama, repos/OmniPaint.
"""
import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:                 # `python object_edit/object_edit.py` puts object_edit/ first; the package must win
    sys.path.insert(0, str(ROOT))

from object_edit import flux_omnipaint, sd15                                # noqa: E402
from object_edit.common import DEV, OUT_ROOT, Stage, StageLog, draw_points, empty_cache  # noqa: E402
from object_edit.geometry import estimate_perspective_scale                 # noqa: E402
from object_edit.insertion import INSERTERS, make_inserter                  # noqa: E402
from object_edit.removal import REMOVERS, make_remover                      # noqa: E402
from object_edit.resolution import WorkingResolution                        # noqa: E402
from object_edit.segment import dilate_object_masks, segment_objects        # noqa: E402
from object_edit.task import MoveTask, default_config, load_task            # noqa: E402

__all__ = ["run", "load_task", "default_config", "MoveTask", "REMOVERS", "INSERTERS"]


def run(task, cfg=None, out_dir=None, on_stage=None):
    """Run the four blocks on `task` (task.MoveTask). `cfg` from task.default_config(); `on_stage(row)` is called after
    every finished stage (server progress). Returns the metrics dict; files are written to `out_dir`."""
    cfg = cfg or default_config()
    log = StageLog(on_stage)
    od = Path(out_dir) if out_dir else OUT_ROOT / task.scene_path.stem
    od.mkdir(parents=True, exist_ok=True)
    seed = cfg["seed"] if cfg["seed"] >= 0 else random.randint(0, 2 ** 31)

    # ---- 0. pre-clean (server uploads are already pre-cleaned; the CLI opts in with --preclean)
    if cfg["preclean"]:
        from object_edit.preclean import preclean
        task.image = Image.fromarray(preclean(np.asarray(task.image), log))
        task.image.save(od / "input_preclean.png")

    # ---- 1. masks (skipped when the task already carries them, e.g. chosen in the UI) + dilation
    if not task.has_masks:
        segment_objects(task, log)
    remover, inserter = make_remover(cfg), make_inserter(cfg)
    src_region = cfg["src_region"]
    if src_region == "full" and not remover.generative:
        print(f"[removal] {remover.name} can only fill a hole, not re-render the frame: source region 'full' -> 'box'")
        src_region = "box"
    dilate_object_masks(task, cfg["dilate_radius"], log, src_region, cfg["region_margin"])
    Image.fromarray(task.mask.astype(np.uint8) * 255).save(od / "mask_object.png")
    Image.fromarray(task.mask_dilated.astype(np.uint8) * 255).save(od / "mask_dilated.png")
    Image.fromarray(task.region.astype(np.uint8) * 255).save(od / "region_source.png")
    draw_points(task.image, task.moves).save(od / "task_points.png")

    # ---- 2. perspective scale
    if cfg["scale"] == "auto":
        estimate_perspective_scale(task, log)
    else:
        for mv in task.moves:
            mv.scale = float(cfg["scale"])

    # ---- 2b. working resolution: blur + subsample the frame (and masks / points) for the generative blocks
    native = np.asarray(task.image)
    wr = WorkingResolution(cfg["sr_factor"], cfg["sr_steps"], cfg["sr_noise"]) if int(cfg["sr_factor"]) > 1 else None
    if wr is not None:
        with Stage(log, "downsample", f"block average x{wr.sf} -> {native.shape[1] // wr.sf}x{native.shape[0] // wr.sf}"):
            _, low_u8, task = wr.down(task)
        Image.fromarray(low_u8).save(od / "input_lowres.png")

    # ---- 3. removal on the full (working) frame: the hole is the dilated silhouette, `region` (>= hole) is what the
    # remover may repaint (source edit region).  The remover is unloaded before the inserter's model is loaded so the two
    # never share the accelerator (LaMa's refiner back-propagates; SD-1.5 is 2 GB fp16) - unless both use the same model.
    original = np.asarray(task.image)
    remover.out_dir = od                      # backends that want to save what they feed the model may use it
    background = remover.remove(original, task.mask_dilated, seed=seed, log=log, region=task.region)
    Image.fromarray(background).save(od / "background.png")
    if remover.shared_model is None or remover.shared_model != inserter.shared_model:
        remover.unload()
    inserter.load(log)

    # ---- 4. insertion, one object at a time (later objects see earlier insertions)
    for i, mv in enumerate(task.moves):
        background = inserter.insert(original, background, mv, seed=seed, log=log, out_dir=od, idx=i, total=len(task.moves))
    inserter.unload()
    sd15.release(); flux_omnipaint.release(); empty_cache()

    # ---- 5. back to native resolution: GS-PnP super-resolution of the edited low-res frame; only the pixels the edit
    # changed are taken from it, the rest stays the original
    if wr is not None:
        Image.fromarray(background).save(od / "output_lowres.png")
        with Stage(log, "super_resolution", wr.describe()):
            background, hr_u8 = wr.up(native, low_u8, background)
        wr.sr.unload()
        Image.fromarray(hr_u8).save(od / "output_sr_full.png")

    Image.fromarray(background).save(od / "output.jpg", quality=95)
    meta = {"device": str(DEV), "stages": list(log), "seed": seed,
            "moves": [{"src": list(m.src), "dst": list(m.dst), "scale": round(m.scale, 3),
                       **({"src_box": m.src_box, "dst_box": m.dst_box} if m.src_box else {})} for m in task.moves],
            "removal": remover.name, "insertion": inserter.name, "src_region": src_region, "dst_region": cfg["dst_region"],
            "region_margin": cfg["region_margin"], "sr_factor": int(cfg["sr_factor"]), "sr_steps": int(cfg["sr_steps"]),
            "start_step": cfg["start_step"], "end_scale": cfg["end_scale"],
            "total_s": round(sum(r.get("time_s", 0) for r in log), 2)}
    (od / "metrics.json").write_text(json.dumps(meta, indent=2))
    print("saved", od / "output.jpg")
    return meta


def parse():
    d = default_config()
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--coords", default=str(ROOT / "input" / "object_move.json"))
    ap.add_argument("--scene", default=None, help="override the scene path in the coords file")
    ap.add_argument("--out", default=None, help="output directory (default outputs/object_edit/<scene>)")
    ap.add_argument("--scale", default=d["scale"], help="object scale at the destination: number or 'auto' (MoGe-2 depth ratio)")
    ap.add_argument("--prompt", default=d["prompt"], help='refine-stage guidance text (FreeFine benchmark: "")')
    ap.add_argument("--bg-prompt", default=d["bg_prompt"], help="background-generation text (FreeFine benchmark value)")
    ap.add_argument("--removal", default=d["removal"], choices=sorted(REMOVERS), help="object-removal backend")
    ap.add_argument("--insertion", default=d["insertion"], choices=sorted(INSERTERS), help="object-insertion backend")
    ap.add_argument("--no-lama-refine", action="store_true", help="alias for --removal lama-plain")
    ap.add_argument("--dilate-radius", type=int, default=d["dilate_radius"], help="stage 1: object mask grown by this many native px (6)")
    ap.add_argument("--start-step", type=int, default=d["start_step"], help="FreeFine: benchmark uses 35 (keeps more pasted pixels); 15 regenerates more")
    ap.add_argument("--end-scale", type=float, default=d["end_scale"])
    ap.add_argument("--omnipaint-steps", type=int, default=d["omnipaint_steps"], help="OmniPaint backends: diffusion steps (28)")
    ap.add_argument("--omnipaint-grow", type=int, default=d["omnipaint_grow"], help="OmniPaint insertion: target rectangle margin px (8)")
    ap.add_argument("--src-region", default=d["src_region"], choices=["mask", "dilated", "box", "full"],
                    help="what the remover may repaint at the source (mask = cut-paste; dilated/box = shadow band; full = whole canvas)")
    ap.add_argument("--dst-region", default=d["dst_region"], choices=["mask", "dilated", "box", "full"],
                    help="what the inserter may repaint at the target")
    ap.add_argument("--region-margin", type=float, default=d["region_margin"], help="dilated/box margin as a fraction of the object size (0.35)")
    ap.add_argument("--preclean", action="store_true", help="DDRM identity pre-clean of the input first (what the web UI does on upload)")
    ap.add_argument("--sr-factor", type=int, default=d["sr_factor"], choices=[1, 2, 3, 4],
                    help="working resolution: block-average the frame by this factor for removal/insertion, DDRM SR back (1 = native)")
    ap.add_argument("--sr-steps", type=int, default=d["sr_steps"], help="DDRM super-resolution: number of DDIM timesteps (15)")
    ap.add_argument("--sr-noise", type=float, default=d["sr_noise"], help="DDRM: measurement noise level (0.01)")
    ap.add_argument("--omnipaint-mode", default=d["omnipaint_mode"], choices=["window", "full"],
                    help="window: one 512 square per hole/object (default); full: whole frame at --omnipaint-res")
    ap.add_argument("--omnipaint-res", type=int, default=d["omnipaint_res"], help="full mode: long side of the generation canvas (1024)")
    ap.add_argument("--seed", type=int, default=d["seed"])
    return ap.parse_args()


if __name__ == "__main__":
    a = parse()
    scale = a.scale if a.scale == "auto" else float(a.scale)
    cfg = default_config(scale=scale, dilate_radius=a.dilate_radius, removal="lama-plain" if a.no_lama_refine else a.removal,
                         insertion=a.insertion, prompt=a.prompt, bg_prompt=a.bg_prompt, start_step=a.start_step,
                         end_scale=a.end_scale, seed=a.seed, omnipaint_steps=a.omnipaint_steps, omnipaint_grow=a.omnipaint_grow,
                         omnipaint_mode=a.omnipaint_mode, omnipaint_res=a.omnipaint_res,
                         src_region=a.src_region, dst_region=a.dst_region, region_margin=a.region_margin,
                         sr_factor=a.sr_factor, sr_steps=a.sr_steps, sr_noise=a.sr_noise, preclean=a.preclean)
    run(load_task(a.scene, a.coords), cfg, a.out)
