"""The single GPU worker: every model call in the process goes through one thread, one at a time.

Interactive SAM requests (encode / propose) have priority over jobs but never pre-empt a running job - while a 2-minute
object edit runs, proposals wait.  The SAM model is dropped before a job starts (16 GB shared iGPU) and reloaded lazily."""
import importlib.util
import queue
import threading
import time
import traceback
from concurrent.futures import Future
from pathlib import Path

import numpy as np
from PIL import Image

from object_edit import flux_omnipaint, sd15
from object_edit.common import ROOT, empty_cache
from object_edit.object_edit import run as run_object_edit
from object_edit.task import Move, MoveTask, default_config
from object_edit.preclean import DDRMClean
from server.sam_session import SamSession

# floor_edit/ is a plain script directory (no package): load the module from its file
_spec = importlib.util.spec_from_file_location("floor_edit_pipeline", ROOT / "floor_edit" / "floor_edit.py")
floor_edit = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(floor_edit)

PRIO_INTERACTIVE, PRIO_JOB = 0, 1

# expected stage order per job kind, to show the stage *in progress* (Stage rows are only logged when a stage ends)
STAGE_ORDER = {
    "floor": ["segmentation", "geometry", "irradiance", "render"],
    "object": ["segmentation", "mask_dilation", "geometry", "downsample", "background_generation", "load", "regeneration", "super_resolution"],
    "image3d": ["preprocess", "reconstruct"],
}
INTERMEDIATES = {
    "floor": ["mask_floor.png", "irradiance.png", "ao.png", "albedo_old.png"],
    "object": ["task_points.png", "mask_object.png", "mask_dilated.png", "region_source.png", "input_lowres.png", "background.png",
               "output_lowres.png", "output_sr_full.png"],
    "image3d": ["subject.png", "input_processed.png", "preview.png"],
}


class GpuWorker:
    def __init__(self, registry, uploads):
        self.registry, self.uploads = registry, uploads
        self.sam = SamSession()
        self.clean = DDRMClean()
        self.q = queue.PriorityQueue()
        self.seq = 0
        self.busy_with = None                     # job id while a job runs
        self.thread = threading.Thread(target=self._loop, name="gpu-worker", daemon=True)
        self.thread.start()

    # ---- submission
    def submit(self, fn, priority=PRIO_JOB):
        fut = Future()
        self.seq += 1
        self.q.put((priority, self.seq, fn, fut))
        return fut

    def preclean(self, image_id):
        """DDRM pre-clean of an uploaded scene -> data/uploads/<id>_clean.png (the working image from here on)."""
        def fn():
            src = self.uploads.orig_path(image_id)
            t = time.time()
            out = self.clean(np.asarray(Image.open(src).convert("RGB")))
            dst = self.uploads.clean_path(image_id)
            Image.fromarray(out).save(dst)
            print(f"[preclean] {image_id}: {self.clean.describe()} in {time.time() - t:.1f} s")
            return dst
        return self.submit(fn, PRIO_INTERACTIVE)

    def encode(self, image_id):
        return self.submit(lambda: self.sam.encode(image_id, self.uploads.open(image_id)), PRIO_INTERACTIVE)

    def propose(self, image_id, box):
        def fn():
            if not self.sam.has(image_id):
                self.sam.encode(image_id, self.uploads.open(image_id))
            return self.sam.propose(image_id, box)
        return self.submit(fn, PRIO_INTERACTIVE)

    def run_job(self, job):
        return self.submit(lambda: self._run_job(job), PRIO_JOB)

    def _loop(self):
        while True:
            _, _, fn, fut = self.q.get()
            try:
                fut.set_result(fn())
            except BaseException as e:       # noqa: BLE001 - the worker must survive any failure
                traceback.print_exc()
                fut.set_exception(e)

    # ---- jobs
    def _run_job(self, job):
        job.status, job.started = "running", time.time()
        self.busy_with = job.id
        order = STAGE_ORDER[job.kind]
        job.stage = order[0]

        def on_stage(row):
            job.stages_done.append({k: v for k, v in row.items() if k in ("stage", "model", "time_s", "gen_s", "gpu_peak_gb")})
            i = order.index(row["stage"]) if row["stage"] in order else -1
            job.stage = order[i + 1] if 0 <= i < len(order) - 1 else None

        try:
            self.sam.unload(); self.clean.unload()
            if job.kind == "floor":
                meta = self._floor(job, on_stage)
            elif job.kind == "image3d":
                meta = self._image3d(job, on_stage)
            else:
                meta = self._object(job, on_stage)
            files = [p.name for p in sorted(job.out_dir.iterdir()) if p.suffix == ".png"]
            ordered = [f for f in INTERMEDIATES[job.kind] if f in files] + [f for f in files if f not in INTERMEDIATES[job.kind]]
            # image3d produces a mesh, not an edited photo: its "output" image is the turntable preview
            out_name = "preview.png" if job.kind == "image3d" else "output.jpg"
            job.result = {"output_url": f"/files/jobs/{job.id}/{out_name}",
                          "intermediates": [{"name": f, "url": f"/files/jobs/{job.id}/{f}"} for f in ordered],
                          "metrics": meta}
            if job.kind == "image3d":
                # The splat is what the UI shows: the mesh decoder's per-vertex colour channel
                # loses ~37% saturation, and DIRECT itself only ever renders the Gaussian.
                job.result["model_url"] = f"/files/jobs/{job.id}/mesh.glb"
                job.result["splat_url"] = f"/files/jobs/{job.id}/gaussian.ply"
            job.status = "done"
        except Exception as e:                # noqa: BLE001
            traceback.print_exc()
            job.status, job.error = "failed", f"{type(e).__name__}: {e}"
        finally:
            job.stage, job.finished, self.busy_with = None, time.time(), None
            sd15.release(); flux_omnipaint.release(); empty_cache()      # a failed job must not leave a model (or the FLUX worker) behind

    def _floor(self, job, on_stage):
        s = job.spec
        return floor_edit.run_floor_edit(self.uploads.path(s["scene_id"]), self.uploads.path(s["pattern_id"]), job.out_dir,
                                         material=s.get("material", "smooth-matte"),
                                         m_per_1000px=s.get("m_per_1000px", floor_edit.M_PER_1000PX),
                                         on_stage=on_stage)

    def _image3d(self, job, on_stage):
        """One SAM-selected object -> a 3D proxy (imageto3D / TRELLIS via DIRECT's pipeline)."""
        from imageto3D.image_to_3d import configure_backends, image_to_3d, subject_from_mask

        configure_backends()                  # must precede the trellis import inside image_to_3d
        s = job.spec
        scene = Image.open(self.uploads.path(s["scene_id"])).convert("RGB")
        mask = Image.open(self.uploads.mask_path(s["mask_id"])).convert("L")
        if mask.size != scene.size:
            raise ValueError(f"mask is {mask.size}, scene is {scene.size}")
        subject = subject_from_mask(scene, mask)
        subject_path = job.out_dir / "subject.png"
        subject.save(subject_path)
        _, meta = image_to_3d(subject_path, out_dir=job.out_dir, seed=s.get("seed", 42),
                              ss_steps=s.get("ss_steps"), slat_steps=s.get("slat_steps"),
                              on_stage=on_stage)
        return meta

    def _object(self, job, on_stage):
        s = job.spec
        scene = self.uploads.path(s["scene_id"])
        image = Image.open(scene).convert("RGB")
        W, H = image.size
        moves = []
        for m in s["moves"]:
            mask = np.asarray(Image.open(self.uploads.mask_path(m["mask_id"])).convert("L")) > 127
            if mask.shape != (H, W):
                raise ValueError(f"mask {m['mask_id']} is {mask.shape[::-1]}, scene is {W}x{H}")
            if not mask.any():
                raise ValueError(f"mask {m['mask_id']} is empty")
            sb, db = m["src_box"], m["dst_box"]
            src = (int(round((sb[0] + sb[2]) / 2)), int(round((sb[1] + sb[3]) / 2)))
            dst = (int(round((db[0] + db[2]) / 2)), int(round((db[1] + db[3]) / 2)))
            moves.append(Move(src, dst, mask=mask, src_box=list(sb), dst_box=list(db)))
        task = MoveTask(Path(scene), image, moves)
        cfg = default_config(removal=s.get("removal", "omnipaint"), insertion=s.get("insertion", "omnipaint"),
                             omnipaint_mode=s.get("omnipaint_mode", "full"), omnipaint_res=s.get("omnipaint_res", 1024),
                             omnipaint_steps=s.get("omnipaint_steps", 28), src_region=s.get("src_region", "full"),
                             dst_region=s.get("dst_region", "full"), region_margin=s.get("region_margin", 0.35),
                             sr_factor=s.get("sr_factor", 1))
        return run_object_edit(task, cfg, job.out_dir, on_stage=on_stage)
