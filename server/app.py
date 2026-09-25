"""HTTP API + static frontend for the two pipelines.

    python -m uvicorn server.app:app --port 8000            # API (+ built UI from web/dist when present)
    cd web && npm run dev                                   # dev UI on :5173, proxies /api and /files to :8000

Everything model-related runs on the single GpuWorker thread (server/gpu.py); handlers only validate, store files and
enqueue.  Files live under data/ (uploads/, masks/, jobs/<id>/) and are served at /files/...
"""
import asyncio
import io
import sys
import uuid
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image, ImageOps

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from server import schemas as S                       # noqa: E402
from server.gpu import GpuWorker                       # noqa: E402
from server.jobs import JobRegistry                    # noqa: E402

DATA = ROOT / "data"
MAX_UPLOAD_MB = 20
MAX_SIDE = 2000                                        # longer uploads are downscaled (pipelines were tuned at 1080-1600 px)


# ============================================================================ file store
class Uploads:
    def __init__(self, root):
        self.dir, self.masks = root / "uploads", root / "masks"
        self.dir.mkdir(parents=True, exist_ok=True); self.masks.mkdir(parents=True, exist_ok=True)

    def save_image(self, data: bytes):
        """Decode, apply EXIF orientation (so natural pixels == what the browser shows), cap the size, store as PNG/JPEG."""
        try:
            im = Image.open(io.BytesIO(data)); im.load()
        except Exception:
            raise HTTPException(400, "not a decodable image")
        fmt = im.format
        im = ImageOps.exif_transpose(im).convert("RGB")
        if max(im.size) > MAX_SIDE:
            im.thumbnail((MAX_SIDE, MAX_SIDE), Image.LANCZOS)
        uid = uuid.uuid4().hex[:12]
        ext = "jpg" if fmt == "JPEG" else "png"
        p = self.dir / f"{uid}.{ext}"
        im.save(p, quality=95) if ext == "jpg" else im.save(p)
        return uid, p, im.size

    def orig_path(self, uid):
        for p in self.dir.glob(f"{uid}.*"):
            if not p.stem.endswith("_clean"):
                return p
        raise HTTPException(404, f"unknown image {uid}")

    def clean_path(self, uid):
        return self.dir / f"{uid}_clean.png"

    def path(self, uid):
        """The working image: the pre-cleaned version when it exists (scene uploads), else the original (patterns)."""
        c = self.clean_path(uid)
        return c if c.exists() else self.orig_path(uid)

    def open(self, uid):
        return Image.open(self.path(uid)).convert("RGB")

    def save_mask(self, mask_u8_or_bytes):
        uid = uuid.uuid4().hex[:12]
        p = self.masks / f"{uid}.png"
        if isinstance(mask_u8_or_bytes, bytes):
            p.write_bytes(mask_u8_or_bytes)
        else:
            Image.fromarray(mask_u8_or_bytes).save(p)
        return uid, p

    def mask_path(self, uid):
        p = self.masks / f"{uid}.png"
        if not p.exists():
            raise HTTPException(404, f"unknown mask {uid}")
        return p


uploads = Uploads(DATA)
registry = JobRegistry()
worker = GpuWorker(registry, uploads)
app = FastAPI(title="imageEditor")


def url_of(p: Path):
    return "/files/" + p.relative_to(DATA).as_posix()


def clamp_box(box, w, h):
    x0, y0, x1, y1 = box
    x0, x1 = sorted((max(0, min(int(x0), w - 1)), max(0, min(int(x1), w))))
    y0, y1 = sorted((max(0, min(int(y0), h - 1)), max(0, min(int(y1), h))))
    if x1 - x0 < 2 or y1 - y0 < 2:
        raise HTTPException(400, f"box {box} is degenerate")
    return x0, y0, x1, y1


async def on_worker(fut):
    try:
        return await asyncio.wrap_future(fut)
    except HTTPException:
        raise
    except Exception as e:                    # noqa: BLE001
        raise HTTPException(500, f"{type(e).__name__}: {e}")


# ============================================================================ uploads
@app.post("/api/uploads", response_model=S.UploadOut)
async def upload(kind: str = "scene", file: UploadFile = File(...)):
    data = await file.read()
    if len(data) > MAX_UPLOAD_MB * 2 ** 20:
        raise HTTPException(413, f"file larger than {MAX_UPLOAD_MB} MB")
    uid, p, (w, h) = uploads.save_image(data)
    if kind == "scene":
        # pre-clean stage (DDRM identity, 5 DDIM steps) - everything downstream uses its output; then SAM encoding
        clean = await on_worker(worker.preclean(uid))
        worker.encode(uid)                    # SAM image embedding in the background; proposals wait on it
        return S.UploadOut(id=uid, url=url_of(clean), original_url=url_of(p), width=w, height=h, precleaned=True)
    return S.UploadOut(id=uid, url=url_of(p), original_url=url_of(p), width=w, height=h)


@app.post("/api/sam/proposals", response_model=S.ProposalsOut)
async def sam_proposals(body: S.ProposalsIn):
    im = uploads.open(body.image_id)
    box = clamp_box(body.box, *im.size)
    props, ms = await on_worker(worker.propose(body.image_id, box))
    out = []
    for m, iou in props:
        uid, p = uploads.save_mask((m.astype("uint8") * 255))
        out.append(S.Proposal(mask_id=uid, mask_url=url_of(p), iou=round(iou, 3), area=int(m.sum())))
    return S.ProposalsOut(proposals=out, decode_ms=round(ms, 1))


@app.post("/api/masks", response_model=S.MaskOut)
async def upload_mask(image_id: str, file: UploadFile = File(...)):
    """A user-drawn mask: PNG at the scene's natural size, white = object."""
    data = await file.read()
    try:
        m = Image.open(io.BytesIO(data)); m.load()
    except Exception:
        raise HTTPException(400, "mask is not a decodable image")
    im = uploads.open(image_id)
    if m.size != im.size:
        raise HTTPException(400, f"mask is {m.size}, scene is {im.size}")
    # accept RGBA/RGB/L: anything non-black (or alpha > 0 for RGBA) is object
    if m.mode == "RGBA":
        a = m.getchannel("A").point(lambda v: 255 if v > 127 else 0)
    else:
        a = m.convert("L").point(lambda v: 255 if v > 127 else 0)
    buf = io.BytesIO(); a.save(buf, format="PNG")
    uid, p = uploads.save_mask(buf.getvalue())
    return S.MaskOut(id=uid, url=url_of(p))


# ============================================================================ jobs
@app.post("/api/jobs", response_model=S.JobCreated, status_code=202)
async def create_job(body: S.JobIn):
    scene = uploads.open(body.scene_id)
    if body.kind == "floor":
        uploads.path(body.pattern_id)
        spec = body.model_dump()
    elif body.kind == "image3d":
        uploads.mask_path(body.mask_id)
        spec = body.model_dump()
    else:
        from object_edit.insertion import INSERTERS
        from object_edit.removal import REMOVERS
        if body.removal not in REMOVERS or body.insertion not in INSERTERS:
            raise HTTPException(400, f"removal must be one of {sorted(REMOVERS)}, insertion one of {sorted(INSERTERS)}")
        spec = body.model_dump()
        for m in spec["moves"]:
            uploads.mask_path(m["mask_id"])
            m["src_box"], m["dst_box"] = clamp_box(m["src_box"], *scene.size), clamp_box(m["dst_box"], *scene.size)
    job = registry.create(body.kind, spec, DATA / "jobs")
    worker.run_job(job)
    return S.JobCreated(job_id=job.id, status=job.status, position=registry.position(job))


@app.get("/api/jobs/{job_id}", response_model=S.JobStatus)
async def job_status(job_id: str):
    job = registry.get(job_id)
    if job is None:
        raise HTTPException(404, "unknown job")
    return S.JobStatus(job_id=job.id, kind=job.kind, status=job.status, stage=job.stage, stages_done=job.stages_done,
                       elapsed_s=job.elapsed_s, position=registry.position(job), result=job.result, error=job.error)


@app.get("/api/health")
async def health():
    return {"ok": True, "sam_loaded": worker.sam.model is not None, "busy_with": worker.busy_with}


# ============================================================================ static
app.mount("/files", StaticFiles(directory=DATA), name="files")
DIST = ROOT / "web" / "dist"
if DIST.exists():
    app.mount("/assets", StaticFiles(directory=DIST / "assets"), name="assets")

    @app.get("/{path:path}", include_in_schema=False)
    async def spa(path: str):
        f = DIST / path
        return FileResponse(f if path and f.is_file() else DIST / "index.html")
