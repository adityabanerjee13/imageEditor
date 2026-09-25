"""Request / response models of the HTTP API. All pixel coordinates are in the uploaded image's natural pixels."""
from typing import Literal

from pydantic import BaseModel, Field

Box = tuple[int, int, int, int]          # x0, y0, x1, y1 (inclusive-exclusive corners, x0 < x1, y0 < y1)


class UploadOut(BaseModel):
    id: str
    url: str                 # the working image (pre-cleaned for scenes)
    original_url: str        # the upload as received
    width: int
    height: int
    precleaned: bool = False


class ProposalsIn(BaseModel):
    image_id: str
    box: Box


class Proposal(BaseModel):
    mask_id: str
    mask_url: str
    iou: float
    area: int            # pixels


class ProposalsOut(BaseModel):
    proposals: list[Proposal]
    decode_ms: float


class MaskOut(BaseModel):
    id: str
    url: str


class MoveSpec(BaseModel):
    src_box: Box
    dst_box: Box
    mask_id: str


class FloorJobIn(BaseModel):
    kind: Literal["floor"]
    scene_id: str
    pattern_id: str
    material: Literal["smooth-matte", "smooth-glossy"] = "smooth-matte"
    # how much floor 1000 px of the pattern covers; sets the tile size, since a pattern PNG
    # carries no scale of its own. UI: "1000 px = <m> of floor".
    m_per_1000px: float = Field(default=5.0, gt=0.0, le=100.0)


class ObjectJobIn(BaseModel):
    kind: Literal["object"]
    scene_id: str
    moves: list[MoveSpec] = Field(min_length=1)
    removal: str = "omnipaint"
    insertion: str = "omnipaint"
    src_region: Literal["mask", "dilated", "box", "full"] = "full"
    dst_region: Literal["mask", "dilated", "box", "full"] = "full"
    region_margin: float = Field(default=0.35, ge=0.0, le=2.0)
    sr_factor: Literal[1, 2, 3, 4] = 1
    omnipaint_mode: Literal["window", "full"] = "full"
    omnipaint_res: int = Field(default=1024, ge=256, le=1536)
    omnipaint_steps: int = Field(default=28, ge=1, le=50)


class Image3dJobIn(BaseModel):
    """Reconstruct a 3D proxy of one SAM-selected object (imageto3D / TRELLIS)."""
    kind: Literal["image3d"]
    scene_id: str
    mask_id: str
    seed: int = 42
    ss_steps: int | None = Field(default=None, ge=1, le=50)
    slat_steps: int | None = Field(default=None, ge=1, le=50)


JobIn = FloorJobIn | ObjectJobIn | Image3dJobIn


class JobCreated(BaseModel):
    job_id: str
    status: str
    position: int        # 0 = running / next


class JobResult(BaseModel):
    output_url: str
    intermediates: list[dict]        # [{name, url}]
    metrics: dict
    model_url: str | None = None     # image3d: the .glb mesh (geometry)
    splat_url: str | None = None     # image3d: the 3DGS .ply - the representation DIRECT renders


class JobStatus(BaseModel):
    job_id: str
    kind: Literal["floor", "object", "image3d"]
    status: Literal["queued", "running", "done", "failed"]
    stage: str | None = None
    stages_done: list[dict] = []
    elapsed_s: float
    position: int
    result: JobResult | None = None
    error: str | None = None
