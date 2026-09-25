"""Task description: the image and the list of object moves (points / boxes / optional precomputed masks)."""
import json
from pathlib import Path

import numpy as np
from PIL import Image

from object_edit.common import ROOT


class Move:
    """One object: anchor point, destination point, masks and scale (all in the task's pixel frame).
    `src_box` / `dst_box` (x0, y0, x1, y1) are kept when the move was specified as boxes (UI)."""

    def __init__(self, src, dst, mask=None, mask_dilated=None, scale=1.0, src_box=None, dst_box=None):
        self.src, self.dst, self.mask, self.mask_dilated, self.scale = src, dst, mask, mask_dilated, scale
        self.src_box, self.dst_box = src_box, dst_box
        self.region = None                 # source edit region (common.edit_region of mask_dilated): what the remover may repaint

    @property
    def shift(self):
        return self.dst[0] - self.src[0], self.dst[1] - self.src[1]


class MoveTask:
    def __init__(self, scene_path, image, moves):
        self.scene_path, self.image, self.moves = scene_path, image, moves
        self.meta = {}

    @property
    def size(self):
        return self.image.size            # (W, H)

    @property
    def mask(self):                        # union of tight masks
        return np.any([m.mask for m in self.moves], axis=0)

    @property
    def mask_dilated(self):                # union of dilated masks = removal hole
        return np.any([m.mask_dilated for m in self.moves], axis=0)

    @property
    def region(self):                      # union of source edit regions = what the remover may repaint (superset of the hole)
        return np.any([m.region if m.region is not None else m.mask_dilated for m in self.moves], axis=0)

    @property
    def has_masks(self):
        return all(m.mask is not None for m in self.moves)


def _pt(d):
    return int(d["x"]), int(d["y"])


def _box_center(b):
    return int(round((b[0] + b[2]) / 2)), int(round((b[1] + b[3]) / 2))


def load_task(scene, coords_json):
    """input/*.json: {"scene": ..., "moves": [{"initial": {x, y}, "final": {x, y}}, ...]}
    (a single {"initial", "final"} pair at the top level is also accepted).
    Per move, optional: "src_box" / "dst_box" [x0, y0, x1, y1] (used instead of the points: box centres)
    and "mask" (path of a binary PNG, skips SAM for that object)."""
    c = json.loads(Path(coords_json).read_text())
    base = Path(coords_json).resolve().parent
    scene = Path(scene) if scene else ROOT / c["scene"]
    pairs = c.get("moves") or [{"initial": c["initial"], "final": c["final"]}]
    moves = []
    for m in pairs:
        src = _pt(m["initial"]) if "initial" in m else _box_center(m["src_box"])
        dst = _pt(m["final"]) if "final" in m else _box_center(m["dst_box"])
        mask = None
        if m.get("mask"):
            p = Path(m["mask"])
            p = p if p.is_absolute() else (base / p if (base / p).exists() else ROOT / p)
            mask = np.asarray(Image.open(p).convert("L")) > 127
        moves.append(Move(src, dst, mask=mask, src_box=m.get("src_box"), dst_box=m.get("dst_box")))
    return MoveTask(scene, Image.open(scene).convert("RGB"), moves)


def default_config(**overrides):
    """Pipeline options (mirrors the CLI defaults). Unknown keys are rejected so typos surface early."""
    cfg = dict(
        scale="auto",             # object scale at the destination: number or "auto" (MoGe-2 depth ratio)
        dilate_radius=6,          # stage 2: object mask grown by this many native px
        removal="omnipaint",      # object_edit.removal.REMOVERS key
        insertion="omnipaint",    # object_edit.insertion.INSERTERS key
        prompt="",                # refine-stage guidance text (FreeFine benchmark: "")
        bg_prompt="empty scene",  # FreeFine background-generation text (benchmark value)
        start_step=15,            # FreeFine: benchmark uses 35 (keeps more pasted pixels); 15 regenerates more
        end_scale=0.0,
        src_region="full",        # what the remover may repaint at the source: mask (silhouette + dilate_radius, cut-paste) |
                                  # dilated (silhouette grown by region_margin, downward-biased: cast shadow) | box (bbox +
                                  # region_margin) | full (the whole window / frame the model renders; generative backends only)
        dst_region="full",        # what the inserter may repaint at the target: same choices around the pasted footprint
        region_margin=0.35,       # dilated / box: margin as a fraction of the object's size
        preclean=False,           # DDRM identity pre-clean of the input (5 DDIM steps, noise 0.005) before everything else
        sr_factor=1,              # working resolution: 1 = native; 2/3/4 = block-average the frame by this factor before
                                  # removal / insertion, then DDRM super-resolution back (object_edit/resolution.py)
        sr_steps=15,              # DDRM: number of DDIM timesteps (t = k*(1000//n)); ~2 s/step at 1080^2, ~5 s/step at 2000x1500
        sr_noise=0.01,            # DDRM: measurement noise level handed to the sampler (decides which directions count as observed)
        omnipaint_steps=28,       # OmniPaint (FLUX) removal / insertion: diffusion steps (28 = paper; ~10 s/step on the iGPU)
        omnipaint_grow=8,         # OmniPaint insertion: rectangular target = paste bbox + this many px (512 canvas)
        omnipaint_mode="full",    # "full": the whole frame downscaled to omnipaint_res on the long side (whole-scene context,
                                  #         half-res fill on a 2000 px photo, ~25-30 min per generation at 1024) - default;
                                  # "window": one 512 square per hole / object (near-native detail, ~7-12 min each)
        omnipaint_res=1024,       # full mode: long side of the generation canvas (multiple of 16; 1024 = OmniPaint's cap)
        seed=42,                  # < 0: random
    )
    unknown = set(overrides) - set(cfg)
    if unknown:
        raise KeyError(f"unknown object_edit options: {sorted(unknown)}")
    cfg.update(overrides)
    return cfg
