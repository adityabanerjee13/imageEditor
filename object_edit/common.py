"""Shared paths, device, instrumentation and geometry helpers for the object_edit package."""
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
WEIGHTS = ROOT / "weights"
REPOS = ROOT / "repos"
OUT_ROOT = ROOT / "outputs" / "object_edit"
for p in (REPOS / "MoGe", REPOS / "FreeFine"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

DEV = torch.device("cuda" if torch.cuda.is_available() else "xpu" if torch.xpu.is_available() else "cpu")
RES = 512


# ============================================================================ instrumentation
def sync():
    if DEV.type == "cuda":
        torch.cuda.synchronize()
    elif DEV.type == "xpu":
        torch.xpu.synchronize()


def empty_cache():
    if DEV.type == "cuda":
        torch.cuda.empty_cache()
    elif DEV.type == "xpu":
        torch.xpu.empty_cache()


def peak_mem_gb():
    if DEV.type == "cuda":
        return torch.cuda.max_memory_allocated() / 1e9
    if DEV.type == "xpu":
        return torch.xpu.max_memory_allocated() / 1e9
    return 0.0


def reset_peak():
    if DEV.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    elif DEV.type == "xpu":
        torch.xpu.reset_peak_memory_stats()


class Stage:
    """Times a stage (device-synced) and records peak accelerator memory into `log`."""

    def __init__(self, log, name, model=""):
        self.log, self.row = log, {"stage": name, "model": model}

    def __enter__(self):
        sync(); reset_peak(); self.t = time.perf_counter()
        return self

    def __exit__(self, *_):
        sync()
        self.row["time_s"] = round(time.perf_counter() - self.t, 2)
        self.row["gpu_peak_gb"] = round(peak_mem_gb(), 2)
        self.log.append(self.row)
        print(f"[{self.row['stage']}] {self.row['time_s']} s, peak {self.row['gpu_peak_gb']} GB  {self.row['model']}")


class StageLog(list):
    """A stage list that notifies `on_append(row)` whenever a stage finishes (used by the server for progress)."""

    def __init__(self, on_append=None):
        super().__init__()
        self.on_append = on_append

    def append(self, row):
        super().append(row)
        if self.on_append:
            self.on_append(row)


def timed(fn):
    t = time.perf_counter(); r = fn(); return r, round(time.perf_counter() - t, 2)


# ============================================================================ window helpers
def bbox(mask):
    ys, xs = np.where(mask)
    return xs.min(), ys.min(), xs.max(), ys.max()


def square_window(center, side, W, H):
    """(x0, y0, side) square of `side` px around `center`, clamped inside the frame."""
    side = int(min(side, W, H))
    x0 = int(np.clip(round(center[0] - side / 2), 0, W - side))
    y0 = int(np.clip(round(center[1] - side / 2), 0, H - side))
    return x0, y0, side


def window_side(mv, factor=3.0, min_side=RES):
    """Same side for the source and destination windows: ~`factor` x the larger of the object's source
    bbox and its scaled target bbox, at least `min_side`."""
    x0, y0, x1, y1 = bbox(mv.mask)
    diag = float(np.hypot(x1 - x0, y1 - y0)) * max(1.0, mv.scale)
    return int(max(min_side, np.ceil(factor * diag)))


def crop(img, win):
    x0, y0, s = win
    return np.asarray(img)[y0:y0 + s, x0:x0 + s]


def to_512(img):
    return cv2.resize(np.asarray(img), (RES, RES), interpolation=cv2.INTER_LANCZOS4)


def mask_to_512(m):
    return cv2.resize(m.astype(np.uint8), (RES, RES), interpolation=cv2.INTER_NEAREST)


def shadow_region(mask, frac=0.35, up=0.3, down=1.5):
    """The area around an object where its cast shadow / contact shading can live: `mask` grown by m = frac x the
    object's bbox size sideways, up*m upwards and down*m downwards (shadows fall on the ground below the object).
    frac=0 returns `mask` unchanged.  One dilation with an off-centre anchor."""
    mask = mask.astype(bool)
    if frac <= 0 or not mask.any():
        return mask
    x0, y0, x1, y1 = bbox(mask)
    m = int(round(frac * max(x1 - x0, y1 - y0)))
    if m < 1:
        return mask
    u, d = int(round(up * m)), int(round(down * m))
    k = np.ones((u + d + 1, 2 * m + 1), np.uint8)
    # anchor row = d: output(y) = max over src rows y-d .. y+u, i.e. a source pixel spreads d rows down and u rows up
    return cv2.dilate(mask.astype(np.uint8), k, anchor=(m, d)) > 0


REGION_KINDS = ("mask", "dilated", "box", "full")


def edit_region(mask, kind, margin=0.35):
    """The pixels a model may change around an object, chosen by the user per stage:
      mask    - the object silhouette itself (cut-paste: nothing around it changes)
      dilated - the silhouette grown by `margin` x object size, biased downwards (shadow_region) - cast shadows
      box     - the bounding box of the silhouette, expanded by `margin` x object size on every side
      full    - the whole canvas the model sees (window or frame): everything it renders is kept"""
    mask = mask.astype(bool)
    if kind == "mask" or not mask.any():
        return mask
    if kind == "dilated":
        return shadow_region(mask, margin)
    if kind == "box":
        x0, y0, x1, y1 = bbox(mask)
        m = int(round(margin * max(x1 - x0, y1 - y0)))
        r = np.zeros_like(mask)
        r[max(0, y0 - m):y1 + m + 1, max(0, x0 - m):x1 + m + 1] = True
        return r
    if kind == "full":
        return np.ones_like(mask)
    raise ValueError(f"unknown edit region {kind!r}; choose from {REGION_KINDS}")


def composite_back(original, edited, region, feather_px=3):
    """Paste `edited` onto `original` inside `region` through a feathered edge."""
    o, e = np.asarray(original).astype(np.float32), np.asarray(edited).astype(np.float32)
    a = region.astype(np.float32)
    if feather_px:
        a = cv2.GaussianBlur(a, (0, 0), feather_px)
    return Image.fromarray(np.clip(o * (1 - a[..., None]) + e * a[..., None], 0, 255).astype(np.uint8))


def draw_points(img, moves, r=10):
    a = np.array(img).copy()
    for mv in moves:
        cv2.arrowedLine(a, mv.src, mv.dst, (255, 220, 0), 3, tipLength=0.05)
        cv2.circle(a, mv.src, r, (255, 0, 0), 3); cv2.circle(a, mv.dst, r, (0, 255, 0), 3)
    return Image.fromarray(a)
