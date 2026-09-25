"""Model-free insertion: window the object, affine-copy it (translate + perspective scale), feather-composite back.
Also the shared coarse step used by model-based inserters (they refine `coarse` inside `tgt` before compositing)."""
from dataclasses import dataclass

import cv2
import numpy as np
from PIL import Image

from object_edit.common import RES, Stage, bbox, composite_back, crop, mask_to_512, square_window, to_512, window_side
from object_edit.insertion.base import Inserter


@dataclass
class Coarse:
    """Everything a refiner needs, all in the 512 canvases of two same-sided square windows."""
    swin: tuple          # (x0, y0, side) source window on the original
    dwin: tuple          # (x0, y0, side) destination window on the background
    src512: np.ndarray   # source window crop, 512x512x3
    srcmask512: np.ndarray  # object mask in the source window, 512x512 uint8 {0,1}
    dst512: np.ndarray   # destination window crop of the background, 512x512x3
    coarse: np.ndarray   # dst512 with the object affine-pasted, 512x512x3
    tgt: np.ndarray      # bool 512x512, the pasted object's footprint

    @property
    def side(self):
        return self.swin[2]


def coarse_paste(original, background, mv):
    """Stages 5-6: two square windows of the same side (~3x the object, >= 512 px, clamped to the frame) - a SOURCE window
    on the original (the reference) and a DESTINATION window on the background (the edit canvas), each resized to 512 -
    then an affine copy of the object from the source to the destination canvas (translate so the anchor lands on the
    destination, scale about it).  The object therefore fills ~1/3 of the canvas however far it moves."""
    H, W = original.shape[:2]
    side = window_side(mv)
    sx0, sy0, sx1, sy1 = bbox(mv.mask)
    src_c = ((sx0 + sx1) / 2, (sy0 + sy1) / 2)
    dst_c = (mv.src[0] + (src_c[0] - mv.src[0]) * mv.scale + mv.shift[0],       # scaled about the anchor point
             mv.src[1] + (src_c[1] - mv.src[1]) * mv.scale + mv.shift[1])
    swin, dwin = square_window(src_c, side, W, H), square_window(dst_c, side, W, H)
    f = RES / swin[2]                                                             # both windows share `side`
    src512, srcmask512 = to_512(crop(original, swin)), mask_to_512(crop(mv.mask, swin))
    dst512 = to_512(crop(background, dwin))
    ps = ((mv.src[0] - swin[0]) * f, (mv.src[1] - swin[1]) * f)
    pd = ((mv.dst[0] - dwin[0]) * f, (mv.dst[1] - dwin[1]) * f)
    M = np.array([[mv.scale, 0, pd[0] - mv.scale * ps[0]], [0, mv.scale, pd[1] - mv.scale * ps[1]]], np.float32)
    moved = cv2.warpAffine(src512, M, (RES, RES), flags=cv2.INTER_LANCZOS4)
    tgt = cv2.warpAffine(srcmask512, M, (RES, RES), flags=cv2.INTER_NEAREST) > 0
    coarse = np.where(tgt[..., None], moved, dst512).astype(np.uint8)
    return Coarse(swin, dwin, src512, srcmask512, dst512, coarse, tgt)


def save_coarse(c, out_dir, idx):
    if out_dir is None:
        return
    Image.fromarray(c.src512).save(out_dir / f"obj{idx}_source_512.png")
    Image.fromarray(c.coarse).save(out_dir / f"obj{idx}_coarse_512.png")
    Image.fromarray(c.tgt.astype(np.uint8) * 255).save(out_dir / f"obj{idx}_target_512.png")


def composite(background, c, result, grow_kernel=15, feather_px=4, region=None):
    """Stage 8: write `result` (512 canvas) back into the destination window of `background` through a feathered edge,
    inside `region` (bool 512 canvas; a region covering the whole canvas is feathered at the window border) or, by
    default, the target footprint dilated with a `grow_kernel` square.  Returns a new array."""
    background = background.copy()
    dx0, dy0, ds = c.dwin
    up = cv2.resize(result, (ds, ds), interpolation=cv2.INTER_LANCZOS4)
    if region is None:
        region = cv2.dilate(cv2.resize(c.tgt.astype(np.uint8), (ds, ds), interpolation=cv2.INTER_NEAREST),
                            np.ones((grow_kernel, grow_kernel), np.uint8)) > 0
    else:
        r = cv2.resize(region.astype(np.uint8), (ds, ds), interpolation=cv2.INTER_NEAREST)
        if region.all():
            r = cv2.erode(r, np.ones((2 * feather_px + 1, 2 * feather_px + 1), np.uint8))
        region = r > 0
    patch = composite_back(background[dy0:dy0 + ds, dx0:dx0 + ds], up, region, feather_px=feather_px)
    background[dy0:dy0 + ds, dx0:dx0 + ds] = np.asarray(patch)
    return background


class PasteInserter(Inserter):
    """Reference implementation: the coarse paste is the result (no harmonisation, ~0 s, no model)."""
    name = "paste"

    def insert(self, original, background, move, *, seed, log, out_dir=None, idx=0, total=1):
        c = coarse_paste(original, background, move)
        save_coarse(c, out_dir, idx)
        if not c.tgt.any():
            print(f"[object {idx}] target lies outside the frame - skipped"); return background
        with Stage(log, "regeneration", f"paste object {idx + 1} (window {c.side}px)") as st:
            out = composite(background, c, c.coarse)
        st.row.update(source_window=list(c.swin), dest_window=list(c.dwin), scale=round(move.scale, 3))
        if out_dir is not None:
            Image.fromarray(c.coarse).save(out_dir / f"obj{idx}_result_512.png")
        return out
