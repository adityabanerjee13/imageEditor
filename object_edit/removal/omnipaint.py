"""OmniPaint object removal (FLUX.1-dev + removal LoRA): the scene with the hole blacked out is the only condition; the
model regenerates the whole canvas, and we keep its pixels inside the hole only.

Two ways to map the frame onto the model's canvas:
- mode="window" (default): each hole (connected component of the removal mask) gets its own square window of
  max(512, `factor` x its extent) px, resized to 512 - a 1080 px frame with two small objects is two generations at
  ~object resolution.  ~9 min per hole at 28 steps on the iGPU.
- mode="full": the whole frame, downscaled so the long side is `res` (1024 = OmniPaint's cap; sides rounded to
  multiples of 16), one generation for all holes.  Whole-scene context, but the fill is generated at frame/res
  resolution and upsampled (half-res on a 2000 px photo).  ~25-30 min at 1024 / 28 steps on the iGPU.
Either way the transformer is streamed layer by layer (< 1 GB resident) and only the hole pixels change."""
import cv2
import numpy as np
from PIL import Image

from object_edit import flux_omnipaint as fx
from object_edit.common import RES, Stage, composite_back, crop, mask_to_512, square_window, to_512
from object_edit.removal.base import Remover


class OmniPaintRemover(Remover):
    """The removal LoRA was trained to erase objects *and their effects* (shadows, reflections) from the object mask
    alone: the hole that is blacked out is always the dilated silhouette, and `region` decides how much of the re-rendered
    canvas is kept - "mask": the hole only (cut-paste), "dilated"/"box": a band around it (the old shadow goes),
    "full": the whole window / frame (feathered at its border)."""
    name = "omnipaint"
    shared_model = "flux"
    generative = True

    def __init__(self, steps=28, factor=2.0, mode="window", res=1024):
        self.steps, self.factor, self.mode, self.res = steps, factor, mode, res
        self.out_dir = None                     # set by the orchestrator: where to save the conditions the model sees

    def load(self, log=None):
        fx.get(log)

    def unload(self):
        fx.release()

    def remove(self, image, hole, *, seed, log, region=None):
        region = hole if region is None else (hole | region)
        return self._full(image, hole, region, seed, log) if self.mode == "full" else self._windows(image, hole, region, seed, log)

    @staticmethod
    def _keep(base, fill, region, feather_px=4):
        """fill inside `region` (feathered); a region covering the whole canvas is feathered at the canvas border."""
        r = region.astype(np.uint8)
        if region.all():
            r = cv2.erode(r, np.ones((2 * feather_px + 1, 2 * feather_px + 1), np.uint8))
        return np.asarray(composite_back(base, fill, r > 0, feather_px=feather_px))

    # ---- one generation for the whole frame
    def _full(self, image, hole, region, seed, log):
        H, W = hole.shape
        cw, ch = fx.canvas_size(W, H, self.res)
        with Stage(log, "background_generation", f"OmniPaint remove, full frame {cw}x{ch} ({self.steps} steps)"):
            img = Image.fromarray(image).resize((cw, ch), Image.LANCZOS)
            mask = Image.fromarray(hole.astype(np.uint8) * 255).resize((cw, ch), Image.NEAREST)
            masked = Image.composite(Image.new("RGB", (cw, ch), (0, 0, 0)), img, mask)
            if self.out_dir is not None:
                masked.save(self.out_dir / "removal_condition_full.png")
            out = fx.run("removal", [fx.condition("removal", masked)], seed, steps=self.steps, log=log, size=(cw, ch))
        fill = cv2.resize(np.asarray(out.convert("RGB")), (W, H), interpolation=cv2.INTER_LANCZOS4)
        return self._keep(image, fill, region)

    # ---- one 512 window per hole (window sized to the edit region so a shadow band fits)
    def _windows(self, image, hole, region, seed, log):
        H, W = hole.shape
        n, lab = cv2.connectedComponents(hole.astype(np.uint8))
        background = image.copy()
        for k in range(1, n):
            comp = lab == k
            ys, xs = np.where(comp)
            x0, y0, x1, y1 = xs.min(), ys.min(), xs.max(), ys.max()
            if not region.all():                            # the region component(s) touching this hole
                nr, labr = cv2.connectedComponents(region.astype(np.uint8))
                reg = np.isin(labr, np.unique(labr[comp])) & (labr > 0)
                ry, rx = np.where(reg)
                cx, cy, ext = (rx.min() + rx.max()) / 2, (ry.min() + ry.max()) / 2, max(rx.max() - rx.min(), ry.max() - ry.min())
            else:
                reg = region
                cx, cy, ext = (x0 + x1) / 2, (y0 + y1) / 2, max(x1 - x0, y1 - y0)
            win = square_window((cx, cy), max(RES, self.factor * ext), W, H)
            hole_w, reg_w = crop(comp, win), crop(reg, win)
            with Stage(log, "background_generation", f"OmniPaint remove {k}/{n - 1} (window {win[2]}px, {self.steps} steps)"):
                img512 = Image.fromarray(to_512(crop(background, win)))
                mask512 = Image.fromarray(mask_to_512(hole_w) * 255)
                masked = Image.composite(Image.new("RGB", img512.size, (0, 0, 0)), img512, mask512)
                if self.out_dir is not None:
                    masked.save(self.out_dir / f"removal_condition_{k}_512.png")
                out = fx.run("removal", [fx.condition("removal", masked)], seed, steps=self.steps, log=log)
            wx, wy, ws = win
            fill = cv2.resize(np.asarray(out.convert("RGB")), (ws, ws), interpolation=cv2.INTER_LANCZOS4)
            background[wy:wy + ws, wx:wx + ws] = self._keep(background[wy:wy + ws, wx:wx + ws], fill, reg_w)
        return background
