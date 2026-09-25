"""LaMa (big-lama) object removal on the full frame, optionally with Geomagical feature refinement."""
import importlib.util
import sys
import types

import cv2
import numpy as np
import torch

from object_edit.common import DEV, REPOS, WEIGHTS, Stage, empty_cache
from object_edit.removal.base import Remover


class LamaRemover(Remover):
    """refine=True runs LaMa's feature refinement (Geomagical, arXiv 2206.13644; repos/lama-with-refiner ==
    upstream `refine=True`): an image pyramid where the generator's inner features are optimised for
    `n_iters` Adam steps per scale so the full-res fill agrees with the low-res prediction (~+45 s)."""

    def __init__(self, refine=True, n_iters=15):
        self.refine, self.n_iters, self.model = refine, n_iters, None
        self.name = "lama" if refine else "lama-plain"

    def load(self, log=None):
        if self.model is None:
            sys.path.insert(0, str(REPOS / "lama"))
            # saicinpainting.training.data.aug needs an albumentations<1.0 API; the predict path never uses it
            aug = types.ModuleType("saicinpainting.training.data.aug"); aug.IAAAffine2 = aug.IAAPerspective2 = object
            aug.__spec__ = importlib.util.spec_from_loader(aug.__name__, loader=None)   # None here makes find_spec() raise
            sys.modules.setdefault("saicinpainting.training.data.aug", aug)
            import yaml
            from omegaconf import OmegaConf
            import saicinpainting.training.trainers as trainers
            root = WEIGHTS / "big-lama" / "big-lama"
            cfg = OmegaConf.create(yaml.safe_load((root / "config.yaml").read_text()))
            cfg.training_model.predict_only = True
            cfg.visualizer.kind = "noop"
            # the Lightning checkpoint pickles callback objects -> torch>=2.6 needs weights_only=False
            state = torch.load(str(root / "models" / "best.ckpt"), map_location="cpu", weights_only=False)
            model = trainers.make_training_model(cfg)
            model.load_state_dict({k: v for k, v in state["state_dict"].items() if k.startswith("generator.")}, strict=False)
            model.freeze()
            self.model = model.to(DEV)
        return self.model

    def unload(self):
        self.model = None
        empty_cache()

    def remove(self, image, hole, *, seed, log, region=None):
        """LaMa has no notion of an object's shadow, so the whole edit region is treated as the hole to fill."""
        hole = hole if region is None else (hole | region)
        H, W = hole.shape
        n_holes = cv2.connectedComponents(hole.astype(np.uint8))[0] - 1
        with Stage(log, "background_generation", f"LaMa big-lama (native {W}x{H}, {n_holes} holes{', refined' if self.refine else ''})"):
            return self._inpaint(image, hole, self.refine)

    def _inpaint(self, img_rgb, hole, refine):
        """img_rgb uint8 HxWx3, hole bool HxW (True = fill); whole frame, padded to a multiple of 8."""
        model = self.load()
        H, W = hole.shape
        ph, pw = (8 - H % 8) % 8, (8 - W % 8) % 8
        img = torch.from_numpy(np.pad(img_rgb, ((0, ph), (0, pw), (0, 0)), mode="reflect")).float().permute(2, 0, 1)[None] / 255.0
        m = torch.from_numpy(np.pad(hole.astype(np.float32), ((0, ph), (0, pw)), mode="constant"))[None, None]
        if refine:
            import saicinpainting.evaluation.refinement as R

            class _TorchOnDev:                      # the refiner builds "cuda:<id>" device strings
                def __getattr__(self, k): return getattr(torch, k)
                def device(self, d): return torch.device(str(DEV)) if str(d).startswith("cuda") else torch.device(d)
            R.torch = _TorchOnDev()
            try:
                # px_budget: the refiner back-propagates through the FFC stack at this many pixels; 1.1 MP is
                # what the 16 GB iGPU sustains (the repo default is 1.8 MP).  Larger ROIs are refined at the
                # budget size and resized back.
                out = R.refine_predict({"image": img, "mask": m, "unpad_to_size": [torch.tensor([H]), torch.tensor([W])]}, model,
                                       gpu_ids="0,", modulo=8, n_iters=self.n_iters, lr=0.002, min_side=512, max_scales=3, px_budget=1100000)
                out = out[0].permute(1, 2, 0).detach().cpu().numpy()
            except torch.OutOfMemoryError:
                print("[lama] refiner out of memory -> plain LaMa pass"); empty_cache()
                return self._inpaint(img_rgb, hole, refine=False)
        else:
            with torch.no_grad():
                out = model({"image": img.to(DEV), "mask": m.to(DEV)})["inpainted"][0].permute(1, 2, 0).cpu().numpy()
        out = np.clip(out * 255, 0, 255).astype(np.uint8)
        if out.shape[:2] != (H + ph, W + pw):
            out = cv2.resize(out, (W + pw, H + ph), interpolation=cv2.INTER_LANCZOS4)
        out = out[:H, :W]
        return np.where(hole[..., None], out, img_rgb)             # original pixels outside the hole
