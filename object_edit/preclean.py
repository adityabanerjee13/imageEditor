"""Pre-clean stage: DDRM (deepinv) with the identity operator on the input photo, run once when an image is added.

    y = x + N(0, sigma^2)                       sigma = 0.005 (1.3/255)
    DDRM(denoiser=DRUNet, sigmas=ddim5)         5 DDIM timesteps of the DDPM T=1000 linear-beta schedule: t = 800, 600,
                                                400, 200, 0 -> variance-exploding sigma_t = sqrt((1 - abar_t) / abar_t)
                                                = 25.7, 6.17, 2.04, 0.72, 0.01; eta 0.85, etab 1.0, seed 0
Everything downstream (SAM proposals, floor_edit, object_edit) runs on the returned image.  Measured on the iGPU:
1080^2 ~11 s, 2000x1500 ~26 s, peak 2.3-5.8 GB; the output stays within ~0.8/255 of the input (47-49 dB).
Weights: weights/dpir/drunet_deepinv_color_finetune_22k.pth (deepinv's DRUNet(pretrained="download") checkpoint)."""
import time

import numpy as np
import torch

from object_edit.common import DEV, WEIGHTS, empty_cache

DRUNET = WEIGHTS / "dpir" / "drunet_deepinv_color_finetune_22k.pth"


def ddim_timesteps(n, T=1000):
    """DDIM uniform timestep selection: t = k * (T // n) for k = n-1 ... 0 (exactly n steps, ending at 0)."""
    return (np.arange(n) * (T // n))[::-1]


def ve_sigmas(ts, T=1000, beta_start=1e-4, beta_end=0.02):
    abar = np.cumprod(1.0 - np.linspace(beta_start, beta_end, T))
    return np.sqrt((1.0 - abar[ts]) / abar[ts])


class DDRMClean:
    def __init__(self, sigma=0.005, steps=5, eta=0.85, etab=1.0, seed=0):
        self.sigma, self.steps, self.eta, self.etab, self.seed = sigma, steps, eta, etab, seed
        self.sigmas = ve_sigmas(ddim_timesteps(steps))
        self.denoiser = None

    def load(self):
        if self.denoiser is None:
            from deepinv.models import DRUNet
            self.denoiser = DRUNet(pretrained=str(DRUNET), device=DEV)
        return self.denoiser

    def unload(self):
        self.denoiser = None
        empty_cache()

    def __call__(self, image_u8):
        """uint8 HxWx3 -> uint8 HxWx3 (same size)."""
        import deepinv as dinv
        denoiser = self.load()
        H, W = image_u8.shape[:2]
        ph, pw = (8 - H % 8) % 8, (8 - W % 8) % 8
        x = torch.from_numpy(np.pad(image_u8, ((0, ph), (0, pw), (0, 0)), mode="reflect")).permute(2, 0, 1)[None].float().div(255).to(DEV)
        physics = dinv.physics.Denoising(noise_model=dinv.physics.GaussianNoise(sigma=self.sigma), device=DEV)
        g = torch.Generator(device="cpu").manual_seed(self.seed)
        y = x + self.sigma * torch.randn(x.shape, generator=g).to(DEV)
        model = dinv.sampling.DDRM(denoiser=denoiser, sigmas=self.sigmas, eta=self.eta, etab=self.etab, verbose=False)
        with torch.no_grad():
            out = model(y, physics, seed=self.seed)
        out = (out[0].permute(1, 2, 0).cpu().numpy() * 255 + 0.5).clip(0, 255).astype(np.uint8)
        return out[:H, :W]

    def describe(self):
        return f"DDRM identity, noise {self.sigma}, {self.steps} DDIM steps (t {', '.join(str(t) for t in ddim_timesteps(self.steps))})"


def preclean(image_u8, log=None, **kw):
    """One-shot helper for the CLIs: run DDRMClean and log a 'preclean' stage row."""
    from object_edit.common import Stage
    c = DDRMClean(**kw)
    if log is not None:
        with Stage(log, "preclean", c.describe()):
            out = c(image_u8)
    else:
        out = c(image_u8)
    c.unload()
    return out
