"""Working-resolution stage: downsample the frame before the generative blocks, super-resolve the result afterwards with
DDRM (deepinv.sampling.DDRM, Kawar et al. 2022) and the DRUNet denoiser.

Degradation:  y = block mean of every sf x sf pixel block (the frame is first modcrop'ed to a multiple of sf).  DDRM needs
              the operator's SVD (a deepinv DecomposablePhysics); block averaging has a closed form - per block an
              orthonormal 2-D DCT whose DC coefficient is (1/sf) * sum(block): singular value 1/sf on the DC
              coefficients, 0 on all AC coefficients (BlockAverageSR).  (deepinv's Gaussian-filter Downsampling has no
              SVD in the library, so it cannot be used with DDRM.)
Super-resolution: DDRM with sigmas from `sr_steps` DDIM timesteps of the DDPM T=1000 linear-beta schedule
              (t = k*(T//n), descending; variance-exploding sigma_t = sqrt((1 - abar_t) / abar_t)), eta 0.85, etab 1.0,
              seed 0; the measurement noise level given to DDRM is `sr_noise` (0.01; it sets which singular directions
              count as observed).  15 steps: ~30 s at 1080^2, ~80 s at 2000x1500 on the iGPU, peak ~2.4 GB.
Only the pixels the edit changed are taken from the super-resolved frame; everything else stays the original.
Weights: weights/dpir/drunet_deepinv_color_finetune_22k.pth (deepinv's DRUNet(pretrained="download") checkpoint)."""
import time
from pathlib import Path

import cv2
import numpy as np
import torch

from object_edit.common import DEV, empty_cache
from object_edit.preclean import DRUNET, ddim_timesteps, ve_sigmas
from object_edit.task import Move, MoveTask


# ============================================================================ degradation
def modcrop(a, sf):
    H, W = a.shape[:2]
    return a[:H - H % sf, :W - W % sf]


def degrade(image_u8, sf):
    """Block mean of the modcrop'ed frame.  Returns (low float32 HxWx3 in [0,1], low uint8)."""
    x = modcrop(image_u8, sf).astype(np.float32) / 255.
    H, W = x.shape[:2]
    y = x.reshape(H // sf, sf, W // sf, sf, 3).mean((1, 3))
    return y.astype(np.float32), np.clip(y * 255 + 0.5, 0, 255).astype(np.uint8)


def subsample_mask(m, sf):
    """Masks: a low-res pixel is on when any pixel of its block is on."""
    m = modcrop(m, sf)
    H, W = m.shape
    return m.reshape(H // sf, sf, W // sf, sf).any((1, 3))


def downsample_task(task, low_u8, sf):
    """The task in the low-resolution frame: image, points / boxes divided by sf, masks on the block grid."""
    from PIL import Image
    moves = []
    for mv in task.moves:
        m = Move((mv.src[0] // sf, mv.src[1] // sf), (mv.dst[0] // sf, mv.dst[1] // sf),
                 mask=subsample_mask(mv.mask, sf), scale=mv.scale,
                 src_box=[v // sf for v in mv.src_box] if mv.src_box else None,
                 dst_box=[v // sf for v in mv.dst_box] if mv.dst_box else None)
        m.mask_dilated = subsample_mask(mv.mask_dilated, sf)
        m.region = subsample_mask(mv.region if mv.region is not None else mv.mask_dilated, sf)
        moves.append(m)
    return MoveTask(task.scene_path, Image.fromarray(low_u8), moves)


# ============================================================================ DDRM super-resolution
def _dct_matrix(n):
    """orthonormal DCT-II basis, rows = basis vectors; row 0 = 1/sqrt(n) constant."""
    k = torch.arange(n, dtype=torch.float64)[:, None]; i = torch.arange(n, dtype=torch.float64)[None, :]
    D = torch.cos(np.pi * (2 * i + 1) * k / (2 * n)) * np.sqrt(2.0 / n)
    D[0] /= np.sqrt(2.0)
    return D.float()


def make_block_average_physics(img_size, sf, sigma_noise, device):
    """deepinv DecomposablePhysics for y = block mean over sf x sf: V_adjoint = blockwise 2-D DCT laid out on the HR grid
    (coefficient (u, v) of block (i, j) at pixel (i*sf+u, j*sf+v)), mask = 1/sf at the DC positions, U picks that grid."""
    import deepinv as dinv

    class BlockAverageSR(dinv.physics.DecomposablePhysics):
        def __init__(self):
            C, H, W = img_size
            self.sf = sf
            mask = torch.zeros((1, C, H, W), device=device); mask[..., ::sf, ::sf] = 1.0 / sf
            super().__init__(mask=mask, device=device, noise_model=dinv.physics.GaussianNoise(sigma=sigma_noise))
            self.D = _dct_matrix(sf).to(device)

        def _blocks(self, x):
            B, C, H, W = x.shape; s = self.sf
            return x.reshape(B, C, H // s, s, W // s, s)

        def V_adjoint(self, x):
            return torch.einsum("uk,bcikjl,vl->bciujv", self.D, self._blocks(x), self.D).reshape(x.shape)

        def V(self, c):
            return torch.einsum("uk,bciujv,vl->bcikjl", self.D, self._blocks(c), self.D).reshape(c.shape)

        def U(self, c):
            return c[..., ::self.sf, ::self.sf]

        def U_adjoint(self, y):
            s = self.sf
            c = torch.zeros((*y.shape[:2], y.shape[2] * s, y.shape[3] * s), device=y.device, dtype=y.dtype)
            c[..., ::s, ::s] = y
            return c

    return BlockAverageSR()


class DDRMSR:
    def __init__(self, steps=15, sigma_noise=0.01, eta=0.85, etab=1.0, seed=0):
        self.steps, self.sigma_noise, self.eta, self.etab, self.seed = steps, sigma_noise, eta, etab, seed
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

    def describe(self):
        return f"DDRM x{{sf}} block-average, {self.steps} DDIM steps (sigma {self.sigmas[0]:.2f} -> {self.sigmas[-1]:.3f}), eta {self.eta}"

    def super_resolve(self, low, sf):
        """low: float32 hxwx3 in [0,1] (block means) -> float32 (h*sf)x(w*sf)x3 in [0,1]."""
        import deepinv as dinv
        denoiser = self.load()
        h, w = low.shape[:2]
        y = torch.from_numpy(low).permute(2, 0, 1)[None].to(DEV)
        physics = make_block_average_physics((3, h * sf, w * sf), sf, self.sigma_noise, DEV)
        model = dinv.sampling.DDRM(denoiser=denoiser, sigmas=self.sigmas, eta=self.eta, etab=self.etab, verbose=False)
        with torch.no_grad():
            out = model(y, physics, seed=self.seed)
        return np.clip(out[0].permute(1, 2, 0).cpu().numpy(), 0, 1).astype(np.float32)


# ============================================================================ pipeline helper
class WorkingResolution:
    """sf > 1: the generative blocks run on the block-averaged frame; the result is super-resolved with DDRM and only
    the pixels the edit changed are taken from it - everything else stays the original native pixels."""

    def __init__(self, sf, steps=15, sigma_noise=0.01):
        self.sf = int(sf)
        self.sr = DDRMSR(steps=steps, sigma_noise=sigma_noise)

    def describe(self):
        return self.sr.describe().format(sf=self.sf)

    def down(self, task):
        """(low float, low uint8, low-res task)"""
        low_f, low_u8 = degrade(np.asarray(task.image), self.sf)
        return low_f, low_u8, downsample_task(task, low_u8, self.sf)

    def up(self, original_u8, low_in_u8, low_out_u8, feather_px=4, grow_px=3):
        """Super-resolve the edited low-res frame and paste only its changed pixels over the original.  Returns
        (final uint8 at the original size, SR uint8 at the modcrop'ed size)."""
        sf = self.sf
        hr = self.sr.super_resolve(low_out_u8.astype(np.float32) / 255., sf)
        hr_u8 = np.clip(hr * 255 + 0.5, 0, 255).astype(np.uint8)
        changed = np.abs(low_out_u8.astype(int) - low_in_u8.astype(int)).max(-1) > 2
        changed = cv2.dilate(changed.astype(np.uint8), np.ones((2 * grow_px + 1, 2 * grow_px + 1), np.uint8))
        H2, W2 = hr_u8.shape[:2]
        changed_hr = cv2.resize(changed, (W2, H2), interpolation=cv2.INTER_NEAREST).astype(np.float32)
        changed_hr = cv2.GaussianBlur(changed_hr, (0, 0), feather_px * sf / 2)
        base = modcrop(original_u8, sf).astype(np.float32)
        final = original_u8.copy()
        final[:H2, :W2] = np.clip(base * (1 - changed_hr[..., None]) + hr_u8.astype(np.float32) * changed_hr[..., None], 0, 255).astype(np.uint8)
        return final, hr_u8


# ============================================================================ resume: SR of a saved low-res result
def resume(job_dir, scene, sf, steps=15):
    """Finish a job whose generative stages completed but whose super-resolution did not: super-resolve
    <job_dir>/output_lowres.png against <job_dir>/input_lowres.png and the native `scene`, write output.jpg."""
    from PIL import Image
    job_dir = Path(job_dir)
    native = np.asarray(Image.open(scene).convert("RGB"))
    low_in = np.asarray(Image.open(job_dir / "input_lowres.png").convert("RGB"))
    low_out = np.asarray(Image.open(job_dir / "output_lowres.png").convert("RGB"))
    wr = WorkingResolution(sf, steps)
    exp = modcrop(native, sf).shape[0] // sf, modcrop(native, sf).shape[1] // sf
    assert low_in.shape[:2] == exp, f"{scene} downsampled by {sf} would be {exp[::-1]}, but input_lowres.png is {low_in.shape[1::-1]}"
    t = time.perf_counter()
    final, hr_u8 = wr.up(native, low_in, low_out)
    Image.fromarray(hr_u8).save(job_dir / "output_sr_full.png")
    Image.fromarray(final).save(job_dir / "output.jpg", quality=95)
    print(f"[sr] {wr.describe()}: {time.perf_counter() - t:.0f} s -> {job_dir / 'output.jpg'}")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="DDRM super-resolution of a job's saved low-res result (finish a job whose SR stage was interrupted).")
    ap.add_argument("--job", required=True, help="job folder with input_lowres.png and output_lowres.png (e.g. data/jobs/<id>)")
    ap.add_argument("--scene", required=True, help="the native-resolution scene the job used (e.g. data/uploads/<id>_clean.png)")
    ap.add_argument("--sf", type=int, required=True, help="the job's processing-resolution factor (2, 3 or 4)")
    ap.add_argument("--steps", type=int, default=15)
    a = ap.parse_args()
    resume(a.job, a.scene, a.sf, a.steps)
