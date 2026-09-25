"""FreeFine background generation (SD-1.5) as an object remover: runs in a 512 window around all holes."""
import cv2
import numpy as np

from object_edit import sd15
from object_edit.common import RES, Stage, bbox, crop, mask_to_512, square_window, to_512
from object_edit.removal.base import Remover


class FreeFineBgRemover(Remover):
    name = "freefine"
    shared_model = "sd15"

    def __init__(self, prompt="empty scene"):
        self.prompt = prompt

    def load(self, log=None):
        sd15.get(log)

    def unload(self):
        pass                                  # kept for the FreeFine inserter; run() releases sd15 at the end

    def remove(self, image, hole, *, seed, log, region=None):
        """Background generation fills what is masked, so the whole edit region is the hole."""
        hole = hole if region is None else (hole | region)
        model = sd15.get(log)
        H, W = hole.shape
        # FreeFine's generator works at 512: run it in a window around all holes and paste the fill back
        x0, y0, x1, y1 = bbox(hole)
        win = square_window(((x0 + x1) / 2, (y0 + y1) / 2), max(RES, 1.5 * max(x1 - x0, y1 - y0)), W, H)
        with Stage(log, "background_generation", f"FreeFine bg-gen 50 steps (window {win[2]}px)"):
            bg512 = self._generate(model, to_512(crop(image, win)),
                                   np.repeat(mask_to_512(crop(hole, win))[..., None], 3, -1), seed)
        background = image.copy()
        wx, wy, ws = win
        background[wy:wy + ws, wx:wx + ws] = np.where(crop(hole, win)[..., None],
                                                      cv2.resize(bg512, (ws, ws), interpolation=cv2.INTER_LANCZOS4), crop(image, win))
        return background

    def _generate(self, model, ori_img, dil_mask, seed):
        """Settings of evaluation/FreeFine/freefine_batch_infer_bggen_2d.py; original pixels kept outside the hole."""
        sd15.install_controller(model, for_background=True)
        bg = model.FreeFine_background_generation(ori_img, dil_mask, self.prompt, guidance_scale=7.5, eta=1.0, end_step=35,
                                                  num_step=50, end_scale=0.5, start_step=1, share_attn=True, method_type="tca",
                                                  local_text_edit=True, local_perturbation=True, verbose=False, seed=seed,
                                                  return_intermediates=False, latent_blended=False)
        m = dil_mask.astype(np.float32)
        return (ori_img * (1 - m) + bg * m).astype(np.uint8)
