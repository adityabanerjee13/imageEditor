"""FreeFine (SD-1.5, training-free) insertion: coarse affine paste, then detail-preserving regeneration inside the
target footprint with the source window as reference (settings of evaluation/FreeFine/freefine_batch_infer_2d.py)."""
import numpy as np
from PIL import Image

from object_edit import sd15
from object_edit.common import Stage, edit_region
from object_edit.insertion.base import Inserter
from object_edit.insertion.paste import coarse_paste, composite, save_coarse


class FreeFineInserter(Inserter):
    """Target edit region: "mask" = FreeFine's own auto-draw band (15 px around the footprint, `cons_area` pins the rest);
    otherwise the region is handed to FreeFine as its `draw_mask` (the completion area it may repaint around the object)
    and composited back.  FreeFine harmonises edges and contact shading there; it is not trained to synthesise cast
    shadows, so expect less than OmniPaint."""
    name = "freefine"
    shared_model = "sd15"
    generative = True

    def __init__(self, start_step=15, end_scale=0.0, prompt="", region="mask", margin=0.35):
        super().__init__(region, margin)
        self.start_step, self.end_scale, self.prompt = start_step, end_scale, prompt

    def load(self, log=None):
        sd15.get(log)

    def unload(self):
        sd15.release()

    def insert(self, original, background, move, *, seed, log, out_dir=None, idx=0, total=1):
        c = coarse_paste(original, background, move)
        save_coarse(c, out_dir, idx)
        if not c.tgt.any():
            print(f"[object {idx}] target lies outside the frame - skipped"); return background
        model = sd15.get(log)
        reg = edit_region(c.tgt, self.region, self.margin)
        with Stage(log, "regeneration", f"FreeFine gen object {idx + 1}/{total} (window {c.side}px, region {self.region}) start_step={self.start_step}") as st:
            result = self._regenerate(model, c.src512, np.repeat(c.srcmask512[..., None], 3, -1), c.coarse,
                                      c.tgt.astype(np.uint8) * 255, seed, draw=None if self.region == "mask" else reg & ~c.tgt)
        st.row.update(source_window=list(c.swin), dest_window=list(c.dwin), scale=round(move.scale, 3))
        if out_dir is not None:
            Image.fromarray(result).save(out_dir / f"obj{idx}_result_512.png")
        return composite(background, c, result, region=None if self.region == "mask" else reg)

    def _regenerate(self, model, ori_img, ori_mask, coarse, target_mask, seed, draw=None):
        """`ori_img`/`ori_mask` is the reference (source window) - a separate batch item, so it need not share the
        coarse image's frame; reduce_inp_artifacts=False keeps the source mask out of the edit frame's
        perturbation region (the hole is not in this canvas).  `draw` (bool 512, outside the target) is FreeFine's
        user-drawn completion area: with it the model may repaint that band (use_auto_draw=False); without it FreeFine
        auto-draws a 15 px band and `cons_area` pins everything else."""
        sd15.install_controller(model, for_background=False, start_layer=10)
        kw = (dict(draw_mask=None, use_auto_draw=True, cons_area=target_mask) if draw is None else
              dict(draw_mask=draw.astype(np.uint8) * 255, use_auto_draw=False, cons_area=None))
        out = model.FreeFine_generation(ori_img=ori_img, ori_mask=ori_mask, coarse_input=coarse, target_mask=target_mask,
                                        guidance_text=self.prompt, guidance_scale=7.5, eta=1.0, end_scale=self.end_scale,
                                        end_step=50, num_step=50, start_step=self.start_step, seed=seed,
                                        return_intermediates=False, reduce_inp_artifacts=False, **kw)
        return np.asarray(out).astype(np.uint8)
