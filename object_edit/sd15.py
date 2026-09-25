"""Shared, lazily loaded FreeFine pipeline (SD-1.5 fp16) so the FreeFine remover and inserter never load it twice."""
import sys
import types

import torch

from object_edit.common import DEV, WEIGHTS, empty_cache, timed

_model = None


def _import_freefine():
    # FreeFine imports rembg only for an optional matting helper that is never called -> stub it
    sys.modules.setdefault("rembg", types.SimpleNamespace(remove=lambda *a, **k: (_ for _ in ()).throw(RuntimeError("rembg stub"))))
    from diffusers import DDIMScheduler
    from src.demo.model import FreeFinePipeline
    from src.utils.attention import Attention_Modulator, register_attention_control, register_attention_control_4bggen
    return DDIMScheduler, FreeFinePipeline, Attention_Modulator, register_attention_control, register_attention_control_4bggen


def get(log=None):
    """The FreeFine pipeline on the device; loaded on first use (logged as a 'load' stage when `log` is given)."""
    global _model
    if _model is None:
        DDIMScheduler, FreeFinePipeline, *_ = _import_freefine()

        def load():
            model = FreeFinePipeline.from_pretrained(str(WEIGHTS / "stable-diffusion-v1-5"), torch_dtype=torch.float16,
                                                     variant="fp16", safety_checker=None).to(DEV)
            model.scheduler = DDIMScheduler.from_config(model.scheduler.config)

            # upstream hard-codes .cuda() for the text encoder; route through the pipeline device instead
            @torch.no_grad()
            def get_text_embeddings(self, prompt):
                ti = self.tokenizer(prompt, padding="max_length", max_length=77, return_tensors="pt")
                return self.text_encoder(ti.input_ids.to(self.device))[0]
            model.get_text_embeddings = types.MethodType(get_text_embeddings, model)
            return model
        _model, load_s = timed(load)
        if log is not None:
            log.append({"stage": "load", "model": "SD-1.5 fp16 (FreeFine)", "time_s": load_s})
    return _model


def release():
    global _model
    if _model is not None:
        _model = None
        empty_cache()


def is_loaded():
    return _model is not None


def install_controller(model, for_background, start_layer=None):
    _, _, Attention_Modulator, register_attention_control, register_attention_control_4bggen = _import_freefine()
    controller = Attention_Modulator() if start_layer is None else Attention_Modulator(start_layer=start_layer)
    model.controller = controller
    (register_attention_control_4bggen if for_background else register_attention_control)(model, controller)
    model.modify_unet_forward()
    model.enable_attention_slicing()          # xformers is unavailable on XPU; the hooks replace the processors anyway
