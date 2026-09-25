"""Shared, lazily loaded OmniPaint pipeline: FLUX.1-dev (Q8_0 GGUF transformer, dequantised on the fly) + both OmniPaint
LoRAs (removal, insertion), no text encoders - the prompts are the precomputed T5/CLIP embeddings shipped with OmniPaint.

OmniPaint (ICCV 2025, repos/OmniPaint) conditions FLUX on extra *image tokens*: the VAE latents of the condition images
are packed like the noisy latents and concatenated to the attention sequence (see repos/OmniPaint/src/flux_core.py);
the LoRA (r=4, 29 MB) is what teaches the transformer to read them.  Removal takes one condition (the scene with the
hole blacked out), insertion takes two (scene with the target region blacked out + the subject on white, whose position
ids are shifted by (0, -32) so its tokens do not collide with the scene's).

Base-model note: black-forest-labs/FLUX.1-dev is gated on this account -> weights/flux1-dev-gguf/flux1-dev-Q8_0.gguf
(second-state/FLUX.1-dev-GGUF) and weights/flux-vae (nerualdreming/flux_vae).  ~13 GB resident on the iGPU, so nothing
transformer is streamed layer by layer (leaf-level group offloading, < 1 GB resident).  512x512, 28 steps at ~18 s/step:
~9 min per removal (1 condition) / ~12 min per insertion (2 conditions) on the iGPU.

Process model: the pipeline lives in a child process (object_edit/flux_worker.py) so OmniPaint's diffusers monkey-patching
and its `src` package stay out of the main process and nothing lingers after release().  This module is the client:
`get()` spawns the worker (~40 s load), `run()` exchanges PNGs through a temp dir, `release()` terminates it.
"""
import importlib.util
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import torch

from object_edit.common import DEV, REPOS, WEIGHTS, empty_cache

RES = 512
FLUX_CONFIG = {"_class_name": "FluxTransformer2DModel", "_diffusers_version": "0.30.0", "attention_head_dim": 128,
               "guidance_embeds": True, "in_channels": 64, "joint_attention_dim": 4096, "num_attention_heads": 24,
               "num_layers": 19, "num_single_layers": 38, "patch_size": 1, "pooled_projection_dim": 768,
               "axes_dims_rope": [16, 56, 56]}
W = WEIGHTS / "omnipaint"
_omni = None
_proc = None
_tmp = None


def _import_omnipaint():
    """OmniPaint's package is called `src` (as is FreeFine's) -> load it under an alias; and its flux_core imports helper
    names that newer diffusers no longer re-exports from transformer_flux -> patch them in."""
    global _omni
    if _omni is None:
        import diffusers.models.transformers.transformer_flux as tfx
        from diffusers.utils import USE_PEFT_BACKEND, is_torch_version, logging, scale_lora_layers, unscale_lora_layers
        for n, v in {"USE_PEFT_BACKEND": USE_PEFT_BACKEND, "is_torch_version": is_torch_version, "scale_lora_layers": scale_lora_layers,
                     "unscale_lora_layers": unscale_lora_layers, "logger": logging.get_logger("omnipaint")}.items():
            if not hasattr(tfx, n):
                setattr(tfx, n, v)
        src = REPOS / "OmniPaint" / "src"
        spec = importlib.util.spec_from_file_location("omnipaint_src", str(src / "__init__.py"), submodule_search_locations=[str(src)])
        pkg = importlib.util.module_from_spec(spec); sys.modules["omnipaint_src"] = pkg; spec.loader.exec_module(pkg)
        from omnipaint_src.condition import Condition
        from omnipaint_src.embedding_loader import load_npz_embeddings
        from omnipaint_src.generate import generate, seed_everything
        _omni = dict(Condition=Condition, generate=generate, seed_everything=seed_everything, load_npz_embeddings=load_npz_embeddings)
    return _omni


def _load():
    from diffusers import AutoencoderKL, FlowMatchEulerDiscreteScheduler, FluxPipeline, FluxTransformer2DModel, GGUFQuantizationConfig
    cfg_dir = WEIGHTS / "flux1-dev-gguf" / "transformer"; cfg_dir.mkdir(exist_ok=True)
    (cfg_dir / "config.json").write_text(json.dumps(FLUX_CONFIG, indent=2))
    transformer = FluxTransformer2DModel.from_single_file(str(WEIGHTS / "flux1-dev-gguf" / "flux1-dev-Q8_0.gguf"), config=str(cfg_dir),
                                                          quantization_config=GGUFQuantizationConfig(compute_dtype=torch.bfloat16),
                                                          torch_dtype=torch.bfloat16)
    vae = AutoencoderKL.from_pretrained(str(WEIGHTS / "flux-vae"), subfolder="vae", torch_dtype=torch.bfloat16)
    sched = FlowMatchEulerDiscreteScheduler(base_image_seq_len=256, max_image_seq_len=4096, base_shift=0.5, max_shift=1.15,
                                            shift=3.0, use_dynamic_shifting=True)
    pipe = FluxPipeline(scheduler=sched, vae=vae, text_encoder=None, tokenizer=None, text_encoder_2=None, tokenizer_2=None,
                        transformer=transformer)
    pipe.load_lora_weights(str(W / "weights" / "omnipaint_remove.safetensors"), adapter_name="removal")
    pipe.load_lora_weights(str(W / "weights" / "omnipaint_insert.safetensors"), adapter_name="insertion")
    # Leaf-level group offloading: each Linear/Norm is copied to the device right before its forward and dropped after,
    # so the 12 GB Q8 transformer needs < 1 GB of device memory (peak 0.68 GB at 512).  Costs ~2x per step (~18 s vs ~10 s)
    # but is the only mode that fits: the 16 GB XPU budget is shared with the desktop and the parent process's caches.
    # Block-level / sequential offload do not work here: OmniPaint calls each block's sub-layers directly (block hooks never
    # fire) and accelerate's sequential offload rebuilds GGUF parameters without their quant type.
    transformer.enable_group_offload(onload_device=DEV, offload_device=torch.device("cpu"), offload_type="leaf_level")
    vae.to(DEV)
    return pipe


# ============================================================================ client side
class WorkerError(RuntimeError):
    pass


def _recv():
    line = _proc.stdout.readline()
    if not line:
        raise WorkerError(f"OmniPaint worker exited (code {_proc.poll()})")
    msg = json.loads(line)
    if "error" in msg:
        raise WorkerError(msg["error"])
    return msg


def get(log=None):
    """Start the worker process (loads FLUX + LoRAs, ~40 s; logged as a 'load' stage) unless it is already running."""
    global _proc, _tmp
    if _proc is None or _proc.poll() is not None:
        empty_cache()                                   # give the child as much of the shared budget as possible
        _tmp = tempfile.TemporaryDirectory(prefix="omnipaint_")
        _proc = subprocess.Popen([sys.executable, "-u", str(Path(__file__).with_name("flux_worker.py"))],
                                 stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, encoding="utf-8", bufsize=1)
        msg = _recv()
        if log is not None:
            log.append({"stage": "load", "model": "FLUX.1-dev Q8 + OmniPaint LoRAs (worker process)", "time_s": msg["load_s"]})
    return _proc


def release():
    global _proc, _tmp
    if _proc is not None:
        try:
            _proc.stdin.close(); _proc.wait(timeout=30)
        except Exception:                               # noqa: BLE001
            _proc.kill()
        _proc = None
    if _tmp is not None:
        _tmp.cleanup(); _tmp = None


def is_loaded():
    return _proc is not None and _proc.poll() is None


def condition(kind, pil, position_delta=None):
    return {"kind": kind, "image": pil, "position_delta": list(position_delta) if position_delta else None}


_seq = 0


def canvas_size(w, h, long_side):
    """Generation size for a w x h frame: long side scaled to `long_side` (never upscaled), both sides multiples of 16
    (VAE /8 and 2x2 latent packing).  FLUX.1-dev is trained at 0.25-2 MP and OmniPaint's scripts cap the long side at
    1024, so 1024 is the useful maximum."""
    f = min(1.0, long_side / max(w, h))
    return max(16, int(round(w * f / 16)) * 16), max(16, int(round(h * f / 16)) * 16)


def run(kind, conditions, seed, steps=28, guidance_scale=3.5, log=None, size=(RES, RES)):
    """One OmniPaint generation in the worker at `size` = (width, height) (512x512 default; multiples of 16).  `kind` =
    "removal" | "insertion" picks the LoRA and the static prompt embeddings (remove.npz / insert.npz); `conditions` from
    condition() - the first must have the canvas size, the subject is always 512x512.  Returns a PIL image."""
    global _seq
    from PIL import Image
    get(log)
    d = Path(_tmp.name); _seq += 1
    req = {"kind": kind, "seed": seed, "steps": steps, "guidance_scale": guidance_scale, "width": int(size[0]), "height": int(size[1]),
           "conditions": [], "out": str(d / f"out_{_seq}.png")}
    for i, c in enumerate(conditions):
        p = d / f"cond_{_seq}_{i}.png"; c["image"].save(p)
        req["conditions"].append({"path": str(p), "position_delta": c["position_delta"]})
    _proc.stdin.write(json.dumps(req) + "\n"); _proc.stdin.flush()
    _recv()
    return Image.open(req["out"]).convert("RGB")
