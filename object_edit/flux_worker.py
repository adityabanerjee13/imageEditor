"""OmniPaint worker process: loads FLUX.1-dev (Q8 GGUF) + the OmniPaint LoRAs once and serves generations over stdin/stdout.

Why a separate process: OmniPaint monkey-patches diffusers' transformer_flux module and ships a package called `src`
(as does FreeFine); keeping it in a child means nothing of that leaks into the main process and release() is a clean
exit.  (Note: the 16 GB XPU budget is *global* across processes, so isolation does not buy memory - the transformer is
streamed layer by layer instead, see flux_omnipaint._load.)  `object_edit.flux_omnipaint` is the client.

Protocol (one JSON object per line):
  -> {"kind": "removal"|"insertion", "conditions": [{"path": png, "position_delta": [dx, dy] | null}, ...],
      "seed": int, "steps": int, "guidance_scale": float, "width": int, "height": int, "out": png}
  <- {"ok": true} | {"error": "..."}
On start the worker prints {"ready": true, "load_s": ...} (or {"error": ...}) and exits when stdin closes.
All library prints go to stderr; stdout carries only the protocol.
"""
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

PROTO = sys.stdout                      # keep the real stdout for the protocol ...
sys.stdout = sys.stderr                 # ... and send every library print (OmniPaint's "Active adapter now") to stderr


def send(obj):
    PROTO.write(json.dumps(obj) + "\n"); PROTO.flush()


def main():
    import torch
    from PIL import Image
    from object_edit import flux_omnipaint as fx
    from object_edit.common import DEV
    try:
        t = time.perf_counter()
        o = fx._import_omnipaint()
        pipe = fx._load()
        embeds = {}
        send({"ready": True, "load_s": round(time.perf_counter() - t, 1)})
    except Exception as e:                # noqa: BLE001
        send({"error": f"{type(e).__name__}: {e}"}); return
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
            kind = req["kind"]
            if kind not in embeds:
                embeds[kind] = o["load_npz_embeddings"](str(fx.W / "embeddings" / ("remove.npz" if kind == "removal" else "insert.npz")),
                                                        device=DEV, dtype=torch.bfloat16)
            pe, ppe, tids = embeds[kind]
            conds = [o["Condition"](kind, Image.open(c["path"]).convert("RGB"),
                                    position_delta=tuple(c["position_delta"]) if c.get("position_delta") else None)
                     for c in req["conditions"]]
            o["seed_everything"](req["seed"])
            img = o["generate"](pipe, conditions=conds, width=req.get("width", fx.RES), height=req.get("height", fx.RES),
                                num_inference_steps=req["steps"], prompt=None,
                                prompt_embeds=pe, pooled_prompt_embeds=ppe, text_ids=tids,
                                guidance_scale=req.get("guidance_scale", 3.5)).images[0]
            img.save(req["out"])
            send({"ok": True, "peak_gb": round(torch.xpu.max_memory_allocated() / 1e9, 2) if DEV.type == "xpu" else None})
        except Exception as e:            # noqa: BLE001
            import traceback; traceback.print_exc()
            send({"error": f"{type(e).__name__}: {e}"})


if __name__ == "__main__":
    main()
