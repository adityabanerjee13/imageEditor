"""Stage 3: perspective scale from MoGe-2 metric depth."""
import numpy as np
import torch

from object_edit.common import DEV, WEIGHTS, Stage, empty_cache, timed


def estimate_perspective_scale(task, log):
    """Size ratio for a rigid object whose anchor moves from src to dst: z_src / z_dst (metric depth)."""
    from moge.model.v2 import MoGeModel
    model, load_s = timed(lambda: MoGeModel.from_pretrained(str(WEIGHTS / "moge-2-vitl-normal" / "model.pt")).to(DEV).eval())
    x = torch.tensor(np.asarray(task.image) / 255.0, dtype=torch.float32, device=DEV).permute(2, 0, 1)
    with Stage(log, "geometry", "MoGe-2 ViT-L") as st:
        with torch.no_grad():
            o = model.infer(x, use_fp16=False)          # fp16 autocast is broken on XPU for this model
    st.row["load_s"] = load_s
    z = o["points"][..., 2].cpu().numpy(); valid = o["mask"].cpu().numpy().astype(bool)
    del model; empty_cache()

    def zat(p, r=6):
        x0, y0 = p
        win, vw = z[max(0, y0 - r):y0 + r + 1, max(0, x0 - r):x0 + r + 1], valid[max(0, y0 - r):y0 + r + 1, max(0, x0 - r):x0 + r + 1]
        return float(np.median(win[vw])) if vw.any() else float(np.median(win))
    st.row["objects"] = []
    for mv in task.moves:
        z_src, z_dst = zat(mv.src), zat(mv.dst)
        mv.scale = float(np.clip(z_src / max(z_dst, 1e-3), 0.25, 4.0))
        st.row["objects"].append({"z_src": round(z_src, 3), "z_dst": round(z_dst, 3), "scale": round(mv.scale, 3)})
        print(f"[geometry] {mv.src} -> {mv.dst}: z_src={z_src:.2f} m z_dst={z_dst:.2f} m -> perspective scale {mv.scale:.2f}")
