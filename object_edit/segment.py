"""Stage 1-2: object masks (SAM 3, one click per object) and mask dilation."""
import cv2
import numpy as np
import torch

from object_edit.common import DEV, WEIGHTS, Stage, edit_region, empty_cache, timed


# ============================================================================ 1. object mask (SAM 3, one click)
def segment_objects(task, log, max_frac=0.3, max_ratio=4.0, min_iou=0.3):
    from transformers import Sam3TrackerModel, Sam3TrackerProcessor
    W, H = task.size
    (proc, model), load_s = timed(lambda: (Sam3TrackerProcessor.from_pretrained(str(WEIGHTS / "sam3")),
                                           Sam3TrackerModel.from_pretrained(str(WEIGHTS / "sam3")).to(DEV).eval()))
    with Stage(log, "segmentation", f"SAM 3 point prompt x{len(task.moves)}") as st:
        st.row["objects"] = []
        for mv in task.moves:
            x, y = mv.src
            inp = proc(images=task.image, input_points=[[[[x, y]]]], input_labels=[[[1]]], return_tensors="pt").to(DEV)
            with torch.no_grad():
                o = model(**inp, multimask_output=True)
            masks = proc.post_process_masks(o.pred_masks.cpu(), inp["original_sizes"])[0][0].numpy() > 0    # (3, H, W)
            iou = o.iou_scores.flatten().cpu().numpy()
            best = pick_proposal(masks, iou, H * W, max_frac, max_ratio, min_iou)
            mv.mask = clean_mask(masks[best], (x, y))
            st.row["objects"].append({"src": list(mv.src), "iou": np.round(iou, 3).tolist(),
                                      "areas": masks.reshape(3, -1).sum(1).tolist(), "picked": best})
    st.row["load_s"] = load_s
    del model; empty_cache()


def pick_proposal(masks, iou, frame_px, max_frac=0.3, max_ratio=4.0, min_iou=0.3):
    """SAM's proposals are sub-part / part / whole.  Start from the best-IoU one and accept a larger
    proposal only if it is a modest enlargement (< max_ratio x area, e.g. plant+pot over pot) and
    not implausibly large (< max_frac of the frame) - otherwise a click on a cushion returns the sofa."""
    areas = masks.reshape(len(masks), -1).sum(1)
    top = int(iou.argmax()); best = top
    for j in np.argsort(-areas):
        if areas[j] > areas[top] and areas[j] <= max_ratio * areas[top] and areas[j] < max_frac * frame_px and iou[j] >= min_iou:
            best = int(j); break
    return best


def clean_mask(m, point, close_px=7):
    """Morphological close, keep the component containing `point` (else the largest), fill holes."""
    k = np.ones((close_px, close_px), np.uint8)
    m8 = cv2.morphologyEx(m.astype(np.uint8), cv2.MORPH_CLOSE, k)
    n, lab = cv2.connectedComponents(m8)
    if n > 1:
        x, y = point
        keep = lab[y, x] if lab[y, x] > 0 else 1 + np.bincount(lab[lab > 0]).argmax()
        m8 = (lab == keep).astype(np.uint8)
    flood = m8.copy(); h, w = flood.shape
    cv2.floodFill(flood, np.zeros((h + 2, w + 2), np.uint8), (0, 0), 1)
    return (m8 | (1 - flood)).astype(bool)


# ============================================================================ 2. mask dilation
def dilate_object_masks(task, radius, log, src_region="mask", margin=0.35):
    """Removal hole = each object mask grown by `radius` px (kernel 2r+1), unioned.  Each move also gets its source edit
    region (common.edit_region of the hole with `src_region` / `margin`): the pixels the remover may repaint."""
    with Stage(log, "mask_dilation", f"radius {radius} px, source region {src_region}") as st:
        k = np.ones((2 * radius + 1, 2 * radius + 1), np.uint8)
        for mv in task.moves:
            mv.mask_dilated = cv2.dilate(mv.mask.astype(np.uint8), k) > 0
            mv.region = edit_region(mv.mask_dilated, src_region, margin)
    st.row.update(mask_px=int(task.mask.sum()), dilated_px=int(task.mask_dilated.sum()), region_px=int(task.region.sum()))
