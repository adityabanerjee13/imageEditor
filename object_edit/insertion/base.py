"""Object-insertion contract.

An Inserter places one object (cut from the original frame with `move.mask`) into the background so that the
object's anchor `move.src` lands on `move.dst`, scaled by `move.scale` about the anchor.

    insert(original, background, move, *, seed, log, out_dir, idx, total) -> background

- `original`:   uint8 HxWx3 RGB, the untouched input frame (source of the object pixels / reference)
- `background`: uint8 HxWx3 RGB, the frame with the objects removed, possibly already holding earlier insertions
- `move`:       task.Move with `.mask` (bool HxW), `.src`, `.dst`, `.scale`
- returns a new uint8 HxWx3; only pixels inside the *target edit region* may differ from `background`.  The region is
  the user's choice (constructor `region` = mask | dilated | box | full, `margin`; see common.edit_region), taken around
  the pasted footprint: "mask" = the footprint (+ a few px), "dilated"/"box" = a band where a cast shadow may be
  rendered, "full" = the whole canvas the model rendered
- `out_dir` / `idx` / `total`: where to save per-object intermediates (`obj<idx>_*.png`) and the object index / count for logging
- `log`: stage list - wrap the model call in `common.Stage(log, "regeneration", "<model>")`
- `load()` / `unload()` bracket the accelerator residency; multi-object runs are a fold:
      bg = inserter.insert(orig, bg, mv) for mv in moves
  so later objects see earlier insertions.

`paste.py` is the model-free reference implementation and provides the shared coarse step (windows + affine copy)
that model-based inserters refine.  Register new implementations in `object_edit/insertion/__init__.py`.
"""


class Inserter:
    name = "base"
    shared_model = None
    generative = False       # can render context around the object (region "full" meaningful)

    def __init__(self, region="mask", margin=0.35):
        self.region, self.margin = region, margin      # name of a base model shared with the other module (e.g. "sd15", "flux")

    def load(self, log=None):
        pass

    def unload(self):
        pass

    def insert(self, original, background, move, *, seed, log, out_dir=None, idx=0, total=1):
        raise NotImplementedError
