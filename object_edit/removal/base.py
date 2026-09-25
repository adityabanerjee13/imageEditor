"""Object-removal contract.

A Remover fills the `hole` region of a full-resolution frame with plausible background.

    remove(image, hole, *, seed, log, region=None) -> image

- `image`:  uint8 HxWx3 RGB, the original frame at native resolution
- `hole`:   bool HxW, True = the object pixels to fill (silhouette dilated a few px)
- `region`: bool HxW, superset of `hole` (default: `hole`) - the *source edit region* chosen by the user
            (common.edit_region: mask | dilated | box | full): the only pixels the remover may change.  A generative
            remover blacks out `hole`, re-renders its canvas and keeps its output inside `region` (so a cast shadow
            in the band around the object is removed); an inpainter without that ability (LaMa) fills all of `region`.
- returns uint8 HxWx3 of the same size; pixels outside `region` must be byte-identical to `image`
- `generative`: class flag - True if the backend re-renders context (supports region "full"); the orchestrator
            downgrades "full" to "box" for the others
- `log`:   list of stage rows - wrap the model call in `common.Stage(log, "background_generation", "<model>")`
- `load()` / `unload()` bracket the model's residency on the accelerator; `run()` calls `unload()` before the
  inserter is loaded (16 GB shared iGPU: never two large models resident).  A remover that shares a model with an
  inserter (see `object_edit.sd15`) may keep it and report `shared_model = "sd15"`.

Register new implementations in `object_edit/removal/__init__.py`.
"""


class Remover:
    name = "base"
    shared_model = None      # name of a base model shared with the other module (e.g. "sd15", "flux")
    generative = False       # can re-render context outside the hole (region "full" allowed)

    def load(self, log=None):
        pass

    def unload(self):
        pass

    def remove(self, image, hole, *, seed, log, region=None):
        raise NotImplementedError
