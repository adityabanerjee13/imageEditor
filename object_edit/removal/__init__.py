"""Object-removal backends. Each entry is `name -> factory(cfg) -> Remover`; see base.py for the contract."""
from object_edit.removal.base import Remover
from object_edit.removal.freefine_bg import FreeFineBgRemover
from object_edit.removal.lama import LamaRemover
from object_edit.removal.omnipaint import OmniPaintRemover

REMOVERS = {
    "lama":       lambda cfg: LamaRemover(refine=True),
    "lama-plain": lambda cfg: LamaRemover(refine=False),
    "freefine":   lambda cfg: FreeFineBgRemover(prompt=cfg["bg_prompt"]),
    "omnipaint":  lambda cfg: OmniPaintRemover(steps=cfg["omnipaint_steps"], mode=cfg["omnipaint_mode"], res=cfg["omnipaint_res"]),
}


def make_remover(cfg) -> Remover:
    try:
        return REMOVERS[cfg["removal"]](cfg)
    except KeyError:
        raise ValueError(f"unknown removal backend {cfg['removal']!r}; choose from {sorted(REMOVERS)}") from None
