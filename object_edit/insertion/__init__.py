"""Object-insertion backends. Each entry is `name -> factory(cfg) -> Inserter`; see base.py for the contract."""
from object_edit.insertion.base import Inserter
from object_edit.insertion.freefine import FreeFineInserter
from object_edit.insertion.omnipaint import OmniPaintInserter
from object_edit.insertion.paste import PasteInserter

INSERTERS = {
    "freefine": lambda cfg: FreeFineInserter(start_step=cfg["start_step"], end_scale=cfg["end_scale"], prompt=cfg["prompt"],
                                             region=cfg["dst_region"], margin=cfg["region_margin"]),
    "paste":    lambda cfg: PasteInserter(),
    "omnipaint": lambda cfg: OmniPaintInserter(steps=cfg["omnipaint_steps"], grow=cfg["omnipaint_grow"], mode=cfg["omnipaint_mode"],
                                               res=cfg["omnipaint_res"], region=cfg["dst_region"], margin=cfg["region_margin"]),
}


def make_inserter(cfg) -> Inserter:
    try:
        return INSERTERS[cfg["insertion"]](cfg)
    except KeyError:
        raise ValueError(f"unknown insertion backend {cfg['insertion']!r}; choose from {sorted(INSERTERS)}") from None
