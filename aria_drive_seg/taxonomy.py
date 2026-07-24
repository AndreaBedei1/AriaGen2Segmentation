"""Canonical taxonomy: load classes.yaml, provide id<->name, palette, colorize."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import yaml


@dataclass(frozen=True)
class CanonClass:
    id: int
    name: str
    group: str            # special | road | cockpit
    color: tuple          # (r, g, b)
    eval: bool            # part of the common method-vs-method comparison set


class Taxonomy:
    def __init__(self, classes: List[CanonClass]):
        self.classes = sorted(classes, key=lambda c: c.id)
        self.by_id: Dict[int, CanonClass] = {c.id: c for c in self.classes}
        self.by_name: Dict[str, CanonClass] = {c.name: c for c in self.classes}
        self.max_id = max(self.by_id)
        if 0 not in self.by_id:
            raise ValueError("Canonical taxonomy must define id 0 (unknown)")

    # ---- constructors ----
    @classmethod
    def load(cls, path: str | Path) -> "Taxonomy":
        data = yaml.safe_load(Path(path).read_text())
        classes = [
            CanonClass(
                id=int(c["id"]), name=str(c["name"]), group=str(c["group"]),
                color=tuple(int(x) for x in c["color"]), eval=bool(c.get("eval", False)),
            )
            for c in data["classes"]
        ]
        return cls(classes)

    # ---- lookups ----
    def id_of(self, name: str) -> int:
        return self.by_name[name].id

    def name_of(self, cid: int) -> str:
        c = self.by_id.get(int(cid))
        return c.name if c else f"id_{cid}"

    def names(self) -> List[str]:
        return [c.name for c in self.classes]

    def eval_ids(self) -> List[int]:
        return [c.id for c in self.classes if c.eval]

    def group_ids(self, group: str) -> List[int]:
        return [c.id for c in self.classes if c.group == group]

    # ---- rendering ----
    def palette(self) -> np.ndarray:
        """(max_id+1, 3) uint8 lookup table; unknown ids -> black."""
        lut = np.zeros((self.max_id + 1, 3), dtype=np.uint8)
        for c in self.classes:
            lut[c.id] = c.color
        return lut

    def colorize(self, mask: np.ndarray) -> np.ndarray:
        """Map a uint16 id-mask to an RGB image using the deterministic palette."""
        lut = self.palette()
        m = np.clip(mask.astype(np.int64), 0, self.max_id)
        return lut[m]


def load_default(config_dir: str | Path = "configs") -> Taxonomy:
    return Taxonomy.load(Path(config_dir) / "classes.yaml")
