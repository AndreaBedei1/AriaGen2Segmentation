"""Multi-layer semantic outputs + structured gaze-target resolution (Phase 3).

A single canonical mask cannot describe "road seen through the windshield" vs "the
windshield glass" vs "a mirror". We split detections into functional layers:

  exterior   : outdoor scene classes (road, cars, people, buildings, sky, signs, ...)
               — INCLUDING what is seen THROUGH the glass.
  cockpit    : vehicle-interior objects (wheel, dashboard, cluster, hands, trim, ...).
  transparent: windshield + side windows (a see-through backdrop, NOT the content).
  mirror     : rear-view + side mirrors (functional regions; may reflect exterior).

Gaze resolution rule (§3): the driver looking at a windshield pixel is really looking
at whatever exterior class is behind it; the glass never masks the content. If no
exterior class was detected there, the target is `unknown_exterior` (never silently
`windshield`). Mirrors are their own primary target, with an optional reflected content.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from ..taxonomy import Taxonomy

TRANSPARENT_NAMES = {"windshield", "left_window", "right_window"}
MIRROR_NAMES = {"rear_view_mirror", "side_mirror", "left_side_mirror", "right_side_mirror"}

LAYERS = ("exterior", "cockpit", "transparent", "mirror")


def layer_of_name(tax: Taxonomy, name: str) -> str:
    if name in TRANSPARENT_NAMES:
        return "transparent"
    if name in MIRROR_NAMES:
        return "mirror"
    c = tax.by_name.get(name)
    if c is None:
        return "exterior"
    return "cockpit" if c.group == "cockpit" else "exterior"


def layer_of_id(tax: Taxonomy, cid: int) -> str:
    c = tax.by_id.get(int(cid))
    return layer_of_name(tax, c.name) if c else "exterior"


def class_at(mask: np.ndarray, u: float, v: float, tax: Taxonomy,
             disc: int = 0) -> Tuple[int, str, float]:
    """(id, name, fraction) of the dominant class at (u,v). fraction is coverage in the disc."""
    if u is None or v is None or not np.isfinite(u) or not np.isfinite(v):
        return 0, "unknown", 0.0
    h, w = mask.shape[:2]
    ui, vi = int(round(u)), int(round(v))
    if not (0 <= ui < w and 0 <= vi < h):
        return 0, "unknown", 0.0
    if disc <= 0:
        cid = int(mask[vi, ui])
        return cid, tax.name_of(cid), 1.0 if cid else 0.0
    y0, y1 = max(0, vi - disc), min(h, vi + disc + 1)
    x0, x1 = max(0, ui - disc), min(w, ui + disc + 1)
    patch = mask[y0:y1, x0:x1].ravel()
    vals, counts = np.unique(patch[patch > 0], return_counts=True)
    if len(vals) == 0:
        return 0, "unknown", 0.0
    k = int(np.argmax(counts))
    cid = int(vals[k])
    return cid, tax.name_of(cid), float(counts[k] / patch.size)


def resolve_gaze_target(layer_masks: Dict[str, np.ndarray], u: float, v: float,
                        tax: Taxonomy, disc: int = 0,
                        conf_maps: Optional[Dict[str, np.ndarray]] = None) -> Dict[str, Any]:
    """Structured gaze target across layers (§3). Returns primary_target + per-layer
    observations + a resolution reason (not just a single string)."""
    conf_maps = conf_maps or {}

    def obs(layer):
        m = layer_masks.get(layer)
        if m is None:
            return {"class": None, "id": 0, "coverage": 0.0, "confidence": None}
        cid, name, frac = class_at(m, u, v, tax, disc)
        conf = None
        cm = conf_maps.get(layer)
        if cm is not None and u is not None and np.isfinite(u):
            ui, vi = int(round(u)), int(round(v))
            if 0 <= ui < cm.shape[1] and 0 <= vi < cm.shape[0]:
                conf = float(cm[vi, ui])
        return {"class": (name if cid else None), "id": cid, "coverage": frac, "confidence": conf}

    ext = obs("exterior")
    cock = obs("cockpit")
    trans = obs("transparent")
    mirr = obs("mirror")

    # resolution priority
    if mirr["class"]:
        primary, reason = mirr["class"], "gaze_on_mirror"
        secondary = ext["class"]  # optional reflected/behind content if any
    elif ext["class"]:
        primary, reason = ext["class"], ("through_glass" if trans["class"] else "direct_exterior")
        secondary = None
    elif cock["class"]:
        primary, reason = cock["class"], "cockpit_object"
        secondary = None
    elif trans["class"]:
        primary, reason = "unknown_exterior", "transparent_only_no_content"
        secondary = None
    else:
        primary, reason = "unknown", "nothing_at_gaze"
        secondary = None

    return {
        "primary_target": primary,
        "secondary_target": secondary,
        "resolution_reason": reason,
        "exterior_content": ext["class"],
        "cockpit_object": cock["class"],
        "transparent_surface": trans["class"],
        "mirror_region": mirr["class"],
        "confidence": {"exterior": ext["confidence"], "cockpit": cock["confidence"],
                       "transparent": trans["confidence"], "mirror": mirr["confidence"]},
        "coverage": {"exterior": ext["coverage"], "cockpit": cock["coverage"],
                     "transparent": trans["coverage"], "mirror": mirr["coverage"]},
    }
