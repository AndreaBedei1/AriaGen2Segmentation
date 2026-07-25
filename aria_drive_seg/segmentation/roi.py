"""Hierarchical ROI + multi-scale helpers (Phase 4 & 5).

ROIs (windshield / windows / mirror zones / interior zones) are DERIVED from the
structural detections of the full-frame pass (no per-frame hardcoded coordinates).
For the same person+car they are temporally stable, so an optional EMA smooths them
and only updates when the structural box moves (camera motion) — documented, not fixed.

Each ROI re-runs a TARGETED class subset on its crop (optionally upscaled / tiled) so
small / distant objects seen through the windshield or in a mirror get more pixels.
Crop detections are remapped to full-frame coordinates with recorded provenance.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# Which detected structural class defines each ROI (first present wins).
ROI_STRUCTURAL = {
    "windshield_view": ["windshield"],
    "left_window_view": ["left_window"],
    "right_window_view": ["right_window"],
    "rear_view_mirror_zone": ["rear_view_mirror"],
    "left_side_mirror_zone": ["left_side_mirror", "side_mirror"],
    "right_side_mirror_zone": ["right_side_mirror"],
    "dashboard_zone": ["dashboard"],
    "instrument_cluster_zone": ["instrument_cluster"],
    "center_console_zone": ["center_console", "infotainment_screen"],
}

# Class subsets re-detected inside each ROI (filled from the taxonomy at runtime).
EXTERIOR_SMALL = [
    "traffic_light", "traffic_sign", "stop_sign", "yield_sign", "speed_limit_sign",
    "warning_sign", "direction_sign", "pedestrian", "person", "child", "cyclist",
    "motorcyclist", "rider", "bicycle", "motorcycle", "bollard", "traffic_cone",
    "delineator_post", "pole", "street_light", "lane_marking", "stop_line",
    "direction_arrow", "car", "truck", "bus", "van",
]
MIRROR_REFLECTED = ["car", "truck", "bus", "van", "motorcycle", "bicycle", "person",
                    "rider", "pedestrian", "cyclist"]
INTERIOR_SUBSETS = {
    "dashboard_zone": ["dashboard", "center_console", "left_air_vent", "right_air_vent",
                       "warning_indicator"],
    "instrument_cluster_zone": ["instrument_cluster", "speedometer_display",
                                "warning_indicator"],
    "center_console_zone": ["infotainment_screen", "navigation_display",
                            "climate_controls", "gear_selector", "phone_mount"],
}


def roi_class_subset(roi_name: str, available: set) -> List[str]:
    if roi_name in ("windshield_view", "left_window_view", "right_window_view"):
        base = EXTERIOR_SMALL
    elif roi_name.endswith("mirror_zone"):
        base = MIRROR_REFLECTED
    else:
        base = INTERIOR_SUBSETS.get(roi_name, [])
    return [c for c in base if c in available]


def _largest_box(dets: List[Any], names: List[str]) -> Optional[Tuple[float, float, float, float]]:
    best, best_area = None, 0.0
    for d in dets:
        nm = d["name"] if isinstance(d, dict) else getattr(d, "canonical_name", None)
        if nm in names:
            box = d["box"] if isinstance(d, dict) else d.box
            area = (box[2] - box[0]) * (box[3] - box[1])
            if area > best_area:
                best, best_area = tuple(float(x) for x in box), area
    return best


def _pad_clip(box, pad_frac, h, w) -> Tuple[int, int, int, int]:
    x0, y0, x1, y1 = box
    pw, ph = (x1 - x0) * pad_frac, (y1 - y0) * pad_frac
    return (max(0, int(x0 - pw)), max(0, int(y0 - ph)),
            min(w, int(x1 + pw)), min(h, int(y1 + ph)))


def derive_rois(detections: List[Any], h: int, w: int,
                pad_frac: float = 0.05) -> Dict[str, Tuple[int, int, int, int]]:
    """Return {roi_name: (x0,y0,x1,y1)} from structural detections present this frame."""
    rois: Dict[str, Tuple[int, int, int, int]] = {}
    for name, structural in ROI_STRUCTURAL.items():
        box = _largest_box(detections, structural)
        if box is not None:
            rois[name] = _pad_clip(box, pad_frac, h, w)
    return rois


@dataclass
class ROIState:
    """Optional temporal EMA of ROI boxes (same car/person -> stable ROIs)."""
    alpha: float = 0.4
    move_thresh_frac: float = 0.15
    boxes: Dict[str, Tuple[float, float, float, float]] = field(default_factory=dict)

    def update(self, rois: Dict[str, Tuple[int, int, int, int]], h: int, w: int
               ) -> Dict[str, Tuple[int, int, int, int]]:
        out = {}
        for name, box in rois.items():
            prev = self.boxes.get(name)
            if prev is None:
                self.boxes[name] = tuple(float(x) for x in box)
            else:
                # if the box moved a lot (camera motion) snap; else EMA-smooth
                moved = abs(box[0] - prev[0]) + abs(box[1] - prev[1]) > self.move_thresh_frac * w
                a = 1.0 if moved else self.alpha
                self.boxes[name] = tuple(a * b + (1 - a) * p for b, p in zip(box, prev))
            out[name] = tuple(int(round(x)) for x in self.boxes[name])
        # keep previously-known ROIs not seen this frame (stability)
        for name, prev in self.boxes.items():
            out.setdefault(name, tuple(int(round(x)) for x in prev))
        return out


def tiles_of(box: Tuple[int, int, int, int], n_tiles: int, overlap: float
             ) -> List[Tuple[int, int, int, int]]:
    """Split a box into a horizontal strip of overlapping tiles (small-object pass)."""
    x0, y0, x1, y1 = box
    if n_tiles <= 1:
        return [box]
    bw = x1 - x0
    step = bw / n_tiles
    ov = step * overlap
    out = []
    for i in range(n_tiles):
        tx0 = int(max(x0, x0 + i * step - ov))
        tx1 = int(min(x1, x0 + (i + 1) * step + ov))
        out.append((tx0, y0, tx1, y1))
    return out


def remap_box(box_crop, offset: Tuple[int, int], scale: float
              ) -> Tuple[float, float, float, float]:
    ox, oy = offset
    return (box_crop[0] / scale + ox, box_crop[1] / scale + oy,
            box_crop[2] / scale + ox, box_crop[3] / scale + oy)


def paste_mask(mask_crop: np.ndarray, offset: Tuple[int, int],
               full_h: int, full_w: int, scale: float) -> np.ndarray:
    """Place a crop (possibly upscaled) boolean mask into a full-frame boolean mask."""
    import cv2
    ox, oy = offset
    m = mask_crop.astype(np.uint8)
    if scale != 1.0:
        ch, cw = m.shape
        m = cv2.resize(m, (int(round(cw / scale)), int(round(ch / scale))),
                       interpolation=cv2.INTER_NEAREST)
    full = np.zeros((full_h, full_w), dtype=bool)
    hh, ww = m.shape
    y1, x1 = min(full_h, oy + hh), min(full_w, ox + ww)
    full[oy:y1, ox:x1] = m[: y1 - oy, : x1 - ox].astype(bool)
    return full
