"""Measure what each stage does to the frame, and how much of it survives.

The question this module answers is deliberately narrow and empirical: for every
pixel of the source RGB frame, does it still exist in the geometry a given stage
produces, and if not, where was it lost?

Nothing here assumes a crop exists. The retained region is computed by actually
pushing source pixels through the transform and asking whether they land inside
the destination, so "there is no crop" and "there is a crop of this exact shape"
are both results the same code can produce.

Two different notions of "valid" matter and are kept apart:

* **model-valid**: the calibration can unproject the pixel at all. A fisheye's
  projection model has an angular limit, so the corners of the sensor rectangle
  are outside it and carry no usable ray.
* **retained**: the pixel is model-valid *and* the stage's transform places it
  inside the destination image.

A pixel that is model-valid but not retained is field of view that the pipeline
threw away, which is exactly what a mirror near the image edge is made of.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


@dataclass
class StageGeometry:
    """What one stage of the chain does to the frame."""

    stage: str
    description: str
    width: int
    height: int
    aspect_ratio: float
    transform: str
    inverse_transform: str
    crop_box: Optional[List[int]] = None          # x0, y0, x1, y1 in the input
    padding: Optional[List[int]] = None           # left, top, right, bottom
    scale_x: Optional[float] = None
    scale_y: Optional[float] = None
    source_frame_retained_fraction: Optional[float] = None
    valid_region_retained_fraction: Optional[float] = None
    pixels_lost_left: Optional[int] = None
    pixels_lost_right: Optional[int] = None
    pixels_lost_top: Optional[int] = None
    pixels_lost_bottom: Optional[int] = None
    aspect_preserved: Optional[bool] = None
    notes: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def aspect(width: int, height: int) -> float:
    return float(width) / float(height)


def _pixel_grid(width: int, height: int) -> np.ndarray:
    us, vs = np.meshgrid(np.arange(width), np.arange(height))
    return np.stack([us.ravel(), vs.ravel()], axis=1).astype(np.float64)


def model_valid_mask(calib, width: int, height: int) -> np.ndarray:
    """Pixels the camera model can turn into a ray.

    For a fisheye this excludes the corners of the sensor rectangle, which lie
    beyond the model's angular limit and contain no scene information.
    """
    pix = _pixel_grid(width, height)
    valid = np.zeros(pix.shape[0], dtype=bool)
    try:
        rays = np.asarray(calib.unproject(pix), dtype=np.float64)
        valid = np.isfinite(rays).all(axis=1)
    except Exception:
        for i in range(pix.shape[0]):
            ray = calib.unproject(pix[i])
            valid[i] = ray is not None and np.isfinite(np.asarray(ray)).all()
    return valid.reshape(height, width)


def retained_mask(source_calib, destination_calib,
                  source_size: Tuple[int, int],
                  destination_size: Tuple[int, int]) -> np.ndarray:
    """Source pixels that land inside the destination image.

    Forward direction on purpose: the rectifier itself is built backwards (for
    each destination pixel, where does it come from), which cannot answer "what
    did we throw away".
    """
    sw, sh = source_size
    dw, dh = destination_size
    pix = _pixel_grid(sw, sh)

    inside = np.zeros(pix.shape[0], dtype=bool)
    try:
        rays = np.asarray(source_calib.unproject(pix), dtype=np.float64)
        finite = np.isfinite(rays).all(axis=1)
        projected = np.full((pix.shape[0], 2), np.nan)
        if finite.any():
            out = np.asarray(destination_calib.project(rays[finite]), dtype=np.float64)
            projected[finite] = out
        ok = np.isfinite(projected).all(axis=1)
        inside = ok & (projected[:, 0] >= 0) & (projected[:, 0] <= dw - 1) \
            & (projected[:, 1] >= 0) & (projected[:, 1] <= dh - 1)
    except Exception:
        for i in range(pix.shape[0]):
            ray = source_calib.unproject(pix[i])
            if ray is None:
                continue
            p = destination_calib.project(np.asarray(ray, dtype=np.float64))
            if p is None or not np.isfinite(p).all():
                continue
            inside[i] = 0 <= p[0] <= dw - 1 and 0 <= p[1] <= dh - 1
    return inside.reshape(sh, sw)


def edge_losses(valid: np.ndarray, retained: np.ndarray) -> Dict[str, int]:
    """How far the retained region falls short of the valid region on each side.

    Measured as the difference in extent along the middle row and column, which is
    the number a reader can check against the overlay image by eye.
    """
    h, w = valid.shape
    mid_row, mid_col = h // 2, w // 2

    def extent(mask_row: np.ndarray) -> Tuple[int, int]:
        idx = np.flatnonzero(mask_row)
        return (int(idx[0]), int(idx[-1])) if idx.size else (0, -1)

    vl, vr = extent(valid[mid_row])
    rl, rr = extent(retained[mid_row])
    vt, vb = extent(valid[:, mid_col])
    rt, rb = extent(retained[:, mid_col])
    return {
        "pixels_lost_left": max(0, rl - vl),
        "pixels_lost_right": max(0, vr - rr),
        "pixels_lost_top": max(0, rt - vt),
        "pixels_lost_bottom": max(0, vb - rb),
        "valid_extent_horizontal": [vl, vr],
        "retained_extent_horizontal": [rl, rr],
        "valid_extent_vertical": [vt, vb],
        "retained_extent_vertical": [rt, rb],
    }


def angular_coverage(calib, width: int, height: int) -> Dict[str, Any]:
    """Field of view the model covers along the image axes, in degrees."""
    cx, cy = width // 2, height // 2
    probes = {
        "left": (0, cy), "right": (width - 1, cy),
        "top": (cx, 0), "bottom": (cx, height - 1),
        "corner_top_left": (0, 0), "corner_bottom_right": (width - 1, height - 1),
    }
    out: Dict[str, Any] = {}
    for name, (u, v) in probes.items():
        try:
            ray = calib.unproject(np.array([float(u), float(v)]))
        except Exception:
            ray = None
        if ray is None:
            out[name + "_deg"] = None
            continue
        r = np.asarray(ray, dtype=np.float64)
        n = np.linalg.norm(r)
        out[name + "_deg"] = (float(np.degrees(np.arccos(np.clip(r[2] / n, -1, 1))))
                              if n > 0 else None)
    horizontal = [out.get("left_deg"), out.get("right_deg")]
    vertical = [out.get("top_deg"), out.get("bottom_deg")]
    out["horizontal_fov_deg"] = (sum(x for x in horizontal if x is not None)
                                 if all(x is not None for x in horizontal) else None)
    out["vertical_fov_deg"] = (sum(x for x in vertical if x is not None)
                               if all(x is not None for x in vertical) else None)
    return out


def summarise_stage(stage: str, description: str, transform: str,
                    inverse_transform: str, size: Tuple[int, int],
                    source_size: Tuple[int, int],
                    valid: Optional[np.ndarray] = None,
                    retained: Optional[np.ndarray] = None,
                    crop_box: Optional[List[int]] = None,
                    padding: Optional[List[int]] = None) -> StageGeometry:
    w, h = size
    sw, sh = source_size
    geometry = StageGeometry(
        stage=stage, description=description, width=w, height=h,
        aspect_ratio=aspect(w, h), transform=transform,
        inverse_transform=inverse_transform, crop_box=crop_box, padding=padding,
        scale_x=w / sw, scale_y=h / sh,
        aspect_preserved=abs(aspect(w, h) - aspect(sw, sh)) < 1e-6,
    )
    if valid is not None and retained is not None:
        total = valid.size
        n_valid = int(np.count_nonzero(valid))
        geometry.source_frame_retained_fraction = float(
            np.count_nonzero(retained) / total)
        geometry.valid_region_retained_fraction = float(
            np.count_nonzero(retained & valid) / n_valid) if n_valid else None
        geometry.__dict__.update({k: v for k, v in edge_losses(valid, retained).items()
                                  if k.startswith("pixels_lost")})
    return geometry


def detect_centre_crop(source_size: Tuple[int, int],
                       destination_size: Tuple[int, int],
                       crop_box: Optional[List[int]]) -> Dict[str, Any]:
    """Classify whether a stage performs a centre crop or an aspect change."""
    sw, sh = source_size
    dw, dh = destination_size
    source_aspect, destination_aspect = aspect(sw, sh), aspect(dw, dh)
    changed = abs(source_aspect - destination_aspect) > 1e-6
    result = {
        "source_aspect": source_aspect,
        "destination_aspect": destination_aspect,
        "aspect_changed": changed,
        "is_centre_crop": False,
        "widescreen_conversion": bool(changed and destination_aspect > source_aspect),
    }
    if crop_box:
        x0, y0, x1, y1 = crop_box
        centred = (abs(x0 - (sw - (x1 - x0)) / 2) <= 1
                   and abs(y0 - (sh - (y1 - y0)) / 2) <= 1)
        result["is_centre_crop"] = bool(centred and (x1 - x0 < sw or y1 - y0 < sh))
    return result
