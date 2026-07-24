"""Assign each frame's gaze to semantic classes for one method (§10):
  1. class at the exact gaze pixel
  2. dominant class in a configurable disc
  3. class distribution under a 2D gaussian (sigma_px, optionally from visual degrees)
  4. confidence of the observed class
  5. gaze distance to the class-region boundary
  6. unknown / unsupported flags
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from ..config import Config
from ..logging_utils import get_logger
from ..taxonomy import Taxonomy

log = get_logger("gaze_labels")


def _gaussian_distribution(mask: np.ndarray, u: float, v: float, sigma: float,
                           max_id: int) -> Dict[int, float]:
    h, w = mask.shape[:2]
    r = int(np.ceil(3 * sigma))
    x0, x1 = max(0, int(u) - r), min(w, int(u) + r + 1)
    y0, y1 = max(0, int(v) - r), min(h, int(v) + r + 1)
    if x0 >= x1 or y0 >= y1:
        return {}
    ys, xs = np.mgrid[y0:y1, x0:x1]
    g = np.exp(-(((xs - u) ** 2 + (ys - v) ** 2) / (2 * sigma * sigma)))
    patch = mask[y0:y1, x0:x1]
    total = g.sum()
    if total <= 0:
        return {}
    dist: Dict[int, float] = {}
    for cid in np.unique(patch):
        dist[int(cid)] = float(g[patch == cid].sum() / total)
    return dist


def _dist_to_boundary(mask: np.ndarray, u: int, v: int, cid: int) -> float:
    import cv2
    region = (mask == cid).astype(np.uint8)
    if region[v, u] == 0:
        return 0.0
    dt = cv2.distanceTransform(region, cv2.DIST_L2, 3)
    return float(dt[v, u])


def compute_gaze_labels(input_dir: str, method: str, cfg: Config,
                        tax: Taxonomy, unsupported_ids: Optional[set] = None) -> Path:
    import pandas as pd
    from ..io_utils import read_mask_u16
    from ..segmentation.base import SegLayout

    out = Path(input_dir)
    layout = SegLayout(input_dir, method)
    gz = pd.read_parquet(out / "gaze" / "aligned_gaze.parquet").set_index("frame_index")

    disc_r = int(cfg.get("gaze.disc_radius_px", 40))
    sigma = float(cfg.get("gaze.gauss_sigma_px", 60.0))
    sigma_deg = cfg.get("gaze.gauss_sigma_deg")
    if sigma_deg:
        focal = float(cfg.get("rectify.focal", 879.0))
        sigma = float(np.tan(np.deg2rad(sigma_deg)) * focal)

    rows: List[Dict[str, Any]] = []
    for mp in sorted(layout.metadata.glob("frame_*.json")):
        fi = int(mp.stem.split("_")[1])
        if fi not in gz.index:
            continue
        g = gz.loc[fi]
        rec: Dict[str, Any] = {
            "frame_index": fi, "capture_timestamp_ns": int(g["capture_timestamp_ns"]),
            "gaze_valid": bool(g["valid"]), "rect_u": g.get("rect_u"), "rect_v": g.get("rect_v"),
            "method": method,
        }
        cpath = layout.canonical_path(fi)
        if not cpath.exists() or not bool(g["valid"]) or g.get("rect_u") is None or not np.isfinite(g.get("rect_u")):
            rec.update(class_at_pixel="unknown", class_at_pixel_id=0, dominant_disc="unknown",
                       gauss_top="unknown", gauss_top_w=None, gauss_unknown_w=None,
                       confidence=None, dist_boundary_px=None, is_unknown=True, is_unsupported=False)
            rows.append(rec)
            continue
        mask = read_mask_u16(cpath)
        h, w = mask.shape[:2]
        u, v = float(g["rect_u"]), float(g["rect_v"])
        ui, vi = int(round(u)), int(round(v))
        ui, vi = min(max(ui, 0), w - 1), min(max(vi, 0), h - 1)

        cid = int(mask[vi, ui])
        # disc dominant
        y0, y1 = max(0, vi - disc_r), min(h, vi + disc_r + 1)
        x0, x1 = max(0, ui - disc_r), min(w, ui + disc_r + 1)
        patch = mask[y0:y1, x0:x1].ravel()
        vals, counts = np.unique(patch, return_counts=True)
        dom = int(vals[int(np.argmax(counts))])
        # gaussian distribution
        dist = _gaussian_distribution(mask, u, v, sigma, tax.max_id)
        gauss_top = max(dist, key=dist.get) if dist else 0
        # confidence
        conf = None
        cpath_conf = layout.confidence_path(fi)
        if cpath_conf.exists():
            import cv2
            cimg = cv2.imread(str(cpath_conf), cv2.IMREAD_UNCHANGED)
            if cimg is not None:
                conf = float(cimg[vi, ui]) / 255.0
        db = _dist_to_boundary(mask, ui, vi, cid)
        unsup = bool(unsupported_ids and cid in unsupported_ids)
        rec.update(
            class_at_pixel=tax.name_of(cid), class_at_pixel_id=cid,
            dominant_disc=tax.name_of(dom), dominant_disc_id=dom,
            gauss_top=tax.name_of(int(gauss_top)), gauss_top_w=dist.get(int(gauss_top)),
            gauss_unknown_w=dist.get(0, 0.0),
            confidence=conf, dist_boundary_px=db,
            is_unknown=(cid == 0), is_unsupported=unsup,
        )
        rows.append(rec)

    df = pd.DataFrame(rows).sort_values("frame_index")
    outp = out / "comparison" / "gaze" / f"gaze_labels_{method}.parquet"
    outp.parent.mkdir(parents=True, exist_ok=True)
    tmp = outp.with_suffix(".tmp.parquet")
    df.to_parquet(tmp, index=False)
    tmp.replace(outp)
    log.info("wrote %s (%d frames)", outp, len(df))
    return outp
