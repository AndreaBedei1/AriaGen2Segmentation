"""Pixel-level overlap resolution for prompted masks (§7).

Rules encoded via a (priority, score) key per pixel — HIGHER wins:
  * high-priority thin/safety-critical classes (lane_marking, crosswalk, person,
    rider, bicycle, motorcycle, traffic_light, traffic_sign) beat large background
    classes (road_surface, sidewalk, building, sky, vegetation) regardless of area;
  * at EQUAL priority, the higher combined score wins (a low-confidence mask never
    overwrites a higher-confidence one);
  * unassigned pixels stay `unknown` (id 0) — never invented from a residual.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np


@dataclass
class Instance:
    mask: np.ndarray      # bool (H, W)
    canonical_id: int
    score: float          # combined score in [0, 1]
    priority: int         # higher wins a pixel conflict


def resolve_overlaps(height: int, width: int,
                     instances: List[Instance]) -> Tuple[np.ndarray, np.ndarray]:
    """Return (id_map uint16, score_map float32). id 0 == unknown."""
    id_map = np.zeros((height, width), dtype=np.uint16)
    prio_map = np.full((height, width), -1, dtype=np.int32)
    score_map = np.zeros((height, width), dtype=np.float32)

    # Deterministic order (does not affect result, but keeps runs reproducible):
    # paint low->high priority, low->high score.
    order = sorted(range(len(instances)),
                   key=lambda k: (instances[k].priority, instances[k].score))
    for k in order:
        inst = instances[k]
        m = inst.mask
        if m.dtype != np.bool_:
            m = m.astype(bool)
        better = m & (
            (inst.priority > prio_map)
            | ((inst.priority == prio_map) & (np.float32(inst.score) > score_map))
        )
        if not better.any():
            continue
        id_map[better] = np.uint16(inst.canonical_id)
        prio_map[better] = np.int32(inst.priority)
        score_map[better] = np.float32(inst.score)
    return id_map, score_map


def class_aware_nms(boxes: np.ndarray, scores: np.ndarray, class_ids: np.ndarray,
                    iou_thr: float = 0.7, iou_by_class: "dict | None" = None) -> List[int]:
    """Greedy per-class NMS. boxes: (N,4) x0y0x1y1. Returns kept indices.

    iou_by_class optionally overrides the IoU threshold per class id (falls back to
    iou_thr)."""
    keep: List[int] = []
    for cid in np.unique(class_ids):
        thr = iou_thr
        if iou_by_class and int(cid) in iou_by_class and iou_by_class[int(cid)] is not None:
            thr = float(iou_by_class[int(cid)])
        idx = np.where(class_ids == cid)[0]
        idx = idx[np.argsort(-scores[idx])]
        b = boxes[idx]
        suppressed = np.zeros(len(idx), dtype=bool)
        for i in range(len(idx)):
            if suppressed[i]:
                continue
            keep.append(int(idx[i]))
            for j in range(i + 1, len(idx)):
                if suppressed[j]:
                    continue
                if _iou(b[i], b[j]) > thr:
                    suppressed[j] = True
    return keep


def mask_iou(a: np.ndarray, b: np.ndarray) -> float:
    a = a.astype(bool); b = b.astype(bool)
    inter = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()
    return float(inter / union) if union else 0.0


def _iou(a: np.ndarray, b: np.ndarray) -> float:
    x0 = max(a[0], b[0]); y0 = max(a[1], b[1])
    x1 = min(a[2], b[2]); y1 = min(a[3], b[3])
    iw = max(0.0, x1 - x0); ih = max(0.0, y1 - y0)
    inter = iw * ih
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    union = area_a + area_b - inter
    return float(inter / union) if union > 0 else 0.0
