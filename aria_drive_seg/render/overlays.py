"""Overlay primitives (§11). Overlays are ALWAYS drawn on a COPY — the frame fed
to inference is never mutated. Palette is the shared deterministic taxonomy palette
so both methods colour identical classes identically."""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from ..taxonomy import Taxonomy


def colorize(mask: np.ndarray, tax: Taxonomy) -> np.ndarray:
    return tax.colorize(mask)


def blend_mask(image_rgb: np.ndarray, mask: np.ndarray, tax: Taxonomy,
               alpha: float = 0.5) -> np.ndarray:
    """Alpha-blend the coloured mask over assigned pixels only (unknown stays raw)."""
    out = image_rgb.copy()
    color = tax.colorize(mask)
    assigned = mask > 0
    out[assigned] = (alpha * color[assigned] + (1 - alpha) * out[assigned]).astype(np.uint8)
    return out


def draw_gaze(img: np.ndarray, u: Optional[float], v: Optional[float],
              valid: bool, radius: int = 22) -> np.ndarray:
    import cv2
    if u is None or v is None or not np.isfinite(u) or not np.isfinite(v):
        return img
    u, v = int(round(u)), int(round(v))
    color = (0, 255, 0) if valid else (0, 0, 255)  # RGB: green valid / red invalid
    cv2.circle(img, (u, v), radius, color, 3, lineType=cv2.LINE_AA)
    cv2.circle(img, (u, v), 2, color, -1, lineType=cv2.LINE_AA)
    cv2.line(img, (u - radius - 8, v), (u - radius + 4, v), color, 2, cv2.LINE_AA)
    cv2.line(img, (u + radius - 4, v), (u + radius + 8, v), color, 2, cv2.LINE_AA)
    cv2.line(img, (u, v - radius - 8), (u, v - radius + 4), color, 2, cv2.LINE_AA)
    cv2.line(img, (u, v + radius - 4), (u, v + radius + 8), color, 2, cv2.LINE_AA)
    return img


def class_at_gaze(mask: np.ndarray, u: Optional[float], v: Optional[float],
                  tax: Taxonomy, radius: int = 0) -> Tuple[int, str]:
    if u is None or v is None or not np.isfinite(u) or not np.isfinite(v):
        return 0, "unknown"
    h, w = mask.shape[:2]
    ui, vi = int(round(u)), int(round(v))
    if not (0 <= ui < w and 0 <= vi < h):
        return 0, "unknown"
    if radius <= 0:
        cid = int(mask[vi, ui])
    else:
        y0, y1 = max(0, vi - radius), min(h, vi + radius + 1)
        x0, x1 = max(0, ui - radius), min(w, ui + radius + 1)
        patch = mask[y0:y1, x0:x1].ravel()
        vals, counts = np.unique(patch, return_counts=True)
        cid = int(vals[int(np.argmax(counts))])
    return cid, tax.name_of(cid)


def draw_hud(img: np.ndarray, lines: List[str], org=(12, 8),
             font_scale: float = 0.6, bg: bool = True) -> np.ndarray:
    import cv2
    x, y = org
    fh = int(28 * font_scale / 0.6)
    pad = 6
    if bg:
        h = fh * len(lines) + pad * 2
        w = max((len(s) for s in lines), default=1)
        w = int(w * 11 * font_scale / 0.6) + pad * 2
        sub = img[y:y + h, x:x + w]
        if sub.size:
            img[y:y + h, x:x + w] = (0.45 * sub).astype(np.uint8)
    yy = y + pad + fh - 6
    for s in lines:
        cv2.putText(img, s, (x + pad, yy), cv2.FONT_HERSHEY_SIMPLEX, font_scale,
                    (255, 255, 255), 1, cv2.LINE_AA)
        yy += fh
    return img


def draw_legend(img: np.ndarray, class_ids: List[int], tax: Taxonomy,
                font_scale: float = 0.5) -> np.ndarray:
    import cv2
    if not class_ids:
        return img
    ids = [c for c in class_ids if c != 0]
    rowh = 22
    sw = 18
    box_w = 210
    box_h = rowh * len(ids) + 10
    H, W = img.shape[:2]
    x0 = W - box_w - 10
    y0 = 10
    sub = img[y0:y0 + box_h, x0:x0 + box_w]
    if sub.size:
        img[y0:y0 + box_h, x0:x0 + box_w] = (0.45 * sub).astype(np.uint8)
    y = y0 + 6
    lut = tax.palette()
    for cid in ids:
        col = tuple(int(c) for c in lut[cid])
        cv2.rectangle(img, (x0 + 6, y + 2), (x0 + 6 + sw, y + 2 + sw - 4), col, -1)
        cv2.putText(img, tax.name_of(cid), (x0 + 6 + sw + 8, y + sw - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), 1, cv2.LINE_AA)
        y += rowh
    return img


def hstack_pad(imgs: List[np.ndarray], gap: int = 8) -> np.ndarray:
    h = max(i.shape[0] for i in imgs)
    pieces = []
    for k, im in enumerate(imgs):
        if im.shape[0] != h:
            pad = np.zeros((h - im.shape[0], im.shape[1], 3), np.uint8)
            im = np.vstack([im, pad])
        pieces.append(im)
        if k < len(imgs) - 1:
            pieces.append(np.zeros((h, gap, 3), np.uint8))
    return np.hstack(pieces)


def ts_seconds(capture_ts_ns: int, t0_ns: int) -> float:
    return (capture_ts_ns - t0_ns) / 1e9
