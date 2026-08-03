"""Overlay drawing for the multimodal analysis videos.

The scene has to stay readable: the point of these videos is to let a human check
what the pipeline claims, and an overlay that covers the road defeats that. So the
semantic layer is drawn at low opacity, the telemetry lives in a band along the
bottom rather than over the image, and the gaze marker is a ring rather than a
filled disc.

Every panel states what it is. A number with no source is worse than no number:
a reviewer watching this must be able to tell head motion from vehicle motion,
and a proxy class from a reliable one, without leaving the video.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

#: Classes flagged in the overlay as exploratory proxies.
PROXY_CLASSES = ("mirror", "instrument_display", "control_and_ego_vehicle")

PANEL_H = 190
FONT = 0


def _put(img, text, org, scale=0.55, colour=(255, 255, 255), thick=1) -> None:
    import cv2
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0),
                thick + 2, cv2.LINE_AA)
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, colour,
                thick, cv2.LINE_AA)


def semantic_overlay(rgb: np.ndarray, mask: Optional[np.ndarray],
                     colours: Dict[int, Sequence[int]],
                     alpha: float = 0.28) -> np.ndarray:
    """Blend the class map over the frame, lightly."""
    import cv2
    out = rgb.copy()
    if mask is None:
        return out
    if mask.shape[:2] != rgb.shape[:2]:
        mask = cv2.resize(mask, (rgb.shape[1], rgb.shape[0]),
                          interpolation=cv2.INTER_NEAREST)
    layer = np.zeros_like(rgb)
    for cid, bgr in colours.items():
        layer[mask == cid] = bgr
    painted = mask > 0
    out[painted] = cv2.addWeighted(rgb, 1.0 - alpha, layer, alpha, 0)[painted]
    return out


def draw_gaze(img: np.ndarray, u: Optional[float], v: Optional[float],
              radius_px: int, is_fixation: bool, valid: bool) -> None:
    """A ring at the foveal window, open so the scene under it stays visible."""
    import cv2
    if u is None or v is None or not np.isfinite(u) or not np.isfinite(v):
        return
    p = (int(round(u)), int(round(v)))
    colour = (0, 255, 255) if valid else (120, 120, 120)
    cv2.circle(img, p, radius_px, colour, 3 if is_fixation else 1, cv2.LINE_AA)
    cv2.drawMarker(img, p, colour, cv2.MARKER_CROSS, 18, 2, cv2.LINE_AA)


def telemetry_panel(width: int, fields: List[Tuple[str, str, str]],
                    title: str, height: int = PANEL_H) -> np.ndarray:
    """A caption band under the frame.

    `fields` are (label, value, provenance) triples. The provenance column is not
    decoration: it is what stops a viewer reading head motion as vehicle motion,
    or a cockpit proxy as a measured class.
    """
    import cv2
    panel = np.full((height, width, 3), 24, np.uint8)
    _put(panel, title, (14, 26), 0.62, (230, 230, 230), 1)
    cv2.line(panel, (10, 36), (width - 10, 36), (70, 70, 70), 1)

    cols = 3
    per_col = int(np.ceil(len(fields) / cols)) or 1
    col_w = (width - 28) // cols
    for i, (label, value, prov) in enumerate(fields):
        c, r = divmod(i, per_col)
        x = 14 + c * col_w
        y = 62 + r * 30
        if y > height - 12:
            continue
        _put(panel, f"{label}", (x, y), 0.46, (150, 150, 150), 1)
        _put(panel, f"{value}", (x + 155, y), 0.52, (255, 255, 255), 1)
        if prov:
            _put(panel, prov, (x + 155, y + 13), 0.34, (140, 170, 210), 1)
    return panel


def compose(frame: np.ndarray, panel: np.ndarray) -> np.ndarray:
    return np.vstack([frame, panel])


def side_by_side(left: np.ndarray, right: np.ndarray,
                 left_label: str, right_label: str) -> np.ndarray:
    """Two frames with a divider, for the position-matched comparison."""
    import cv2
    h = min(left.shape[0], right.shape[0])
    l = cv2.resize(left, (int(left.shape[1] * h / left.shape[0]), h))
    r = cv2.resize(right, (int(right.shape[1] * h / right.shape[0]), h))
    _put(l, left_label, (18, 34), 0.8, (255, 255, 255), 2)
    _put(r, right_label, (18, 34), 0.8, (255, 255, 255), 2)
    divider = np.full((h, 6, 3), 200, np.uint8)
    return np.hstack([l, divider, r])


class VideoWriter:
    """ffmpeg-backed writer; falls back to OpenCV if imageio is unavailable."""

    def __init__(self, path, fps: float, size: Tuple[int, int]):
        self.path = str(path)
        self.fps = float(fps)
        self.size = size
        self._w = None
        self._backend = None

    def __enter__(self):
        try:
            import imageio_ffmpeg
            self._w = imageio_ffmpeg.write_frames(
                self.path, self.size, fps=self.fps, quality=6,
                # H.264/yuv420p only requires even dimensions.  Using 2 keeps
                # the calibrated 1008x756 semantic-camera geometry unchanged.
                macro_block_size=2)
            self._w.send(None)
            self._backend = "imageio_ffmpeg"
        except Exception:
            import cv2
            self._w = cv2.VideoWriter(self.path,
                                      cv2.VideoWriter_fourcc(*"mp4v"),
                                      self.fps, self.size)
            self._backend = "opencv"
        return self

    def write(self, bgr: np.ndarray) -> None:
        import cv2
        if bgr.shape[1] != self.size[0] or bgr.shape[0] != self.size[1]:
            bgr = cv2.resize(bgr, self.size)
        if self._backend == "imageio_ffmpeg":
            self._w.send(np.ascontiguousarray(bgr[:, :, ::-1]))
        else:
            self._w.write(bgr)

    def __exit__(self, *exc) -> None:
        if self._backend == "imageio_ffmpeg":
            self._w.close()
        else:
            self._w.release()
