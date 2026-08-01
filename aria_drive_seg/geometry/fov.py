"""Field-of-view preserving geometry for the Article 1 pipeline.

The audit showed that rectifying this fisheye onto a pinhole at the same focal
length discards 43% of the valid field of view, including the region where a
motorcycle's bar-end mirrors sit. This module provides the alternative: keep the
frame in its **source geometry**, carry an explicit valid-pixel mask, and give the
models a letterboxed view so that nothing is cropped and the aspect ratio is never
deformed.

Design rules:

* no centre crop and no aspect change anywhere;
* when a model needs a fixed input size, pad rather than crop, and record the
  padding so the inverse is exact;
* a pixel outside the camera model's angular limit is marked invalid rather than
  quietly treated as scene content;
* the transform is reversible to sub-pixel accuracy, and the round trip is tested
  rather than asserted.

Nothing here depends on the vehicle domain. The same transform runs on the car and
on the motorcycle; only the calibration differs.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, Optional, Tuple

import numpy as np


@dataclass(frozen=True)
class LetterboxTransform:
    """A resize-and-pad that preserves the aspect ratio exactly.

    `scale` is applied to both axes, so the aspect ratio cannot change. Padding is
    split as evenly as the parity allows and recorded in full, which is what makes
    the inverse exact instead of approximate.
    """

    source_width: int
    source_height: int
    target_width: int
    target_height: int
    scale: float
    pad_left: int
    pad_top: int
    pad_right: int
    pad_bottom: int

    @property
    def content_width(self) -> int:
        return self.target_width - self.pad_left - self.pad_right

    @property
    def content_height(self) -> int:
        return self.target_height - self.pad_top - self.pad_bottom

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["content_size"] = [self.content_width, self.content_height]
        d["aspect_preserved"] = True
        d["crop"] = False
        return d

    # ---- coordinate mapping ------------------------------------------- #
    def forward_points(self, points: np.ndarray) -> np.ndarray:
        """Source pixel coordinates -> letterboxed coordinates."""
        p = np.asarray(points, dtype=np.float64)
        return np.stack([p[..., 0] * self.scale + self.pad_left,
                         p[..., 1] * self.scale + self.pad_top], axis=-1)

    def inverse_points(self, points: np.ndarray) -> np.ndarray:
        """Letterboxed coordinates -> source pixel coordinates."""
        p = np.asarray(points, dtype=np.float64)
        return np.stack([(p[..., 0] - self.pad_left) / self.scale,
                         (p[..., 1] - self.pad_top) / self.scale], axis=-1)

    # ---- image mapping ------------------------------------------------- #
    def apply(self, image: np.ndarray, pad_value: int = 0,
              nearest: bool = False) -> np.ndarray:
        """Resize and pad. Label maps must pass `nearest=True`.

        The channel layout of the input is preserved: OpenCV silently drops a
        singleton channel, which would change a (H, W, 1) label map into (H, W)
        and break the caller's indexing.
        """
        import cv2
        interpolation = (cv2.INTER_NEAREST if nearest else
                         (cv2.INTER_AREA if self.scale < 1 else cv2.INTER_LINEAR))
        resized = cv2.resize(image, (self.content_width, self.content_height),
                             interpolation=interpolation)
        channels = image.shape[2] if image.ndim == 3 else 1
        padded = cv2.copyMakeBorder(
            resized, self.pad_top, self.pad_bottom, self.pad_left, self.pad_right,
            cv2.BORDER_CONSTANT, value=[pad_value] * channels)
        if image.ndim == 3 and padded.ndim == 2:
            padded = padded[:, :, None]
        return padded

    def invert_mask(self, mask: np.ndarray) -> np.ndarray:
        """Letterboxed label mask -> source geometry, nearest neighbour."""
        import cv2
        cropped = mask[self.pad_top:self.pad_top + self.content_height,
                       self.pad_left:self.pad_left + self.content_width]
        return cv2.resize(cropped, (self.source_width, self.source_height),
                          interpolation=cv2.INTER_NEAREST)

    def invert_map(self, values: np.ndarray) -> np.ndarray:
        """Letterboxed continuous map -> source geometry, bilinear."""
        import cv2
        cropped = values[self.pad_top:self.pad_top + self.content_height,
                         self.pad_left:self.pad_left + self.content_width]
        return cv2.resize(cropped, (self.source_width, self.source_height),
                          interpolation=cv2.INTER_LINEAR)


def build_letterbox(source_size: Tuple[int, int],
                    target_size: Tuple[int, int]) -> LetterboxTransform:
    """Fit the source into the target without cropping or deforming it."""
    sw, sh = source_size
    tw, th = target_size
    if sw <= 0 or sh <= 0 or tw <= 0 or th <= 0:
        raise ValueError("sizes must be positive")
    scale = min(tw / sw, th / sh)
    content_w = int(round(sw * scale))
    content_h = int(round(sh * scale))
    # rounding can overshoot by a pixel; clamp so the content always fits
    content_w = min(content_w, tw)
    content_h = min(content_h, th)
    pad_x = tw - content_w
    pad_y = th - content_h
    return LetterboxTransform(
        source_width=sw, source_height=sh,
        target_width=tw, target_height=th, scale=scale,
        pad_left=pad_x // 2, pad_top=pad_y // 2,
        pad_right=pad_x - pad_x // 2, pad_bottom=pad_y - pad_y // 2)


# --------------------------------------------------------------------------- #
# Valid-pixel mask
# --------------------------------------------------------------------------- #
def valid_pixel_mask(calib, width: int, height: int) -> np.ndarray:
    """Pixels inside the camera model's angular limit.

    Outside this region the optics carry no scene information. Those pixels are
    still segmented (the fusion contract is dense), but they are excluded from
    every scientific statistic through this mask, and they are never counted as
    evidence for a class.
    """
    us, vs = np.meshgrid(np.arange(width), np.arange(height))
    pix = np.stack([us.ravel(), vs.ravel()], axis=1).astype(np.float64)
    valid = np.zeros(pix.shape[0], dtype=bool)
    try:
        rays = np.asarray(calib.unproject(pix), dtype=np.float64)
        valid = np.isfinite(rays).all(axis=1)
    except Exception:
        for i in range(pix.shape[0]):
            ray = calib.unproject(pix[i])
            valid[i] = ray is not None and np.isfinite(np.asarray(ray)).all()
    return valid.reshape(height, width)


def vignette_mask(image: np.ndarray, luminance_threshold: int = 12) -> np.ndarray:
    """Optically black pixels, as a fallback when no calibration is available."""
    import cv2
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) if image.ndim == 3 else image
    return gray > luminance_threshold


# --------------------------------------------------------------------------- #
# Round-trip verification
# --------------------------------------------------------------------------- #
def probe_points(width: int, height: int,
                 valid: Optional[np.ndarray] = None) -> Dict[str, np.ndarray]:
    """Named probe points for round-trip checks.

    Deliberately includes the places where a transform is most likely to be wrong:
    the exact centre, the four edge midpoints, the corners of the valid region, the
    lateral bands where motorcycle mirrors sit, and the bottom strip.
    """
    cx, cy = (width - 1) / 2.0, (height - 1) / 2.0
    points = {
        "centre": np.array([[cx, cy]]),
        "edge_left": np.array([[0.0, cy]]),
        "edge_right": np.array([[width - 1.0, cy]]),
        "edge_top": np.array([[cx, 0.0]]),
        "edge_bottom": np.array([[cx, height - 1.0]]),
        "mirror_band_left": np.array([[width * 0.08, height * 0.72],
                                      [width * 0.14, height * 0.80]]),
        "mirror_band_right": np.array([[width * 0.92, height * 0.72],
                                       [width * 0.86, height * 0.80]]),
        "bottom_strip": np.array([[cx, height * 0.92],
                                  [width * 0.30, height * 0.95]]),
    }
    if valid is not None:
        ys, xs = np.nonzero(valid)
        if xs.size:
            points["valid_corner_top_left"] = np.array(
                [[float(xs[ys == ys.min()].min()), float(ys.min())]])
            points["valid_corner_bottom_right"] = np.array(
                [[float(xs[ys == ys.max()].max()), float(ys.max())]])
            points["valid_extreme_left"] = np.array(
                [[float(xs.min()), float(ys[xs == xs.min()].mean())]])
            points["valid_extreme_right"] = np.array(
                [[float(xs.max()), float(ys[xs == xs.max()].mean())]])
    return points


def round_trip_error(transform: LetterboxTransform,
                     points: np.ndarray) -> np.ndarray:
    """Euclidean error of source -> letterbox -> source, in source pixels."""
    p = np.asarray(points, dtype=np.float64)
    back = transform.inverse_points(transform.forward_points(p))
    return np.linalg.norm(back - p, axis=-1)


def round_trip_report(transform: LetterboxTransform, width: int, height: int,
                      valid: Optional[np.ndarray] = None) -> Dict[str, Any]:
    """Round-trip error at the probe points plus a dense sample."""
    named = {}
    all_errors = []
    for name, pts in probe_points(width, height, valid).items():
        err = round_trip_error(transform, pts)
        named[name] = {"max_error_px": float(err.max()),
                       "points": [[float(a), float(b)] for a, b in pts]}
        all_errors.append(err)

    grid_x, grid_y = np.meshgrid(np.linspace(0, width - 1, 64),
                                 np.linspace(0, height - 1, 64))
    dense = round_trip_error(
        transform, np.stack([grid_x.ravel(), grid_y.ravel()], axis=-1))
    all_errors.append(dense)
    errors = np.concatenate([e.ravel() for e in all_errors])

    return {
        "transform": transform.to_dict(),
        "linear": True,
        "median_error_px": float(np.median(errors)),
        "p99_error_px": float(np.percentile(errors, 99)),
        "max_error_px": float(errors.max()),
        "probe_points": named,
        "thresholds": {"median_px": 0.5, "p99_px": 1.0},
        "passes": bool(np.median(errors) < 0.5 and np.percentile(errors, 99) < 1.0),
    }
