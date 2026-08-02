"""Keeping a commute's endpoints out of the committed record.

A GPS track of someone's regular journey is personal data. Its two ends are the
most sensitive part: they are where the participant lives and where they were
going. This module is the single place that decides what a figure or a table may
show, so the rule is applied once rather than remembered at twenty call sites.

The rule: committed artefacts carry **relative** coordinates — metres from the
track's own origin — and both ends of the track are trimmed. Absolute latitude and
longitude stay in the local output tree, which is not committed.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import numpy as np


@dataclass
class RedactionResult:
    x_m: np.ndarray
    y_m: np.ndarray
    keep: np.ndarray
    trimmed_start: int
    trimmed_end: int
    redaction_m: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "points_total": int(self.keep.size),
            "points_published": int(self.keep.sum()),
            "trimmed_start_points": self.trimmed_start,
            "trimmed_end_points": self.trimmed_end,
            "endpoint_redaction_m": self.redaction_m,
            "coordinates": "relative_metres_from_track_origin",
            "absolute_coordinates_present": False,
        }


def redact_track(x_m, y_m, endpoint_redaction_m: float = 250.0
                 ) -> RedactionResult:
    """Re-origin a track and drop the first and last stretch of it.

    Re-origining alone is not enough: a track shape plus any single landmark
    identifies the whole route, and the endpoints are what matter most. Both are
    trimmed by distance travelled, not by point count, so the guarantee is metric
    and does not depend on the sampling rate.
    """
    x = np.asarray(x_m, float)
    y = np.asarray(y_m, float)
    n = x.size
    if n == 0:
        return RedactionResult(x, y, np.zeros(0, bool), 0, 0, endpoint_redaction_m)

    step = np.concatenate([[0.0], np.hypot(np.diff(x), np.diff(y))])
    travelled = np.cumsum(step)
    total = float(travelled[-1])

    r = float(endpoint_redaction_m)
    if total <= 2 * r:
        # A track shorter than the redaction it would need cannot be published
        # with its endpoints hidden, so none of it is published.
        keep = np.zeros(n, bool)
        return RedactionResult(x - x[0], y - y[0], keep, n, 0, r)

    keep = (travelled >= r) & (travelled <= total - r)
    idx = np.flatnonzero(keep)
    ox, oy = x[idx[0]], y[idx[0]]
    return RedactionResult(
        x_m=x - ox, y_m=y - oy, keep=keep,
        trimmed_start=int(np.sum(travelled < r)),
        trimmed_end=int(np.sum(travelled > total - r)),
        redaction_m=r)


def assert_no_absolute_coordinates(frame, context: str = "") -> None:
    """Raise if a table about to be committed still carries lat/lon.

    Named columns are checked rather than values: a column called `latitude` is a
    coordinate whatever it happens to contain, and catching it by name catches it
    before the file is written rather than after it is pushed.
    """
    banned = {"latitude", "longitude", "lat", "lon", "lng",
              "matched_lat", "matched_lon"}
    present = sorted(banned & {str(c).lower() for c in frame.columns})
    if present:
        raise ValueError(
            f"{context or 'this table'} carries absolute coordinates {present}; "
            "committed artefacts must use relative metres. Write it under the "
            "local output tree instead, or redact it first.")


def public_bounds(x_m: np.ndarray, y_m: np.ndarray) -> Dict[str, float]:
    """Extent of a redacted track, for axis limits. Relative metres only."""
    return {"x_min_m": float(np.nanmin(x_m)), "x_max_m": float(np.nanmax(x_m)),
            "y_min_m": float(np.nanmin(y_m)), "y_max_m": float(np.nanmax(y_m))}
