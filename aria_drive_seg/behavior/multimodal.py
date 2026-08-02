"""The timestamp-aligned multimodal master table.

One row per **real RGB frame**. Every other stream is attached by its own capture
timestamp, and every attachment carries three things: which real sample was used,
how far away in time it was, and whether that distance was small enough to call
the association meaningful.

Rules this module exists to enforce:

* no interpolation, no resampling, no synthetic sample. A row always points at a
  sample that was actually recorded;
* the association tolerance is a **duration**, derived from the two streams' own
  measured cadences, so the same configuration behaves correctly against a 10 fps
  and a 15 fps recording;
* an association that fails the tolerance is marked invalid **with a reason** and
  kept in the table, rather than dropped silently or filled;
* derived 1 Hz / 5 Hz views select real rows of this table. They never create a
  new time point that has no real sample behind it.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from ..ingestion.timeline import NS_PER_S, measure_rate, nearest_indices
from .sensors import SensorSeries

#: Absolute ceiling on any association, whatever the cadences say. A sample more
#: than this far from the frame describes a different moment of a drive.
DEFAULT_ABSOLUTE_TOLERANCE_S = 2.0

#: Floor, so a very fast stream does not get an unusably tight tolerance.
DEFAULT_MINIMUM_TOLERANCE_S = 0.020


def association_tolerance_s(reference_median_dt_s: Optional[float],
                            target_median_dt_s: Optional[float],
                            absolute_ceiling_s: float = DEFAULT_ABSOLUTE_TOLERANCE_S,
                            minimum_s: float = DEFAULT_MINIMUM_TOLERANCE_S) -> float:
    """Half a sampling interval of each stream, capped and floored.

    The nearest real sample of a target stream can be at most half of that
    stream's own period away from any instant, plus half of the reference period
    for the reference instant itself. Beyond that the "nearest" sample belongs to
    a different sampling slot and the association is not co-temporal.

    Deriving the tolerance from measured cadences rather than from a constant is
    what makes one configuration correct at both 10 fps and 15 fps.
    """
    ref = float(reference_median_dt_s or 0.0)
    tgt = float(target_median_dt_s or 0.0)
    tol = 0.5 * ref + 0.5 * tgt
    return float(min(max(tol, minimum_s), absolute_ceiling_s))


@dataclass
class StreamAttachment:
    """How one stream was attached to the RGB reference."""

    name: str
    available: bool
    count: int
    measured_hz: Optional[float]
    tolerance_s: float
    valid_fraction: float
    median_abs_dt_ms: Optional[float]
    max_abs_dt_ms: Optional[float]
    exclusion_reason: Optional[str] = None
    meta: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name, "available": self.available, "count": self.count,
            "measured_hz": self.measured_hz, "tolerance_s": self.tolerance_s,
            "valid_fraction": self.valid_fraction,
            "median_abs_dt_ms": self.median_abs_dt_ms,
            "max_abs_dt_ms": self.max_abs_dt_ms,
            "exclusion_reason": self.exclusion_reason,
            "meta": self.meta,
        }


def attach_stream(reference_ns: np.ndarray,
                  target_ns: np.ndarray,
                  name: str,
                  tolerance_s: float) -> Dict[str, np.ndarray]:
    """Attach one stream to the reference timestamps by nearest real sample.

    Returns index / signed dt / validity arrays. An out-of-tolerance row keeps its
    nearest index so a reviewer can still see what was closest, but is flagged
    invalid: the flag, not the index, is what downstream code must gate on.
    """
    ref = np.asarray(reference_ns, dtype=np.int64)
    tgt = np.asarray(target_ns, dtype=np.int64)
    if tgt.size == 0:
        return {
            f"{name}_index": np.full(ref.shape, -1, np.int64),
            f"{name}_timestamp_ns": np.full(ref.shape, -1, np.int64),
            f"{name}_dt_ms": np.full(ref.shape, np.nan, np.float64),
            f"{name}_valid": np.zeros(ref.shape, bool),
            f"{name}_quality": np.zeros(ref.shape, np.float64),
        }
    idx = nearest_indices(ref, tgt)
    dt_ns = tgt[idx] - ref                       # signed: sample - frame
    tol_ns = float(tolerance_s) * NS_PER_S
    valid = np.abs(dt_ns) <= tol_ns
    # Quality falls linearly from 1 at a perfect match to 0 at the tolerance.
    quality = np.clip(1.0 - np.abs(dt_ns) / max(tol_ns, 1.0), 0.0, 1.0)
    return {
        f"{name}_index": idx.astype(np.int64),
        f"{name}_timestamp_ns": tgt[idx].astype(np.int64),
        f"{name}_dt_ms": dt_ns / 1e6,
        f"{name}_valid": valid,
        f"{name}_quality": np.where(valid, quality, 0.0),
    }


@dataclass
class MultimodalTimeline:
    """One row per real RGB frame, with every stream attached by timestamp."""

    recording_id: str
    domain: str
    reference_label: str
    columns: Dict[str, np.ndarray]
    attachments: List[StreamAttachment]
    reference_hz: Optional[float]
    notes: List[str] = field(default_factory=list)

    def __len__(self) -> int:
        return int(self.columns["timestamp_ns"].size)

    @property
    def duration_s(self) -> float:
        ts = self.columns["timestamp_ns"]
        return float((ts[-1] - ts[0]) / NS_PER_S) if ts.size > 1 else 0.0

    def to_frame(self):
        import pandas as pd
        df = pd.DataFrame(self.columns)
        df.insert(0, "domain", self.domain)
        df.insert(0, "recording_id", self.recording_id)
        return df

    def summary(self) -> Dict[str, Any]:
        return {
            "recording_id": self.recording_id,
            "domain": self.domain,
            "reference_label": self.reference_label,
            "reference_hz": self.reference_hz,
            "rows": len(self),
            "duration_s": self.duration_s,
            "interpolated": False,
            "resampled": False,
            "synthetic_samples": 0,
            "attachments": [a.to_dict() for a in self.attachments],
            "notes": self.notes,
        }


def build_multimodal_timeline(recording_id: str,
                              domain: str,
                              reference_ns: Sequence[int] | np.ndarray,
                              streams: Dict[str, SensorSeries],
                              extra_streams: Optional[Dict[str, np.ndarray]] = None,
                              reference_label: str = "camera-rgb",
                              reference_indices: Optional[Sequence[int]] = None,
                              absolute_ceiling_s: float = DEFAULT_ABSOLUTE_TOLERANCE_S,
                              ) -> MultimodalTimeline:
    """Build the master table for one recording.

    `streams` are `SensorSeries` (IMU, PPG, ...); `extra_streams` are bare
    timestamp arrays for streams read elsewhere (gaze, GPS, hand tracking, the
    SLAM cameras), keyed by the column prefix to use.
    """
    ref = np.asarray(reference_ns, dtype=np.int64)
    if ref.size == 0:
        raise ValueError("the RGB reference stream is empty")
    if np.any(np.diff(ref) < 0):
        raise ValueError("the RGB reference timestamps are not monotonic")

    ref_rate = measure_rate(ref)
    ref_median_dt_s = ((ref_rate.median_dt_ms or 0.0) / 1e3) or None

    idx = (np.asarray(reference_indices, dtype=np.int64)
           if reference_indices is not None else np.arange(ref.size, dtype=np.int64))
    columns: Dict[str, np.ndarray] = {
        "frame_index": idx,
        "timestamp_ns": ref,
        "timestamp_s": ref / NS_PER_S,
        "rel_time_s": (ref - ref[0]) / NS_PER_S,
    }

    attachments: List[StreamAttachment] = []
    notes: List[str] = []

    targets: List[tuple] = []
    for name, series in sorted(streams.items()):
        targets.append((_column_name(name), series.timestamp_ns,
                        dict(series.meta)))
    for name, ts in sorted((extra_streams or {}).items()):
        targets.append((_column_name(name), np.asarray(ts, dtype=np.int64), {}))

    for col, ts, meta in targets:
        rate = measure_rate(ts)
        tgt_median_dt_s = ((rate.median_dt_ms or 0.0) / 1e3) or None
        tol = association_tolerance_s(ref_median_dt_s, tgt_median_dt_s,
                                      absolute_ceiling_s=absolute_ceiling_s)
        attached = attach_stream(ref, ts, col, tol)
        columns.update(attached)

        valid = attached[f"{col}_valid"]
        abs_dt = np.abs(attached[f"{col}_dt_ms"])
        finite = abs_dt[np.isfinite(abs_dt)]
        reason = None
        if ts.size == 0:
            reason = "stream not present in this recording"
        elif not valid.any():
            reason = (f"no frame found a {col} sample within {tol * 1e3:.0f} ms; "
                      "the stream does not overlap the RGB span")
        elif valid.mean() < 0.5:
            reason = (f"only {valid.mean() * 100:.1f}% of frames found a {col} "
                      f"sample within {tol * 1e3:.0f} ms")

        attachments.append(StreamAttachment(
            name=col, available=bool(ts.size), count=int(ts.size),
            measured_hz=rate.effective_fps, tolerance_s=tol,
            valid_fraction=float(valid.mean()) if valid.size else 0.0,
            median_abs_dt_ms=float(np.median(finite)) if finite.size else None,
            max_abs_dt_ms=float(np.max(finite)) if finite.size else None,
            exclusion_reason=reason, meta=meta,
        ))
        if reason:
            notes.append(f"{col}: {reason}")

    return MultimodalTimeline(
        recording_id=recording_id, domain=domain,
        reference_label=reference_label, columns=columns,
        attachments=attachments, reference_hz=ref_rate.effective_fps, notes=notes,
    )


def _column_name(stream_label: str) -> str:
    """`imu-left` -> `imu_left`, so the label survives into a column name."""
    return stream_label.replace("-", "_").replace(".", "_")


# --------------------------------------------------------------------------- #
# Derived views
# --------------------------------------------------------------------------- #
@dataclass
class DerivedView:
    """A coarser view that still points only at real samples of the master table."""

    grid_hz: float
    purpose: str
    row_indices: np.ndarray          # into the master table
    grid_ns: np.ndarray              # the requested grid points
    temporal_error_ms: np.ndarray    # grid point -> selected real frame
    duplicate_row_count: int
    dropped_grid_points: int
    interpolated: bool = False
    synthetic_samples: int = 0

    def to_dict(self) -> Dict[str, Any]:
        err = self.temporal_error_ms
        return {
            "grid_hz": self.grid_hz, "purpose": self.purpose,
            "rows": int(self.row_indices.size),
            "duplicate_row_count": self.duplicate_row_count,
            "dropped_grid_points": self.dropped_grid_points,
            "max_temporal_error_ms": float(np.max(err)) if err.size else 0.0,
            "median_temporal_error_ms": float(np.median(err)) if err.size else 0.0,
            "interpolated": self.interpolated,
            "synthetic_samples": self.synthetic_samples,
        }


def derive_view(timeline: MultimodalTimeline, grid_hz: float, purpose: str,
                max_temporal_error_s: Optional[float] = None) -> DerivedView:
    """Select the real master rows nearest to a regular grid.

    A grid point whose nearest real frame is farther than the tolerance is
    **dropped**, never filled: filling it would invent an observation. Reuse of an
    already-selected row is counted and reported rather than hidden.
    """
    if grid_hz <= 0:
        raise ValueError(f"grid_hz must be positive, got {grid_hz!r}")
    ts = timeline.columns["timestamp_ns"]
    step = int(round(NS_PER_S / grid_hz))
    grid = np.arange(int(ts[0]), int(ts[-1]) + 1, step, dtype=np.int64)
    idx = nearest_indices(grid, ts)
    err_ms = np.abs(grid - ts[idx]) / 1e6

    if max_temporal_error_s is None:
        # Half a grid step, or half an RGB period, whichever is larger: a real
        # frame farther than that is simply not this grid point's observation.
        rgb_dt_ms = 1e3 / (timeline.reference_hz or grid_hz)
        max_temporal_error_s = max(0.5 * step / 1e6, 0.5 * rgb_dt_ms) / 1e3
    keep = err_ms <= float(max_temporal_error_s) * 1e3

    sel = idx[keep]
    return DerivedView(
        grid_hz=float(grid_hz), purpose=purpose,
        row_indices=sel, grid_ns=grid[keep], temporal_error_ms=err_ms[keep],
        duplicate_row_count=int(sel.size - np.unique(sel).size),
        dropped_grid_points=int(np.sum(~keep)),
    )


#: The derived views the behaviour analysis is allowed to build, and why.
STANDARD_VIEWS = (
    (1.0, "route statistics and scouting"),
    (5.0, "slow events"),
)


def window_indices(series_ts: np.ndarray, centre_ns: int,
                   before_s: float, after_s: float) -> slice:
    """Index slice of the real samples inside [centre-before, centre+after].

    Used to pull an IMU or PPG burst around a frame. Returns a slice over real
    samples only: an empty slice means the window genuinely holds no data, which
    the caller must report rather than paper over.
    """
    lo = int(np.searchsorted(series_ts, int(centre_ns - before_s * NS_PER_S), "left"))
    hi = int(np.searchsorted(series_ts, int(centre_ns + after_s * NS_PER_S), "right"))
    return slice(lo, hi)
