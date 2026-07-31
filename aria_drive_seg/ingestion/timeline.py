"""Timestamp-driven timeline primitives shared by every ingestion stage.

Design rules enforced here (see the Article 1 ingestion constraints):

* the sampling rate of a stream is **measured** from its capture timestamps, never
  taken from the recording profile name or from a constant;
* every temporal window is declared in **seconds** and converted to a frame count
  against the recording it is applied to, so the same configuration behaves
  identically on a 10 fps and on a 15 fps recording;
* metric normalisation is **per second**, so raw per-frame counts of recordings
  with different sampling rates are never compared;
* the "comparable timeline" is a *nearest real sample* grid. It never interpolates,
  never synthesises a frame and never silently duplicates one: every duplicate
  assignment is reported explicitly together with its temporal error.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

NS_PER_S = 1_000_000_000


# --------------------------------------------------------------------------- #
# Measured rate
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class RateEstimate:
    """Sampling rate measured from timestamps only."""

    count: int
    first_ns: Optional[int]
    last_ns: Optional[int]
    duration_s: float
    # span rate = (n-1)/duration: robust to jitter, sensitive to long pauses
    effective_fps_span: Optional[float]
    # median rate = 1/median(dt): robust to pauses, the "instantaneous" cadence
    effective_fps_median: Optional[float]
    median_dt_ms: Optional[float]
    mean_dt_ms: Optional[float]
    std_dt_ms: Optional[float]
    min_dt_ms: Optional[float]
    max_dt_ms: Optional[float]
    p01_dt_ms: Optional[float]
    p05_dt_ms: Optional[float]
    p50_dt_ms: Optional[float]
    p95_dt_ms: Optional[float]
    p99_dt_ms: Optional[float]
    non_monotonic_count: int
    duplicate_timestamp_count: int

    @property
    def effective_fps(self) -> Optional[float]:
        """Primary effective rate: the median cadence.

        The median is preferred over the span rate because a recording that pauses
        and restarts still samples at its configured cadence; the pause is reported
        separately as a gap rather than smeared into the rate.
        """
        return self.effective_fps_median

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["effective_fps"] = self.effective_fps
        return d


def _pct(values: np.ndarray, q: float) -> Optional[float]:
    return float(np.percentile(values, q)) if values.size else None


def measure_rate(timestamps_ns: Sequence[int] | np.ndarray) -> RateEstimate:
    """Measure the sampling rate of a stream from its capture timestamps."""
    ts = np.asarray(timestamps_ns, dtype=np.int64)
    if ts.size == 0:
        return RateEstimate(0, None, None, 0.0, None, None, None, None, None, None,
                            None, None, None, None, None, None, 0, 0)
    if ts.size == 1:
        return RateEstimate(1, int(ts[0]), int(ts[0]), 0.0, None, None, None, None,
                            None, None, None, None, None, None, None, None, 0, 0)

    dt = np.diff(ts)
    dt_ms = dt / 1e6
    duration_s = float((ts[-1] - ts[0]) / NS_PER_S)
    median_dt = float(np.median(dt))
    positive = dt[dt > 0]

    return RateEstimate(
        count=int(ts.size),
        first_ns=int(ts[0]),
        last_ns=int(ts[-1]),
        duration_s=duration_s,
        effective_fps_span=float((ts.size - 1) / duration_s) if duration_s > 0 else None,
        effective_fps_median=(float(NS_PER_S / median_dt) if median_dt > 0 else None),
        median_dt_ms=median_dt / 1e6,
        mean_dt_ms=float(np.mean(dt_ms)),
        std_dt_ms=float(np.std(dt_ms)),
        min_dt_ms=float(np.min(dt_ms)),
        max_dt_ms=float(np.max(dt_ms)),
        p01_dt_ms=_pct(dt_ms, 1),
        p05_dt_ms=_pct(dt_ms, 5),
        p50_dt_ms=_pct(dt_ms, 50),
        p95_dt_ms=_pct(dt_ms, 95),
        p99_dt_ms=_pct(dt_ms, 99),
        non_monotonic_count=int(np.sum(dt <= 0)),
        duplicate_timestamp_count=int(np.sum(dt == 0)),
    )


def frames_for_seconds(seconds: float, effective_fps: float,
                       minimum: int = 1) -> int:
    """Convert a window expressed in seconds into a frame count for THIS recording.

    This is the only sanctioned way to turn a temporal parameter into a frame
    count. A configuration says "±1.0 s"; the number of frames that represents is
    a property of the recording, not of the configuration.
    """
    if not np.isfinite(effective_fps) or effective_fps <= 0:
        raise ValueError(f"effective_fps must be positive and finite, got {effective_fps!r}")
    if seconds < 0:
        raise ValueError(f"seconds must be non-negative, got {seconds!r}")
    return max(int(minimum), int(round(float(seconds) * float(effective_fps))))


def per_second(count: float, duration_s: float) -> Optional[float]:
    """Normalise a raw count to a per-second rate.

    Comparing raw per-frame counts between recordings sampled at different rates is
    forbidden; every cross-recording count must go through this function.
    """
    if duration_s is None or duration_s <= 0:
        return None
    return float(count) / float(duration_s)


# --------------------------------------------------------------------------- #
# Gaps and continuous segments
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Gap:
    index_before: int
    index_after: int
    t_before_ns: int
    t_after_ns: int
    dt_ms: float
    periods: float
    missing_estimate: int
    over_1p5_periods: bool
    over_2_periods: bool
    over_250ms: bool
    non_monotonic: bool

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def effective_absolute_threshold_ns(median_dt_ns: float,
                                    absolute_threshold_ms: float) -> float:
    """Apply the absolute pause threshold only where it is meaningful.

    A 250 ms threshold detects a stall in a 15 fps camera, but a 1 Hz GPS samples
    every 1000 ms by design: applying the absolute rule there would flag every
    single normal interval as a gap. The absolute rule is therefore disabled for
    streams whose own nominal period already exceeds it; such streams are judged
    against their own cadence only.
    """
    abs_ns = float(absolute_threshold_ms) * 1e6
    return abs_ns if (median_dt_ns > 0 and median_dt_ns < abs_ns) else float("inf")


def find_gaps(timestamps_ns: Sequence[int] | np.ndarray,
              median_dt_ns: Optional[float] = None,
              absolute_threshold_ms: float = 250.0) -> List[Gap]:
    """Report every interval that is anomalous w.r.t. the stream's own cadence.

    A gap is reported when the interval exceeds 1.5 nominal periods, or exceeds the
    absolute pause threshold where that threshold applies (see
    `effective_absolute_threshold_ns`).
    """
    ts = np.asarray(timestamps_ns, dtype=np.int64)
    if ts.size < 2:
        return []
    dt = np.diff(ts)
    med = float(median_dt_ns) if median_dt_ns else float(np.median(dt))
    if med <= 0:
        med = float(np.median(dt[dt > 0])) if np.any(dt > 0) else 0.0

    gaps: List[Gap] = []
    abs_ns = effective_absolute_threshold_ns(med, absolute_threshold_ms)
    for i in range(dt.size):
        d = int(dt[i])
        periods = (d / med) if med > 0 else float("inf")
        flagged = (med > 0 and d > 1.5 * med) or d > abs_ns or d <= 0
        if not flagged:
            continue
        gaps.append(Gap(
            index_before=i,
            index_after=i + 1,
            t_before_ns=int(ts[i]),
            t_after_ns=int(ts[i + 1]),
            dt_ms=d / 1e6,
            periods=float(periods),
            missing_estimate=int(max(0, round(periods) - 1)) if med > 0 and d > 0 else 0,
            over_1p5_periods=bool(med > 0 and d > 1.5 * med),
            over_2_periods=bool(med > 0 and d > 2.0 * med),
            over_250ms=bool(d > abs_ns),
            non_monotonic=bool(d <= 0),
        ))
    return gaps


@dataclass(frozen=True)
class Segment:
    """A maximal run of samples with no break larger than the break threshold."""

    start_index: int
    end_index: int          # inclusive
    start_ns: int
    end_ns: int

    @property
    def count(self) -> int:
        return self.end_index - self.start_index + 1

    @property
    def duration_s(self) -> float:
        return (self.end_ns - self.start_ns) / NS_PER_S

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["count"] = self.count
        d["duration_s"] = self.duration_s
        return d


def continuous_segments(timestamps_ns: Sequence[int] | np.ndarray,
                        break_periods: float = 2.0,
                        break_absolute_ms: float = 250.0,
                        median_dt_ns: Optional[float] = None) -> List[Segment]:
    """Split a stream into maximal continuous segments.

    The break threshold is the stricter of `break_periods` nominal periods and the
    absolute millisecond threshold, so both a dropped frame burst and a wall-clock
    pause terminate a segment.
    """
    ts = np.asarray(timestamps_ns, dtype=np.int64)
    if ts.size == 0:
        return []
    if ts.size == 1:
        return [Segment(0, 0, int(ts[0]), int(ts[0]))]

    dt = np.diff(ts)
    med = float(median_dt_ns) if median_dt_ns else float(np.median(dt))
    thresholds = [effective_absolute_threshold_ns(med, break_absolute_ms)]
    if med > 0:
        thresholds.append(break_periods * med)
    threshold = min(thresholds)
    if not np.isfinite(threshold):
        return [Segment(0, int(ts.size - 1), int(ts[0]), int(ts[-1]))]

    segments: List[Segment] = []
    start = 0
    for i in range(dt.size):
        if dt[i] > threshold or dt[i] <= 0:
            segments.append(Segment(start, i, int(ts[start]), int(ts[i])))
            start = i + 1
    segments.append(Segment(start, int(ts.size - 1), int(ts[start]), int(ts[-1])))
    return segments


# --------------------------------------------------------------------------- #
# Cross-stream synchronisation
# --------------------------------------------------------------------------- #
def nearest_indices(reference_ns: Sequence[int] | np.ndarray,
                    target_ns: Sequence[int] | np.ndarray) -> np.ndarray:
    """For each reference timestamp, the index of the nearest target sample."""
    ref = np.asarray(reference_ns, dtype=np.int64)
    tgt = np.asarray(target_ns, dtype=np.int64)
    if tgt.size == 0:
        return np.full(ref.shape, -1, dtype=np.int64)
    pos = np.searchsorted(tgt, ref)
    left = np.clip(pos - 1, 0, tgt.size - 1)
    right = np.clip(pos, 0, tgt.size - 1)
    take_left = np.abs(ref - tgt[left]) <= np.abs(tgt[right] - ref)
    return np.where(take_left, left, right)


def synchronisation_report(reference_ns: Sequence[int] | np.ndarray,
                           target_ns: Sequence[int] | np.ndarray) -> Dict[str, Any]:
    """Temporal distance statistics of a stream relative to the RGB reference."""
    ref = np.asarray(reference_ns, dtype=np.int64)
    tgt = np.asarray(target_ns, dtype=np.int64)
    if ref.size == 0 or tgt.size == 0:
        return {"available": False, "reason": "empty stream"}
    idx = nearest_indices(ref, tgt)
    delta_ms = np.abs(ref - tgt[idx]) / 1e6
    return {
        "available": True,
        "reference_count": int(ref.size),
        "target_count": int(tgt.size),
        "abs_dt_to_reference_ms": {
            "mean": float(np.mean(delta_ms)),
            "median": float(np.median(delta_ms)),
            "p95": float(np.percentile(delta_ms, 95)),
            "max": float(np.max(delta_ms)),
        },
        "coverage_start_offset_s": float((tgt[0] - ref[0]) / NS_PER_S),
        "coverage_end_offset_s": float((tgt[-1] - ref[-1]) / NS_PER_S),
    }


# --------------------------------------------------------------------------- #
# Derived comparable timeline (QA only)
# --------------------------------------------------------------------------- #
@dataclass
class ComparableTimeline:
    """A nearest-real-sample grid used only for preliminary QA visualisation.

    Marked `derived` and `preliminary` on purpose: it is not a dataset format, it
    must not feed a vehicle classifier, and it never creates a frame that was not
    actually captured.
    """

    grid_hz: float
    grid_ns: List[int]
    selected_indices: List[int]
    selected_ns: List[int]
    temporal_error_ms: List[float]
    duplicate_selection_count: int
    max_temporal_error_ms: float
    derived: bool = True
    preliminary: bool = True
    interpolated: bool = False
    synthetic_frames: int = 0
    notes: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def build_comparable_timeline(timestamps_ns: Sequence[int] | np.ndarray,
                              grid_hz: float,
                              start_ns: Optional[int] = None,
                              end_ns: Optional[int] = None,
                              max_temporal_error_ms: Optional[float] = None
                              ) -> ComparableTimeline:
    """Snap a real stream onto a common grid by nearest-neighbour selection.

    `max_temporal_error_ms` (when given) marks grid points whose nearest real sample
    is too far away; those points are dropped rather than filled, because filling
    them would mean inventing data.
    """
    ts = np.asarray(timestamps_ns, dtype=np.int64)
    if ts.size == 0:
        raise ValueError("cannot build a comparable timeline from an empty stream")
    if grid_hz <= 0:
        raise ValueError(f"grid_hz must be positive, got {grid_hz!r}")

    t0 = int(start_ns if start_ns is not None else ts[0])
    t1 = int(end_ns if end_ns is not None else ts[-1])
    step = int(round(NS_PER_S / grid_hz))
    grid = np.arange(t0, t1 + 1, step, dtype=np.int64)

    idx = nearest_indices(grid, ts)
    err_ms = np.abs(grid - ts[idx]) / 1e6

    keep = np.ones(grid.shape, dtype=bool)
    notes: List[str] = []
    if max_temporal_error_ms is not None:
        keep = err_ms <= float(max_temporal_error_ms)
        dropped = int(np.sum(~keep))
        if dropped:
            notes.append(
                f"{dropped} grid points dropped: nearest real sample farther than "
                f"{max_temporal_error_ms} ms; not filled by design")

    sel = idx[keep]
    duplicates = int(sel.size - np.unique(sel).size)
    if duplicates:
        notes.append(
            f"{duplicates} grid points reuse an already-selected real sample; "
            "reported explicitly, never silently duplicated")

    return ComparableTimeline(
        grid_hz=float(grid_hz),
        grid_ns=[int(x) for x in grid[keep]],
        selected_indices=[int(x) for x in sel],
        selected_ns=[int(ts[i]) for i in sel],
        temporal_error_ms=[float(x) for x in err_ms[keep]],
        duplicate_selection_count=duplicates,
        max_temporal_error_ms=float(np.max(err_ms[keep])) if np.any(keep) else 0.0,
        notes=notes,
    )
