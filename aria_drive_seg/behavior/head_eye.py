"""Head–eye coordination: lateral checks, gaze excursions and how they combine.

The head IMU runs at ~805 Hz and the gaze stream at 30 Hz, so nothing is
compared until both are on the gaze stream's own real timestamps — head rate is
carried to each gaze sample from its nearest real IMU sample, never interpolated,
and the association distance is kept.

**A lateral head check is not a mirror check.** This module counts a sustained
yaw excursion of the head. A head turn to a mirror, a head turn to a side road, a
head turn to look at a shop window and a head turn because the vehicle leaned are
identical in this signal. Every count is named `lateral_head_check` and never
`mirror_check`, and no output of this module may be relabelled as one downstream:
`tests/` pins that.

The scan taxonomy is likewise mechanical. An **eye-only scan** is a gaze
excursion with no concurrent head excursion; a **head-assisted scan** is one with
a concurrent head excursion. Those are descriptions of two measured signals, not
of intent.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

NS_PER_S = 1_000_000_000

MIRROR_ATTRIBUTION_NOTE = (
    "a lateral head check is a sustained yaw excursion of the head. It is not "
    "attributed to a mirror, a side road or any other target: this signal cannot "
    "distinguish them.")


def _runs(flag: np.ndarray) -> List[Tuple[int, int]]:
    f = np.asarray(flag, bool)
    if f.size == 0:
        return []
    edges = np.diff(f.astype(np.int8))
    starts = list(np.flatnonzero(edges == 1) + 1)
    ends = list(np.flatnonzero(edges == -1) + 1)
    if f[0]:
        starts.insert(0, 0)
    if f[-1]:
        ends.append(f.size)
    return list(zip(starts, ends))


@dataclass
class Excursion:
    """A sustained signed angular excursion of the head or of the eyes."""

    start_ns: int
    end_ns: int
    duration_s: float
    direction: str                   # "left" | "right"
    peak_rate_rad_s: float
    mean_rate_rad_s: float

    def to_dict(self) -> Dict[str, Any]:
        return {"start_ns": self.start_ns, "end_ns": self.end_ns,
                "duration_s": self.duration_s, "direction": self.direction,
                "peak_rate_rad_s": self.peak_rate_rad_s,
                "mean_rate_rad_s": self.mean_rate_rad_s}


def find_excursions(timestamp_ns: Sequence[int], rate: Sequence[float],
                    threshold: float, min_duration_s: float,
                    positive_is: str = "left") -> List[Excursion]:
    """Runs where a signed angular rate stays beyond a threshold.

    Direction is taken from the run's mean signed rate, so a run that crosses
    zero — which is a return, not a check — cannot be counted twice.
    """
    ts = np.asarray(timestamp_ns, np.int64)
    values = np.asarray(rate, float)
    beyond = np.isfinite(values) & (np.abs(values) >= float(threshold))
    negative_is = "right" if positive_is == "left" else "left"
    out: List[Excursion] = []
    for a, b in _runs(beyond):
        duration_s = float((ts[b - 1] - ts[a]) / NS_PER_S)
        if duration_s < float(min_duration_s):
            continue
        segment = values[a:b]
        mean = float(np.mean(segment))
        peak = float(segment[int(np.argmax(np.abs(segment)))])
        out.append(Excursion(
            start_ns=int(ts[a]), end_ns=int(ts[b - 1]), duration_s=duration_s,
            direction=(positive_is if mean >= 0 else negative_is),
            peak_rate_rad_s=peak, mean_rate_rad_s=mean))
    return out


def gaze_yaw_rate(timestamp_ns: Sequence[int], yaw_rad: Sequence[float],
                  valid: Optional[Sequence[bool]] = None) -> np.ndarray:
    """Signed yaw rate of the gaze direction, in rad/s, on its own timestamps."""
    ts = np.asarray(timestamp_ns, np.int64)
    yaw = np.asarray(yaw_rad, float)
    ok = (np.asarray(valid, bool) if valid is not None else np.isfinite(yaw))
    out = np.full(ts.size, np.nan)
    if ts.size < 2:
        return out
    dt = np.diff(ts) / NS_PER_S
    dyaw = np.diff(yaw)
    with np.errstate(divide="ignore", invalid="ignore"):
        rate = np.where(dt > 0, dyaw / dt, np.nan)
    out[:-1] = rate
    out[-1] = rate[-1]
    # A rate spanning an invalid sample is a rate across an unmeasured gap.
    pair_ok = ok.copy()
    pair_ok[:-1] &= ok[1:]
    out[~pair_ok] = np.nan
    return out


def attach_head_rate(gaze_ts: Sequence[int], head_ts: Sequence[int],
                     head_yaw_rate: Sequence[float], max_dt_s: float = 0.02
                     ) -> Tuple[np.ndarray, np.ndarray]:
    """Head yaw rate at each gaze sample, from the nearest real IMU sample."""
    target = np.asarray(gaze_ts, np.int64)
    source = np.asarray(head_ts, np.int64)
    values = np.asarray(head_yaw_rate, float)
    if source.size == 0:
        return np.full(target.size, np.nan), np.full(target.size, np.nan)
    hi = np.clip(np.searchsorted(source, target, side="left"), 0, source.size - 1)
    lo = np.clip(hi - 1, 0, source.size - 1)
    use_hi = np.abs(source[hi] - target) < np.abs(target - source[lo])
    idx = np.where(use_hi, hi, lo)
    dt_s = (source[idx] - target) / NS_PER_S
    out = values[idx].astype(float)
    out[np.abs(dt_s) > float(max_dt_s)] = np.nan
    return out, dt_s


def _overlaps(a: Excursion, others: Sequence[Excursion], tolerance_s: float) -> bool:
    pad = int(round(float(tolerance_s) * NS_PER_S))
    return any(other.start_ns - pad <= a.end_ns and a.start_ns <= other.end_ns + pad
               for other in others)


def head_eye_metrics(gaze_timestamp_ns: Sequence[int],
                     gaze_yaw_rate_rad_s: Sequence[float],
                     head_yaw_rate_rad_s: Sequence[float],
                     head_timestamp_ns: Sequence[int],
                     head_native_yaw_rate_rad_s: Sequence[float],
                     head_threshold_rad_s: float = 0.35,
                     head_min_duration_s: float = 0.15,
                     gaze_threshold_rad_s: float = 1.0,
                     gaze_min_duration_s: float = 0.066,
                     concurrency_tolerance_s: float = 0.30,
                     ) -> Dict[str, Any]:
    """Lateral checks, gaze excursions, the scan split and their asymmetry.

    Head checks are found on the head stream's **own** ~805 Hz samples, because a
    0.15 s excursion is three samples at the gaze rate and would be lost. The
    per-gaze-sample head rate is used only for the direction-agreement statistic,
    where the two must be on one clock.
    """
    gaze_ts = np.asarray(gaze_timestamp_ns, np.int64)
    duration_s = (float((gaze_ts[-1] - gaze_ts[0]) / NS_PER_S)
                  if gaze_ts.size > 1 else 0.0)

    head_checks = find_excursions(head_timestamp_ns, head_native_yaw_rate_rad_s,
                                  head_threshold_rad_s, head_min_duration_s)
    gaze_excursions = find_excursions(gaze_ts, gaze_yaw_rate_rad_s,
                                      gaze_threshold_rad_s, gaze_min_duration_s)

    head_assisted = [e for e in gaze_excursions
                     if _overlaps(e, head_checks, concurrency_tolerance_s)]
    eye_only = [e for e in gaze_excursions
                if not _overlaps(e, head_checks, concurrency_tolerance_s)]

    head_rate = np.asarray(head_yaw_rate_rad_s, float)
    eye_rate = np.asarray(gaze_yaw_rate_rad_s, float)
    both = np.isfinite(head_rate) & np.isfinite(eye_rate)
    moving = both & (np.abs(head_rate) >= head_threshold_rad_s)
    same_direction = (np.sign(head_rate[moving]) == np.sign(eye_rate[moving])
                      if moving.any() else np.zeros(0, bool))

    def rate_per_minute(count: int) -> Optional[float]:
        return float(count * 60.0 / duration_s) if duration_s > 0 else None

    def asymmetry(events: Sequence[Excursion]) -> Dict[str, Any]:
        left = sum(1 for e in events if e.direction == "left")
        right = sum(1 for e in events if e.direction == "right")
        total = left + right
        return {
            "left": left, "right": right,
            "left_per_minute": rate_per_minute(left),
            "right_per_minute": rate_per_minute(right),
            # (left - right) / (left + right): 0 is symmetric, +1 all left.
            "asymmetry_index": (float((left - right) / total) if total else None),
        }

    return {
        "schema": "article1_head_eye_coordination_v1",
        "observed_duration_s": duration_s,
        "thresholds": {
            "head_yaw_rate_rad_s": head_threshold_rad_s,
            "head_min_duration_s": head_min_duration_s,
            "gaze_yaw_rate_rad_s": gaze_threshold_rad_s,
            "gaze_min_duration_s": gaze_min_duration_s,
            "concurrency_tolerance_s": concurrency_tolerance_s,
        },
        "lateral_head_checks": {
            "count": len(head_checks),
            "per_minute": rate_per_minute(len(head_checks)),
            "median_duration_s": (float(np.median([e.duration_s for e in head_checks]))
                                  if head_checks else None),
            "asymmetry": asymmetry(head_checks),
            "attribution": MIRROR_ATTRIBUTION_NOTE,
            "is_mirror_check_count": False,
        },
        "gaze_lateral_excursions": {
            "count": len(gaze_excursions),
            "per_minute": rate_per_minute(len(gaze_excursions)),
            "median_duration_s": (
                float(np.median([e.duration_s for e in gaze_excursions]))
                if gaze_excursions else None),
            "asymmetry": asymmetry(gaze_excursions),
        },
        "scan_split": {
            "eye_only_scans": len(eye_only),
            "eye_only_per_minute": rate_per_minute(len(eye_only)),
            "head_assisted_scans": len(head_assisted),
            "head_assisted_per_minute": rate_per_minute(len(head_assisted)),
            "head_assisted_fraction": (
                float(len(head_assisted) / len(gaze_excursions))
                if gaze_excursions else None),
            "definition": ("a gaze excursion is head-assisted when a lateral head "
                           "check overlaps it within the concurrency tolerance, "
                           "and eye-only otherwise"),
        },
        "gaze_relative_to_head_motion": {
            "paired_samples": int(moving.sum()),
            "same_direction_fraction": (float(same_direction.mean())
                                        if same_direction.size else None),
            "opposite_direction_fraction": (float(1.0 - same_direction.mean())
                                            if same_direction.size else None),
            "median_gaze_over_head_rate_ratio": (
                float(np.median(np.abs(eye_rate[moving]) /
                                np.maximum(np.abs(head_rate[moving]), 1e-9)))
                if moving.any() else None),
            "note": ("gaze counter-rotating against the head is the expected "
                     "vestibulo-ocular response to head motion, not a separate "
                     "behaviour; gaze rotating with the head is a combined "
                     "head-and-eye shift"),
        },
        "head_motion_is_not_vehicle_motion": True,
    }


def head_motion_before(head_timestamp_ns: Sequence[int],
                       head_angular_speed_rad_s: Sequence[float],
                       event_start_ns: int, lead_s: float = 5.0
                       ) -> Dict[str, Optional[float]]:
    """Head motion in the seconds leading up to an event's start."""
    ts = np.asarray(head_timestamp_ns, np.int64)
    values = np.asarray(head_angular_speed_rad_s, float)
    start = int(event_start_ns) - int(round(float(lead_s) * NS_PER_S))
    sel = (ts >= start) & (ts < int(event_start_ns))
    window = values[sel]
    window = window[np.isfinite(window)]
    return {
        "lead_s": float(lead_s),
        "samples": int(window.size),
        "median_angular_speed_rad_s": (float(np.median(window))
                                       if window.size else None),
        "p95_angular_speed_rad_s": (float(np.percentile(window, 95))
                                    if window.size else None),
    }
