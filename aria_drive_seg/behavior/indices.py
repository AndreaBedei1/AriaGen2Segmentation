"""Composite indices built from the primary signals.

Each index is a normalised combination of things measured elsewhere. Two rules
keep them honest:

* an index reports **which components it actually had**. A road-complexity score
  built from three of its eight inputs is not the same quantity as one built from
  all eight, and silently averaging over whatever happened to be present would
  hide that;
* normalisation is against the pooled range of *both* domains, so a value is
  comparable between car and motorcycle. Normalising each domain separately would
  guarantee both look average.

None of these is a validated instrument. They are descriptive composites for a
pilot, and `attentional_tunnelling` in particular names a candidate interval for
review, not a diagnosis of distraction.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np


def _normalise(values: np.ndarray, lo: Optional[float] = None,
               hi: Optional[float] = None) -> np.ndarray:
    """Scale to [0, 1] against a supplied range, robust to outliers."""
    v = np.asarray(values, float)
    finite = v[np.isfinite(v)]
    if finite.size == 0:
        return np.full(v.shape, np.nan)
    lo = float(np.percentile(finite, 5)) if lo is None else float(lo)
    hi = float(np.percentile(finite, 95)) if hi is None else float(hi)
    if hi <= lo:
        return np.where(np.isfinite(v), 0.5, np.nan)
    return np.clip((v - lo) / (hi - lo), 0.0, 1.0)


@dataclass
class CompositeIndex:
    name: str
    values: np.ndarray
    components_used: List[str]
    components_missing: List[str]
    definition: str

    @property
    def completeness(self) -> float:
        total = len(self.components_used) + len(self.components_missing)
        return float(len(self.components_used) / total) if total else 0.0

    def to_dict(self) -> Dict[str, Any]:
        v = self.values[np.isfinite(self.values)]
        return {
            "name": self.name,
            "components_used": self.components_used,
            "components_missing": self.components_missing,
            "completeness": self.completeness,
            "definition": self.definition,
            "n": int(v.size),
            "median": float(np.median(v)) if v.size else None,
            "p05": float(np.percentile(v, 5)) if v.size else None,
            "p95": float(np.percentile(v, 95)) if v.size else None,
            "caveat": (None if self.completeness >= 0.999 else
                       f"built from {len(self.components_used)} of "
                       f"{len(self.components_used) + len(self.components_missing)} "
                       "components; not comparable to a fully-populated index"),
        }


def _combine(components: Dict[str, Optional[np.ndarray]], name: str,
             definition: str, ranges: Optional[Dict[str, Tuple[float, float]]] = None
             ) -> CompositeIndex:
    used, missing, stack = [], [], []
    for key, values in components.items():
        if values is None:
            missing.append(key)
            continue
        arr = np.asarray(values, float)
        if not np.isfinite(arr).any():
            missing.append(key)
            continue
        rng = (ranges or {}).get(key, (None, None))
        stack.append(_normalise(arr, rng[0], rng[1]))
        used.append(key)
    if not stack:
        return CompositeIndex(name, np.zeros(0), used, missing, definition)
    return CompositeIndex(name, np.nanmean(np.vstack(stack), axis=0),
                          used, missing, definition)


def road_complexity_index(curvature=None, junction_density=None, lanes=None,
                          vehicle_density=None, pedestrian_density=None,
                          sign_density=None, boundary_density=None,
                          signal_density=None, scene_change=None,
                          ranges=None) -> CompositeIndex:
    """How much the road itself is asking of the driver."""
    return _combine({
        "curvature": curvature,
        "junction_density": junction_density,
        "lanes": lanes,
        "vehicle_density": vehicle_density,
        "pedestrian_density": pedestrian_density,
        "sign_density": sign_density,
        "boundary_density": boundary_density,
        "signal_density": signal_density,
        "scene_change": scene_change,
    }, "road_complexity_index",
        "mean of range-normalised road and scene demand components", ranges)


def visual_demand_index(gaze_switching=None, gaze_entropy=None,
                        short_fixation_rate=None, scanpath_length=None,
                        head_motion=None, semantic_density=None,
                        ranges=None) -> CompositeIndex:
    """How hard the eyes and head are working."""
    return _combine({
        "gaze_switching": gaze_switching,
        "gaze_entropy": gaze_entropy,
        "short_fixation_rate": short_fixation_rate,
        "scanpath_length": scanpath_length,
        "head_motion": head_motion,
        "semantic_density": semantic_density,
    }, "visual_demand_index",
        "mean of range-normalised visual and head-motion demand components",
        ranges)


def traffic_interaction_index(vehicle_count=None, vehicle_pixel_share=None,
                              gaze_on_vehicles=None, speed_variation=None,
                              braking_rate=None, ranges=None) -> CompositeIndex:
    """How much of the drive is spent dealing with other traffic."""
    return _combine({
        "vehicle_count": vehicle_count,
        "vehicle_pixel_share": vehicle_pixel_share,
        "gaze_on_vehicles": gaze_on_vehicles,
        "speed_variation": speed_variation,
        "braking_rate": braking_rate,
    }, "traffic_interaction_index",
        "mean of range-normalised traffic-interaction components", ranges)


def attentional_tunnelling_candidates(timestamp_ns: Sequence[int],
                                      gaze_entropy: Sequence[float],
                                      fixation_duration_s: Sequence[float],
                                      class_switch_rate: Sequence[float],
                                      road_complexity: Sequence[float],
                                      entropy_percentile: float = 20.0,
                                      complexity_percentile: float = 60.0,
                                      min_duration_s: float = 1.5
                                      ) -> List[Dict[str, Any]]:
    """Intervals where gaze narrowed while the road was demanding.

    This names a **candidate interval for review**. Narrow gaze on a complex road
    is equally consistent with a driver correctly locking onto the one thing that
    matters, and nothing in these signals separates the two. Calling it
    distraction would be an interpretation the data cannot carry.
    """
    ts = np.asarray(timestamp_ns, np.int64)
    ent = np.asarray(gaze_entropy, float)
    fix = np.asarray(fixation_duration_s, float)
    sw = np.asarray(class_switch_rate, float)
    cx = np.asarray(road_complexity, float)
    if ts.size == 0:
        return []

    ent_ok = ent[np.isfinite(ent)]
    cx_ok = cx[np.isfinite(cx)]
    if ent_ok.size < 5 or cx_ok.size < 5:
        return []
    ent_thr = float(np.percentile(ent_ok, entropy_percentile))
    cx_thr = float(np.percentile(cx_ok, complexity_percentile))

    flag = (np.isfinite(ent) & (ent <= ent_thr) &
            np.isfinite(cx) & (cx >= cx_thr))
    out: List[Dict[str, Any]] = []
    start = None
    for i, f in enumerate(flag):
        if f and start is None:
            start = i
        elif not f and start is not None:
            out.append(_tunnel_record(ts, ent, fix, sw, cx, start, i,
                                      ent_thr, cx_thr, min_duration_s))
            start = None
    if start is not None:
        out.append(_tunnel_record(ts, ent, fix, sw, cx, start, flag.size,
                                  ent_thr, cx_thr, min_duration_s))
    return [r for r in out if r is not None]


def _tunnel_record(ts, ent, fix, sw, cx, a, b, ent_thr, cx_thr, min_duration_s):
    duration = float((ts[b - 1] - ts[a]) / 1e9)
    if duration < min_duration_s:
        return None
    return {
        "start_ns": int(ts[a]), "end_ns": int(ts[b - 1]), "duration_s": duration,
        "mean_gaze_entropy": float(np.nanmean(ent[a:b])),
        "mean_fixation_duration_s": float(np.nanmean(fix[a:b])),
        "mean_class_switch_rate": float(np.nanmean(sw[a:b])),
        "mean_road_complexity": float(np.nanmean(cx[a:b])),
        "entropy_threshold": ent_thr, "complexity_threshold": cx_thr,
        "status": "candidate_for_review",
        "not_a_distraction_claim": True,
        "note": ("narrow gaze on a demanding road is equally consistent with "
                 "correct selective attention; this interval is flagged for a "
                 "human to look at, not classified"),
    }


def head_eye_coordination(head_yaw_rate: Sequence[float],
                          gaze_yaw_rate: Sequence[float]) -> Dict[str, Any]:
    """How much of a gaze shift the head contributed, versus the eyes.

    Both series must already be on a common time base; the caller is responsible
    for that, because doing it here would hide which samples were really paired.
    """
    h = np.asarray(head_yaw_rate, float)
    g = np.asarray(gaze_yaw_rate, float)
    n = min(h.size, g.size)
    h, g = h[:n], g[:n]
    ok = np.isfinite(h) & np.isfinite(g)
    if ok.sum() < 8:
        return {"samples": int(ok.sum()), "head_contribution": None,
                "correlation": None,
                "reason": "fewer than 8 paired samples"}
    total = np.abs(h[ok]) + np.abs(g[ok])
    contribution = np.where(total > 1e-9, np.abs(h[ok]) / np.maximum(total, 1e-9),
                            np.nan)
    return {
        "samples": int(ok.sum()),
        "head_contribution": float(np.nanmedian(contribution)),
        "eye_contribution": float(1.0 - np.nanmedian(contribution)),
        "correlation": float(np.corrcoef(h[ok], g[ok])[0, 1]),
        "note": ("contribution is the head's share of the combined absolute "
                 "angular rate; it is a ratio, not a decomposition of a single "
                 "gaze shift into its parts"),
    }


def cross_correlation(a: Sequence[float], b: Sequence[float],
                      sample_interval_s: float,
                      max_lag_s: float = 30.0) -> Dict[str, Any]:
    """Lagged correlation between two signals on a common time base.

    Physiological responses lag their trigger by seconds, so the peak of this
    curve is more informative than the zero-lag value. A peak is still only a
    temporal association: it is not evidence that one signal caused the other.
    """
    x = np.asarray(a, float)
    y = np.asarray(b, float)
    n = min(x.size, y.size)
    x, y = x[:n], y[:n]
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 16 or sample_interval_s <= 0:
        return {"samples": int(ok.sum()), "peak_lag_s": None,
                "peak_correlation": None, "zero_lag_correlation": None,
                "reason": "fewer than 16 paired finite samples"}
    x = x[ok] - np.mean(x[ok])
    y = y[ok] - np.mean(y[ok])
    max_lag = int(round(max_lag_s / sample_interval_s))
    max_lag = max(1, min(max_lag, x.size // 2))
    lags = np.arange(-max_lag, max_lag + 1)
    corr = []
    for lag in lags:
        if lag < 0:
            xa, ya = x[-lag:], y[:lag]
        elif lag > 0:
            xa, ya = x[:-lag], y[lag:]
        else:
            xa, ya = x, y
        if xa.size < 8 or np.std(xa) < 1e-12 or np.std(ya) < 1e-12:
            corr.append(np.nan)
        else:
            corr.append(float(np.corrcoef(xa, ya)[0, 1]))
    corr = np.asarray(corr)
    if not np.isfinite(corr).any():
        return {"samples": int(ok.sum()), "peak_lag_s": None,
                "peak_correlation": None, "zero_lag_correlation": None,
                "reason": "correlation undefined at every lag"}
    k = int(np.nanargmax(np.abs(corr)))
    return {
        "samples": int(ok.sum()),
        "peak_lag_s": float(lags[k] * sample_interval_s),
        "peak_correlation": float(corr[k]),
        "zero_lag_correlation": float(corr[max_lag]),
        "lags_s": (lags * sample_interval_s).tolist(),
        "correlation": corr.tolist(),
        "note": ("temporal association only; a lagged correlation between two "
                 "signals recorded on one person in one session is not evidence "
                 "of causation in either direction"),
    }
