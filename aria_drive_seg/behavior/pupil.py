"""Pupil-diameter metrics, and an exploratory correction for ambient light.

**No clinical reading.** Pupil diameter here is an ocular measurement from a
consumer eye tracker on a pair of glasses. It is not a diagnostic sign, not a
measure of cognitive load, not a measure of arousal and not a measure of
impairment. Every summary this module produces carries that with it.

The one substantive analysis choice: the pupillary light reflex dominates
everything else a pupil does outdoors. Between a shaded street and open sky the
illuminance changes by orders of magnitude, and the diameter follows. Comparing a
raw diameter between a car (behind glass, under a roof) and a motorcycle (in the
open, +4 000 lux) would therefore be comparing the two vehicles' *glazing*.

So a light term is removed first. Diameter is regressed on `log10(lux + 1)` — log
because the reflex is approximately logarithmic in luminance over the working
range, not because the fit is good — and the residual is what the rest of the
analysis uses. The regression is labelled `exploratory` everywhere: it is a
single-covariate ordinary least squares fit on one session per vehicle, it does
not model the reflex's latency or its hysteresis, and its R² is reported so a
reader can see how little of the variance it actually removes.

The ambient-light sensor is itself head-mounted, so it follows where the wearer
looks as well as the real scene illumination. That is a limitation of the
correction, not a property that the correction fixes, and it is stated with every
residual.
"""
from __future__ import annotations

from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np

NS_PER_S = 1_000_000_000

CLINICAL_NOTE = (
    "pupil diameter from a consumer head-mounted eye tracker. Not a clinical "
    "measurement, not a diagnostic sign, and not interpreted as cognitive load, "
    "arousal, drowsiness or impairment.")

LIGHT_CORRECTION_NOTE = (
    "exploratory single-covariate correction: diameter regressed on "
    "log10(lux + 1) by ordinary least squares. It does not model the pupillary "
    "light reflex's latency or hysteresis, and the ambient-light sensor is "
    "head-mounted, so it follows gaze direction as well as scene illumination. "
    "The residual is a light-adjusted diameter, not a light-free one.")


# --------------------------------------------------------------------------- #
# Association with ambient light
# --------------------------------------------------------------------------- #
def attach_nearest(target_ns: Sequence[int], source_ns: Sequence[int],
                   values: Sequence[float], max_dt_s: float
                   ) -> Tuple[np.ndarray, np.ndarray]:
    """Nearest real source sample per target, blanked beyond `max_dt_s`.

    Nothing is interpolated. A target with no source sample inside the tolerance
    gets NaN, and the signed distance is returned so a consumer can see how far
    the association had to reach.
    """
    tgt = np.asarray(target_ns, np.int64)
    src = np.asarray(source_ns, np.int64)
    val = np.asarray(values, float)
    if src.size == 0:
        return np.full(tgt.size, np.nan), np.full(tgt.size, np.nan)
    hi = np.clip(np.searchsorted(src, tgt, side="left"), 0, src.size - 1)
    lo = np.clip(hi - 1, 0, src.size - 1)
    use_hi = np.abs(src[hi] - tgt) < np.abs(tgt - src[lo])
    idx = np.where(use_hi, hi, lo)
    dt_s = (src[idx] - tgt) / NS_PER_S
    out = val[idx].astype(float)
    out[np.abs(dt_s) > float(max_dt_s)] = np.nan
    return out, dt_s


# --------------------------------------------------------------------------- #
# Local baseline
# --------------------------------------------------------------------------- #
def local_baseline(timestamp_ns: Sequence[int], values: Sequence[float],
                   window_s: float = 60.0) -> np.ndarray:
    """Centred rolling median over a window defined in seconds.

    A median rather than a mean, because a blink edge or a tracking dropout would
    drag a mean and leave the "deviation from baseline" reading as a pupil change.
    Windows are defined in seconds so the same configuration means the same thing
    whatever the sampling rate.
    """
    ts = np.asarray(timestamp_ns, np.int64)
    v = np.asarray(values, float)
    half = int(round(float(window_s) / 2.0 * NS_PER_S))
    lo = np.searchsorted(ts, ts - half, side="left")
    hi = np.searchsorted(ts, ts + half, side="right")
    out = np.full(v.size, np.nan)
    for i in range(v.size):
        window = v[lo[i]:hi[i]]
        window = window[np.isfinite(window)]
        if window.size:
            out[i] = float(np.median(window))
    return out


# --------------------------------------------------------------------------- #
# Light correction
# --------------------------------------------------------------------------- #
def fit_log_lux_model(diameter_m: Sequence[float], lux: Sequence[float]
                      ) -> Dict[str, Any]:
    """Ordinary least squares of diameter on log10(lux + 1).

    `+1` rather than a clamp: lux is genuinely zero-ish in a tunnel, and log(0)
    would drop exactly the samples the correction most needs.
    """
    d = np.asarray(diameter_m, float)
    x = np.asarray(lux, float)
    ok = np.isfinite(d) & np.isfinite(x) & (x >= 0)
    if ok.sum() < 32:
        return {"fitted": False, "n": int(ok.sum()),
                "reason": f"only {int(ok.sum())} paired samples; too few to fit"}
    log_lux = np.log10(x[ok] + 1.0)
    if float(np.std(log_lux)) < 1e-6:
        return {"fitted": False, "n": int(ok.sum()),
                "reason": "illuminance is constant; the model is unidentifiable"}
    slope, intercept = np.polyfit(log_lux, d[ok], 1)
    predicted = slope * log_lux + intercept
    residual = d[ok] - predicted
    ss_res = float(np.sum(residual ** 2))
    ss_tot = float(np.sum((d[ok] - d[ok].mean()) ** 2))
    return {
        "fitted": True,
        "n": int(ok.sum()),
        "slope_m_per_log10lux": float(slope),
        "intercept_m": float(intercept),
        "r_squared": float(1.0 - ss_res / ss_tot) if ss_tot > 0 else None,
        "pearson_r": float(np.corrcoef(log_lux, d[ok])[0, 1]),
        "lux_range": [float(x[ok].min()), float(x[ok].max())],
        "model": "diameter ~ a * log10(lux + 1) + b",
        "estimator": "ordinary least squares",
        "status": "exploratory",
        "note": LIGHT_CORRECTION_NOTE,
    }


def apply_log_lux_model(diameter_m: Sequence[float], lux: Sequence[float],
                        model: Dict[str, Any]) -> np.ndarray:
    """Residual diameter after removing the fitted light term."""
    d = np.asarray(diameter_m, float)
    x = np.asarray(lux, float)
    if not model.get("fitted"):
        return np.full(d.size, np.nan)
    with np.errstate(invalid="ignore"):
        log_lux = np.log10(np.where(x >= 0, x, np.nan) + 1.0)
    predicted = model["slope_m_per_log10lux"] * log_lux + model["intercept_m"]
    return d - predicted


# --------------------------------------------------------------------------- #
# Sample table
# --------------------------------------------------------------------------- #
def build_pupil_table(eye_state, als_timestamp_ns: Optional[Sequence[int]] = None,
                      als_lux: Optional[Sequence[float]] = None,
                      closed: Optional[Sequence[bool]] = None,
                      max_light_dt_s: float = 0.25,
                      baseline_window_s: float = 60.0):
    """Per-sample pupil table: raw diameter, light, local baseline and residual.

    Samples flagged invalid by the device, and samples taken while an eye is
    closed, are excluded from every derived quantity — a diameter measured
    through an eyelid is not a pupil measurement — but they stay in the table with
    their flags so the exclusion is visible rather than silent.
    """
    frame = eye_state.copy()
    ts = frame["timestamp_ns"].to_numpy(np.int64)
    closed_mask = (np.asarray(closed, bool) if closed is not None
                   else np.zeros(ts.size, bool))
    frame["eye_closed"] = closed_mask

    if als_timestamp_ns is not None and als_lux is not None:
        lux, dt_s = attach_nearest(ts, als_timestamp_ns, als_lux, max_light_dt_s)
        frame["lux"] = lux
        frame["lux_dt_s"] = dt_s
    else:
        frame["lux"] = np.nan
        frame["lux_dt_s"] = np.nan

    for side in ("left", "right"):
        raw = frame[f"{side}_pupil_diameter_meter"].to_numpy(float)
        usable = np.where(
            frame[f"{side}_pupil_diameter_valid"].to_numpy(bool) & ~closed_mask,
            raw, np.nan)
        frame[f"{side}_pupil_usable_m"] = usable
        frame[f"{side}_pupil_baseline_m"] = local_baseline(ts, usable,
                                                           baseline_window_s)
        frame[f"{side}_pupil_delta_baseline_m"] = (
            usable - frame[f"{side}_pupil_baseline_m"].to_numpy(float))

    left = frame["left_pupil_usable_m"].to_numpy(float)
    right = frame["right_pupil_usable_m"].to_numpy(float)
    # Mean of whichever eyes are usable; NaN when neither is. Written out rather
    # than with nanmean, which warns on the all-invalid rows that blinks create.
    stacked = np.stack([left, right])
    usable = np.isfinite(stacked)
    count = usable.sum(axis=0)
    total = np.where(usable, stacked, 0.0).sum(axis=0)
    frame["pupil_mean_m"] = np.where(count > 0, total / np.maximum(count, 1), np.nan)
    frame["pupil_left_minus_right_m"] = left - right
    frame.attrs["baseline_window_s"] = float(baseline_window_s)
    frame.attrs["max_light_dt_s"] = float(max_light_dt_s)
    return frame


def add_light_residual(frame, column: str = "pupil_mean_m") -> Dict[str, Any]:
    """Fit and subtract the light term in place; returns the fitted model."""
    model = fit_log_lux_model(frame[column].to_numpy(float),
                              frame["lux"].to_numpy(float))
    frame["pupil_light_adjusted_residual_m"] = apply_log_lux_model(
        frame[column].to_numpy(float), frame["lux"].to_numpy(float), model)
    frame.attrs["light_model"] = model
    return model


# --------------------------------------------------------------------------- #
# Summaries
# --------------------------------------------------------------------------- #
def _describe(values: Sequence[float], scale: float = 1000.0
              ) -> Dict[str, Optional[float]]:
    """Descriptive statistics in millimetres, from metres."""
    v = np.asarray(values, float)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return {"n": 0, "median_mm": None, "iqr_mm": None, "p05_mm": None,
                "p95_mm": None, "mean_mm": None, "sd_mm": None, "cv": None}
    q1, q3 = np.percentile(v, [25, 75])
    mean = float(v.mean())
    sd = float(v.std(ddof=1)) if v.size > 1 else None
    return {
        "n": int(v.size),
        "median_mm": float(np.median(v)) * scale,
        "iqr_mm": float(q3 - q1) * scale,
        "p05_mm": float(np.percentile(v, 5)) * scale,
        "p95_mm": float(np.percentile(v, 95)) * scale,
        "mean_mm": mean * scale,
        "sd_mm": sd * scale if sd is not None else None,
        "cv": float(sd / mean) if sd is not None and abs(mean) > 1e-12 else None,
    }


def pupil_metrics(frame, model: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Per-eye diameter, variability, asymmetry, coverage and light coupling."""
    total = int(len(frame))
    per_eye: Dict[str, Any] = {}
    for side in ("left", "right"):
        device_valid = frame[f"{side}_pupil_diameter_valid"].to_numpy(bool)
        usable = frame[f"{side}_pupil_usable_m"].to_numpy(float)
        per_eye[side] = {
            **_describe(usable),
            "device_valid_fraction": float(device_valid.mean()) if total else None,
            "usable_fraction": (float(np.isfinite(usable).sum() / total)
                                if total else None),
            "excluded_by_eye_closure": int(
                (device_valid & ~np.isfinite(usable)).sum()),
        }

    difference = frame["pupil_left_minus_right_m"].to_numpy(float)
    delta = {side: _describe(
        frame[f"{side}_pupil_delta_baseline_m"].to_numpy(float))
        for side in ("left", "right")}

    lux = frame["lux"].to_numpy(float)
    mean_d = frame["pupil_mean_m"].to_numpy(float)
    both = np.isfinite(lux) & np.isfinite(mean_d)
    light_relation: Dict[str, Any] = {
        "paired_samples": int(both.sum()),
        "lux_coverage_fraction": float(np.isfinite(lux).mean()) if total else None,
        "lux": {
            "median": float(np.nanmedian(lux)) if np.isfinite(lux).any() else None,
            "p05": float(np.nanpercentile(lux, 5)) if np.isfinite(lux).any() else None,
            "p95": float(np.nanpercentile(lux, 95)) if np.isfinite(lux).any() else None,
        },
        "spearman_diameter_vs_lux": (_spearman(mean_d[both], lux[both])
                                     if both.sum() >= 32 else None),
        "sensor_caveat": ("the ambient-light sensor is head-mounted, so it "
                          "measures where the wearer is looking as much as the "
                          "scene's illumination"),
    }

    return {
        "schema": "article1_pupil_metrics_v1",
        "samples": total,
        "baseline_window_s": float(frame.attrs.get("baseline_window_s", 0.0)),
        "per_eye": per_eye,
        "left_minus_right": {
            **_describe(difference),
            "note": ("a left-right difference of this size is within the tracker's "
                     "own per-eye error; it is reported as a measurement, and no "
                     "anisocoria or any other clinical reading is implied"),
        },
        "deviation_from_local_baseline": delta,
        "mean_of_eyes": _describe(mean_d),
        "light_relation": light_relation,
        "light_model": model or frame.attrs.get("light_model"),
        "light_adjusted_residual": _describe(
            frame["pupil_light_adjusted_residual_m"].to_numpy(float))
        if "pupil_light_adjusted_residual_m" in frame else None,
        "interpolated": False,
        "clinical_interpretation": CLINICAL_NOTE,
    }


def _spearman(a: np.ndarray, b: np.ndarray) -> float:
    """Rank correlation without a scipy dependency."""
    return float(np.corrcoef(_rank(a), _rank(b))[0, 1])


def _rank(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="stable")
    ranks = np.empty(values.size, float)
    ranks[order] = np.arange(values.size, dtype=float)
    return ranks


def pupil_in_window(frame, start_ns: int, end_ns: int,
                    column: str = "pupil_light_adjusted_residual_m"
                    ) -> Dict[str, Optional[float]]:
    """Pupil summary inside one time window, for event-related analysis."""
    ts = frame["timestamp_ns"].to_numpy(np.int64)
    sel = (ts >= int(start_ns)) & (ts < int(end_ns))
    raw = frame["pupil_mean_m"].to_numpy(float)[sel]
    raw = raw[np.isfinite(raw)]
    adjusted = (frame[column].to_numpy(float)[sel]
                if column in frame else np.zeros(0))
    adjusted = adjusted[np.isfinite(adjusted)]
    return {
        "samples": int(sel.sum()),
        "usable_samples": int(raw.size),
        "median_diameter_mm": float(np.median(raw)) * 1000.0 if raw.size else None,
        "median_light_adjusted_residual_mm": (float(np.median(adjusted)) * 1000.0
                                              if adjusted.size else None),
    }
