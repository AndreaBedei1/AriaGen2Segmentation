"""PPG: quality gate first, beats second, heart-rate variability only if earned.

This is a **physiological proxy**, not a medical measurement. The sensor is a
consumer photoplethysmograph on a pair of glasses worn by someone riding a
motorcycle. Nothing here supports a clinical statement, and the vocabulary
downstream is deliberately "autonomic response candidate" and "physiological
arousal proxy" rather than "stress".

The gate order matters. Every beat is checked before it becomes an interval,
every interval before it becomes a heart rate, and every window before it becomes
a variability figure. HRV in particular is only computed where the window is long
enough for the statistic to mean anything: 30 s minimum, 60 s for SDNN, and pNN50
only with enough beats to have a resolution better than the quantity it reports.

Motion artefact is judged from the head IMU rather than from the PPG alone,
because a motion artefact and a genuine perfusion change look alike in one
channel and quite different across two.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from ..ingestion.timeline import NS_PER_S


@dataclass
class PpgQuality:
    """Per-window signal quality, and the reason a window failed."""

    start_ns: np.ndarray
    end_ns: np.ndarray
    sqi: np.ndarray
    sqi_band_power: np.ndarray
    sqi_template: np.ndarray
    sqi_regularity: np.ndarray
    saturated_fraction: np.ndarray
    dropout_fraction: np.ndarray
    motion_artefact: np.ndarray
    motion_accel_std: np.ndarray
    usable: np.ndarray
    reason: List[str]

    def to_frame(self):
        import pandas as pd
        return pd.DataFrame({
            "start_ns": self.start_ns, "end_ns": self.end_ns,
            "sqi": self.sqi, "sqi_band_power": self.sqi_band_power,
            "sqi_template": self.sqi_template, "sqi_regularity": self.sqi_regularity,
            "saturated_fraction": self.saturated_fraction,
            "dropout_fraction": self.dropout_fraction,
            "motion_artefact": self.motion_artefact,
            "motion_accel_std_msec2": self.motion_accel_std,
            "usable": self.usable, "reason": self.reason,
            "interpretation": "physiological_proxy_not_medical",
        })


@dataclass
class Beats:
    """Detected beats and the intervals between them."""

    timestamp_ns: np.ndarray
    amplitude: np.ndarray
    ibi_ms: np.ndarray                # interval to the PREVIOUS beat
    heart_rate_bpm: np.ndarray
    valid: np.ndarray
    reason: List[str]
    sqi_at_beat: np.ndarray

    def __len__(self) -> int:
        return int(self.timestamp_ns.size)

    def to_frame(self):
        import pandas as pd
        return pd.DataFrame({
            "timestamp_ns": self.timestamp_ns, "amplitude": self.amplitude,
            "ibi_ms": self.ibi_ms, "heart_rate_bpm": self.heart_rate_bpm,
            "valid": self.valid, "reason": self.reason,
            "sqi_at_beat": self.sqi_at_beat,
            "interpretation": "physiological_proxy_not_medical",
        })


def _bandpass(x: np.ndarray, fs: float, band: Tuple[float, float],
              order: int = 3) -> np.ndarray:
    from scipy.signal import butter, sosfiltfilt
    nyq = 0.5 * fs
    lo, hi = band[0] / nyq, min(band[1] / nyq, 0.99)
    if not (0 < lo < hi < 1):
        return np.full_like(np.asarray(x, float), np.nan)
    sos = butter(order, [lo, hi], btype="band", output="sos")
    y = np.asarray(x, float).copy()
    bad = ~np.isfinite(y)
    if bad.all():
        return np.full_like(y, np.nan)
    if bad.any():
        y[bad] = np.interp(np.flatnonzero(bad), np.flatnonzero(~bad), y[~bad])
    out = sosfiltfilt(sos, y)
    out[bad] = np.nan
    return out


def detect_beats(timestamp_ns: Sequence[int], value: Sequence[float],
                 bandpass_hz: Tuple[float, float] = (0.5, 8.0),
                 min_hr_bpm: float = 40.0,
                 max_hr_bpm: float = 180.0) -> Tuple[Beats, np.ndarray, float]:
    """Detect pulse peaks on the band-passed signal.

    Returns the beats, the filtered signal and the measured sampling rate. An
    interval implying a heart rate outside the physiological gate is kept but
    marked invalid with its reason, so a reviewer can see what was rejected.
    """
    from scipy.signal import find_peaks

    ts = np.asarray(timestamp_ns, dtype=np.int64)
    v = np.asarray(value, float)
    if ts.size < 16:
        return (Beats(np.zeros(0, np.int64), np.zeros(0), np.zeros(0), np.zeros(0),
                      np.zeros(0, bool), [], np.zeros(0)), np.zeros(0), float("nan"))

    dt = np.diff(ts) / NS_PER_S
    fs = float(1.0 / np.median(dt[dt > 0])) if np.any(dt > 0) else float("nan")
    if not np.isfinite(fs) or fs <= 0:
        raise ValueError("cannot measure the PPG sampling rate from its timestamps")

    filtered = _bandpass(v, fs, bandpass_hz)
    # Minimum spacing between peaks comes from the maximum plausible heart rate,
    # expressed in seconds and converted to samples for THIS recording's rate.
    min_distance = max(1, int(round(fs * 60.0 / float(max_hr_bpm))))
    finite = np.isfinite(filtered)
    work = np.where(finite, filtered, 0.0)
    prominence = float(np.nanstd(filtered[finite])) * 0.3 if finite.any() else 0.0
    peaks, props = find_peaks(work, distance=min_distance,
                              prominence=max(prominence, 1e-9))

    beat_ts = ts[peaks]
    amp = np.asarray(props.get("prominences", np.full(peaks.size, np.nan)), float)
    ibi = np.full(peaks.size, np.nan)
    if peaks.size >= 2:
        ibi[1:] = np.diff(beat_ts) / 1e6
    hr = np.where(np.isfinite(ibi) & (ibi > 0), 60_000.0 / ibi, np.nan)

    valid = np.isfinite(hr) & (hr >= min_hr_bpm) & (hr <= max_hr_bpm)
    reason = []
    for i in range(peaks.size):
        if not np.isfinite(ibi[i]):
            reason.append("first beat: no preceding interval")
        elif hr[i] < min_hr_bpm:
            reason.append(f"implied {hr[i]:.0f} bpm below the {min_hr_bpm:.0f} gate")
        elif hr[i] > max_hr_bpm:
            reason.append(f"implied {hr[i]:.0f} bpm above the {max_hr_bpm:.0f} gate")
        else:
            reason.append("ok")

    return (Beats(beat_ts, amp, ibi, hr, valid, reason, np.full(peaks.size, np.nan)),
            filtered, fs)


def _template_sqi(filtered: np.ndarray, peaks_idx: np.ndarray, fs: float) -> float:
    """Mean correlation of individual pulses with the window's average pulse.

    A clean PPG repeats nearly the same waveform every beat; motion noise does
    not. This is the single most informative quality statistic for the signal.
    """
    if peaks_idx.size < 3 or not np.isfinite(fs):
        return float("nan")
    half = int(round(0.3 * fs))
    segs = []
    for p in peaks_idx:
        a, b = p - half, p + half
        if a < 0 or b >= filtered.size:
            continue
        seg = filtered[a:b]
        if not np.isfinite(seg).all():
            continue
        sd = np.std(seg)
        if sd <= 0:
            continue
        segs.append((seg - np.mean(seg)) / sd)
    if len(segs) < 3:
        return float("nan")
    arr = np.asarray(segs)
    template = arr.mean(axis=0)
    tsd = np.std(template)
    if tsd <= 0:
        return float("nan")
    template = (template - template.mean()) / tsd
    corr = [float(np.dot(s, template) / s.size) for s in arr]
    return float(np.clip(np.mean(corr), 0.0, 1.0))


def assess_quality(timestamp_ns: Sequence[int], value: Sequence[float],
                   filtered: np.ndarray, fs: float, beats: Beats,
                   window_s: float = 10.0,
                   bandpass_hz: Tuple[float, float] = (0.5, 8.0),
                   min_sqi: float = 0.5,
                   saturation_fraction_limit: float = 0.02,
                   imu_timestamp_ns: Optional[Sequence[int]] = None,
                   imu_accel_magnitude: Optional[Sequence[float]] = None,
                   motion_accel_std_threshold: float = 2.0) -> PpgQuality:
    """Sliding-window quality assessment; nothing downstream skips this."""
    ts = np.asarray(timestamp_ns, dtype=np.int64)
    v = np.asarray(value, float)
    if ts.size == 0:
        return PpgQuality(*[np.zeros(0) for _ in range(9)], np.zeros(0, bool), [])

    step_ns = int(window_s * NS_PER_S)
    starts = np.arange(int(ts[0]), int(ts[-1]), step_ns, dtype=np.int64)
    n = starts.size

    # Saturation / clipping: the ADC pinned at an extreme value it repeats.
    vmin, vmax = np.nanmin(v), np.nanmax(v)
    pinned = (np.isclose(v, vmin) | np.isclose(v, vmax))

    imu_ts = (np.asarray(imu_timestamp_ns, np.int64)
              if imu_timestamp_ns is not None else None)
    imu_mag = (np.asarray(imu_accel_magnitude, float)
               if imu_accel_magnitude is not None else None)

    out = {k: np.full(n, np.nan) for k in
           ("sqi", "band", "template", "regularity", "sat", "drop", "accel_std")}
    motion = np.zeros(n, bool)
    usable = np.zeros(n, bool)
    reasons: List[str] = []

    nominal_dt = 1.0 / fs if np.isfinite(fs) and fs > 0 else np.nan
    for i, s0 in enumerate(starts):
        s1 = s0 + step_ns
        lo = int(np.searchsorted(ts, s0, "left"))
        hi = int(np.searchsorted(ts, s1, "right"))
        seg_ts, seg_v = ts[lo:hi], v[lo:hi]
        seg_f = filtered[lo:hi]

        expected = window_s / nominal_dt if np.isfinite(nominal_dt) else np.nan
        out["drop"][i] = (float(max(0.0, 1.0 - seg_ts.size / expected))
                          if np.isfinite(expected) and expected > 0 else np.nan)
        out["sat"][i] = (float(np.mean(pinned[lo:hi])) if seg_v.size else np.nan)

        if seg_f.size < int(2 * fs):
            reasons.append("window too short after dropout")
            continue

        # Band power: how much of the signal lives where a pulse lives.
        spec = np.abs(np.fft.rfft(np.nan_to_num(seg_f - np.nanmean(seg_f)))) ** 2
        freqs = np.fft.rfftfreq(seg_f.size, d=1.0 / fs)
        total = float(spec.sum())
        band = float(spec[(freqs >= bandpass_hz[0]) & (freqs <= bandpass_hz[1])].sum())
        out["band"][i] = float(band / total) if total > 0 else np.nan

        in_win = (beats.timestamp_ns >= s0) & (beats.timestamp_ns < s1)
        good = in_win & beats.valid
        ibis = beats.ibi_ms[good]
        ibis = ibis[np.isfinite(ibis)]
        out["regularity"][i] = (
            float(np.clip(1.0 - np.std(ibis) / max(np.mean(ibis), 1e-9), 0.0, 1.0))
            if ibis.size >= 3 else np.nan)

        peak_idx = np.searchsorted(seg_ts, beats.timestamp_ns[in_win])
        peak_idx = peak_idx[(peak_idx > 0) & (peak_idx < seg_f.size)]
        out["template"][i] = _template_sqi(seg_f, peak_idx, fs)

        parts = [out[k][i] for k in ("band", "template", "regularity")]
        finite = [p for p in parts if np.isfinite(p)]
        out["sqi"][i] = (float(np.prod(finite) ** (1.0 / len(finite)))
                         if finite else np.nan)

        if imu_ts is not None and imu_mag is not None:
            a = int(np.searchsorted(imu_ts, s0, "left"))
            b = int(np.searchsorted(imu_ts, s1, "right"))
            if b > a:
                out["accel_std"][i] = float(np.nanstd(imu_mag[a:b]))
                motion[i] = bool(out["accel_std"][i] > motion_accel_std_threshold)

        why: List[str] = []
        if np.isfinite(out["sat"][i]) and out["sat"][i] > saturation_fraction_limit:
            why.append(f"saturated {out['sat'][i] * 100:.1f}% of samples")
        if np.isfinite(out["drop"][i]) and out["drop"][i] > 0.1:
            why.append(f"{out['drop'][i] * 100:.0f}% of samples missing")
        if not np.isfinite(out["sqi"][i]):
            why.append("signal quality index not computable")
        elif out["sqi"][i] < min_sqi:
            why.append(f"SQI {out['sqi'][i]:.2f} below {min_sqi}")
        if motion[i]:
            why.append(f"head motion artefact: accel std "
                       f"{out['accel_std'][i]:.1f} m/s^2")
        usable[i] = not why
        reasons.append("; ".join(why) or "ok")

    while len(reasons) < n:
        reasons.append("not assessed")

    return PpgQuality(
        start_ns=starts, end_ns=starts + step_ns, sqi=out["sqi"],
        sqi_band_power=out["band"], sqi_template=out["template"],
        sqi_regularity=out["regularity"], saturated_fraction=out["sat"],
        dropout_fraction=out["drop"], motion_artefact=motion,
        motion_accel_std=out["accel_std"], usable=usable, reason=reasons)


def tag_beats_with_quality(beats: Beats, quality: PpgQuality) -> Beats:
    """Carry each window's SQI onto the beats inside it, and drop unusable ones."""
    if len(beats) == 0 or quality.start_ns.size == 0:
        return beats
    idx = np.clip(np.searchsorted(quality.start_ns, beats.timestamp_ns,
                                  side="right") - 1, 0, quality.start_ns.size - 1)
    beats.sqi_at_beat = quality.sqi[idx]
    unusable = ~quality.usable[idx]
    for i in np.flatnonzero(unusable & beats.valid):
        beats.reason[i] = f"window rejected: {quality.reason[idx[i]]}"
    beats.valid = beats.valid & ~unusable
    return beats


# --------------------------------------------------------------------------- #
# Heart rate and variability
# --------------------------------------------------------------------------- #
def hrv_window(beats: Beats, start_ns: int, end_ns: int,
               min_window_s: float = 30.0,
               sdnn_min_window_s: float = 60.0,
               min_valid_beat_fraction: float = 0.8,
               pnn50_min_beats: int = 50) -> Dict[str, Any]:
    """Heart-rate and variability statistics for one window, with refusals.

    Every variability statistic that the window cannot support is returned as
    `None` next to the reason, rather than computed on too little data. A short
    window does not give a noisy SDNN; it gives a meaningless one.
    """
    duration_s = float((end_ns - start_ns) / NS_PER_S)
    in_win = (beats.timestamp_ns >= start_ns) & (beats.timestamp_ns < end_ns)
    total = int(in_win.sum())
    good = in_win & beats.valid
    ibis = beats.ibi_ms[good]
    ibis = ibis[np.isfinite(ibis)]
    hr = beats.heart_rate_bpm[good]
    hr = hr[np.isfinite(hr)]
    frac = float(good.sum() / total) if total else 0.0

    out: Dict[str, Any] = {
        "start_ns": int(start_ns), "end_ns": int(end_ns),
        "duration_s": duration_s,
        "beats_total": total, "beats_valid": int(good.sum()),
        "valid_beat_fraction": frac,
        "invalid_beat_rate_per_minute": (
            float((total - int(good.sum())) * 60.0 / duration_s)
            if duration_s > 0 else None),
        "heart_rate_bpm": None, "heart_rate_median_bpm": None,
        "heart_rate_min_bpm": None, "heart_rate_max_bpm": None,
        "heart_rate_p05_bpm": None, "heart_rate_p95_bpm": None,
        "mean_ibi_ms": None,
        "pulse_amplitude": None, "pulse_amplitude_variability": None,
        "rmssd_ms": None, "sdnn_ms": None, "pnn50": None,
        "refusals": {},
        "interpretation": "physiological_proxy_not_medical",
    }

    if hr.size >= 3 and frac >= min_valid_beat_fraction:
        out["heart_rate_bpm"] = float(np.mean(hr))
        out["heart_rate_median_bpm"] = float(np.median(hr))
        out["heart_rate_min_bpm"] = float(np.min(hr))
        out["heart_rate_max_bpm"] = float(np.max(hr))
        out["heart_rate_p05_bpm"] = float(np.percentile(hr, 5))
        out["heart_rate_p95_bpm"] = float(np.percentile(hr, 95))
        out["mean_ibi_ms"] = float(np.mean(ibis)) if ibis.size else None
        amp = beats.amplitude[good]
        amp = amp[np.isfinite(amp)]
        if amp.size >= 3:
            out["pulse_amplitude"] = float(np.median(amp))
            out["pulse_amplitude_variability"] = float(
                np.std(amp) / max(np.mean(amp), 1e-9))
    else:
        out["refusals"]["heart_rate"] = (
            f"{hr.size} valid beats and a {frac:.0%} valid fraction; needs at "
            f"least 3 beats and {min_valid_beat_fraction:.0%}")

    # --- variability, gated on window length --------------------------------
    if duration_s < min_window_s:
        out["refusals"]["rmssd"] = (
            f"window is {duration_s:.0f} s, below the {min_window_s:.0f} s minimum")
        out["refusals"]["sdnn"] = out["refusals"]["rmssd"]
    elif frac < min_valid_beat_fraction:
        out["refusals"]["rmssd"] = (
            f"only {frac:.0%} of beats in the window passed the quality gate")
        out["refusals"]["sdnn"] = out["refusals"]["rmssd"]
    elif ibis.size < 4:
        out["refusals"]["rmssd"] = f"only {ibis.size} usable intervals"
        out["refusals"]["sdnn"] = out["refusals"]["rmssd"]
    else:
        out["rmssd_ms"] = float(np.sqrt(np.mean(np.diff(ibis) ** 2)))
        if duration_s >= sdnn_min_window_s:
            out["sdnn_ms"] = float(np.std(ibis, ddof=1))
        else:
            out["refusals"]["sdnn"] = (
                f"window is {duration_s:.0f} s; SDNN needs at least "
                f"{sdnn_min_window_s:.0f} s to describe anything")
        if ibis.size >= pnn50_min_beats:
            out["pnn50"] = float(np.mean(np.abs(np.diff(ibis)) > 50.0))
        else:
            out["refusals"]["pnn50"] = (
                f"{ibis.size} intervals; below {pnn50_min_beats} the statistic "
                "has a coarser resolution than the value it reports")
    return out


def rolling_hrv(beats: Beats, start_ns: int, end_ns: int,
                window_s: float = 60.0, step_s: float = 30.0,
                **kwargs) -> List[Dict[str, Any]]:
    """HRV over sliding windows across a recording."""
    out = []
    w = int(window_s * NS_PER_S)
    s = int(step_s * NS_PER_S)
    for t0 in range(int(start_ns), int(end_ns) - w + 1, s):
        out.append(hrv_window(beats, t0, t0 + w, **kwargs))
    return out


def beat_detection_diagnostics(beats: Beats) -> Dict[str, Any]:
    """Check whether the detector is counting each pulse once.

    The classic PPG failure is picking up the dicrotic notch as a second beat,
    which roughly doubles the reported heart rate. It leaves a specific
    signature: intervals alternate short-long, so the lag-1 autocorrelation of
    the interval series goes strongly **negative**, and every second peak is much
    smaller than its neighbours.

    An elevated heart rate that passes this check is not proof it is real, but a
    heart rate that fails it is certainly not.
    """
    ok = beats.valid & np.isfinite(beats.ibi_ms)
    ibi = beats.ibi_ms[ok]
    amp = beats.amplitude[ok]
    out: Dict[str, Any] = {
        "valid_intervals": int(ibi.size),
        "ibi_lag1_autocorrelation": None,
        "amplitude_coefficient_of_variation": None,
        "small_amplitude_beat_fraction": None,
        "alternation_suggests_double_detection": None,
        "note": ("a strongly negative lag-1 interval autocorrelation is the "
                 "signature of counting the dicrotic notch as a beat"),
    }
    if ibi.size < 8:
        out["note"] = "too few intervals to run the double-detection check"
        return out

    r1 = float(np.corrcoef(ibi[:-1], ibi[1:])[0, 1])
    out["ibi_lag1_autocorrelation"] = r1
    out["alternation_suggests_double_detection"] = bool(r1 < -0.3)

    finite_amp = amp[np.isfinite(amp)]
    if finite_amp.size >= 8:
        out["amplitude_coefficient_of_variation"] = float(
            np.std(finite_amp) / max(np.mean(finite_amp), 1e-9))
        half = max(2, min(10, finite_amp.size // 4))
        local = np.array([np.median(finite_amp[max(0, i - half):i + half + 1])
                          for i in range(finite_amp.size)])
        out["small_amplitude_beat_fraction"] = float(
            np.mean(finite_amp < 0.5 * local))
    return out


def ppg_summary(beats: Beats, quality: PpgQuality,
                duration_s: float) -> Dict[str, Any]:
    """Recording-level PPG usability. The headline number is usable fraction."""
    usable_windows = int(quality.usable.sum())
    total_windows = int(quality.usable.size)
    reasons: Dict[str, int] = {}
    for ok, r in zip(quality.usable, quality.reason):
        if not ok:
            head = str(r).split(";")[0].strip()
            reasons[head] = reasons.get(head, 0) + 1
    hr = beats.heart_rate_bpm[beats.valid]
    hr = hr[np.isfinite(hr)]
    return {
        "interpretation": "physiological_proxy_not_medical",
        "duration_s": duration_s,
        "beats_detected": len(beats),
        "beats_passing_gate": int(beats.valid.sum()),
        "beat_pass_fraction": (float(beats.valid.mean()) if len(beats) else 0.0),
        "quality_windows": total_windows,
        "usable_windows": usable_windows,
        "usable_fraction": (usable_windows / total_windows) if total_windows else 0.0,
        "rejection_reasons": reasons,
        "beat_detection_diagnostics": beat_detection_diagnostics(beats),
        "median_sqi": (float(np.nanmedian(quality.sqi))
                       if np.isfinite(quality.sqi).any() else None),
        "motion_artefact_window_fraction": (float(np.mean(quality.motion_artefact))
                                            if total_windows else None),
        "heart_rate_bpm": {
            "count": int(hr.size),
            "mean": float(np.mean(hr)) if hr.size else None,
            "median": float(np.median(hr)) if hr.size else None,
            "p05": float(np.percentile(hr, 5)) if hr.size else None,
            "p95": float(np.percentile(hr, 95)) if hr.size else None,
        },
        "caveat": ("PPG is an optical perfusion proxy from a consumer sensor on "
                   "moving glasses. It carries no diagnostic meaning and an "
                   "increase in heart rate is not evidence of stress."),
    }
