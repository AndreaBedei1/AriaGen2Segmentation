"""Two separate families of motion: the vehicle's, and the rider's head.

They must never be mixed. The only inertial sensors in this recording sit on the
temples of a pair of glasses. What they measure is gravity, plus how the head
moves on the neck, plus how the vehicle shakes the whole assembly — three things
superimposed. Reading a longitudinal vehicle acceleration off that signal would be
wrong in a way that looks plausible, which is the dangerous kind.

So:

* **vehicle dynamics** come from position and speed — here, from GPS, because
  this device recorded no usable VIO/SLAM pose stream. Every value is labelled
  with the source it came from;
* **head dynamics** come from the IMU, and are described as head motion, never as
  vehicle motion.

The two are computed by different functions, stored in different tables and
compared side by side rather than merged.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from ..ingestion.timeline import NS_PER_S

#: Standard gravity, used only to sanity-check the accelerometer magnitude.
G_MSEC2 = 9.80665


# --------------------------------------------------------------------------- #
# Vehicle dynamics
# --------------------------------------------------------------------------- #
@dataclass
class VehicleDynamics:
    """Vehicle motion, per real GPS fix. Never derived from the head IMU."""

    timestamp_ns: np.ndarray
    speed_source: str
    speed_mps: np.ndarray
    speed_raw_mps: np.ndarray
    speed_from_position_mps: np.ndarray
    acceleration_mps2: np.ndarray
    acceleration_raw_mps2: np.ndarray
    jerk_mps3: np.ndarray
    lateral_acceleration_mps2: np.ndarray
    curvature_1_per_m: np.ndarray
    heading_deg: np.ndarray
    yaw_rate_deg_s: np.ndarray
    is_stopped: np.ndarray
    valid: np.ndarray
    smoothing_window_s: float = 0.0
    notes: List[str] = field(default_factory=list)

    def to_frame(self):
        import pandas as pd
        return pd.DataFrame({
            "timestamp_ns": self.timestamp_ns,
            "speed_source": self.speed_source,
            "speed_mps": self.speed_mps,
            "speed_raw_mps": self.speed_raw_mps,
            "speed_from_position_mps": self.speed_from_position_mps,
            "acceleration_mps2": self.acceleration_mps2,
            "acceleration_raw_mps2": self.acceleration_raw_mps2,
            "jerk_mps3": self.jerk_mps3,
            "lateral_acceleration_mps2": self.lateral_acceleration_mps2,
            "curvature_1_per_m": self.curvature_1_per_m,
            "heading_deg": self.heading_deg,
            "yaw_rate_deg_s": self.yaw_rate_deg_s,
            "is_stopped": self.is_stopped,
            "valid": self.valid,
            "measures": "vehicle_motion",
            "derived_from_head_imu": False,
        })


def _moving_average(values: np.ndarray, t_s: np.ndarray,
                    window_s: float) -> np.ndarray:
    """Centred moving average over a window in SECONDS, not in samples.

    A sample-count window would mean a different duration in each recording; a
    duration window means the same physical smoothing everywhere.
    """
    v = np.asarray(values, float)
    t = np.asarray(t_s, float)
    out = np.full(v.shape, np.nan)
    if v.size == 0 or window_s <= 0:
        return v.copy()
    half = 0.5 * float(window_s)
    lo = np.searchsorted(t, t - half, side="left")
    hi = np.searchsorted(t, t + half, side="right")
    finite = np.isfinite(v)
    filled = np.where(finite, v, 0.0)
    csum = np.concatenate([[0.0], np.cumsum(filled)])
    ccnt = np.concatenate([[0.0], np.cumsum(finite.astype(float))])
    n = ccnt[hi] - ccnt[lo]
    with np.errstate(invalid="ignore", divide="ignore"):
        out = np.where(n > 0, (csum[hi] - csum[lo]) / np.maximum(n, 1e-9), np.nan)
    return out


def _gradient(values: np.ndarray, t_s: np.ndarray) -> np.ndarray:
    """d(value)/dt on a possibly irregular time base."""
    v = np.asarray(values, float)
    t = np.asarray(t_s, float)
    if v.size < 2:
        return np.full(v.shape, np.nan)
    out = np.full(v.shape, np.nan)
    ok = np.isfinite(v) & np.isfinite(t)
    if ok.sum() < 2:
        return out
    out[ok] = np.gradient(v[ok], t[ok])
    return out


def compute_vehicle_dynamics(timestamp_ns: Sequence[int],
                             x_m: Sequence[float],
                             y_m: Sequence[float],
                             gps_speed_mps: Optional[Sequence[float]] = None,
                             curvature_1_per_m: Optional[Sequence[float]] = None,
                             heading_deg: Optional[Sequence[float]] = None,
                             speed_priority: Sequence[str] = ("vio_pose",
                                                              "gps_speed_field",
                                                              "gps_position_derivative"),
                             vio_speed_mps: Optional[Sequence[float]] = None,
                             smoothing_window_s: float = 1.0,
                             stopped_speed_mps: float = 0.5,
                             max_plausible_speed_mps: float = 60.0,
                             ) -> VehicleDynamics:
    """Vehicle speed, acceleration, jerk and lateral acceleration.

    The speed source is chosen by the configured priority and recorded on every
    row, so a reader never has to guess whether a number came from a pose stream,
    from the receiver's own Doppler speed, or from differencing positions.
    """
    ts = np.asarray(timestamp_ns, dtype=np.int64)
    t_s = ts / NS_PER_S
    x = np.asarray(x_m, float)
    y = np.asarray(y_m, float)
    n = ts.size
    notes: List[str] = []

    # Position-derivative speed is always computed, even when it is not chosen:
    # it is the independent cross-check on the receiver's own speed field.
    speed_pos = np.full(n, np.nan)
    if n >= 2:
        dt = np.diff(t_s)
        step = np.hypot(np.diff(x), np.diff(y))
        with np.errstate(divide="ignore", invalid="ignore"):
            v = np.where(dt > 0, step / dt, np.nan)
        speed_pos[:-1] = v
        speed_pos[-1] = v[-1] if v.size else np.nan

    candidates = {
        "vio_pose": (np.asarray(vio_speed_mps, float)
                     if vio_speed_mps is not None else None),
        "gps_speed_field": (np.asarray(gps_speed_mps, float)
                            if gps_speed_mps is not None else None),
        "gps_position_derivative": speed_pos,
    }
    source = None
    raw = None
    for name in speed_priority:
        cand = candidates.get(name)
        if cand is not None and np.isfinite(cand).any():
            source, raw = name, cand.astype(float).copy()
            break
    if raw is None:
        source, raw = "unavailable", np.full(n, np.nan)
        notes.append("no usable speed source: every candidate was empty")
    if source != "vio_pose":
        notes.append("no VIO/SLAM pose stream in this recording; vehicle speed "
                     f"falls back to {source}")

    implausible = raw > float(max_plausible_speed_mps)
    if implausible.any():
        notes.append(f"{int(implausible.sum())} speed samples above "
                     f"{max_plausible_speed_mps} m/s discarded as fix artefacts")
        raw = np.where(implausible, np.nan, raw)

    speed = _moving_average(raw, t_s, smoothing_window_s)
    accel_raw = _gradient(raw, t_s)
    accel = _gradient(speed, t_s)
    jerk = _gradient(accel, t_s)

    curv = (np.asarray(curvature_1_per_m, float) if curvature_1_per_m is not None
            else np.full(n, np.nan))
    # a_lat = v^2 * kappa. The curvature comes from the road geometry, so this is
    # the lateral acceleration the road demands at this speed, not a measurement.
    lateral = speed ** 2 * curv

    head = (np.asarray(heading_deg, float) if heading_deg is not None
            else np.full(n, np.nan))
    yaw_rate = np.full(n, np.nan)
    if n >= 2:
        unwrapped = np.degrees(np.unwrap(np.radians(head)))
        yaw_rate = _gradient(unwrapped, t_s)

    return VehicleDynamics(
        timestamp_ns=ts, speed_source=source,
        speed_mps=speed, speed_raw_mps=raw, speed_from_position_mps=speed_pos,
        acceleration_mps2=accel, acceleration_raw_mps2=accel_raw, jerk_mps3=jerk,
        lateral_acceleration_mps2=lateral, curvature_1_per_m=curv,
        heading_deg=head, yaw_rate_deg_s=yaw_rate,
        is_stopped=(speed < float(stopped_speed_mps)),
        valid=np.isfinite(speed),
        smoothing_window_s=float(smoothing_window_s), notes=notes,
    )


@dataclass
class Episode:
    """A contiguous stretch where a condition held."""

    start_ns: int
    end_ns: int
    duration_s: float
    peak_value: float
    mean_value: float

    def to_dict(self) -> Dict[str, Any]:
        return {"start_ns": self.start_ns, "end_ns": self.end_ns,
                "duration_s": self.duration_s, "peak_value": self.peak_value,
                "mean_value": self.mean_value}


def find_episodes(timestamp_ns: np.ndarray, values: np.ndarray,
                  condition: np.ndarray, min_duration_s: float = 0.0
                  ) -> List[Episode]:
    """Contiguous runs where `condition` holds, longer than a minimum duration."""
    ts = np.asarray(timestamp_ns, np.int64)
    v = np.asarray(values, float)
    c = np.asarray(condition, bool) & np.isfinite(v)
    out: List[Episode] = []
    if c.size == 0:
        return out
    edges = np.diff(c.astype(np.int8))
    starts = list(np.flatnonzero(edges == 1) + 1)
    ends = list(np.flatnonzero(edges == -1) + 1)
    if c[0]:
        starts.insert(0, 0)
    if c[-1]:
        ends.append(c.size)
    for s, e in zip(starts, ends):
        dur = float((ts[e - 1] - ts[s]) / NS_PER_S)
        if dur < min_duration_s:
            continue
        seg = v[s:e]
        out.append(Episode(
            start_ns=int(ts[s]), end_ns=int(ts[e - 1]), duration_s=dur,
            peak_value=float(np.nanmax(np.abs(seg)) * np.sign(np.nanmean(seg))),
            mean_value=float(np.nanmean(seg)),
        ))
    return out


def vehicle_dynamics_summary(dyn: VehicleDynamics,
                             hard_braking_mps2: float = -2.5,
                             strong_acceleration_mps2: float = 2.0,
                             min_episode_s: float = 0.5) -> Dict[str, Any]:
    """Descriptive statistics, all normalised per second where they are counts."""
    v = dyn.speed_mps
    a = dyn.acceleration_mps2
    ok = np.isfinite(v)
    duration_s = (float((dyn.timestamp_ns[-1] - dyn.timestamp_ns[0]) / NS_PER_S)
                  if dyn.timestamp_ns.size > 1 else 0.0)

    braking = find_episodes(dyn.timestamp_ns, a, a <= hard_braking_mps2, min_episode_s)
    accelerating = find_episodes(dyn.timestamp_ns, a, a >= strong_acceleration_mps2,
                                 min_episode_s)
    stopped = find_episodes(dyn.timestamp_ns, v, dyn.is_stopped, 0.0)

    cross = np.nan
    both = np.isfinite(dyn.speed_raw_mps) & np.isfinite(dyn.speed_from_position_mps)
    if both.sum() > 2:
        cross = float(np.corrcoef(dyn.speed_raw_mps[both],
                                  dyn.speed_from_position_mps[both])[0, 1])

    # What acceleration can this sampling rate actually see? A 1 Hz speed series
    # averages a brake application over a whole second, so a short, hard stop
    # appears as a modest mean deceleration. Reporting "0 hard-braking events"
    # without this number would read as a claim about the driver rather than
    # about the sensor.
    dt = np.diff(dyn.timestamp_ns) / NS_PER_S
    median_dt = float(np.median(dt)) if dt.size else float("nan")
    observed_extreme = (float(np.nanmin(dyn.acceleration_mps2))
                        if np.isfinite(dyn.acceleration_mps2).any() else np.nan)
    detectable = bool(np.isfinite(observed_extreme)
                      and observed_extreme <= hard_braking_mps2)

    return {
        "detectability": {
            "speed_sampling_interval_s": median_dt,
            "smoothing_window_s": dyn.smoothing_window_s,
            "hard_braking_threshold_mps2": hard_braking_mps2,
            "most_negative_observed_acceleration_mps2": observed_extreme,
            "threshold_reached_at_all": detectable,
            "caveat": (
                None if detectable else
                f"the most negative acceleration this {1 / median_dt:.1f} Hz speed "
                f"series resolves is {observed_extreme:.2f} m/s^2, above the "
                f"{hard_braking_mps2} m/s^2 threshold. A count of zero hard-braking "
                "episodes is therefore a limit of the sampling rate, not evidence "
                "that no hard braking occurred."),
        },
        "speed_source": dyn.speed_source,
        "derived_from_head_imu": False,
        "duration_s": duration_s,
        "valid_samples": int(ok.sum()),
        "speed_mps": _percentiles(v),
        "acceleration_mps2": _percentiles(a),
        "jerk_mps3": _percentiles(dyn.jerk_mps3),
        "lateral_acceleration_mps2": _percentiles(dyn.lateral_acceleration_mps2),
        "hard_braking_episodes": len(braking),
        "hard_braking_per_minute": _per_minute(len(braking), duration_s),
        "hard_braking_total_s": float(sum(e.duration_s for e in braking)),
        "strong_acceleration_episodes": len(accelerating),
        "strong_acceleration_per_minute": _per_minute(len(accelerating), duration_s),
        "strong_acceleration_total_s": float(sum(e.duration_s for e in accelerating)),
        "stopped_time_s": float(sum(e.duration_s for e in stopped)),
        "stopped_fraction": (float(sum(e.duration_s for e in stopped) / duration_s)
                             if duration_s > 0 else None),
        "stop_and_go_events": len([e for e in stopped if e.duration_s >= 1.0]),
        "speed_field_vs_position_derivative_r": cross,
        "notes": dyn.notes,
    }


def _percentiles(a: np.ndarray) -> Dict[str, Optional[float]]:
    a = np.asarray(a, float)
    a = a[np.isfinite(a)]
    if a.size == 0:
        return {k: None for k in ("count", "mean", "p05", "p25", "median",
                                  "p75", "p95", "min", "max")}
    return {
        "count": int(a.size), "mean": float(np.mean(a)),
        "p05": float(np.percentile(a, 5)), "p25": float(np.percentile(a, 25)),
        "median": float(np.median(a)), "p75": float(np.percentile(a, 75)),
        "p95": float(np.percentile(a, 95)),
        "min": float(np.min(a)), "max": float(np.max(a)),
    }


def _per_minute(count: int, duration_s: float) -> Optional[float]:
    """Counts are compared per unit TIME, never per frame."""
    return float(count * 60.0 / duration_s) if duration_s > 0 else None


# --------------------------------------------------------------------------- #
# Head dynamics
# --------------------------------------------------------------------------- #
@dataclass
class HeadDynamics:
    """Head motion from the glasses IMU. Explicitly not vehicle motion."""

    timestamp_ns: np.ndarray
    yaw_rate_rad_s: np.ndarray
    pitch_rate_rad_s: np.ndarray
    roll_rate_rad_s: np.ndarray
    angular_speed_rad_s: np.ndarray
    angular_acceleration_rad_s2: np.ndarray
    linear_accel_x: np.ndarray
    linear_accel_y: np.ndarray
    linear_accel_z: np.ndarray
    linear_accel_magnitude: np.ndarray
    head_jerk_m_s3: np.ndarray
    gravity_magnitude: np.ndarray
    head_pitch_deg: np.ndarray
    head_roll_deg: np.ndarray
    head_pitch_rel_deg: np.ndarray
    head_roll_rel_deg: np.ndarray
    vibration_rms: np.ndarray
    control_band_rms: np.ndarray
    meta: Dict[str, Any] = field(default_factory=dict)

    def to_frame(self):
        import pandas as pd
        return pd.DataFrame({
            "timestamp_ns": self.timestamp_ns,
            "yaw_rate_rad_s": self.yaw_rate_rad_s,
            "pitch_rate_rad_s": self.pitch_rate_rad_s,
            "roll_rate_rad_s": self.roll_rate_rad_s,
            "angular_speed_rad_s": self.angular_speed_rad_s,
            "angular_acceleration_rad_s2": self.angular_acceleration_rad_s2,
            "linear_accel_magnitude_msec2": self.linear_accel_magnitude,
            "head_jerk_m_s3": self.head_jerk_m_s3,
            "gravity_magnitude_msec2": self.gravity_magnitude,
            "head_pitch_deg": self.head_pitch_deg,
            "head_roll_deg": self.head_roll_deg,
            "head_pitch_rel_deg": self.head_pitch_rel_deg,
            "head_roll_rel_deg": self.head_roll_rel_deg,
            "vibration_rms_msec2": self.vibration_rms,
            "control_band_rms_rad_s": self.control_band_rms,
            "measures": "head_motion",
            "is_vehicle_acceleration": False,
        })


def _butter_filter(x: np.ndarray, fs: float, band: Tuple[float, float] | float,
                   btype: str, order: int = 4) -> np.ndarray:
    """Zero-phase Butterworth filter. Returns NaN-safe output."""
    from scipy.signal import butter, sosfiltfilt

    nyq = 0.5 * fs
    if btype in ("low", "high"):
        wn = float(band) / nyq
        if not (0 < wn < 1):
            return np.full_like(x, np.nan)
    else:
        lo, hi = float(band[0]) / nyq, float(band[1]) / nyq
        if not (0 < lo < hi < 1):
            return np.full_like(x, np.nan)
        wn = [lo, hi]
    sos = butter(order, wn, btype=btype, output="sos")
    y = np.asarray(x, float).copy()
    bad = ~np.isfinite(y)
    if bad.all():
        return np.full_like(y, np.nan)
    if bad.any():
        y[bad] = np.interp(np.flatnonzero(bad), np.flatnonzero(~bad), y[~bad])
    out = sosfiltfilt(sos, y)
    out[bad] = np.nan
    return out


def _windowed_rms(x: np.ndarray, fs: float, window_s: float) -> np.ndarray:
    """RMS in a centred window of `window_s` seconds, via cumulative sums."""
    v = np.asarray(x, float)
    half = max(1, int(round(0.5 * window_s * fs)))
    sq = np.where(np.isfinite(v), v ** 2, 0.0)
    ok = np.isfinite(v).astype(float)
    csq = np.concatenate([[0.0], np.cumsum(sq)])
    cok = np.concatenate([[0.0], np.cumsum(ok)])
    n = v.size
    lo = np.clip(np.arange(n) - half, 0, n)
    hi = np.clip(np.arange(n) + half + 1, 0, n)
    cnt = cok[hi] - cok[lo]
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.sqrt(np.where(cnt > 0, (csq[hi] - csq[lo]) / np.maximum(cnt, 1e-9),
                                np.nan))


def compute_head_dynamics(timestamp_ns: Sequence[int],
                          accel_msec2: np.ndarray,
                          gyro_rad_s: np.ndarray,
                          camera_forward_device: Optional[Sequence[float]] = None,
                          camera_right_device: Optional[Sequence[float]] = None,
                          gravity_cutoff_hz: float = 0.3,
                          vibration_band_hz: Tuple[float, float] = (5.0, 20.0),
                          control_band_hz: Tuple[float, float] = (0.2, 2.0),
                          rms_window_s: float = 1.0,
                          ) -> HeadDynamics:
    """Head motion, with gravity removed and the rotation axes resolved physically.

    Gravity is estimated as the low-passed accelerometer vector and subtracted, so
    `linear_accel_*` is specific force minus gravity — head motion, still not
    vehicle motion.

    Yaw, pitch and roll are resolved against a frame built from two physical
    directions rather than from a device-axis naming convention: **up** is the
    gravity direction, and **forward** is the RGB camera's optical axis projected
    perpendicular to it. Yaw is then rotation about the true vertical whatever way
    the glasses happen to sit.
    """
    ts = np.asarray(timestamp_ns, dtype=np.int64)
    t_s = ts / NS_PER_S
    acc = np.asarray(accel_msec2, float)
    gyr = np.asarray(gyro_rad_s, float)
    n = ts.size
    if n < 2:
        raise ValueError("head dynamics need at least two IMU samples")

    dt = np.diff(t_s)
    fs = float(1.0 / np.median(dt[dt > 0])) if np.any(dt > 0) else float("nan")
    if not np.isfinite(fs) or fs <= 0:
        raise ValueError("cannot measure the IMU sampling rate from its timestamps")

    # --- gravity and linear acceleration ------------------------------------
    gravity = np.column_stack([
        _butter_filter(acc[:, i], fs, gravity_cutoff_hz, "low") for i in range(3)])
    linear = acc - gravity
    g_mag = np.linalg.norm(gravity, axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        up = gravity / np.maximum(g_mag, 1e-9)[:, None]

    # --- body frame ---------------------------------------------------------
    fwd_ref = (np.asarray(camera_forward_device, float)
               if camera_forward_device is not None else np.array([0.0, 0.0, 1.0]))
    fwd_ref = fwd_ref / max(float(np.linalg.norm(fwd_ref)), 1e-9)
    proj = fwd_ref[None, :] - (up * (up @ fwd_ref)[:, None])
    proj_norm = np.linalg.norm(proj, axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        fwd = proj / np.maximum(proj_norm, 1e-9)[:, None]
    right = np.cross(fwd, up)

    yaw_rate = np.einsum("ij,ij->i", gyr, up)
    pitch_rate = np.einsum("ij,ij->i", gyr, right)
    roll_rate = np.einsum("ij,ij->i", gyr, fwd)
    ang_speed = np.linalg.norm(gyr, axis=1)
    ang_accel = _gradient(ang_speed, t_s)

    lin_mag = np.linalg.norm(linear, axis=1)
    jerk = _gradient(lin_mag, t_s)

    # Head attitude relative to the vertical. Both angles are how far a FIXED
    # device axis has tilted out of horizontal, measured against the gravity
    # direction — pitch for the optical axis, roll for the lateral axis.
    #
    # Roll cannot be built from `fwd` and `up` alone: `fwd` is by construction the
    # projection of `fwd_ref` perpendicular to `up`, so any vector formed from the
    # two lies in their common plane and its component along `up` is identically
    # zero. It needs the device's own lateral axis as an independent reference.
    right_ref = (np.asarray(camera_right_device, float)
                 if camera_right_device is not None else None)
    if right_ref is None or float(np.linalg.norm(right_ref)) < 1e-9:
        seed = np.array([0.0, 1.0, 0.0])
        if abs(float(seed @ fwd_ref)) > 0.9:
            seed = np.array([1.0, 0.0, 0.0])
        right_ref = np.cross(fwd_ref, seed)
    right_ref = right_ref / max(float(np.linalg.norm(right_ref)), 1e-9)
    # Remove any forward component so the lateral axis is genuinely lateral.
    right_ref = right_ref - fwd_ref * float(right_ref @ fwd_ref)
    right_ref = right_ref / max(float(np.linalg.norm(right_ref)), 1e-9)

    head_pitch = np.degrees(np.arcsin(np.clip(up @ fwd_ref, -1.0, 1.0)))
    head_roll = np.degrees(np.arcsin(np.clip(up @ right_ref, -1.0, 1.0)))

    # Those two absolute angles contain a large constant offset that is not
    # behaviour: the RGB sensor is mounted rotated on the device, and the glasses
    # then sit on a particular face at a particular angle. Only the *change* in
    # attitude is comparable across recordings, so the posture-relative angles are
    # computed here and are the ones the behavioural metrics use.
    pitch_ref = float(np.nanmedian(head_pitch))
    roll_ref = float(np.nanmedian(head_roll))
    head_pitch_rel = head_pitch - pitch_ref
    head_roll_rel = head_roll - roll_ref

    vib = _windowed_rms(
        _butter_filter(lin_mag - np.nanmean(lin_mag), fs, vibration_band_hz, "band"),
        fs, rms_window_s)
    ctrl = _windowed_rms(
        _butter_filter(yaw_rate, fs, control_band_hz, "band"), fs, rms_window_s)

    return HeadDynamics(
        timestamp_ns=ts,
        yaw_rate_rad_s=yaw_rate, pitch_rate_rad_s=pitch_rate,
        roll_rate_rad_s=roll_rate, angular_speed_rad_s=ang_speed,
        angular_acceleration_rad_s2=ang_accel,
        linear_accel_x=linear[:, 0], linear_accel_y=linear[:, 1],
        linear_accel_z=linear[:, 2], linear_accel_magnitude=lin_mag,
        head_jerk_m_s3=jerk, gravity_magnitude=g_mag,
        head_pitch_deg=head_pitch, head_roll_deg=head_roll,
        head_pitch_rel_deg=head_pitch_rel, head_roll_rel_deg=head_roll_rel,
        vibration_rms=vib, control_band_rms=ctrl,
        meta={
            "sampling_rate_hz": fs,
            "measures": "head_motion",
            "is_vehicle_acceleration": False,
            "gravity_removed": True,
            "gravity_cutoff_hz": gravity_cutoff_hz,
            "gravity_magnitude_median": float(np.nanmedian(g_mag)),
            "gravity_sanity_check": (
                "median gravity magnitude should sit near "
                f"{G_MSEC2:.2f} m/s^2; measured "
                f"{float(np.nanmedian(g_mag)):.2f}"),
            "frame_definition": ("up = gravity direction; forward = RGB optical "
                                 "axis projected perpendicular to up"),
            "absolute_attitude_offset_deg": {"pitch": pitch_ref, "roll": roll_ref},
            "attitude_note": (
                "head_pitch_deg / head_roll_deg are absolute angles from "
                "horizontal and include a fixed offset from how the rotated RGB "
                "sensor is mounted and how the glasses sit on the face. Only "
                "head_pitch_rel_deg / head_roll_rel_deg, taken against this "
                "recording's median posture, are comparable across recordings."),
            "vibration_band_hz": list(vibration_band_hz),
            "control_band_hz": list(control_band_hz),
        },
    )


def head_dynamics_summary(head: HeadDynamics,
                          lateral_check_threshold_rad_s: float = 0.35,
                          lateral_check_min_duration_s: float = 0.15
                          ) -> Dict[str, Any]:
    """Descriptive head-motion statistics, counts normalised per minute."""
    duration_s = (float((head.timestamp_ns[-1] - head.timestamp_ns[0]) / NS_PER_S)
                  if head.timestamp_ns.size > 1 else 0.0)
    checks = find_episodes(head.timestamp_ns, head.yaw_rate_rad_s,
                           np.abs(head.yaw_rate_rad_s) >= lateral_check_threshold_rad_s,
                           lateral_check_min_duration_s)
    stability = float(np.nanstd(head.angular_speed_rad_s))
    return {
        "measures": "head_motion",
        "is_vehicle_acceleration": False,
        "sampling_rate_hz": head.meta.get("sampling_rate_hz"),
        "duration_s": duration_s,
        "yaw_rate_rad_s": _percentiles(np.abs(head.yaw_rate_rad_s)),
        "pitch_rate_rad_s": _percentiles(np.abs(head.pitch_rate_rad_s)),
        "roll_rate_rad_s": _percentiles(np.abs(head.roll_rate_rad_s)),
        "angular_speed_rad_s": _percentiles(head.angular_speed_rad_s),
        "angular_acceleration_rad_s2": _percentiles(
            np.abs(head.angular_acceleration_rad_s2)),
        "linear_accel_magnitude_msec2": _percentiles(head.linear_accel_magnitude),
        "head_jerk_m_s3": _percentiles(np.abs(head.head_jerk_m_s3)),
        "vibration_rms_msec2": _percentiles(head.vibration_rms),
        "control_band_rms_rad_s": _percentiles(head.control_band_rms),
        "head_pitch_deg_absolute_with_mounting_offset":
            _percentiles(head.head_pitch_deg),
        "head_roll_deg_absolute_with_mounting_offset":
            _percentiles(head.head_roll_deg),
        "head_pitch_rel_deg": _percentiles(head.head_pitch_rel_deg),
        "head_roll_rel_deg": _percentiles(head.head_roll_rel_deg),
        "attitude_note": head.meta.get("attitude_note"),
        "lateral_check_events": len(checks),
        "lateral_checks_per_minute": _per_minute(len(checks), duration_s),
        "mean_lateral_check_duration_s": (
            float(np.mean([c.duration_s for c in checks])) if checks else None),
        "head_angular_speed_std_rad_s": stability,
        "gravity": {
            "removed": True,
            "median_magnitude_msec2": head.meta.get("gravity_magnitude_median"),
            "expected_msec2": G_MSEC2,
        },
        "caveat": ("head IMU only: this describes how the rider's head moved, "
                   "including whatever vibration the vehicle transmitted through "
                   "the body. It is not vehicle acceleration."),
    }
