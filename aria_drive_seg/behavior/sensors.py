"""Array readers for the non-image Aria streams used by the behaviour analysis.

Each reader returns a `SensorSeries`: the stream's own capture timestamps plus
named channels, as parallel numpy arrays. Nothing is resampled, interpolated or
snapped to another stream's clock here — that decision belongs to the caller,
which must also record how far it had to reach.

Three facts about this device are encoded in the readers and must survive into
every downstream report:

* the IMUs sit on the **head**, on the two temples of a pair of glasses. They
  measure gravity, head motion and vehicle vibration together. Nothing here is a
  vehicle accelerometer;
* the `temperature` stream is a set of ~25 **device** thermal sensors (nose pad,
  temples, front housing, SoC virtual sensors). It is a device/environment
  context signal, never a body temperature;
* the `ppg` stream is a raw photoplethysmography count with no units and no
  clinical validation. It is a physiological *proxy*.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from ..ingestion.timeline import NS_PER_S, RateEstimate, measure_rate

#: Streams the behaviour analysis knows how to read, with the provider accessor.
SENSOR_ACCESSORS: Dict[str, str] = {
    "imu-left": "get_imu_data_by_index",
    "imu-right": "get_imu_data_by_index",
    "mag0": "get_magnetometer_data_by_index",
    "baro0": "get_barometer_data_by_index",
    "ppg": "get_ppg_data_by_index",
    "als": "get_als_data_by_index",
    "temperature": "get_temperature_data_by_index",
}


@dataclass
class SensorSeries:
    """A stream's real samples: its own timestamps plus named channels."""

    label: str
    timestamp_ns: np.ndarray
    channels: Dict[str, np.ndarray] = field(default_factory=dict)
    #: Free-form per-stream facts (sensor names, units, device placement, ...).
    meta: Dict[str, Any] = field(default_factory=dict)

    def __len__(self) -> int:
        return int(self.timestamp_ns.size)

    @property
    def duration_s(self) -> float:
        if self.timestamp_ns.size < 2:
            return 0.0
        return float((self.timestamp_ns[-1] - self.timestamp_ns[0]) / NS_PER_S)

    def rate(self) -> RateEstimate:
        """Rate measured from this stream's own timestamps, never assumed."""
        return measure_rate(self.timestamp_ns)

    def slice_time(self, start_ns: int, end_ns: int) -> "SensorSeries":
        """Real samples inside a closed time window. Never pads the edges."""
        lo = int(np.searchsorted(self.timestamp_ns, int(start_ns), side="left"))
        hi = int(np.searchsorted(self.timestamp_ns, int(end_ns), side="right"))
        return SensorSeries(
            label=self.label,
            timestamp_ns=self.timestamp_ns[lo:hi],
            channels={k: v[lo:hi] for k, v in self.channels.items()},
            meta=dict(self.meta),
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "label": self.label,
            "count": len(self),
            "duration_s": self.duration_s,
            "channels": sorted(self.channels),
            "meta": self.meta,
        }


def _sorted(series: SensorSeries) -> SensorSeries:
    """Enforce monotonic timestamps, recording how many samples were out of order.

    `searchsorted` is used everywhere downstream, so a non-monotonic stream would
    silently produce wrong windows. Reordering is a repair, so it is counted.
    """
    ts = series.timestamp_ns
    if ts.size < 2 or bool(np.all(np.diff(ts) >= 0)):
        series.meta["reordered_samples"] = 0
        return series
    order = np.argsort(ts, kind="stable")
    series.meta["reordered_samples"] = int(np.sum(order != np.arange(ts.size)))
    series.timestamp_ns = ts[order]
    series.channels = {k: v[order] for k, v in series.channels.items()}
    return series


# --------------------------------------------------------------------------- #
# IMU  — head-mounted, includes gravity
# --------------------------------------------------------------------------- #
def read_imu(provider, label: str = "imu-left") -> SensorSeries:
    """Read one head IMU: accelerometer (m/s^2, gravity included) and gyro (rad/s)."""
    if not provider.has_label(label):
        return SensorSeries(label, np.zeros(0, np.int64),
                            meta={"available": False})
    sid = provider.stream_id(label)
    n = int(provider._dp.get_num_data(sid))
    ts = np.zeros(n, np.int64)
    acc = np.full((n, 3), np.nan, np.float64)
    gyr = np.full((n, 3), np.nan, np.float64)
    temp = np.full(n, np.nan, np.float64)
    acc_ok = np.zeros(n, bool)
    gyr_ok = np.zeros(n, bool)
    for i in range(n):
        r = provider._dp.get_imu_data_by_index(sid, i)
        ts[i] = int(r.capture_timestamp_ns)
        acc_ok[i] = bool(r.accel_valid)
        gyr_ok[i] = bool(r.gyro_valid)
        if acc_ok[i]:
            acc[i] = np.asarray(r.accel_msec2, dtype=np.float64)
        if gyr_ok[i]:
            gyr[i] = np.asarray(r.gyro_radsec, dtype=np.float64)
        temp[i] = float(r.temperature)
    return _sorted(SensorSeries(
        label=label,
        timestamp_ns=ts,
        channels={
            "accel_x_msec2": acc[:, 0], "accel_y_msec2": acc[:, 1],
            "accel_z_msec2": acc[:, 2],
            "gyro_x_radsec": gyr[:, 0], "gyro_y_radsec": gyr[:, 1],
            "gyro_z_radsec": gyr[:, 2],
            "imu_temperature_c": temp,
            "accel_valid": acc_ok, "gyro_valid": gyr_ok,
        },
        meta={
            "available": True,
            "placement": "head_mounted_glasses_temple",
            "gravity_included": True,
            "is_vehicle_accelerometer": False,
            "note": ("head IMU: measures gravity, head motion and vehicle "
                     "vibration together; never used as vehicle acceleration"),
        },
    ))


# --------------------------------------------------------------------------- #
# PPG — physiological proxy, raw counts
# --------------------------------------------------------------------------- #
def read_ppg(provider, label: str = "ppg") -> SensorSeries:
    """Read the raw PPG counts with the LED drive settings that produced them.

    `integration_time_us` and `led_current_ma` matter: a change in either rescales
    `value` for instrumental reasons, which must not be read as a physiological
    change.
    """
    if not provider.has_label(label):
        return SensorSeries(label, np.zeros(0, np.int64), meta={"available": False})
    sid = provider.stream_id(label)
    n = int(provider._dp.get_num_data(sid))
    ts = np.zeros(n, np.int64)
    val = np.full(n, np.nan, np.float64)
    integ = np.full(n, np.nan, np.float64)
    led = np.full(n, np.nan, np.float64)
    for i in range(n):
        r = provider._dp.get_ppg_data_by_index(sid, i)
        ts[i] = int(r.capture_timestamp_ns)
        val[i] = float(r.value)
        integ[i] = float(getattr(r, "integration_time_us", np.nan))
        led[i] = float(getattr(r, "led_current_ma", np.nan))
    return _sorted(SensorSeries(
        label=label,
        timestamp_ns=ts,
        channels={"ppg_value": val,
                  "integration_time_us": integ,
                  "led_current_ma": led},
        meta={
            "available": True,
            "units": "raw_sensor_counts",
            "interpretation": "physiological_proxy_not_medical",
            "note": ("raw PPG counts; amplitude depends on LED current and "
                     "integration time as well as on perfusion"),
        },
    ))


# --------------------------------------------------------------------------- #
# Magnetometer / barometer / ambient light / device temperature
# --------------------------------------------------------------------------- #
def read_magnetometer(provider, label: str = "mag0") -> SensorSeries:
    if not provider.has_label(label):
        return SensorSeries(label, np.zeros(0, np.int64), meta={"available": False})
    sid = provider.stream_id(label)
    n = int(provider._dp.get_num_data(sid))
    ts = np.zeros(n, np.int64)
    mag = np.full((n, 3), np.nan, np.float64)
    ok = np.zeros(n, bool)
    for i in range(n):
        r = provider._dp.get_magnetometer_data_by_index(sid, i)
        ts[i] = int(r.capture_timestamp_ns)
        ok[i] = bool(r.mag_valid)
        if ok[i]:
            mag[i] = np.asarray(r.mag_tesla, dtype=np.float64)
    return _sorted(SensorSeries(
        label=label, timestamp_ns=ts,
        channels={"mag_x_tesla": mag[:, 0], "mag_y_tesla": mag[:, 1],
                  "mag_z_tesla": mag[:, 2], "mag_valid": ok},
        meta={"available": True, "placement": "head_mounted",
              "note": ("uncalibrated head magnetometer; a vehicle is a large "
                       "ferrous body, so absolute heading from it is unreliable")},
    ))


def read_barometer(provider, label: str = "baro0") -> SensorSeries:
    if not provider.has_label(label):
        return SensorSeries(label, np.zeros(0, np.int64), meta={"available": False})
    sid = provider.stream_id(label)
    n = int(provider._dp.get_num_data(sid))
    ts = np.zeros(n, np.int64)
    pres = np.full(n, np.nan, np.float64)
    temp = np.full(n, np.nan, np.float64)
    for i in range(n):
        r = provider._dp.get_barometer_data_by_index(sid, i)
        ts[i] = int(r.capture_timestamp_ns)
        pres[i] = float(r.pressure)
        temp[i] = float(r.temperature)
    return _sorted(SensorSeries(
        label=label, timestamp_ns=ts,
        channels={"pressure_pa": pres, "baro_temperature_c": temp},
        meta={"available": True, "units": {"pressure": "Pa"},
              "note": ("barometric pressure; relative altitude change only, "
                       "absolute altitude needs a sea-level reference")},
    ))


def read_als(provider, label: str = "als") -> SensorSeries:
    """Ambient light. `lux` and `cct` are the two channels used downstream."""
    if not provider.has_label(label):
        return SensorSeries(label, np.zeros(0, np.int64), meta={"available": False})
    sid = provider.stream_id(label)
    n = int(provider._dp.get_num_data(sid))
    ts = np.zeros(n, np.int64)
    out = {k: np.full(n, np.nan, np.float64) for k in
           ("lux", "cct", "red_channel_normalized", "green_channel_normalized",
            "blue_channel_normalized", "ir_channel_normalized",
            "uv_channel_normalized")}
    for i in range(n):
        r = provider._dp.get_als_data_by_index(sid, i)
        ts[i] = int(r.capture_timestamp_ns)
        for k in out:
            out[k][i] = float(getattr(r, k, np.nan))
    return _sorted(SensorSeries(
        label=label, timestamp_ns=ts, channels=out,
        meta={"available": True, "units": {"lux": "lux", "cct": "K"},
              "note": ("head-mounted ambient light; it follows where the rider "
                       "looks as well as the real scene illumination")},
    ))


def read_device_temperature(provider, label: str = "temperature",
                            preferred: Sequence[str] = (
                                "TEMP_NOSE_PAD", "TEMP_FRONT_LEFT_OUTER",
                                "TEMP_LEFT_TEMPLE_OUTER")) -> SensorSeries:
    """Read the device thermal sensors, de-interleaved by sensor name.

    The stream interleaves ~25 named sensors at different cadences, so a naive
    read produces a sawtooth that is an artefact of the interleaving, not a
    temperature change. Each sensor is therefore kept as its own channel on the
    timestamps of a single chosen reference sensor.

    None of these is a body temperature. They are device thermal telemetry and are
    labelled `device_or_environment_temperature_context` downstream.
    """
    if not provider.has_label(label):
        return SensorSeries(label, np.zeros(0, np.int64), meta={"available": False})
    sid = provider.stream_id(label)
    n = int(provider._dp.get_num_data(sid))
    by_sensor: Dict[str, List[tuple]] = {}
    for i in range(n):
        r = provider._dp.get_temperature_data_by_index(sid, i)
        by_sensor.setdefault(str(r.sensor_name), []).append(
            (int(r.capture_timestamp_ns), float(r.temperature_celsius)))

    if not by_sensor:
        return SensorSeries(label, np.zeros(0, np.int64), meta={"available": False})

    # Reference sensor: the first preferred name that is present and dense,
    # otherwise simply the densest sensor in the stream.
    dense = {k: len(v) for k, v in by_sensor.items()}
    ref = next((p for p in preferred if dense.get(p, 0) == max(dense.values())), None)
    if ref is None:
        ref = max(dense, key=lambda k: dense[k])

    ref_ts = np.array([t for t, _ in by_sensor[ref]], dtype=np.int64)
    order = np.argsort(ref_ts, kind="stable")
    ref_ts = ref_ts[order]
    channels: Dict[str, np.ndarray] = {
        f"temp_{ref.lower()}_c":
            np.array([v for _, v in by_sensor[ref]], dtype=np.float64)[order]}

    # Other sensors are carried at their own nearest real sample; the temporal
    # distance is reported so a sparse sensor is never mistaken for a dense one.
    for name, pairs in sorted(by_sensor.items()):
        if name == ref or len(pairs) < 2:
            continue
        t = np.array([p[0] for p in pairs], dtype=np.int64)
        v = np.array([p[1] for p in pairs], dtype=np.float64)
        o = np.argsort(t, kind="stable")
        t, v = t[o], v[o]
        pos = np.clip(np.searchsorted(t, ref_ts), 0, t.size - 1)
        left = np.clip(pos - 1, 0, t.size - 1)
        take_left = np.abs(ref_ts - t[left]) <= np.abs(t[pos] - ref_ts)
        idx = np.where(take_left, left, pos)
        channels[f"temp_{name.lower()}_c"] = v[idx]
        channels[f"temp_{name.lower()}_dt_ms"] = (ref_ts - t[idx]) / 1e6

    return _sorted(SensorSeries(
        label=label, timestamp_ns=ref_ts, channels=channels,
        meta={
            "available": True,
            "reference_sensor": ref,
            "sensor_sample_counts": dense,
            "interpretation": "device_or_environment_temperature_context",
            "is_body_temperature": False,
            "note": ("device thermal telemetry de-interleaved by sensor name; "
                     "not a body temperature and not validated as one"),
        },
    ))


# --------------------------------------------------------------------------- #
# On-device VIO / SLAM pose availability
# --------------------------------------------------------------------------- #
def probe_vio(provider) -> Dict[str, Any]:
    """Report whether a usable on-device VIO/SLAM pose stream exists.

    Vehicle speed prefers a pose-based estimate over GPS, so the absence of pose
    has to be an explicit, recorded finding rather than a silent fallback.
    """
    from projectaria_tools.core.stream_id import StreamId

    out: Dict[str, Any] = {"usable": False, "streams": {}, "reason": None}
    for sid_str in ("502-1", "503-1", "504-1", "505-1"):
        try:
            sid = StreamId(sid_str)
            n = int(provider._dp.get_num_data(sid))
        except Exception as exc:                      # stream absent entirely
            out["streams"][sid_str] = {"present": False, "error": str(exc)[:120]}
            continue
        entry: Dict[str, Any] = {"present": True, "num_data": n, "readable": False}
        for meth in ("get_vio_data_by_index", "get_vio_high_freq_data_by_index"):
            if n <= 0 or not hasattr(provider._dp, meth):
                continue
            try:
                getattr(provider._dp, meth)(sid, 0)
                entry["readable"] = True
                entry["accessor"] = meth
                break
            except Exception as exc:
                entry["error"] = str(exc)[:120]
        out["streams"][sid_str] = entry

    readable = [s for s in out["streams"].values()
                if s.get("readable") and s.get("num_data", 0) > 0]
    if readable:
        out["usable"] = True
    else:
        present = [f"{k}(n={v.get('num_data', 0)})"
                   for k, v in out["streams"].items() if v.get("present")]
        out["reason"] = (
            "no activatable VIO/SLAM pose stream with usable content; "
            f"present but not activated: {', '.join(present) or 'none'}")
    return out


# --------------------------------------------------------------------------- #
# Bulk read
# --------------------------------------------------------------------------- #
READERS = {
    "imu-left": read_imu,
    "imu-right": read_imu,
    "mag0": read_magnetometer,
    "baro0": read_barometer,
    "ppg": read_ppg,
    "als": read_als,
    "temperature": read_device_temperature,
}


def read_all_sensors(provider, labels: Optional[Sequence[str]] = None
                     ) -> Dict[str, SensorSeries]:
    """Read every supported non-image stream, skipping the ones not recorded."""
    out: Dict[str, SensorSeries] = {}
    for label in (labels if labels is not None else READERS):
        reader = READERS.get(label)
        if reader is None:
            continue
        out[label] = (reader(provider, label) if label != "temperature"
                      else reader(provider, label))
    return out
