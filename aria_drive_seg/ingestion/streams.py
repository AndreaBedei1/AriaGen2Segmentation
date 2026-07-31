"""Typed access to the non-RGB Aria streams used by Article 1 ingestion.

Gaze, hand tracking and GPS are read here with their **own** timestamps and are
associated to RGB frames strictly by timestamp, never by index or by assuming a
one-to-one correspondence. Every association keeps the signed temporal distance so
a downstream consumer can decide whether the sample was close enough to be useful.

None of these signals is ground truth: hand tracking in particular is an on-device
estimate used only to build review candidates.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from .timeline import NS_PER_S

# Aria hand-tracking landmark layout (21 points per hand).
HAND_LANDMARK_COUNT = 21


# --------------------------------------------------------------------------- #
# Gaze
# --------------------------------------------------------------------------- #
@dataclass
class GazeSample:
    index: int
    timestamp_ns: int
    combined_valid: bool
    spatial_valid: bool
    yaw_rad: Optional[float]
    pitch_rad: Optional[float]
    depth_m: Optional[float]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def read_gaze_samples(provider, label: str = "eyegaze") -> List[GazeSample]:
    """Read every on-device gaze sample with its own timestamp."""
    if not provider.has_eyegaze(label):
        return []
    n = provider.num_eyegaze(label)
    ts = provider.timestamps_ns(label)
    out: List[GazeSample] = []
    for i in range(n):
        g = provider.eyegaze_by_index(i, label)
        out.append(GazeSample(
            index=i,
            timestamp_ns=int(ts[i]) if i < ts.size else 0,
            combined_valid=bool(getattr(g, "combined_gaze_valid", False)),
            spatial_valid=bool(getattr(g, "spatial_gaze_point_valid", False)),
            yaw_rad=_opt_float(getattr(g, "yaw", None)),
            pitch_rad=_opt_float(getattr(g, "pitch", None)),
            depth_m=_opt_float(getattr(g, "depth_m", None)),
        ))
    return out


def _opt_float(v: Any) -> Optional[float]:
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return f if np.isfinite(f) else None


# --------------------------------------------------------------------------- #
# Hand tracking
# --------------------------------------------------------------------------- #
@dataclass
class HandSideSample:
    present: bool
    confidence: Optional[float] = None
    wrist_device: Optional[List[float]] = None
    palm_device: Optional[List[float]] = None
    landmarks_device: Optional[List[List[float]]] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class HandTrackingSample:
    index: int
    timestamp_ns: int
    left: HandSideSample
    right: HandSideSample

    def to_dict(self) -> Dict[str, Any]:
        return {"index": self.index, "timestamp_ns": self.timestamp_ns,
                "left": self.left.to_dict(), "right": self.right.to_dict()}


def _side(sample) -> HandSideSample:
    if sample is None:
        return HandSideSample(present=False)
    landmarks = None
    raw = getattr(sample, "landmark_positions_device", None)
    if raw is not None:
        try:
            landmarks = [[float(c) for c in p] for p in raw]
        except Exception:
            landmarks = None
    wrist = palm = None
    try:
        wrist = [float(c) for c in sample.get_wrist_position_device()]
    except Exception:
        pass
    try:
        palm = [float(c) for c in sample.get_palm_position_device()]
    except Exception:
        pass
    return HandSideSample(
        present=True,
        confidence=_opt_float(getattr(sample, "confidence", None)),
        wrist_device=wrist, palm_device=palm, landmarks_device=landmarks,
    )


def read_hand_tracking(provider, label: str = "handtracking") -> List[HandTrackingSample]:
    """Read the on-device hand-tracking stream.

    A missing side is recorded as `present=False`. That is a *tracker* outcome and
    must never be reported as "the hand is absent from the image": the rider's hand
    can be perfectly visible while tracking fails, and can be tracked while outside
    the RGB field of view.
    """
    if not provider.has_label(label):
        return []
    sid = provider.stream_id(label)
    n = provider._dp.get_num_data(sid)
    ts = provider.timestamps_ns(label)
    out: List[HandTrackingSample] = []
    for i in range(n):
        r = provider._dp.get_hand_pose_data_by_index(sid, i)
        out.append(HandTrackingSample(
            index=i,
            timestamp_ns=int(ts[i]) if i < ts.size else 0,
            left=_side(getattr(r, "left_hand", None)),
            right=_side(getattr(r, "right_hand", None)),
        ))
    return out


def project_device_points_to_rgb(provider, points_device: Sequence[Sequence[float]],
                                 rgb_label: str = "camera-rgb"
                                 ) -> List[Optional[List[float]]]:
    """Project device-frame 3D points into raw RGB pixel coordinates.

    Returns `None` for points that fall behind the camera or outside the model's
    valid projection domain, so an out-of-view landmark is never silently clamped
    onto the image border.
    """
    calib = provider.rgb_calib(rgb_label)
    T_device_camera = np.asarray(calib.get_transform_device_camera().to_matrix())
    T_camera_device = np.linalg.inv(T_device_camera)
    out: List[Optional[List[float]]] = []
    for p in points_device:
        ph = np.array([float(p[0]), float(p[1]), float(p[2]), 1.0])
        cam = T_camera_device @ ph
        if cam[2] <= 0:
            out.append(None)
            continue
        try:
            pix = calib.project(cam[:3])
        except Exception:
            pix = None
        out.append([float(pix[0]), float(pix[1])] if pix is not None else None)
    return out


# --------------------------------------------------------------------------- #
# GPS
# --------------------------------------------------------------------------- #
@dataclass
class GpsSample:
    index: int
    timestamp_ns: int
    latitude: Optional[float]
    longitude: Optional[float]
    altitude: Optional[float]
    accuracy: Optional[float]
    speed: Optional[float]
    utc_time_ms: Optional[float]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def read_gps(provider, label: str = "gps-app") -> List[GpsSample]:
    if not provider.has_label(label):
        return []
    sid = provider.stream_id(label)
    n = provider._dp.get_num_data(sid)
    out: List[GpsSample] = []
    for i in range(n):
        g = provider._dp.get_gps_data_by_index(sid, i)
        out.append(GpsSample(
            index=i,
            timestamp_ns=int(getattr(g, "capture_timestamp_ns", 0)),
            latitude=_opt_float(getattr(g, "latitude", None)),
            longitude=_opt_float(getattr(g, "longitude", None)),
            altitude=_opt_float(getattr(g, "altitude", None)),
            accuracy=_opt_float(getattr(g, "accuracy", None)),
            speed=_opt_float(getattr(g, "speed", None)),
            utc_time_ms=_opt_float(getattr(g, "utc_time_ms", None)),
        ))
    return out


def valid_gps(samples: Sequence[GpsSample],
              max_accuracy_m: float = 50.0) -> List[GpsSample]:
    """Keep only samples with a plausible fix.

    Latitude/longitude exactly at (0, 0) is treated as "no fix" rather than as a
    position in the Gulf of Guinea.
    """
    out = []
    for s in samples:
        if s.latitude is None or s.longitude is None:
            continue
        if abs(s.latitude) < 1e-7 and abs(s.longitude) < 1e-7:
            continue
        if not (-90.0 <= s.latitude <= 90.0 and -180.0 <= s.longitude <= 180.0):
            continue
        if s.accuracy is not None and s.accuracy > max_accuracy_m:
            continue
        out.append(s)
    return out


# --------------------------------------------------------------------------- #
# Timestamp association
# --------------------------------------------------------------------------- #
@dataclass
class Association:
    """Association of a reference RGB frame to samples of another stream."""

    frame_index: int
    frame_timestamp_ns: int
    nearest_index: Optional[int]
    nearest_timestamp_ns: Optional[int]
    nearest_dt_ms: Optional[float]          # signed: sample - frame
    window_indices: List[int] = field(default_factory=list)
    window_dt_ms: List[float] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def associate_by_timestamp(frame_timestamps_ns: Sequence[int] | np.ndarray,
                           sample_timestamps_ns: Sequence[int] | np.ndarray,
                           window_s: float,
                           frame_indices: Optional[Sequence[int]] = None
                           ) -> List[Association]:
    """Associate each RGB frame with every sample inside a ± window in SECONDS.

    The window is a duration, so the number of samples it contains adapts to the
    real rate of both streams. Multiple samples per frame are preserved: a 30 Hz
    gaze stream legitimately has several samples per 15 fps frame and collapsing
    them to one would throw away signal.
    """
    ft = np.asarray(frame_timestamps_ns, dtype=np.int64)
    st = np.asarray(sample_timestamps_ns, dtype=np.int64)
    idxs = list(frame_indices) if frame_indices is not None else list(range(ft.size))
    if window_s < 0:
        raise ValueError("window_s must be non-negative")
    half = int(round(window_s * NS_PER_S))

    out: List[Association] = []
    for k, t in enumerate(ft):
        if st.size == 0:
            out.append(Association(int(idxs[k]), int(t), None, None, None))
            continue
        lo = int(np.searchsorted(st, t - half, side="left"))
        hi = int(np.searchsorted(st, t + half, side="right"))
        window = list(range(lo, hi))
        pos = int(np.searchsorted(st, t))
        cands = [c for c in (pos - 1, pos) if 0 <= c < st.size]
        nearest = min(cands, key=lambda c: abs(int(st[c]) - int(t)))
        out.append(Association(
            frame_index=int(idxs[k]),
            frame_timestamp_ns=int(t),
            nearest_index=nearest,
            nearest_timestamp_ns=int(st[nearest]),
            nearest_dt_ms=float((int(st[nearest]) - int(t)) / 1e6),
            window_indices=window,
            window_dt_ms=[float((int(st[j]) - int(t)) / 1e6) for j in window],
        ))
    return out
