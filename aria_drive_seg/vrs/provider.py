"""Thin wrapper around projectaria_tools VrsDataProvider.

Centralises: stream discovery, RGB + on-device eyegaze accessors, camera
calibration, and rectification (fisheye624 -> linear pinhole, §5). projectaria_tools
is imported lazily so the rest of the package loads without it.

NB: VrsDataProvider is treated as NON thread-safe (§13) — read sequentially.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


@dataclass
class StreamInfo:
    stream_id: str
    label: str
    num_data: int
    first_ns: Optional[int]
    last_ns: Optional[int]


@dataclass
class RectifyParams:
    out_width: int
    out_height: int
    focal: float
    rotate_ccw90: int = 0


class AriaProvider:
    def __init__(self, vrs_path: str | Path, time_domain: str = "DEVICE_TIME"):
        from projectaria_tools.core import data_provider
        from projectaria_tools.core.sensor_data import TimeDomain

        self.vrs_path = str(vrs_path)
        self._dp = data_provider.create_vrs_data_provider(self.vrs_path)
        if self._dp is None:
            raise IOError(f"projectaria_tools could not open {vrs_path}")
        self._TimeDomain = TimeDomain
        self.time_domain = getattr(TimeDomain, time_domain)

    # ------------------------------------------------------------------ #
    # streams
    # ------------------------------------------------------------------ #
    def list_streams(self) -> List[StreamInfo]:
        infos = []
        for s in self._dp.get_all_streams():
            label = self._dp.get_label_from_stream_id(s)
            n = self._dp.get_num_data(s)
            first = last = None
            try:
                first = self._dp.get_first_time_ns(s, self.time_domain)
                last = self._dp.get_last_time_ns(s, self.time_domain)
            except Exception:
                pass
            infos.append(StreamInfo(str(s), label, n, first, last))
        return infos

    def stream_id(self, label: str):
        from projectaria_tools.core.stream_id import StreamId
        sid = self._dp.get_stream_id_from_label(label)
        if sid is not None:
            return sid
        # fall back to scanning (labels can differ across device generations)
        for s in self._dp.get_all_streams():
            if self._dp.get_label_from_stream_id(s) == label:
                return s
        raise KeyError(f"stream label {label!r} not found")

    def has_label(self, label: str) -> bool:
        try:
            self.stream_id(label)
            return True
        except KeyError:
            return False

    # ------------------------------------------------------------------ #
    # RGB
    # ------------------------------------------------------------------ #
    def rgb_stream(self, label: str = "camera-rgb"):
        return self.stream_id(label)

    def rgb_config(self, label: str = "camera-rgb") -> Dict[str, Any]:
        cfg = self._dp.get_image_configuration(self.stream_id(label))
        return {"width": int(cfg.image_width), "height": int(cfg.image_height),
                "pixel_format": int(cfg.pixel_format)}

    def num_rgb(self, label: str = "camera-rgb") -> int:
        return self._dp.get_num_data(self.stream_id(label))

    def rgb_by_index(self, index: int, label: str = "camera-rgb") -> Tuple[np.ndarray, int]:
        """Return (RGB uint8 HxWx3, capture_timestamp_ns)."""
        sid = self.stream_id(label)
        img_data, rec = self._dp.get_image_data_by_index(sid, index)
        arr = img_data.to_numpy_array()
        return np.ascontiguousarray(arr), int(rec.capture_timestamp_ns)

    def rgb_timestamps_ns(self, label: str = "camera-rgb") -> np.ndarray:
        """All RGB capture timestamps WITHOUT decoding images (fast)."""
        sid = self.stream_id(label)
        ts = self._dp.get_timestamps_ns(sid, self.time_domain)
        return np.asarray(ts, dtype=np.int64)

    def timestamps_ns(self, label: str) -> np.ndarray:
        sid = self.stream_id(label)
        return np.asarray(self._dp.get_timestamps_ns(sid, self.time_domain), dtype=np.int64)

    # ------------------------------------------------------------------ #
    # Eye gaze (on-device stream 373-1)
    # ------------------------------------------------------------------ #
    def has_eyegaze(self, label: str = "eyegaze") -> bool:
        return self.has_label(label)

    def num_eyegaze(self, label: str = "eyegaze") -> int:
        return self._dp.get_num_data(self.stream_id(label))

    def eyegaze_by_index(self, index: int, label: str = "eyegaze"):
        return self._dp.get_eye_gaze_data_by_index(self.stream_id(label), index)

    def eyegaze_by_time(self, ts_ns: int, label: str = "eyegaze", option: str = "closest"):
        from projectaria_tools.core.sensor_data import TimeQueryOptions
        opt = {"closest": TimeQueryOptions.CLOSEST,
               "before": TimeQueryOptions.BEFORE,
               "after": TimeQueryOptions.AFTER}[option]
        return self._dp.get_eye_gaze_data_by_time_ns(
            self.stream_id(label), int(ts_ns), self.time_domain, opt)

    def eyegaze_index_of_time(self, ts_ns: int, label: str = "eyegaze",
                              option: str = "before") -> int:
        from projectaria_tools.core.sensor_data import TimeQueryOptions
        opt = {"before": TimeQueryOptions.BEFORE,
               "after": TimeQueryOptions.AFTER,
               "closest": TimeQueryOptions.CLOSEST}[option]
        return self._dp.get_index_by_time_ns(
            self.stream_id(label), int(ts_ns), self.time_domain, opt)

    # ------------------------------------------------------------------ #
    # Calibration & rectification (§5)
    # ------------------------------------------------------------------ #
    @cached_property
    def device_calib(self):
        return self._dp.get_device_calibration()

    def rgb_calib(self, label: str = "camera-rgb"):
        return self.device_calib.get_camera_calib(label)

    @cached_property
    def T_device_cpf(self) -> np.ndarray:
        return np.asarray(self.device_calib.get_transform_device_cpf().to_matrix())

    def calib_summary(self, label: str = "camera-rgb") -> Dict[str, Any]:
        c = self.rgb_calib(label)
        return {
            "model": str(c.get_model_name()),
            "image_size": [int(x) for x in c.get_image_size()],
            "focal_lengths": [float(x) for x in c.get_focal_lengths()],
            "principal_point": [float(x) for x in c.get_principal_point()],
            "T_device_camera": np.asarray(c.get_transform_device_camera().to_matrix()).tolist(),
        }

    def make_pinhole(self, params: RectifyParams, label: str = "camera-rgb"):
        """Build a linear (pinhole) CameraCalibration for rectification."""
        from projectaria_tools.core import calibration
        src = self.rgb_calib(label)
        pin = calibration.get_linear_camera_calibration(
            params.out_width, params.out_height, params.focal,
            label, src.get_transform_device_camera())
        if params.rotate_ccw90 % 4 != 0:
            # rotate_camera_calib_cw90deg rotates clockwise; CCW = 3x CW
            n_cw = (-params.rotate_ccw90) % 4
            for _ in range(n_cw):
                pin = calibration.rotate_camera_calib_cw90deg(pin)
        return pin

    def rectifier(self, params: RectifyParams, label: str = "camera-rgb") -> "Rectifier":
        return Rectifier(self.rgb_calib(label), self.make_pinhole(params, label), params)


class Rectifier:
    """Rectify fisheye624 -> pinhole. Uses projectaria_tools' C++ distort mapping,
    with a one-time precomputed cv2 remap for speed on many frames."""

    def __init__(self, src_calib, dst_calib, params: RectifyParams):
        self.src = src_calib
        self.dst = dst_calib
        self.params = params
        self._map_x = None
        self._map_y = None

    @property
    def pinhole(self):
        return self.dst

    def _build_maps(self) -> None:
        """Precompute per-pixel source coordinates (dst pixel -> src pixel)."""
        w, h = self.params.out_width, self.params.out_height
        us, vs = np.meshgrid(np.arange(w), np.arange(h))
        pix = np.stack([us.ravel(), vs.ravel()], axis=1).astype(np.float64)
        map_x = np.full(w * h, -1.0, dtype=np.float32)
        map_y = np.full(w * h, -1.0, dtype=np.float32)
        # Try vectorised unproject/project; fall back to per-pixel.
        try:
            rays = self.dst.unproject(pix)              # (N,3) or list
            rays = np.asarray(rays, dtype=np.float64)
            src_pix = self.src.project(rays)            # (N,2)
            src_pix = np.asarray(src_pix, dtype=np.float64)
            valid = np.isfinite(src_pix).all(axis=1)
            map_x[valid] = src_pix[valid, 0].astype(np.float32)
            map_y[valid] = src_pix[valid, 1].astype(np.float32)
        except Exception:
            for i in range(pix.shape[0]):
                ray = self.dst.unproject(pix[i])
                if ray is None:
                    continue
                sp = self.src.project(np.asarray(ray, dtype=np.float64))
                if sp is None:
                    continue
                map_x[i] = float(sp[0])
                map_y[i] = float(sp[1])
        self._map_x = map_x.reshape(h, w)
        self._map_y = map_y.reshape(h, w)

    def rectify(self, raw: np.ndarray, fast: bool = True) -> np.ndarray:
        if fast:
            import cv2
            if self._map_x is None:
                self._build_maps()
            out = cv2.remap(raw, self._map_x, self._map_y, interpolation=cv2.INTER_LINEAR,
                            borderMode=cv2.BORDER_CONSTANT, borderValue=0)
            return np.ascontiguousarray(out)
        from projectaria_tools.core import calibration
        return np.ascontiguousarray(calibration.distort_by_calibration(raw, self.dst, self.src))
