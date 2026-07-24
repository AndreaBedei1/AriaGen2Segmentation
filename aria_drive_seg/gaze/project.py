"""Gaze ray projection into camera geometry (§9).

Uses projectaria_tools' official reprojection so CPF -> Device -> Camera and the
camera distortion model are all handled by the SDK (no hand-rolled transforms with
invented parameters). We project into BOTH:
  * the native fisheye624 RGB calibration  -> "original" pixel
  * the linear pinhole (rectified) calibration -> "rectified" pixel
so masks in either geometry can be queried.

Depth: use eye_gaze.depth when valid (>0), else a configurable fallback distance
that is explicitly flagged as an approximation.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np


@dataclass
class GazePixel:
    u: float
    v: float
    in_image: bool


class GazeProjector:
    def __init__(self, device_calib, fisheye_calib, pinhole_calib,
                 rgb_label: str = "camera-rgb", fallback_depth_m: float = 8.0):
        from projectaria_tools.core.mps import utils as mu
        self._mu = mu
        self.dev = device_calib
        self.fisheye = fisheye_calib
        self.pinhole = pinhole_calib
        self.rgb_label = rgb_label
        self.fallback_depth_m = float(fallback_depth_m)
        # cache transforms for the manual (interpolated) path
        self._T_device_cpf = np.asarray(device_calib.get_transform_device_cpf().to_matrix())
        self._fish_size = [int(x) for x in fisheye_calib.get_image_size()]
        self._pin_size = [int(x) for x in pinhole_calib.get_image_size()]

    # ------------------------------------------------------------------ #
    def depth_of(self, eye_gaze) -> Tuple[float, str]:
        d = float(getattr(eye_gaze, "depth", 0.0) or 0.0)
        if d > 0.0:
            return d, "device"
        return self.fallback_depth_m, "fallback"

    def reproject_official(self, eye_gaze, depth_m: float, which: str) -> Optional[GazePixel]:
        calib = self.pinhole if which == "rectified" else self.fisheye
        px = self._mu.get_gaze_vector_reprojection(eye_gaze, self.rgb_label, self.dev, calib, depth_m)
        return self._wrap(px, which)

    def reproject_yawpitch(self, yaw: float, pitch: float, depth_m: float,
                           which: str) -> Optional[GazePixel]:
        """Manual path (used for interpolated yaw/pitch). Mirrors the SDK math."""
        calib = self.pinhole if which == "rectified" else self.fisheye
        pt_cpf = np.asarray(self._mu.get_eyegaze_point_at_depth(yaw, pitch, depth_m), dtype=np.float64)
        pt_cpf_h = np.array([pt_cpf[0], pt_cpf[1], pt_cpf[2], 1.0])
        pt_device = self._T_device_cpf @ pt_cpf_h
        T_dev_cam = np.asarray(calib.get_transform_device_camera().to_matrix())
        T_cam_dev = np.linalg.inv(T_dev_cam)
        pt_cam = (T_cam_dev @ pt_device)[:3]
        px = calib.project(pt_cam)
        return self._wrap(px, which)

    def _wrap(self, px, which: str) -> Optional[GazePixel]:
        if px is None:
            return None
        px = np.asarray(px).reshape(-1)
        if px.shape[0] < 2 or not np.isfinite(px[:2]).all():
            return None
        w, h = (self._pin_size if which == "rectified" else self._fish_size)
        u, v = float(px[0]), float(px[1])
        return GazePixel(u, v, bool(0 <= u < w and 0 <= v < h))
