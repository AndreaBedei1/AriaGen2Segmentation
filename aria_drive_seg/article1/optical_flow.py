"""Causal optical-flow backends and validity checks for Article 1."""
from __future__ import annotations

from dataclasses import dataclass, field
import time

import cv2
import numpy as np


@dataclass
class FlowResult:
    forward: np.ndarray
    backward: np.ndarray
    valid: np.ndarray
    occlusion: np.ndarray
    photometric_error: np.ndarray
    valid_fraction: float
    median_flow_px: float
    mean_photometric_error: float
    timings_ms: dict = field(default_factory=dict)


def _gray(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return image.astype(np.uint8)
    return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)


def resize_flow(flow: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    """Resize flow to (width, height), scaling vector magnitudes correctly."""
    old_h, old_w = flow.shape[:2]
    width, height = size
    result = cv2.resize(flow, (width, height), interpolation=cv2.INTER_LINEAR)
    result[..., 0] *= width / old_w
    result[..., 1] *= height / old_h
    return result


def warp_with_backward(array: np.ndarray, backward: np.ndarray,
                       interpolation=cv2.INTER_LINEAR,
                       border_value=0) -> np.ndarray:
    """Warp previous-frame data into current coordinates using current→previous flow."""
    h, w = backward.shape[:2]
    x, y = np.meshgrid(np.arange(w, dtype=np.float32),
                       np.arange(h, dtype=np.float32))
    map_x = x + backward[..., 0]
    map_y = y + backward[..., 1]
    if array.ndim == 2:
        return cv2.remap(array, map_x, map_y, interpolation,
                         borderMode=cv2.BORDER_CONSTANT, borderValue=border_value)
    channels = [
        cv2.remap(array[i], map_x, map_y, interpolation,
                  borderMode=cv2.BORDER_CONSTANT, borderValue=border_value)
        for i in range(array.shape[0])
    ]
    return np.stack(channels)


def validate_flow(previous_rgb: np.ndarray, current_rgb: np.ndarray,
                  forward: np.ndarray, backward: np.ndarray, cfg: dict) -> FlowResult:
    """Forward/backward, bounds and photometric validation in current coordinates."""
    h, w = backward.shape[:2]
    x, y = np.meshgrid(np.arange(w, dtype=np.float32),
                       np.arange(h, dtype=np.float32))
    prev_x = x + backward[..., 0]
    prev_y = y + backward[..., 1]
    in_bounds = ((prev_x >= 0) & (prev_x <= w - 1) &
                 (prev_y >= 0) & (prev_y <= h - 1))
    sampled_forward_x = cv2.remap(
        forward[..., 0], prev_x, prev_y, cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT, borderValue=np.nan)
    sampled_forward_y = cv2.remap(
        forward[..., 1], prev_x, prev_y, cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT, borderValue=np.nan)
    fb_error = np.hypot(
        backward[..., 0] + sampled_forward_x,
        backward[..., 1] + sampled_forward_y)
    fb_valid = fb_error <= float(cfg.get("forward_backward_threshold_px", 1.5))
    if not cfg.get("use_forward_backward_check", True):
        fb_valid[:] = True

    previous_gray = _gray(previous_rgb).astype(np.float32) / 255
    current_gray = _gray(current_rgb).astype(np.float32) / 255
    warped_previous = warp_with_backward(previous_gray, backward)
    photometric = np.abs(warped_previous - current_gray)
    photo_valid = photometric <= float(
        cfg.get("photometric_error_threshold", .15))
    finite = np.isfinite(backward).all(axis=2) & np.isfinite(fb_error)
    valid = in_bounds & finite & fb_valid & photo_valid
    magnitude = np.linalg.norm(backward, axis=2)
    median = float(np.median(magnitude[valid])) if valid.any() else float("inf")
    mean_photo = float(photometric[valid].mean()) if valid.any() else 1.0
    return FlowResult(
        forward=forward.astype(np.float32),
        backward=backward.astype(np.float32),
        valid=valid,
        occlusion=~valid,
        photometric_error=photometric.astype(np.float32),
        valid_fraction=float(valid.mean()),
        median_flow_px=median,
        mean_photometric_error=mean_photo,
    )


def compute_optical_flow(previous_rgb: np.ndarray, current_rgb: np.ndarray,
                         cfg: dict) -> FlowResult:
    """Compute pluggable flow; only local OpenCV DIS is mandatory."""
    backend = cfg.get("backend", "opencv_dis")
    if backend != "opencv_dis":
        raise RuntimeError(
            f"optical-flow backend {backend!r} unavailable; "
            "no local verified checkpoint was configured")
    if previous_rgb.shape != current_rgb.shape:
        raise ValueError("optical-flow frames must share geometry")
    scale = float(cfg.get("processing_scale", 1.0))
    h, w = previous_rgb.shape[:2]
    size = (max(2, round(w * scale)), max(2, round(h * scale)))
    previous = cv2.resize(previous_rgb, size, interpolation=cv2.INTER_AREA)
    current = cv2.resize(current_rgb, size, interpolation=cv2.INTER_AREA)
    previous_input = _gray(previous) if cfg.get("grayscale", True) else previous
    current_input = _gray(current) if cfg.get("grayscale", True) else current
    preset = getattr(cv2, "DISOPTICAL_FLOW_PRESET_MEDIUM", 2)
    dis = cv2.DISOpticalFlow_create(preset)
    flow_start = time.perf_counter()
    forward = dis.calc(previous_input, current_input, None)
    backward = dis.calc(current_input, previous_input, None)
    flow_ms = (time.perf_counter() - flow_start) * 1000
    validation_start = time.perf_counter()
    result = validate_flow(previous, current, forward, backward, cfg)
    result.timings_ms = {
        "flow_compute_ms": flow_ms,
        "flow_validation_ms": (time.perf_counter() - validation_start) * 1000,
    }
    return result
