"""Compact, timestamp-driven input stage for fast full semantic gaze.

The VRS provider is read sequentially because it is not thread-safe.  JPEG encode
and atomic writes are bounded and asynchronous.  Only real RGB records nearest to
the requested time grid are retained; no frame is interpolated or synthesised.
"""
from __future__ import annotations

import datetime as dt
import json
import time
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

import numpy as np

from ..config import Config
from ..gaze.project import GazeProjector
from ..hashing import sha256_file
from ..io_utils import (atomic_write_bytes, atomic_write_json,
                        config_fingerprint)
from ..logging_utils import get_logger
from ..vrs.provider import AriaProvider, RectifyParams

log = get_logger("article1.fast_io")
NS_PER_S = 1_000_000_000


def timestamp_grid_indices(timestamps_ns: Sequence[int], frequency_hz: float | str
                           ) -> np.ndarray:
    """Indices of real records nearest a uniform timestamp grid.

    ``native`` returns every source index.  A numeric frequency selects each real
    frame at most once.  The function deliberately accepts timestamps rather than
    a nominal source frame rate, and never creates a timestamp that is later
    treated as an observed frame.
    """
    ts = np.asarray(timestamps_ns, dtype=np.int64)
    if ts.ndim != 1 or ts.size == 0:
        return np.empty(0, dtype=np.int64)
    if np.any(np.diff(ts) <= 0):
        raise ValueError("timestamps must be strictly increasing")
    if isinstance(frequency_hz, str):
        if frequency_hz.lower() != "native":
            raise ValueError("frequency must be numeric or 'native'")
        return np.arange(ts.size, dtype=np.int64)
    hz = float(frequency_hz)
    if not np.isfinite(hz) or hz <= 0:
        raise ValueError("frequency_hz must be positive")
    step_ns = int(round(NS_PER_S / hz))
    grid = np.arange(int(ts[0]), int(ts[-1]) + 1, step_ns, dtype=np.int64)
    hi = np.searchsorted(ts, grid, side="left")
    hi = np.clip(hi, 0, ts.size - 1)
    lo = np.clip(hi - 1, 0, ts.size - 1)
    choose_hi = np.abs(ts[hi] - grid) < np.abs(grid - ts[lo])
    picked = np.where(choose_hi, hi, lo)
    # A low requested frequency cannot repeat, but unique is a fail-safe for
    # unusual sources whose cadence is lower than the request.
    return np.unique(picked).astype(np.int64)


def nearest_indices(source_ns: Sequence[int], target_ns: Sequence[int]
                    ) -> tuple[np.ndarray, np.ndarray]:
    """Nearest source row and signed source-target distance in nanoseconds."""
    src = np.asarray(source_ns, np.int64)
    tgt = np.asarray(target_ns, np.int64)
    if src.size == 0:
        return np.full(tgt.size, -1, np.int64), np.full(tgt.size, 0, np.int64)
    hi = np.clip(np.searchsorted(src, tgt, side="left"), 0, src.size - 1)
    lo = np.clip(hi - 1, 0, src.size - 1)
    use_hi = np.abs(src[hi] - tgt) < np.abs(tgt - src[lo])
    out = np.where(use_hi, hi, lo).astype(np.int64)
    return out, src[out] - tgt


def _encode_write_jpeg(path: Path, image_rgb: np.ndarray, quality: int) -> float:
    import cv2
    t0 = time.perf_counter()
    bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    ok, buf = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, int(quality)])
    if not ok:
        raise RuntimeError(f"JPEG encode failed for {path}")
    atomic_write_bytes(path, buf.tobytes())
    return (time.perf_counter() - t0) * 1000.0


def _drain(pending: set[Future], block: bool, write_ms: List[float]) -> set[Future]:
    if not pending:
        return pending
    if block:
        done, left = wait(pending)
    else:
        done, left = wait(pending, return_when=FIRST_COMPLETED)
    for fut in done:
        write_ms.append(float(fut.result()))
    return set(left)


def _atomic_parquet(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    import pandas as pd
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp.parquet")
    pd.DataFrame(list(rows)).to_parquet(tmp, index=False)
    tmp.replace(path)


def _project_gaze(provider: AriaProvider, selected_indices: np.ndarray,
                  selected_ts: np.ndarray, rectifier, out_width: int,
                  out_height: int, full_width: int, full_height: int,
                  cfg: Config) -> tuple[List[Dict[str, Any]], Dict[str, Any]]:
    label = cfg.get("vrs.eyegaze_label", "eyegaze")
    if not provider.has_eyegaze(label):
        return [], {"available": False, "reason": "eyegaze stream unavailable"}
    pinhole = rectifier.pinhole
    projector = GazeProjector(
        provider.device_calib, provider.rgb_calib(), pinhole,
        cfg.get("vrs.rgb_label", "camera-rgb"),
        fallback_depth_m=float(cfg.get("gaze.fallback_depth_m", 8.0)))
    gaze_ts = provider.timestamps_ns(label)
    nearest, dt_ns = nearest_indices(selected_ts, gaze_ts)
    max_dt_s = float(cfg.get(
        "semantic_gaze_fast_external.gaze.max_segmentation_dt_s", .12))
    sx, sy = out_width / full_width, out_height / full_height
    rows: List[Dict[str, Any]] = []
    valid = in_image = 0
    t0 = time.perf_counter()
    for i, ts in enumerate(gaze_ts):
        gaze = provider.eyegaze_by_index(i, label)
        combined = bool(getattr(gaze, "combined_gaze_valid", False))
        depth, depth_source = projector.depth_of(gaze)
        px = projector.reproject_official(gaze, depth, "rectified")
        u = float(px.u * sx) if px is not None else None
        v = float(px.v * sy) if px is not None else None
        inside = bool(px is not None and px.in_image and
                      0 <= u < out_width and 0 <= v < out_height)
        near_ok = nearest[i] >= 0 and abs(int(dt_ns[i])) <= max_dt_s * NS_PER_S
        semantic_ready = combined and inside and near_ok
        valid += int(semantic_ready)
        in_image += int(inside)
        seg_row = int(nearest[i]) if nearest[i] >= 0 else -1
        rows.append({
            "gaze_index": int(i), "timestamp_ns": int(ts),
            "combined_valid": combined,
            "spatial_valid": bool(getattr(gaze, "spatial_gaze_point_valid", False)),
            "yaw_rad": float(getattr(gaze, "yaw", np.nan)),
            "pitch_rad": float(getattr(gaze, "pitch", np.nan)),
            "depth_m": float(depth), "depth_source": depth_source,
            "rect_u": u, "rect_v": v, "in_image_rect": inside,
            "segmentation_row": seg_row,
            "segmentation_frame_index": (int(selected_indices[seg_row])
                                           if seg_row >= 0 else None),
            "segmentation_timestamp_ns": (int(selected_ts[seg_row])
                                            if seg_row >= 0 else None),
            "segmentation_dt_ms": (float(dt_ns[i] / 1e6)
                                     if seg_row >= 0 else None),
            "semantic_ready": semantic_ready,
        })
    return rows, {
        "available": True, "samples": len(rows), "semantic_ready": valid,
        "semantic_ready_fraction": valid / len(rows) if rows else 0.0,
        "in_image_fraction": in_image / len(rows) if rows else 0.0,
        "max_segmentation_dt_s": max_dt_s,
        "projection_s": time.perf_counter() - t0,
    }


def run_fast_extract(vrs_path: str | Path, output_dir: str | Path,
                     recording_id: str, domain: str, cfg: Config,
                     frequency_hz: Optional[float | str] = None,
                     source_sha256: Optional[str] = None,
                     resume: bool = True) -> Dict[str, Any]:
    """Decode, baseline-rectify and compact one full recording."""
    import cv2
    import pandas as pd

    fast = cfg.get("semantic_gaze_fast_external", {})
    frequency = (frequency_hz if frequency_hz is not None
                 else fast.get("segmentation_frequency_hz", 5.0))
    out = Path(output_dir)
    rect_dir = out / "frames" / "rectified"
    rect_dir.mkdir(parents=True, exist_ok=True)
    provider = AriaProvider(vrs_path, cfg.get("vrs.time_domain", "DEVICE_TIME"))
    rgb_label = cfg.get("vrs.rgb_label", "camera-rgb")
    source_ts = provider.rgb_timestamps_ns(rgb_label)
    indices = timestamp_grid_indices(source_ts, frequency)
    selected_ts = source_ts[indices]

    full_w = int(cfg.get("rectify.out_width", 2016))
    full_h = int(cfg.get("rectify.out_height", 1512))
    rp = RectifyParams(full_w, full_h, float(cfg.get("rectify.focal", 879.0)),
                       int(cfg.get("rectify.rotate_ccw90", 0)))
    rectifier = provider.rectifier(rp, rgb_label)
    out_w = int(fast.get("output_width", full_w))
    out_h = int(fast.get("output_height", full_h))
    quality = int(fast.get("jpeg_quality", 90))
    workers = max(1, int(fast.get("cpu_write_workers", 8)))
    max_pending = max(workers, int(fast.get("max_pending_writes", 32)))
    fingerprint = config_fingerprint(
        "semantic_gaze_fast_external_io", str(Path(vrs_path).resolve()),
        recording_id, domain, frequency, rp.__dict__, out_w, out_h, quality)

    prior: Dict[int, Dict[str, Any]] = {}
    index_path = out / "frames" / "frames.parquet"
    manifest_path = out / "fast_io_manifest.json"
    if resume and manifest_path.exists() and index_path.exists():
        manifest = json.loads(manifest_path.read_text())
        if manifest.get("fingerprint") == fingerprint:
            old = pd.read_parquet(index_path)
            prior = {int(r.frame_index): r._asdict() for r in old.itertuples(index=False)}

    decode_ms: List[float] = []
    rectify_ms: List[float] = []
    resize_ms: List[float] = []
    write_ms: List[float] = []
    records: List[Dict[str, Any]] = []
    pending: set[Future] = set()
    started = time.perf_counter()
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for k, (index, ts) in enumerate(zip(indices, selected_ts)):
            index = int(index); ts = int(ts)
            path = rect_dir / f"frame_{index:06d}.jpg"
            reused = bool(resume and path.exists() and path.stat().st_size > 0)
            if not reused:
                t0 = time.perf_counter()
                raw, decoded_ts = provider.rgb_by_index(index, rgb_label)
                decode_ms.append((time.perf_counter() - t0) * 1000.0)
                if int(decoded_ts) != ts:
                    raise RuntimeError(
                        f"RGB timestamp mismatch at index {index}: {decoded_ts} != {ts}")
                t0 = time.perf_counter()
                image = rectifier.rectify(raw)
                rectify_ms.append((time.perf_counter() - t0) * 1000.0)
                if image.shape[1] != out_w or image.shape[0] != out_h:
                    t0 = time.perf_counter()
                    image = cv2.resize(image, (out_w, out_h), interpolation=cv2.INTER_AREA)
                    resize_ms.append((time.perf_counter() - t0) * 1000.0)
                pending.add(pool.submit(_encode_write_jpeg, path, image, quality))
                if len(pending) >= max_pending:
                    pending = _drain(pending, False, write_ms)
            records.append({
                "recording_id": recording_id, "domain": domain,
                "frame_index": index, "source_frame_index": index,
                "capture_timestamp_ns": ts, "timestamp_ns": ts,
                "time_since_recording_start_s": float((ts - source_ts[0]) / NS_PER_S),
                "rectified_path": str(path.relative_to(out)),
                "width": out_w, "height": out_h,
                "source_is_real_frame": True, "synthetic": False,
                "status": "reused" if reused else "extracted",
            })
            if (k + 1) % 250 == 0:
                log.info("fast extract %s: %d/%d", domain, k + 1, len(indices))
        pending = _drain(pending, True, write_ms)

    frames = pd.DataFrame(records)
    tmp = index_path.with_suffix(".tmp.parquet")
    frames.to_parquet(tmp, index=False); tmp.replace(index_path)
    frames.to_csv(out / "frames" / "frames.csv", index=False)

    gaze_rows, gaze_summary = _project_gaze(
        provider, indices, selected_ts, rectifier, out_w, out_h, full_w, full_h, cfg)
    _atomic_parquet(out / "gaze" / "projected_gaze.parquet", gaze_rows)
    source_hash = source_sha256 or sha256_file(vrs_path)
    calibration = {
        "source_calib": provider.calib_summary(rgb_label),
        "baseline_rectification": rp.__dict__,
        "stored_resolution": {"width": out_w, "height": out_h,
                              "scale_x": out_w / full_w,
                              "scale_y": out_h / full_h},
        "pinhole_focal_at_stored_resolution": rp.focal * out_w / full_w,
    }
    atomic_write_json(out / "frames" / "calibration.json", calibration)
    elapsed = time.perf_counter() - started
    def mean(values: List[float]) -> Optional[float]:
        return float(np.mean(values)) if values else None
    duration_s = float((source_ts[-1] - source_ts[0]) / NS_PER_S)
    summary = {
        "schema": "article1_fast_external_io_v1",
        "generated_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "recording_id": recording_id, "domain": domain,
        "source_file": str(Path(vrs_path).resolve()),
        "source_file_sha256": source_hash,
        "source_frames": int(source_ts.size), "source_duration_s": duration_s,
        "selected_real_frames": int(indices.size), "segmentation_frequency_hz": frequency,
        "measured_selected_hz": (float((indices.size - 1) /
                                       ((selected_ts[-1] - selected_ts[0]) / NS_PER_S))
                                 if indices.size > 1 else None),
        "resampled": False, "interpolated": False, "synthetic_frames": 0,
        "stored_resolution": [out_w, out_h], "gaze": gaze_summary,
        "profiling": {
            "elapsed_s": elapsed, "throughput_selected_frames_s": indices.size / elapsed,
            "mean_decode_ms": mean(decode_ms), "mean_rectify_ms": mean(rectify_ms),
            "mean_resize_ms": mean(resize_ms), "mean_encode_write_ms": mean(write_ms),
            "cpu_write_workers": workers, "max_pending_writes": max_pending,
        },
    }
    atomic_write_json(out / "frames" / "extraction_summary.json", summary)
    atomic_write_json(manifest_path, {"stage": "semantic_gaze_fast_external_io",
                                     "fingerprint": fingerprint,
                                     "frames": int(indices.size)})
    return summary
