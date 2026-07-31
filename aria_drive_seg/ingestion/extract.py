"""Timestamp-driven, frame-rate agnostic RGB extraction.

Differences from the original extractor that this one generalises:

* selection is expressed in seconds against the **real capture timestamps**, so the
  same request yields the same wall-clock interval at 10 fps and at 15 fps;
* every record carries the provenance needed to trace it back to the source file:
  `recording_id`, `domain`, `source_file_sha256`, `source_stream_id`,
  `source_frame_index`, `timestamp_ns`, `timestamp_s` and the measured
  `effective_fps` of the source stream;
* the original stream index is preserved in the record *and* in the file name, so a
  frame is never identified by a local counter;
* nothing is resampled: `frame_step` is only honoured when explicitly requested and
  is recorded in the manifest as a declared decimation.

The written layout is a superset of the one the frozen Article 1 pipeline expects
(`frames/frames.parquet`, `frames/original`, `frames/rectified`,
`frames/calibration.json`), so the frozen stages run unchanged on it.
"""
from __future__ import annotations

import datetime as dt
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from ..config import Config
from ..io_utils import atomic_write_bytes, atomic_write_json
from ..logging_utils import get_logger
from ..vrs.provider import AriaProvider, RectifyParams
from .streams import (associate_by_timestamp, project_device_points_to_rgb,
                      read_gaze_samples, read_hand_tracking)
from .timeline import NS_PER_S, measure_rate

log = get_logger("ingest.extract")


@dataclass
class ExtractionRequest:
    """A time-window request. Never expressed in frames."""

    recording_id: str
    domain: str
    source_file: str
    source_file_sha256: str
    start_time_s: Optional[float] = None
    end_time_s: Optional[float] = None
    # Absolute device timestamps take precedence when given, so a segment chosen by
    # a previous stage is reproduced exactly rather than re-derived from an offset.
    start_timestamp_ns: Optional[int] = None
    end_timestamp_ns: Optional[int] = None
    frame_step: int = 1
    # Explicitly declared temporal decimation for scouting passes: keep the frame
    # nearest to each point of a uniform time grid. It never interpolates and never
    # selects the same source frame twice, and it is recorded in the manifest so a
    # decimated set can never be mistaken for a full-rate extraction.
    sample_interval_s: Optional[float] = None
    max_frames: Optional[int] = None

    def validate(self) -> None:
        if self.frame_step < 1:
            raise ValueError("frame_step must be >= 1")
        if self.sample_interval_s is not None and self.sample_interval_s <= 0:
            raise ValueError("sample_interval_s must be positive")
        if self.sample_interval_s is not None and self.frame_step > 1:
            raise ValueError("use either frame_step or sample_interval_s, not both")
        if self.start_time_s is not None and self.end_time_s is not None \
                and self.end_time_s < self.start_time_s:
            raise ValueError("end_time_s precedes start_time_s")
        if self.start_timestamp_ns is not None and self.end_timestamp_ns is not None \
                and self.end_timestamp_ns < self.start_timestamp_ns:
            raise ValueError("end_timestamp_ns precedes start_timestamp_ns")


def select_indices(timestamps_ns: np.ndarray, request: ExtractionRequest) -> List[int]:
    """Select source indices for a time window, using timestamps only."""
    request.validate()
    ts = np.asarray(timestamps_ns, dtype=np.int64)
    if ts.size == 0:
        return []
    keep = np.ones(ts.shape, dtype=bool)

    if request.start_timestamp_ns is not None:
        keep &= ts >= int(request.start_timestamp_ns)
    elif request.start_time_s is not None:
        keep &= ts >= ts[0] + int(round(float(request.start_time_s) * NS_PER_S))

    if request.end_timestamp_ns is not None:
        keep &= ts <= int(request.end_timestamp_ns)
    elif request.end_time_s is not None:
        keep &= ts <= ts[0] + int(round(float(request.end_time_s) * NS_PER_S))

    idxs = [int(i) for i in np.flatnonzero(keep)]
    if request.frame_step > 1:
        idxs = idxs[::request.frame_step]
    elif request.sample_interval_s is not None and idxs:
        window = ts[idxs]
        step_ns = int(round(float(request.sample_interval_s) * NS_PER_S))
        grid = np.arange(int(window[0]), int(window[-1]) + 1, step_ns, dtype=np.int64)
        picked = {int(idxs[int(np.argmin(np.abs(window - g)))]) for g in grid}
        idxs = sorted(picked)
    if request.max_frames is not None:
        idxs = idxs[: int(request.max_frames)]
    return idxs


def _encode_jpeg(image: np.ndarray, quality: int) -> bytes:
    import cv2
    ok, buf = cv2.imencode(".jpg", image[:, :, ::-1], [cv2.IMWRITE_JPEG_QUALITY, int(quality)])
    if not ok:
        raise RuntimeError("jpeg encode failed")
    return buf.tobytes()


def _pinhole_summary(calib) -> Dict[str, Any]:
    return {
        "model": str(calib.get_model_name()),
        "image_size": [int(x) for x in calib.get_image_size()],
        "focal_lengths": [float(x) for x in calib.get_focal_lengths()],
        "principal_point": [float(x) for x in calib.get_principal_point()],
        "T_device_camera": np.asarray(
            calib.get_transform_device_camera().to_matrix()).tolist(),
    }


def run_timestamped_extract(request: ExtractionRequest, out_dir: str | Path,
                            cfg: Config, resume: bool = True,
                            gaze_window_s: float = 0.10,
                            hand_window_s: float = 0.10) -> Dict[str, Any]:
    """Extract a time window with full provenance plus gaze/hand sidecars."""
    import pandas as pd

    out = Path(out_dir)
    frames_dir = out / "frames"
    orig_dir = frames_dir / "original"
    rect_dir = frames_dir / "rectified"
    for d in (orig_dir, rect_dir):
        d.mkdir(parents=True, exist_ok=True)

    provider = AriaProvider(request.source_file, cfg.get("vrs.time_domain", "DEVICE_TIME"))
    rgb_label = cfg.get("vrs.rgb_label", "camera-rgb")
    quality = int(cfg.get("frames.jpeg_quality", 92))

    rp = RectifyParams(
        out_width=int(cfg.get("rectify.out_width", 2016)),
        out_height=int(cfg.get("rectify.out_height", 1512)),
        focal=float(cfg.get("rectify.focal", 879.0)),
        rotate_ccw90=int(cfg.get("rectify.rotate_ccw90", 0)),
    )
    do_rect = bool(cfg.get("rectify.enabled", True))
    rectifier = provider.rectifier(rp, rgb_label) if do_rect else None

    atomic_write_json(frames_dir / "calibration.json", {
        "recording_id": request.recording_id,
        "domain": request.domain,
        "source_file_sha256": request.source_file_sha256,
        "source_calib": provider.calib_summary(rgb_label),
        "rectify": rp.__dict__,
        "rectified_pinhole": _pinhole_summary(rectifier.pinhole) if rectifier else None,
        "T_device_cpf": provider.T_device_cpf.tolist(),
    })

    all_ts = provider.rgb_timestamps_ns(rgb_label)
    rate = measure_rate(all_ts)
    effective_fps = rate.effective_fps
    if effective_fps is None:
        raise RuntimeError(f"{request.recording_id}: cannot measure the RGB rate")

    indices = select_indices(all_ts, request)
    if not indices:
        raise RuntimeError(f"{request.recording_id}: the requested window selects no frame")
    sid = str(provider.rgb_stream(rgb_label))
    log.info("%s: extracting %d frames (source has %d, measured %.4f fps)",
             request.recording_id, len(indices), rate.count, effective_fps)

    records: List[Dict[str, Any]] = []
    errors = 0
    for k, i in enumerate(indices):
        name = f"frame_{i:06d}.jpg"
        original = orig_dir / name
        rectified = rect_dir / name
        record: Dict[str, Any] = {
            "recording_id": request.recording_id,
            "domain": request.domain,
            "source_file_sha256": request.source_file_sha256,
            "source_stream_id": sid,
            "source_frame_index": int(i),
            # kept for compatibility with the frozen pipeline, always identical to
            # source_frame_index: a frame is never renumbered by a local counter
            "frame_index": int(i),
            "timestamp_ns": int(all_ts[i]),
            "capture_timestamp_ns": int(all_ts[i]),
            "timestamp_s": float(all_ts[i] / NS_PER_S),
            "time_since_recording_start_s": float((all_ts[i] - all_ts[0]) / NS_PER_S),
            "effective_fps": float(effective_fps),
            "original_path": f"frames/original/{name}",
            "rectified_path": f"frames/rectified/{name}" if do_rect else None,
            "rotate_ccw90": rp.rotate_ccw90,
            "valid": False,
            "extraction_status": "pending",
        }
        if resume and original.exists() and (not do_rect or rectified.exists()):
            raw_cfg = provider.rgb_config(rgb_label)
            record.update({"width": int(raw_cfg["width"]), "height": int(raw_cfg["height"]),
                           "valid": True, "extraction_status": "reused"})
            records.append(record)
            continue
        try:
            raw, ts = provider.rgb_by_index(i, rgb_label)
            if int(ts) != int(all_ts[i]):
                # The index->timestamp mapping must be exact; a mismatch means the
                # stream is not what the timestamp table described.
                record["extraction_status"] = "timestamp_mismatch"
                record["decoded_timestamp_ns"] = int(ts)
                records.append(record)
                errors += 1
                continue
            record["width"], record["height"] = int(raw.shape[1]), int(raw.shape[0])
            atomic_write_bytes(original, _encode_jpeg(raw, quality))
            if do_rect:
                atomic_write_bytes(rectified, _encode_jpeg(rectifier.rectify(raw), quality))
            record["valid"] = True
            record["extraction_status"] = "extracted"
        except Exception as exc:
            errors += 1
            record["extraction_status"] = f"error: {exc}"
            log.warning("frame %d failed: %s", i, exc)
        records.append(record)
        if (k + 1) % 50 == 0:
            log.info("  %d/%d", k + 1, len(indices))

    frame_df = pd.DataFrame(records).sort_values("source_frame_index")
    frame_df.to_parquet(frames_dir / "frames.parquet", index=False)
    frame_df.to_csv(frames_dir / "frames.csv", index=False)

    selected_ts = np.array([r["timestamp_ns"] for r in records], dtype=np.int64)
    selected_idx = [r["source_frame_index"] for r in records]

    sidecars = {
        "gaze": _write_gaze_sidecar(provider, frames_dir, selected_ts, selected_idx,
                                    request, gaze_window_s, cfg),
        "hand_tracking": _write_hand_sidecar(provider, frames_dir, selected_ts,
                                             selected_idx, request, hand_window_s, cfg),
    }

    summary = {
        "schema": "article1_timestamped_extraction_v1",
        "generated_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "recording_id": request.recording_id,
        "domain": request.domain,
        "source_file": request.source_file,
        "source_file_sha256": request.source_file_sha256,
        "source_stream_id": sid,
        "source_frame_count": int(rate.count),
        "source_effective_fps": float(effective_fps),
        "source_effective_fps_span": rate.effective_fps_span,
        "request": {
            "start_time_s": request.start_time_s, "end_time_s": request.end_time_s,
            "start_timestamp_ns": request.start_timestamp_ns,
            "end_timestamp_ns": request.end_timestamp_ns,
            "frame_step": request.frame_step, "max_frames": request.max_frames,
        },
        "declared_decimation": request.frame_step > 1,
        "resampled": False,
        "interpolated": False,
        "synthetic_frames": 0,
        "selected_frames": len(records),
        "extracted": int(sum(1 for r in records if r["extraction_status"] == "extracted")),
        "reused": int(sum(1 for r in records if r["extraction_status"] == "reused")),
        "errors": errors,
        "first_timestamp_ns": int(selected_ts[0]),
        "last_timestamp_ns": int(selected_ts[-1]),
        "window_duration_s": float((selected_ts[-1] - selected_ts[0]) / NS_PER_S),
        "sidecars": sidecars,
    }
    atomic_write_json(frames_dir / "extraction_summary.json", summary)
    log.info("%s: %d frames, %d errors -> %s", request.recording_id,
             len(records), errors, frames_dir)
    return summary


def _write_gaze_sidecar(provider, frames_dir: Path, frame_ts: np.ndarray,
                        frame_idx: List[int], request: ExtractionRequest,
                        window_s: float, cfg: Config) -> Dict[str, Any]:
    """Associate gaze samples to frames by timestamp, keeping the whole window."""
    import pandas as pd

    label = cfg.get("vrs.eyegaze_label", "eyegaze")
    samples = read_gaze_samples(provider, label)
    if not samples:
        return {"available": False, "reason": f"no {label} stream"}

    sample_ts = np.array([s.timestamp_ns for s in samples], dtype=np.int64)
    assoc = associate_by_timestamp(frame_ts, sample_ts, window_s, frame_idx)

    rows: List[Dict[str, Any]] = []
    for a in assoc:
        near = samples[a.nearest_index] if a.nearest_index is not None else None
        window = [samples[j] for j in a.window_indices]
        valid_window = [s for s in window if s.combined_valid]
        rows.append({
            "recording_id": request.recording_id,
            "domain": request.domain,
            "source_frame_index": a.frame_index,
            "frame_index": a.frame_index,
            "frame_timestamp_ns": a.frame_timestamp_ns,
            "gaze_window_s": window_s,
            "gaze_samples_in_window": len(window),
            "gaze_valid_samples_in_window": len(valid_window),
            "nearest_gaze_index": a.nearest_index,
            "nearest_gaze_timestamp_ns": a.nearest_timestamp_ns,
            "nearest_gaze_dt_ms": a.nearest_dt_ms,
            "nearest_gaze_combined_valid": bool(near.combined_valid) if near else False,
            "nearest_gaze_spatial_valid": bool(near.spatial_valid) if near else False,
            "nearest_gaze_yaw_rad": near.yaw_rad if near else None,
            "nearest_gaze_pitch_rad": near.pitch_rad if near else None,
            "nearest_gaze_depth_m": near.depth_m if near else None,
            "window_gaze_indices": ",".join(str(j) for j in a.window_indices),
            "window_gaze_dt_ms": ",".join(f"{d:.3f}" for d in a.window_dt_ms),
        })
    df = pd.DataFrame(rows)
    df.to_parquet(frames_dir / "gaze_association.parquet", index=False)
    df.to_csv(frames_dir / "gaze_association.csv", index=False)
    return {
        "available": True, "stream": label, "window_s": window_s,
        "total_samples": len(samples),
        "frames_with_valid_sample_in_window":
            int(sum(1 for r in rows if r["gaze_valid_samples_in_window"] > 0)),
        "path": "frames/gaze_association.parquet",
        "note": ("association is by timestamp only; several gaze samples per frame "
                 "are preserved and never collapsed"),
    }


def _write_hand_sidecar(provider, frames_dir: Path, frame_ts: np.ndarray,
                        frame_idx: List[int], request: ExtractionRequest,
                        window_s: float, cfg: Config) -> Dict[str, Any]:
    """Associate hand-tracking samples to frames by timestamp.

    Hand tracking is a proxy signal, never ground truth: a `present=False` side is
    recorded as a tracker outcome, not as evidence that the hand is not visible.
    """
    import pandas as pd

    label = cfg.get("vrs.handtracking_label", "handtracking")
    samples = read_hand_tracking(provider, label)
    if not samples:
        return {"available": False, "reason": f"no {label} stream"}

    sample_ts = np.array([s.timestamp_ns for s in samples], dtype=np.int64)
    assoc = associate_by_timestamp(frame_ts, sample_ts, window_s, frame_idx)

    rows: List[Dict[str, Any]] = []
    for a in assoc:
        near = samples[a.nearest_index] if a.nearest_index is not None else None
        row: Dict[str, Any] = {
            "recording_id": request.recording_id,
            "domain": request.domain,
            "source_frame_index": a.frame_index,
            "frame_index": a.frame_index,
            "frame_timestamp_ns": a.frame_timestamp_ns,
            "hand_window_s": window_s,
            "hand_samples_in_window": len(a.window_indices),
            "nearest_hand_index": a.nearest_index,
            "nearest_hand_timestamp_ns": a.nearest_timestamp_ns,
            "nearest_hand_dt_ms": a.nearest_dt_ms,
            "hand_tracking_available": True,
        }
        for side in ("left", "right"):
            s = getattr(near, side) if near else None
            row[f"{side}_hand_tracked"] = bool(s.present) if s else False
            row[f"{side}_hand_confidence"] = s.confidence if s and s.present else None
            wrist = px = py = None
            if s and s.present and s.landmarks_device:
                pts = project_device_points_to_rgb(provider, s.landmarks_device)
                inside = [p for p in pts if p is not None]
                row[f"{side}_hand_landmarks_projected"] = len(inside)
                row[f"{side}_hand_landmarks_total"] = len(pts)
                if inside:
                    px = float(np.mean([p[0] for p in inside]))
                    py = float(np.mean([p[1] for p in inside]))
            if s and s.present and s.wrist_device:
                w = project_device_points_to_rgb(provider, [s.wrist_device])[0]
                wrist = w
            row[f"{side}_hand_centroid_x"] = px
            row[f"{side}_hand_centroid_y"] = py
            row[f"{side}_hand_wrist_x"] = wrist[0] if wrist else None
            row[f"{side}_hand_wrist_y"] = wrist[1] if wrist else None
        rows.append(row)

    df = pd.DataFrame(rows)
    df.to_parquet(frames_dir / "hand_tracking_association.parquet", index=False)
    df.to_csv(frames_dir / "hand_tracking_association.csv", index=False)
    return {
        "available": True, "stream": label, "window_s": window_s,
        "total_samples": len(samples),
        "frames_with_left_tracked": int(sum(1 for r in rows if r["left_hand_tracked"])),
        "frames_with_right_tracked": int(sum(1 for r in rows if r["right_hand_tracked"])),
        "path": "frames/hand_tracking_association.parquet",
        "note": ("hand tracking is a proxy: an untracked side is a tracker outcome, "
                 "not evidence that the hand is absent from the image"),
    }
