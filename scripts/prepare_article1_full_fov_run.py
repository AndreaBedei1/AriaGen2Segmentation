#!/usr/bin/env python3
"""Phase 2: build a run directory that keeps the full valid field of view.

The pipeline segments whatever `rectified_path` points at. The baseline points it
at the pinhole-rectified frame, which discards 43% of the valid field of view. This
script builds a parallel run whose pipeline geometry **is the source fisheye
geometry**, so the same frozen code runs unchanged over the full field of view.

Nothing about the previous run is touched: the frames are symlinked, so the
baseline remains byte-identical and reproducible.

The valid-pixel mask is written alongside. Pixels outside the camera model's
angular limit are still segmented, because the fusion contract is dense, but they
carry no scene information and are excluded from every statistic through the mask.

Runs in the VRS I/O environment.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd

from aria_drive_seg.config import Config
from aria_drive_seg.geometry.fov import (build_letterbox, round_trip_report,
                                         valid_pixel_mask)
from aria_drive_seg.io_utils import atomic_write_json
from aria_drive_seg.logging_utils import get_logger, setup_logging
from aria_drive_seg.vrs.provider import AriaProvider

log = get_logger("fov.prepare")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline-run", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--vrs", required=True)
    ap.add_argument("--model-input", type=int, nargs=2, default=[384, 384],
                    help="the fixed model input the letterbox is reported against")
    ap.add_argument("--config", default=None)
    ap.add_argument("--log-level", default="INFO")
    args = ap.parse_args()
    setup_logging(args.log_level)

    import cv2

    baseline = Path(args.baseline_run)
    out = Path(args.output)
    frames_dir = out / "frames"
    (frames_dir / "original").mkdir(parents=True, exist_ok=True)
    (frames_dir / "rectified").mkdir(parents=True, exist_ok=True)

    cfg = Config.load(args.config)
    provider = AriaProvider(args.vrs, cfg.get("vrs.time_domain", "DEVICE_TIME"))
    rgb_label = cfg.get("vrs.rgb_label", "camera-rgb")
    calib = provider.rgb_calib(rgb_label)
    width, height = (int(x) for x in calib.get_image_size())

    log.info("building the valid-pixel mask for %dx%d", width, height)
    valid = valid_pixel_mask(calib, width, height)
    cv2.imwrite(str(frames_dir / "valid_pixel_mask.png"),
                (valid.astype(np.uint8) * 255))

    frame_index = pd.read_parquet(baseline / "frames" / "frames.parquet")
    log.info("linking %d frames in source geometry", len(frame_index))

    linked = 0
    for _, row in frame_index.iterrows():
        name = Path(row["original_path"]).name
        source = (baseline / row["original_path"]).resolve()
        if not source.exists():
            log.warning("missing source frame %s", source)
            continue
        for sub in ("original", "rectified"):
            target = frames_dir / sub / name
            if target.is_symlink() or target.exists():
                target.unlink()
            os.symlink(source, target)
        linked += 1

    # The pipeline reads `rectified_path`; in this run that path is the source
    # geometry itself, which is the whole point.
    new_index = frame_index.copy()
    new_index["rectified_path"] = new_index["original_path"]
    new_index["pipeline_geometry"] = "source_fisheye_full_fov"
    new_index["width"] = width
    new_index["height"] = height
    new_index.to_parquet(frames_dir / "frames.parquet", index=False)
    new_index.to_csv(frames_dir / "frames.csv", index=False)

    letterbox = build_letterbox((width, height), tuple(args.model_input))
    round_trip = round_trip_report(letterbox, width, height, valid)

    baseline_calibration = json.loads(
        (baseline / "frames" / "calibration.json").read_text())
    document: Dict[str, Any] = {
        "schema": "article1_full_fov_run_v1",
        "generated_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "baseline_run": str(baseline.resolve()),
        "baseline_pipeline_geometry": "linear_pinhole_rectified",
        "pipeline_geometry": "source_fisheye_full_fov",
        "rationale": (
            "rectifying this fisheye onto a pinhole at the same focal length "
            "discards 43% of the valid field of view, including the lateral band "
            "where the motorcycle's bar-end mirrors sit; keeping the source "
            "geometry trades peripheral distortion for that field of view"),
        "source_calibration": baseline_calibration.get("source_calib"),
        "width": width, "height": height,
        "aspect_ratio": width / height,
        "centre_crop": False,
        "aspect_deformed": False,
        "valid_pixel_mask": "frames/valid_pixel_mask.png",
        "valid_pixels": int(np.count_nonzero(valid)),
        "valid_fraction": float(np.count_nonzero(valid) / valid.size),
        "invalid_pixel_policy": (
            "pixels outside the camera model's angular limit are still segmented "
            "because the fusion contract is dense, but they carry no scene "
            "information and are excluded from every statistic through the mask"),
        "model_letterbox": round_trip["transform"],
        "letterbox_round_trip": {k: v for k, v in round_trip.items()
                                 if k != "transform"},
        "frames_linked": linked,
        "frames_are_symlinks": True,
        "baseline_untouched": True,
    }
    atomic_write_json(frames_dir / "full_fov_geometry.json", document)

    for name in ("calibration.json", "extraction_summary.json",
                 "gaze_association.parquet", "hand_tracking_association.parquet"):
        source = baseline / "frames" / name
        if source.exists():
            target = frames_dir / name
            if target.is_symlink() or target.exists():
                target.unlink()
            os.symlink(source.resolve(), target)

    print(json.dumps({
        "output": str(out),
        "frames": linked,
        "geometry": f"{width}x{height}",
        "aspect_ratio": width / height,
        "valid_fraction": document["valid_fraction"],
        "letterbox_round_trip_median_px": round_trip["median_error_px"],
        "letterbox_round_trip_p99_px": round_trip["p99_error_px"],
        "round_trip_passes": round_trip["passes"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
