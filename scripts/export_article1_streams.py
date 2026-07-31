#!/usr/bin/env python3
"""Export the multimodal streams of each selected recording to tabular files.

Gaze, hand tracking and GPS are exported at their own native rates with their own
timestamps; nothing is resampled onto the RGB grid here. Downstream stages associate
them to frames by timestamp.

Runs in the VRS I/O environment.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from aria_drive_seg.config import Config
from aria_drive_seg.ingestion.streams import (project_device_points_to_rgb,
                                              read_gaze_samples, read_gps,
                                              read_hand_tracking)
from aria_drive_seg.ingestion.timeline import measure_rate
from aria_drive_seg.io_utils import atomic_write_json
from aria_drive_seg.logging_utils import get_logger, setup_logging
from aria_drive_seg.vrs.provider import AriaProvider

log = get_logger("streams.export")


def export_hands(provider, rec_id: str, domain: str, out: Path) -> dict:
    samples = read_hand_tracking(provider)
    if not samples:
        return {"available": False}
    cfg = provider.rgb_config()
    width, height = int(cfg["width"]), int(cfg["height"])

    stream_label = cfg_label = "handtracking"
    rows = []
    for s in samples:
        row = {"recording_id": rec_id, "domain": domain,
               "hand_sample_index": s.index, "timestamp_ns": s.timestamp_ns,
               "stream_source": stream_label}
        for side in ("left", "right"):
            side_sample = getattr(s, side)
            # handedness is explicit rather than implied by the column prefix
            row[f"{side}_handedness"] = side
            row[f"{side}_hand_tracked"] = bool(side_sample.present)
            row[f"{side}_hand_confidence"] = side_sample.confidence
            in_frame = total = 0
            cx = cy = None
            projected = None
            if side_sample.present and side_sample.landmarks_device:
                pts = project_device_points_to_rgb(provider, side_sample.landmarks_device)
                total = len(pts)
                inside = [p for p in pts
                          if p is not None and 0 <= p[0] < width and 0 <= p[1] < height]
                in_frame = len(inside)
                if inside:
                    cx = float(np.mean([p[0] for p in inside]))
                    cy = float(np.mean([p[1] for p in inside]))
                projected = pts
            # the raw 3D landmarks and their RGB projection are kept in full; a
            # count and a centroid are summaries, not a substitute for the data
            row[f"{side}_landmarks_device"] = json.dumps(
                side_sample.landmarks_device) if side_sample.present else None
            row[f"{side}_landmarks_rgb"] = json.dumps(projected) if projected else None
            row[f"{side}_wrist_device"] = json.dumps(side_sample.wrist_device) \
                if side_sample.present else None
            row[f"{side}_palm_device"] = json.dumps(side_sample.palm_device) \
                if side_sample.present else None
            row[f"{side}_landmarks_total"] = total
            row[f"{side}_landmarks_in_frame"] = in_frame
            row[f"{side}_centroid_x"] = cx
            row[f"{side}_centroid_y"] = cy
        row["any_hand_tracked"] = row["left_hand_tracked"] or row["right_hand_tracked"]
        row["any_landmark_in_frame"] = (row["left_landmarks_in_frame"] > 0
                                        or row["right_landmarks_in_frame"] > 0)
        rows.append(row)

    df = pd.DataFrame(rows)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out, index=False)
    df.to_csv(out.with_suffix(".csv"), index=False)

    rate = measure_rate(df["timestamp_ns"].to_numpy(np.int64))
    return {
        "available": True, "path": str(out), "samples": len(df),
        "effective_hz": rate.effective_fps,
        "left_tracked_fraction": float(df["left_hand_tracked"].mean()),
        "right_tracked_fraction": float(df["right_hand_tracked"].mean()),
        "any_tracked_fraction": float(df["any_hand_tracked"].mean()),
        "any_landmark_in_rgb_fraction": float(df["any_landmark_in_frame"].mean()),
        "note": ("tracked != visible: the device can track a hand that is outside "
                 "the RGB field of view, and can fail on a perfectly visible hand"),
    }


def export_gaze(provider, rec_id: str, domain: str, out: Path) -> dict:
    samples = read_gaze_samples(provider)
    if not samples:
        return {"available": False}
    df = pd.DataFrame([{"recording_id": rec_id, "domain": domain, **s.to_dict()}
                       for s in samples])
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out, index=False)
    rate = measure_rate(df["timestamp_ns"].to_numpy(np.int64))
    return {
        "available": True, "path": str(out), "samples": len(df),
        "effective_hz": rate.effective_fps,
        "combined_valid_fraction": float(df["combined_valid"].mean()),
        "spatial_valid_fraction": float(df["spatial_valid"].mean()),
    }


def export_gps(provider, rec_id: str, domain: str, out: Path) -> dict:
    samples = read_gps(provider)
    if not samples:
        return {"available": False}
    df = pd.DataFrame([{"recording_id": rec_id, "domain": domain, **s.to_dict()}
                       for s in samples])
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out, index=False)
    df.to_csv(out.with_suffix(".csv"), index=False)
    finite = df.dropna(subset=["latitude", "longitude"])
    nonzero = finite[(finite.latitude.abs() > 1e-7) | (finite.longitude.abs() > 1e-7)]
    return {
        "available": True, "path": str(out), "samples": len(df),
        "with_position": int(len(nonzero)),
        "position_fraction": float(len(nonzero) / len(df)) if len(df) else 0.0,
        "median_accuracy_m": (float(nonzero["accuracy"].median())
                              if len(nonzero) and nonzero["accuracy"].notna().any()
                              else None),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest",
                    default="reports/article1_motorcycle_ingestion/acquisition_manifest.json")
    ap.add_argument("--work", default="output/article1/ingestion")
    ap.add_argument("--log-level", default="INFO")
    args = ap.parse_args()
    setup_logging(args.log_level)

    manifest = json.loads(Path(args.manifest).read_text())
    cfg = Config.load()
    work = Path(args.work)
    summary = {}

    for domain, sel in manifest["selection"].items():
        rec_id = sel["selected_recording_id"]
        if not rec_id:
            continue
        entry = next(f for f in manifest["files"] if f.get("recording_id") == rec_id)
        log.info("exporting streams of %s (%s)", rec_id, domain)
        provider = AriaProvider(entry["absolute_path"],
                                cfg.get("vrs.time_domain", "DEVICE_TIME"))
        summary[rec_id] = {
            "domain": domain,
            "hand_tracking": export_hands(provider, rec_id, domain,
                                          work / "hand_tracking" / f"{rec_id}.parquet"),
            "gaze": export_gaze(provider, rec_id, domain,
                                work / "gaze" / f"{rec_id}.parquet"),
            "gps": export_gps(provider, rec_id, domain,
                              work / "gps" / f"{rec_id}.parquet"),
        }

    atomic_write_json(work / "stream_export_summary.json", summary)
    print(json.dumps(summary, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
