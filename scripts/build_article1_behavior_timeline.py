#!/usr/bin/env python3
"""Phase 1: build the timestamp-aligned multimodal master table per recording.

One row per real RGB frame, with gaze, hand tracking, GPS, both IMUs, the
magnetometer, barometer, PPG, ambient light, the device temperature block and the
SLAM cameras attached by their own capture timestamps. Nothing is interpolated and
no sample is invented; every attachment carries the real sample it used, the
signed temporal distance and whether that distance was inside the tolerance
derived from the two streams' measured cadences.

Runs in the VRS I/O environment (projectaria_tools).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict

import numpy as np

from aria_drive_seg.behavior import EXPLORATORY_MARKER
from aria_drive_seg.behavior.multimodal import (build_multimodal_timeline,
                                                derive_view)
from aria_drive_seg.behavior.sensors import probe_vio, read_all_sensors
from aria_drive_seg.config import Config
from aria_drive_seg.ingestion.streams import read_gps, read_hand_tracking, valid_gps
from aria_drive_seg.io_utils import atomic_write_json
from aria_drive_seg.logging_utils import get_logger, setup_logging
from aria_drive_seg.vrs.provider import AriaProvider

log = get_logger("behavior.timeline")

#: Streams read as bare timestamp arrays; their payload is handled by other stages.
TIMESTAMP_ONLY = ("eyegaze", "handtracking", "gps-app",
                  "slam-front-left", "slam-front-right",
                  "slam-side-left", "slam-side-right",
                  "camera-et-left", "camera-et-right")


def build_for_recording(vrs_path: Path, recording_id: str, domain: str,
                        cfg: Config, out_dir: Path) -> Dict[str, Any]:
    log.info("opening %s", vrs_path)
    provider = AriaProvider(vrs_path)

    ref_ns = provider.rgb_timestamps_ns()
    log.info("%s: %d RGB frames", recording_id, ref_ns.size)

    log.info("reading sensor streams (this decodes every sample, not a subsample)")
    sensors = read_all_sensors(provider)
    for label, series in sensors.items():
        rate = series.rate()
        log.info("  %-12s n=%-8d %.2f Hz", label, len(series),
                 rate.effective_fps or float("nan"))

    extra: Dict[str, np.ndarray] = {}
    for label in TIMESTAMP_ONLY:
        if provider.has_label(label):
            extra[label] = provider.timestamps_ns(label)

    timeline = build_multimodal_timeline(
        recording_id=recording_id, domain=domain,
        reference_ns=ref_ns, streams=sensors, extra_streams=extra,
        absolute_ceiling_s=float(cfg.get("synchronisation.absolute_tolerance_s", 2.0)),
    )

    # --- GPS payload, kept separately: it is the only stream with coordinates.
    gps_all = read_gps(provider)
    gps_ok = valid_gps(gps_all, max_accuracy_m=float(cfg.get("gps.max_accuracy_m", 50.0)))
    log.info("GPS: %d samples, %d with a plausible fix", len(gps_all), len(gps_ok))

    # --- Hand tracking presence only; the landmark payload stays in ingestion.
    hands = read_hand_tracking(provider)
    hand_present = {
        "samples": len(hands),
        "left_tracked_fraction": (float(np.mean([h.left.present for h in hands]))
                                  if hands else 0.0),
        "right_tracked_fraction": (float(np.mean([h.right.present for h in hands]))
                                   if hands else 0.0),
        "caveat": ("hand TRACKING presence, not hand visibility: a visible hand "
                   "can fail to track and a tracked hand can be outside the RGB "
                   "field of view"),
    }

    vio = probe_vio(provider)
    log.info("VIO/SLAM pose usable: %s (%s)", vio["usable"], vio.get("reason"))

    out_dir.mkdir(parents=True, exist_ok=True)
    df = timeline.to_frame()
    df.to_parquet(out_dir / "multimodal_timeline.parquet", index=False)

    # Sensor payloads, at their own native rate, for the physiology and dynamics
    # stages. Written as parquet so nothing has to be re-decoded from the VRS.
    import pandas as pd
    sensor_dir = out_dir / "sensors"
    sensor_dir.mkdir(parents=True, exist_ok=True)
    sensor_summary = {}
    for label, series in sensors.items():
        if not len(series):
            sensor_summary[label] = {"available": False}
            continue
        sdf = pd.DataFrame({"timestamp_ns": series.timestamp_ns, **series.channels})
        sdf.to_parquet(sensor_dir / f"{label.replace('-', '_')}.parquet", index=False)
        sensor_summary[label] = {**series.to_dict(), "rate": series.rate().to_dict()}

    pd.DataFrame([g.to_dict() for g in gps_all]).to_parquet(
        out_dir / "gps_raw.parquet", index=False)
    if gps_ok:
        pd.DataFrame([g.to_dict() for g in gps_ok]).to_parquet(
            out_dir / "gps_valid.parquet", index=False)

    views = {}
    for spec in (cfg.get("synchronisation.derived_views") or []):
        view = derive_view(timeline, float(spec["hz"]), str(spec["purpose"]))
        views[f"{spec['hz']}Hz"] = view.to_dict()
        pd.DataFrame({
            "grid_ns": view.grid_ns,
            "temporal_error_ms": view.temporal_error_ms,
            "master_row": view.row_indices,
            "frame_index": timeline.columns["frame_index"][view.row_indices],
            "timestamp_ns": timeline.columns["timestamp_ns"][view.row_indices],
        }).to_parquet(out_dir / f"view_{str(spec['hz']).replace('.', 'p')}hz.parquet",
                      index=False)

    summary = {
        "schema": "article1_behavior_timeline_v1",
        "result_status": EXPLORATORY_MARKER,
        **timeline.summary(),
        "gps": {
            "raw_samples": len(gps_all),
            "valid_fix_samples": len(gps_ok),
            "valid_fraction": (len(gps_ok) / len(gps_all)) if gps_all else 0.0,
            "max_accuracy_m": float(cfg.get("gps.max_accuracy_m", 50.0)),
        },
        "hand_tracking": hand_present,
        "vio_pose": vio,
        "sensors": sensor_summary,
        "derived_views": views,
        "guarantees": {
            "interpolated": False,
            "resampled": False,
            "synthetic_samples": 0,
            "frame_rate_used_as_feature": False,
            "gaze_used_for_segmentation": False,
        },
    }
    atomic_write_json(out_dir / "timeline_summary.json", summary)
    log.info("wrote %s", out_dir / "multimodal_timeline.parquet")
    return summary


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/article1/behavior_analysis.yaml")
    ap.add_argument("--output", default="output/article1/behavior_analysis")
    ap.add_argument("--domain", default=None,
                    help="restrict to one domain (car / motorcycle)")
    ap.add_argument("--log-level", default="INFO")
    args = ap.parse_args()
    setup_logging(args.log_level)

    cfg = Config.load(args.config)
    out_root = Path(args.output)
    recordings = cfg.get("recordings") or {}

    summaries = {}
    for domain, spec in recordings.items():
        if args.domain and domain != args.domain:
            continue
        vrs = cfg.resolve(spec["vrs"])
        if not vrs.exists():
            raise SystemExit(f"VRS not found: {vrs}")
        summaries[domain] = build_for_recording(
            vrs, spec["recording_id"], domain, cfg,
            out_root / spec["recording_id"])

    atomic_write_json(out_root / "timeline_index.json", {
        "schema": "article1_behavior_timeline_index_v1",
        "result_status": EXPLORATORY_MARKER,
        "domains": {d: {"recording_id": s["recording_id"], "rows": s["rows"],
                        "duration_s": s["duration_s"],
                        "reference_hz": s["reference_hz"]}
                    for d, s in summaries.items()},
    })
    print(json.dumps({d: {"rows": s["rows"], "duration_s": round(s["duration_s"], 2),
                          "reference_hz": round(s["reference_hz"], 4)}
                      for d, s in summaries.items()}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
