#!/usr/bin/env python3
"""Phase 8B: build the derived comparable timeline used only for preliminary QA.

Snaps both recordings onto a common grid by picking, for each grid point, the
**nearest real frame**. Nothing is interpolated, no frame is created, the car is not
upsampled, and every reused frame is reported with its temporal error.

The output is marked derived and preliminary. It is not the dataset format and must
not be used to train or evaluate the future vehicle classifier.

Runs in the VRS I/O environment.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd
import yaml

from aria_drive_seg.ingestion.rgb_scan import RgbScan
from aria_drive_seg.ingestion.timeline import (NS_PER_S, build_comparable_timeline,
                                               measure_rate)
from aria_drive_seg.io_utils import atomic_write_json
from aria_drive_seg.logging_utils import get_logger, setup_logging

log = get_logger("comparable.timeline")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest",
                    default="reports/article1_motorcycle_ingestion/acquisition_manifest.json")
    ap.add_argument("--protocol", default="configs/article1/dataset_protocol.yaml")
    ap.add_argument("--work", default="output/article1/ingestion")
    ap.add_argument("--reports", default="reports/article1_motorcycle_ingestion")
    ap.add_argument("--duration-s", type=float, default=60.0,
                    help="length of the demonstration window, in seconds")
    ap.add_argument("--log-level", default="INFO")
    args = ap.parse_args()
    setup_logging(args.log_level)

    protocol = yaml.safe_load(Path(args.protocol).read_text())
    mode = protocol["analysis_modes"]["comparable_timeline"]
    grid_hz = float(mode["grid_hz"])
    max_error_ms = float(mode["max_temporal_error_ms"])

    manifest = json.loads(Path(args.manifest).read_text())
    by_id = {f["recording_id"]: f for f in manifest["files"] if f.get("recording_id")}

    doc: Dict[str, Any] = {
        "status": "derived_preliminary",
        "derived": True,
        "preliminary": True,
        "valid_for_scientific_claims": False,
        "grid_hz": grid_hz,
        "selection_rule": mode["selection_rule"],
        "interpolated": False,
        "synthetic_frames": 0,
        "upsampled": False,
        "forbidden_uses": mode["forbidden_uses"],
        "demonstration_window_s": args.duration_s,
        "recordings": {},
        "native_mode": protocol["analysis_modes"]["native"],
    }

    for domain, sel in manifest["selection"].items():
        rec = sel["selected_recording_id"]
        if not rec:
            continue
        entry = by_id[rec]
        scan = RgbScan.load(Path(args.work) / "rgb_scan" / f"{entry['sha256'][:12]}.npz")
        rate = measure_rate(scan.timestamp_ns)
        start = int(scan.timestamp_ns[0])
        end = start + int(args.duration_s * NS_PER_S)
        timeline = build_comparable_timeline(
            scan.timestamp_ns, grid_hz, start, end, max_error_ms)

        err = np.asarray(timeline.temporal_error_ms)
        doc["recordings"][rec] = {
            "domain": domain,
            "native_effective_fps": rate.effective_fps,
            "native_frames_in_window": int(np.sum(
                (scan.timestamp_ns >= start) & (scan.timestamp_ns <= end))),
            "grid_points": len(timeline.grid_ns),
            "selected_real_frames": len(set(timeline.selected_indices)),
            "duplicate_selection_count": timeline.duplicate_selection_count,
            "temporal_error_ms": {
                "mean": float(err.mean()) if err.size else 0.0,
                "median": float(np.median(err)) if err.size else 0.0,
                "p95": float(np.percentile(err, 95)) if err.size else 0.0,
                "max": timeline.max_temporal_error_ms,
            },
            "notes": timeline.notes,
        }

        rows = [{
            "recording_id": rec, "domain": domain,
            "grid_index": i, "grid_timestamp_ns": g,
            "selected_source_frame_index": int(scan.frame_index[idx]),
            "selected_timestamp_ns": ts,
            "temporal_error_ms": e,
            "interpolated": False, "synthetic": False,
        } for i, (g, idx, ts, e) in enumerate(zip(
            timeline.grid_ns, timeline.selected_indices, timeline.selected_ns,
            timeline.temporal_error_ms))]
        seen: Dict[int, int] = {}
        for r in rows:
            k = r["selected_source_frame_index"]
            seen[k] = seen.get(k, 0) + 1
        for r in rows:
            r["reused_real_frame"] = seen[r["selected_source_frame_index"]] > 1
        pd.DataFrame(rows).to_csv(
            Path(args.reports) / f"comparable_timeline_{domain}.csv", index=False)
        log.info("%s: %d grid points from %d real frames, max error %.2f ms, "
                 "%d reuses", domain, len(rows), len(set(timeline.selected_indices)),
                 timeline.max_temporal_error_ms, timeline.duplicate_selection_count)

    atomic_write_json(Path(args.reports) / "comparable_timeline_summary.json", doc)
    print(json.dumps(doc["recordings"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
