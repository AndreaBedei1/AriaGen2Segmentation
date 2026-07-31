#!/usr/bin/env python3
"""Phase 4: rank candidate 30-second motorcycle segments and select one.

The ranking is written out in full so the decision is reviewable and reversible.
Runs in the VRS I/O environment.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from aria_drive_seg.ingestion.rgb_scan import RgbScan
from aria_drive_seg.ingestion.segment_select import build_candidates, select_segment
from aria_drive_seg.ingestion.route_align import build_track
from aria_drive_seg.ingestion.streams import (GpsSample, associate_by_timestamp,
                                              valid_gps)
from aria_drive_seg.io_utils import atomic_write_json
from aria_drive_seg.logging_utils import get_logger, setup_logging

log = get_logger("segment.select")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest",
                    default="reports/article1_motorcycle_ingestion/acquisition_manifest.json")
    ap.add_argument("--work", default="output/article1/ingestion")
    ap.add_argument("--domain", default="motorcycle")
    ap.add_argument("--duration-s", type=float, default=30.0)
    ap.add_argument("--stride-s", type=float, default=5.0)
    ap.add_argument("--scout", default=None, help="scouting semantics .npz")
    ap.add_argument("--reports", default="reports/article1_motorcycle_ingestion")
    ap.add_argument("--log-level", default="INFO")
    args = ap.parse_args()
    setup_logging(args.log_level)

    manifest = json.loads(Path(args.manifest).read_text())
    rec_id = manifest["selection"][args.domain]["selected_recording_id"]
    if not rec_id:
        raise SystemExit(f"no selected recording for domain {args.domain}")
    entry = next(f for f in manifest["files"] if f.get("recording_id") == rec_id)

    scan = RgbScan.load(Path(args.work) / "rgb_scan" / f"{entry['sha256'][:12]}.npz")
    log.info("loaded scan for %s: %d frames", rec_id, scan.frame_index.size)

    scout = None
    if args.scout and Path(args.scout).exists():
        d = np.load(args.scout, allow_pickle=False)
        scout = {"timestamp_ns": d["timestamp_ns"],
                 "class_fraction": d["class_fraction"],
                 "class_names": [str(x) for x in d["class_names"]]}
        log.info("using scouting semantics for %d frames", len(scout["timestamp_ns"]))

    hand_flags = None
    hand_path = (Path(args.work) / "hand_tracking" / f"{rec_id}.parquet")
    if hand_path.exists():
        ht = pd.read_parquet(hand_path)
        # Hand tracking runs at its own rate with its own indices; it is associated
        # to RGB frames by timestamp, never by index.
        assoc = associate_by_timestamp(
            scan.timestamp_ns, ht["timestamp_ns"].to_numpy(np.int64),
            window_s=0.10, frame_indices=scan.frame_index.tolist())
        tracked = ht["any_hand_tracked"].to_numpy()
        hand_flags = {
            a.frame_index: bool(tracked[a.nearest_index])
            for a in assoc
            if a.nearest_index is not None and abs(a.nearest_dt_ms or 1e9) <= 100.0}
        log.info("using hand-tracking flags for %d of %d RGB frames",
                 len(hand_flags), scan.frame_index.size)

    # GPS speed is the driving gate. Without it the ranking happily picks the
    # descent into a parking garage, whose spiral ramp has huge visual variety and
    # whose parked cars register as traffic.
    speed_flags = None
    gps_path = Path(args.work) / "gps" / f"{rec_id}.parquet"
    if gps_path.exists():
        gps = pd.read_parquet(gps_path)
        usable = valid_gps([
            GpsSample(int(r.Index), int(r.timestamp_ns), r.latitude, r.longitude,
                      r.altitude, r.accuracy, r.speed, r.utc_time_ms)
            for r in gps.itertuples()])
        if len(usable) >= 2:
            track = build_track(rec_id, args.domain,
                               [s.timestamp_ns for s in usable],
                               [s.latitude for s in usable],
                               [s.longitude for s in usable],
                               [s.accuracy for s in usable],
                               [s.speed for s in usable])
            assoc = associate_by_timestamp(
                scan.timestamp_ns, track.timestamp_ns, window_s=1.5,
                frame_indices=scan.frame_index.tolist())
            speed_flags = {
                a.frame_index: float(track.speed_mps[a.nearest_index])
                for a in assoc
                if a.nearest_index is not None
                and abs(a.nearest_dt_ms or 1e9) <= 1500.0}
            log.info("using GPS speed for %d of %d RGB frames",
                     len(speed_flags), scan.frame_index.size)

    candidates = build_candidates(
        scan, args.domain, duration_s=args.duration_s, stride_s=args.stride_s,
        scout=scout, hand_tracked_by_frame=hand_flags, speed_by_frame=speed_flags)
    log.info("evaluated %d candidate windows", len(candidates))

    result = select_segment(candidates)
    result["recording_id"] = rec_id
    result["domain"] = args.domain
    result["source_file_sha256"] = entry["sha256"]
    result["requested_duration_s"] = args.duration_s
    result["stride_s"] = args.stride_s
    result["scout_used"] = scout is not None
    result["hand_tracking_used"] = hand_flags is not None

    reports = Path(args.reports)
    reports.mkdir(parents=True, exist_ok=True)
    atomic_write_json(reports / f"segment_selection_{args.domain}.json", result)
    pd.DataFrame([c.to_dict() for c in candidates]).to_csv(
        reports / f"segment_candidates_{args.domain}.csv", index=False)

    sel = result.get("selected")
    if sel:
        log.info("selected frames %d-%d  t=%.3f-%.3f s  score=%.4f",
                 sel["start_frame_index"], sel["end_frame_index"],
                 sel["start_timestamp_ns"] / 1e9, sel["end_timestamp_ns"] / 1e9,
                 sel["score"])
    print(json.dumps({
        "recording_id": rec_id,
        "selected": None if not sel else {
            "start_frame_index": sel["start_frame_index"],
            "end_frame_index": sel["end_frame_index"],
            "start_timestamp_ns": sel["start_timestamp_ns"],
            "end_timestamp_ns": sel["end_timestamp_ns"],
            "frame_count": sel["frame_count"],
            "duration_s": sel["duration_s"],
            "score": sel["score"],
        },
        "shortlist_size": len(result.get("shortlist", [])),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
