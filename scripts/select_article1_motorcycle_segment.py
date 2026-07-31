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
        hand_flags = {int(r.source_frame_index): bool(r.any_hand_tracked)
                      for r in ht.itertuples()}
        log.info("using hand-tracking flags for %d frames", len(hand_flags))

    candidates = build_candidates(
        scan, args.domain, duration_s=args.duration_s, stride_s=args.stride_s,
        scout=scout, hand_tracked_by_frame=hand_flags)
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
