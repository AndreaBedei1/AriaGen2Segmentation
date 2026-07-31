#!/usr/bin/env python3
"""Extract one time window at the recording's native rate, with full provenance.

The window is given in absolute device timestamps so the segment chosen by the
selection stage is reproduced exactly, without re-deriving it from an offset.

Runs in the VRS I/O environment.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from aria_drive_seg.config import Config
from aria_drive_seg.hashing import sha256_file
from aria_drive_seg.ingestion.extract import ExtractionRequest, run_timestamped_extract
from aria_drive_seg.logging_utils import get_logger, setup_logging

log = get_logger("segment.extract")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--vrs", required=True)
    ap.add_argument("--recording-id", required=True)
    ap.add_argument("--domain", required=True, choices=["car", "motorcycle"])
    ap.add_argument("--output", required=True)
    ap.add_argument("--start-timestamp-ns", type=int, default=None)
    ap.add_argument("--end-timestamp-ns", type=int, default=None)
    ap.add_argument("--start-time-s", type=float, default=None)
    ap.add_argument("--end-time-s", type=float, default=None)
    ap.add_argument("--sha256", default=None,
                    help="known SHA-256 of the source; recomputed if omitted")
    ap.add_argument("--gaze-window-s", type=float, default=0.10)
    ap.add_argument("--hand-window-s", type=float, default=0.10)
    ap.add_argument("--config", default=None)
    ap.add_argument("--log-level", default="INFO")
    args = ap.parse_args()
    setup_logging(args.log_level)

    digest = args.sha256 or sha256_file(args.vrs)
    cfg = Config.load(args.config)
    request = ExtractionRequest(
        recording_id=args.recording_id, domain=args.domain,
        source_file=str(Path(args.vrs).resolve()), source_file_sha256=digest,
        start_timestamp_ns=args.start_timestamp_ns,
        end_timestamp_ns=args.end_timestamp_ns,
        start_time_s=args.start_time_s, end_time_s=args.end_time_s,
    )
    summary = run_timestamped_extract(
        request, args.output, cfg,
        gaze_window_s=args.gaze_window_s, hand_window_s=args.hand_window_s)
    print(json.dumps({k: v for k, v in summary.items() if k != "sidecars"}, indent=2))
    print(json.dumps(summary["sidecars"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
