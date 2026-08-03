#!/usr/bin/env python3
"""Extract real timestamp-selected RGB frames and project the full gaze stream."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from aria_drive_seg.article1.fast_io import run_fast_extract
from aria_drive_seg.config import Config
from aria_drive_seg.logging_utils import setup_logging


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--vrs", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--recording-id", required=True)
    ap.add_argument("--domain", required=True, choices=["car", "motorcycle"])
    ap.add_argument("--frequency-hz", default=None)
    ap.add_argument("--source-sha256", default=None)
    ap.add_argument("--config", default="configs/article1/fast_semantic_gaze.yaml")
    ap.add_argument("--no-resume", action="store_true")
    ap.add_argument("--log-level", default="INFO")
    args = ap.parse_args()
    setup_logging(args.log_level)
    cfg = Config.load(args.config)
    frequency = None
    if args.frequency_hz is not None:
        frequency = ("native" if args.frequency_hz.lower() == "native"
                     else float(args.frequency_hz))
    summary = run_fast_extract(
        args.vrs, args.output, args.recording_id, args.domain, cfg,
        frequency_hz=frequency, source_sha256=args.source_sha256,
        resume=not args.no_resume)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
