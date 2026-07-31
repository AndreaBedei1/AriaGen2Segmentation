#!/usr/bin/env python3
"""Extract a declared, uniformly-spaced scouting subsample of both recordings.

The scouting set exists to *rank* candidate segments and annotation frames. It is
an explicitly declared temporal decimation (recorded as such in the extraction
summary), never a substitute for the full-rate extraction used by the pipeline.

Runs in the VRS I/O environment.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from aria_drive_seg.config import Config
from aria_drive_seg.ingestion.extract import ExtractionRequest, run_timestamped_extract
from aria_drive_seg.logging_utils import get_logger, setup_logging

log = get_logger("scout.extract")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest",
                    default="reports/article1_motorcycle_ingestion/acquisition_manifest.json")
    ap.add_argument("--work", default="output/article1/ingestion")
    ap.add_argument("--interval-s", type=float, default=1.0,
                    help="scouting sample interval in SECONDS (rate agnostic)")
    ap.add_argument("--log-level", default="INFO")
    args = ap.parse_args()
    setup_logging(args.log_level)

    manifest = json.loads(Path(args.manifest).read_text())
    cfg = Config.load()
    selected = {d: v["selected_recording_id"]
                for d, v in manifest["selection"].items()}
    by_id = {f["recording_id"]: f for f in manifest["files"]
             if f.get("recording_id")}

    out = {}
    for domain, rec_id in selected.items():
        if not rec_id:
            log.warning("no selected recording for domain %s", domain)
            continue
        f = by_id[rec_id]
        request = ExtractionRequest(
            recording_id=rec_id, domain=domain,
            source_file=f["absolute_path"],
            source_file_sha256=f["sha256"],
            sample_interval_s=args.interval_s,
        )
        target = Path(args.work) / "scout" / rec_id
        log.info("extracting scouting frames for %s every %.2f s",
                 rec_id, args.interval_s)
        summary = run_timestamped_extract(request, target, cfg)
        out[rec_id] = {"path": str(target), "frames": summary["selected_frames"],
                       "domain": domain}

    print(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
