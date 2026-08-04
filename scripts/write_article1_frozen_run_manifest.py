#!/usr/bin/env python3
"""Content hashes of the 5 Hz and native semantic-gaze runs, as a tracked baseline.

Those two runs are inputs to the final behaviour statistics and must not be
modified by them. `output/` is not under version control, so the guard is a
checked-in manifest of content hashes that `tests/test_final_behavior_statistics.py`
verifies.

Regenerating this manifest is therefore a deliberate act: it means the runs
themselves were re-executed on purpose, and the diff will say so.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
from pathlib import Path
from typing import Any, Dict

from aria_drive_seg.hashing import sha256_file
from aria_drive_seg.io_utils import atomic_write_json

#: Per-run artefacts that pin the run's content without hashing 5.5 GB of masks.
RUN_FILES = ("full_run_summary.json", "paired_route_bins.parquet",
             "event_metrics.parquet")
DOMAIN_FILES = ("frames/extraction_summary.json", "frames/frames.parquet",
                "frames/calibration.json", "segmentation/summary.json",
                "segmentation/segmentation_index.parquet",
                "gaze/semantic_gaze_summary.json", "gaze/semantic_gaze.parquet",
                "gaze/fixations.parquet", "gaze/projected_gaze.parquet")


def hash_run(root: Path) -> Dict[str, Any]:
    entries: Dict[str, str] = {}
    for name in RUN_FILES:
        path = root / name
        if path.exists():
            entries[name] = sha256_file(path)
    for domain in ("car", "motorcycle"):
        for name in DOMAIN_FILES:
            path = root / domain / name
            if path.exists():
                entries[f"{domain}/{name}"] = sha256_file(path)
    return entries


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", default=[
        "output/article1/fast_semantic_gaze",
        "output/article1/fast_semantic_gaze_native"])
    ap.add_argument("--out",
                    default="reports/article1_final_behavior_statistics/"
                            "frozen_run_manifest.json")
    args = ap.parse_args()

    manifest = {
        "schema": "article1_frozen_run_manifest_v1",
        "generated_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "purpose": ("content hashes of the semantic-gaze runs consumed by the "
                    "final behaviour statistics. These runs are read-only inputs; "
                    "a hash change means a run was re-executed, which must be a "
                    "deliberate, reviewed act"),
        "runs": {},
    }
    for run in args.runs:
        root = Path(run)
        entries = hash_run(root)
        if not entries:
            raise SystemExit(f"{root}: no artefacts found to hash")
        manifest["runs"][str(root)] = {"files": entries, "file_count": len(entries)}
    atomic_write_json(Path(args.out), manifest)
    print(json.dumps({k: v["file_count"] for k, v in manifest["runs"].items()},
                     indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
