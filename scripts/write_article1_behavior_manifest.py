#!/usr/bin/env python3
"""Phase 15: trim the committed tables and write the data manifest.

Two of the generated CSVs are full data dumps: every route bin at every bin size,
and every event-response window mean. Committing them adds a megabyte of numbers
that a reader cannot use without the pipeline anyway.

So the committed copy is the analysis slice plus a schema and a sample, and the
manifest records where the full table lives, how many rows it has and what its
columns mean. The full tables stay under the local output tree.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

from aria_drive_seg.behavior import EXPLORATORY_MARKER
from aria_drive_seg.io_utils import atomic_write_json
from aria_drive_seg.logging_utils import get_logger, setup_logging

log = get_logger("behavior.manifest")

SAMPLE_ROWS = 50


def _schema(df: pd.DataFrame) -> List[Dict[str, str]]:
    return [{"column": str(c), "dtype": str(df[c].dtype)} for c in df.columns]


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", default="output/article1/behavior_analysis")
    ap.add_argument("--reports", default="reports/article1_behavior_analysis")
    ap.add_argument("--log-level", default="INFO")
    args = ap.parse_args()
    setup_logging(args.log_level)

    reports = Path(args.reports)
    out_root = Path(args.output)
    full_dir = out_root / "full_tables"
    full_dir.mkdir(parents=True, exist_ok=True)

    entries: List[Dict[str, Any]] = []

    # --- shared_route_bins: commit the analysis bin size only ---------------
    bins_path = reports / "route" / "shared_route_bins.csv"
    if bins_path.exists():
        align = json.loads((reports / "route" /
                            "route_alignment_summary.json").read_text())
        bin_size = float(align["analysis_bin_size_m"])
        df = pd.read_csv(bins_path)
        full = full_dir / "shared_route_bins_all_sizes.csv"
        df.to_csv(full, index=False)
        trimmed = df[df["bin_size_m"] == bin_size]
        trimmed.to_csv(bins_path, index=False)
        entries.append({
            "committed": str(bins_path.relative_to(reports)),
            "committed_rows": int(len(trimmed)),
            "committed_scope": f"analysis bin size only ({bin_size:.0f} m)",
            "full_table_local": str(full),
            "full_rows": int(len(df)),
            "full_scope": "every configured bin size, both domains",
            "schema": _schema(df),
        })
        log.info("shared_route_bins: %d -> %d rows committed", len(df), len(trimmed))

    # --- event_response_metrics: commit a schema and a sample ---------------
    ev_path = reports / "events" / "event_response_metrics.csv"
    if ev_path.exists():
        df = pd.read_csv(ev_path, low_memory=False)
        full = full_dir / "event_response_metrics_full.csv"
        df.to_csv(full, index=False)
        # Keep the identifying columns and the deltas, which are the result; the
        # per-window raw means are reproducible from the parquet.
        keep = [c for c in df.columns
                if not c.startswith(("pre_event_", "approach_", "post_event_",
                                     "phys_delayed_", "event_"))
                or c.startswith("delta_")]
        sample = df[keep].head(SAMPLE_ROWS)
        sample.to_csv(ev_path, index=False)
        entries.append({
            "committed": str(ev_path.relative_to(reports)),
            "committed_rows": int(len(sample)),
            "committed_scope": (f"first {SAMPLE_ROWS} events, identifying columns "
                                "and deltas only"),
            "full_table_local": str(full),
            "full_rows": int(len(df)),
            "full_columns": int(df.shape[1]),
            "full_scope": "every event, every window, every signal family",
            "parquet": str(out_root / "event_response_metrics.parquet"),
            "schema": _schema(df),
        })
        log.info("event_response_metrics: %d x %d -> %d x %d committed",
                 len(df), df.shape[1], len(sample), sample.shape[1])

    # --- everything else is small enough to commit whole --------------------
    for path in sorted(reports.rglob("*.csv")):
        rel = str(path.relative_to(reports))
        if any(e["committed"] == rel for e in entries):
            continue
        df = pd.read_csv(path, low_memory=False)
        entries.append({
            "committed": rel, "committed_rows": int(len(df)),
            "committed_scope": "complete",
            "size_bytes": path.stat().st_size,
            "sha256": _sha256(path),
            "schema": _schema(df),
        })

    manifest = {
        "schema": "article1_behavior_data_manifest_v1",
        "result_status": EXPLORATORY_MARKER,
        "policy": ("committed tables are the analysis slice plus schema and a "
                   "sample; full tables and every parquet stay under the local "
                   "output tree, which is not committed"),
        "privacy": ("no committed table carries a latitude or longitude; "
                    "positions are relative metres and track endpoints are "
                    "trimmed"),
        "local_output_root": str(out_root),
        "tables": entries,
        "parquet_tables_local_only": [
            "multimodal_timeline.parquet", "sensors/*.parquet",
            "gps_raw.parquet", "gps_valid.parquet", "map_matched.parquet",
            "frame_route_bins.parquet", "route_bins_full.parquet",
            "vehicle_dynamics.parquet", "head_dynamics_native.parquet",
            "head_dynamics_per_frame.parquet", "semantic_gaze.parquet",
            "gaze_fixations.parquet", "ppg_beats.parquet",
            "ppg_quality.parquet", "ppg_windows.parquet",
            "event_response_metrics.parquet",
        ],
        "media_local_only": [
            "car_multimodal_analysis.mp4",
            "motorcycle_multimodal_analysis.mp4",
            "paired_route_comparison.mp4",
            "figures/*.png", "dashboard/index.html", "osm_cache/*.json",
        ],
    }
    atomic_write_json(reports / "DATA_MANIFEST.json", manifest)
    log.info("wrote %s with %d tables", reports / "DATA_MANIFEST.json", len(entries))
    print(json.dumps({"tables": len(entries)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
