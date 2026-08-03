#!/usr/bin/env python3
"""Run and persist the fast full-run performance/frequency ablation."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from aria_drive_seg.article1.fast_benchmark import run_benchmark
from aria_drive_seg.config import Config
from aria_drive_seg.io_utils import atomic_write_json


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--car-input", default="outputs/article1/auto_temporal_180_210")
    ap.add_argument("--motorcycle-input", default="output/article1/motorcycle_baseline_30s")
    ap.add_argument("--behavior", default="output/article1/behavior_analysis")
    ap.add_argument("--ingestion-gaze", default="output/article1/ingestion/gaze")
    ap.add_argument("--output", default="output/article1/fast_semantic_gaze/profiling")
    ap.add_argument("--config", default="configs/article1/fast_semantic_gaze.yaml")
    ap.add_argument("--window-s", type=float, default=4.0)
    ap.add_argument("--batch-sizes", default="1,4,8,16")
    ap.add_argument("--repeats", type=int, default=3)
    args = ap.parse_args()
    cfg = Config.load(args.config)
    result = run_benchmark(
        cfg, {"car": args.car_input, "motorcycle": args.motorcycle_input},
        args.behavior, args.ingestion_gaze, window_s=args.window_s,
        batch_sizes=[int(v) for v in args.batch_sizes.split(",")],
        repeats=args.repeats)
    out = Path(args.output); out.mkdir(parents=True, exist_ok=True)
    atomic_write_json(out / "benchmark.json", result)
    pd.DataFrame(result["batch_profiles"]).to_csv(out / "batch_profiles.csv", index=False)
    pd.DataFrame(result["temporal_quality"]).to_csv(out / "temporal_quality.csv", index=False)
    pd.DataFrame(result["frequency_coverage"]).to_csv(
        out / "frequency_coverage.csv", index=False)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
