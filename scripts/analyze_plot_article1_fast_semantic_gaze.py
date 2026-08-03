#!/usr/bin/env python3
"""Build full-run structured tables and the reduced publication figure set."""
from __future__ import annotations

import argparse
import json

from aria_drive_seg.article1.fast_analysis import build_fast_analysis
from aria_drive_seg.article1.fast_plots import plot_all
from aria_drive_seg.config import Config


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--car", default="output/article1/fast_semantic_gaze/car")
    ap.add_argument("--motorcycle", default="output/article1/fast_semantic_gaze/motorcycle")
    ap.add_argument("--output", default="output/article1/fast_semantic_gaze")
    ap.add_argument("--figures", default="reports/article1_fast_semantic_gaze/figures")
    ap.add_argument("--config", default="configs/article1/fast_semantic_gaze.yaml")
    args = ap.parse_args()
    cfg = Config.load(args.config)
    runs = {"car": args.car, "motorcycle": args.motorcycle}
    summary = build_fast_analysis(runs, args.output, cfg)
    figures = plot_all(runs, args.output, args.figures, cfg)
    print(json.dumps({"summary": summary, "figures": figures}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
