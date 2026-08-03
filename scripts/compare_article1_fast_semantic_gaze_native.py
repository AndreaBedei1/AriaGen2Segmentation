#!/usr/bin/env python3
"""Compare native-frequency fast semantic gaze against the frozen 5 Hz run."""
from __future__ import annotations

import argparse
import json

from aria_drive_seg.article1.fast_native_comparison import build_native_comparison
from aria_drive_seg.config import Config


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline", default="output/article1/fast_semantic_gaze")
    ap.add_argument("--native", default="output/article1/fast_semantic_gaze_native")
    ap.add_argument("--output", default="output/article1/fast_semantic_gaze_native/comparison")
    ap.add_argument("--figures",
                    default="reports/article1_fast_semantic_gaze_native/figures")
    ap.add_argument("--config",
                    default="configs/article1/fast_semantic_gaze_native.yaml")
    args = ap.parse_args()
    result = build_native_comparison(
        args.baseline, args.native, args.output, args.figures,
        Config.load(args.config))
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
