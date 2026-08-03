#!/usr/bin/env python3
"""Build native-versus-5 Hz temporal QA cases and contact sheets."""
from __future__ import annotations

import argparse
import json

from aria_drive_seg.article1.fast_native_qa import build_temporal_qa
from aria_drive_seg.config import Config


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline", default="output/article1/fast_semantic_gaze")
    ap.add_argument("--native", default="output/article1/fast_semantic_gaze_native")
    ap.add_argument("--output", default="output/article1/fast_semantic_gaze_native/qa")
    ap.add_argument("--config",
                    default="configs/article1/fast_semantic_gaze_native.yaml")
    args = ap.parse_args()
    result = build_temporal_qa(
        args.baseline, args.native, args.output, Config.load(args.config))
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
