#!/usr/bin/env python3
"""Run the final offline Article 1 semantic-camera presentation pass."""
from __future__ import annotations

import argparse
from pathlib import Path

from aria_drive_seg.article1.semantic_camera_final import \
    run_semantic_camera_final_pass
from aria_drive_seg.config import Config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--frames", required=True)
    parser.add_argument("--semantic-camera", required=True)
    parser.add_argument("--previous-presentation", required=True)
    parser.add_argument(
        "--output",
        default=(
            "outputs/article1/auto_temporal_180_210/"
            "semantic_camera_final_pass"))
    parser.add_argument(
        "--config",
        default="configs/article1/semantic_camera_final.yaml")
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--no-resume", action="store_true",
        help="do not reuse already complete final-pass frames")
    args = parser.parse_args()
    cfg = Config.load(Path(args.config))
    raise SystemExit(run_semantic_camera_final_pass(
        args.frames,
        args.semantic_camera,
        args.previous_presentation,
        cfg,
        args.output,
        resume=not args.no_resume,
        force=args.force,
    ))


if __name__ == "__main__":
    main()
