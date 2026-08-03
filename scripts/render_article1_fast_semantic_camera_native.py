#!/usr/bin/env python3
"""Render the separate timestamp-native semantic-camera deliverables."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from aria_drive_seg.article1.fast_render import render_domain, render_shared_route
from aria_drive_seg.config import Config


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--car", default="output/article1/fast_semantic_gaze_native/car")
    ap.add_argument("--motorcycle",
                    default="output/article1/fast_semantic_gaze_native/motorcycle")
    ap.add_argument("--output", default="output/article1/fast_semantic_gaze_native")
    ap.add_argument("--config",
                    default="configs/article1/fast_semantic_gaze_native.yaml")
    ap.add_argument("--domain", choices=["car", "motorcycle", "both"],
                    default="both")
    ap.add_argument("--skip-shared", action="store_true")
    ap.add_argument("--shared-only", action="store_true")
    args = ap.parse_args()
    cfg = Config.load(args.config); out = Path(args.output)
    info = {}
    if not args.shared_only and args.domain in ("car", "both"):
        info["car"] = render_domain(
            args.car, out / "car_semantic_camera_full_native.mp4", cfg,
            out / "previews" / "car_preview_native_30s.mp4")
    if not args.shared_only and args.domain in ("motorcycle", "both"):
        info["motorcycle"] = render_domain(
            args.motorcycle,
            out / "motorcycle_semantic_camera_full_native.mp4", cfg,
            out / "previews" / "motorcycle_preview_native_30s.mp4")
    if (args.shared_only or args.domain == "both") and not args.skip_shared:
        info["shared_route"] = render_shared_route(
            args.car, args.motorcycle,
            out / "paired_shared_route_comparison_native.mp4", cfg)
    print(json.dumps(info, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
