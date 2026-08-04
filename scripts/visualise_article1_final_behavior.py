#!/usr/bin/env python3
"""Phase 9: the nine publication figures plus the three final additions."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from aria_drive_seg.article1.final_plots import plot_final
from aria_drive_seg.config import Config
from aria_drive_seg.logging_utils import get_logger, setup_logging

log = get_logger("behavior.final.figures")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/article1/behavior_analysis.yaml")
    ap.add_argument("--gaze-config",
                    default="configs/article1/fast_semantic_gaze.yaml")
    ap.add_argument("--run-root", default="output/article1/fast_semantic_gaze")
    ap.add_argument("--analysis", default="output/article1/fast_semantic_gaze")
    ap.add_argument("--final-analysis",
                    default="output/article1/final_behavior_statistics")
    ap.add_argument("--figures",
                    default="reports/article1_final_behavior_statistics/figures")
    ap.add_argument("--log-level", default="INFO")
    args = ap.parse_args()
    setup_logging(args.log_level)

    cfg = Config.load(args.config)
    gaze_cfg = Config.load(args.gaze_config)
    run_root = Path(args.run_root)
    run_dirs = {"car": run_root / "car", "motorcycle": run_root / "motorcycle"}
    recordings = {d: str(spec["recording_id"])
                  for d, spec in (cfg.get("recordings") or {}).items()}

    manifest = plot_final(run_dirs, args.analysis, args.final_analysis,
                          args.figures, gaze_cfg, recordings)
    log.info("%d figures written to %s", len(manifest), args.figures)
    print(json.dumps([entry["stem"] for entry in manifest], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
