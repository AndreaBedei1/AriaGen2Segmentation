#!/usr/bin/env python3
"""Generate pre-GT and post-hoc gaze diagnostics for a completed temporal run."""
from __future__ import annotations

import argparse

from aria_drive_seg.article1.temporal_metrics import (
    compute_gaze_diagnostics, compute_temporal_metrics)
from aria_drive_seg.taxonomy import Taxonomy


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--temporal", required=True)
    parser.add_argument("--frames", required=True)
    parser.add_argument("--taxonomy", default="configs/article1/classes_article1.yaml")
    parser.add_argument("--reports", default="reports")
    args = parser.parse_args()
    metrics = compute_temporal_metrics(
        args.temporal, Taxonomy.load(args.taxonomy), args.reports)
    gaze = compute_gaze_diagnostics(args.temporal, args.frames)
    print({"frames": metrics["frame_count"], "gaze_diagnostics": str(gaze) if gaze else None})


if __name__ == "__main__":
    main()
