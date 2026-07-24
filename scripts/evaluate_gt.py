#!/usr/bin/env python3
"""Score predictions against human ground truth (§17.2).

    python scripts/evaluate_gt.py --input run200 --gt run200/gt

Computes mIoU, per-class IoU, pixel accuracy, mean class accuracy, boundary F1,
per-class precision/recall, accuracy-on-assigned, and gaze-class accuracy for every
method present. If <gt> is empty it reports that infra is ready and exits cleanly —
it never fabricates accuracy.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from aria_drive_seg.analyze.metrics_gt import (evaluate_method,
                                              gaussian_gaze_weighted_accuracy,
                                              gaze_class_accuracy)
from aria_drive_seg.taxonomy import Taxonomy


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--gt", required=True, help="dir of GT canonical uint16 id-masks")
    ap.add_argument("--config", default="configs/classes.yaml")
    args = ap.parse_args()

    tax = Taxonomy.load(args.config)
    gt = Path(args.gt)
    if not gt.exists() or not any(gt.glob("frame_*.png")):
        print(json.dumps({"status": "NO GROUND TRUTH YET — infrastructure ready, no accuracy claimed",
                          "gt_dir": str(gt)}, indent=2))
        return 0

    report = {"gt_dir": str(gt), "methods": {}}
    for method in ("grounded_sam2", "oneformer_mapillary"):
        if (Path(args.input) / method / "canonical_masks").exists():
            r = evaluate_method(args.input, method, str(gt), tax)
            r.update(gaze_class_accuracy(args.input, method, str(gt), tax))
            r.update(gaussian_gaze_weighted_accuracy(args.input, method, str(gt), tax))
            report["methods"][method] = r

    out = Path(args.input) / "comparison" / "metrics" / "metrics_gt.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))
    print(json.dumps({m: {k: v for k, v in r.items()
                          if k in ("gt_frames", "mIoU_eval", "pixel_accuracy",
                                   "mean_class_accuracy", "gaze_class_accuracy")}
                      for m, r in report["methods"].items()}, indent=2))
    print(f"\nfull report -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
