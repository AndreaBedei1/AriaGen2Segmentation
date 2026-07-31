#!/usr/bin/env python3
"""Validate the trained cockpit model against reviewed annotations.

Reports every metric per domain as well as pooled, because the car/motorcycle gap
is the domain-shift indicator the article needs.

Hands are scored only on frames a reviewer marked `visible` or
`partially_visible`. On frames marked `out_of_frame`, `occluded`, `not_visible` or
`uncertain` a missing mask is **not** an error and is not counted; a mask that is
present there is counted separately as a false hand region.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

from aria_drive_seg.article1.cockpit_segformer import (COCKPIT_CLASSES, CockpitDataset,
                                                       CockpitSegFormer, domain_iou,
                                                       require_reviewed_annotations)
from aria_drive_seg.ingestion.hand_audit import EVALUABLE_STATES
from aria_drive_seg.io_utils import atomic_write_json
from aria_drive_seg.logging_utils import get_logger, setup_logging

log = get_logger("cockpit.validate")

HAND_CLASS_INDEX = COCKPIT_CLASSES.index("control_and_ego_vehicle")


def _hand_report(rows, predictions: List[np.ndarray]) -> Dict[str, Any]:
    evaluable = scored = false_regions = 0
    for row, pred in zip(rows.itertuples(), predictions):
        states = {}
        for side in ("left", "right"):
            for state in ("visible", "partially_visible", "occluded",
                          "out_of_frame", "motion_blurred", "uncertain",
                          "not_visible"):
                col = f"{side}_hand_{state}"
                if getattr(row, col, False):
                    states[side] = state
        any_evaluable = any(s in EVALUABLE_STATES for s in states.values())
        has_region = bool((pred == HAND_CLASS_INDEX).any())
        if any_evaluable:
            evaluable += 1
            scored += 1
        elif has_region:
            false_regions += 1
    return {
        "frames_with_an_evaluable_hand": evaluable,
        "frames_scored_for_hands": scored,
        "false_hand_regions_in_non_evaluable_frames": false_regions,
        "policy": (
            "hands are scored only where a reviewer confirmed visible or "
            "partially_visible; elsewhere a missing mask is not penalised, but a "
            "present mask is reported as a false region"),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset-dir", required=True)
    ap.add_argument("--manifest", required=True, help="validation split CSV")
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--log-level", default="INFO")
    args = ap.parse_args()
    setup_logging(args.log_level)

    require_reviewed_annotations(args.dataset_dir)
    root = Path(args.dataset_dir)
    dataset = CockpitDataset(args.manifest, root / "images", root / "masks")
    model = CockpitSegFormer(args.checkpoint).load()

    predictions, targets, domains = [], [], []
    for i in range(len(dataset)):
        image, mask, row = dataset[i]
        out = model.infer(image, row["domain"])
        predictions.append(out.mask.astype(np.int64))
        targets.append(mask)
        domains.append(row["domain"])
        if (i + 1) % 20 == 0:
            log.info("  %d/%d", i + 1, len(dataset))

    per_domain = domain_iou(predictions, targets, domains, len(COCKPIT_CLASSES))
    report = {
        "status": "validated_against_reviewed_annotations",
        "classes": list(COCKPIT_CLASSES),
        "frames": len(dataset),
        "per_domain_iou": {d: dict(zip(COCKPIT_CLASSES, v))
                           for d, v in per_domain.items()},
        "domain_gap": {
            c: (abs(per_domain["car"][i] - per_domain["motorcycle"][i])
                if np.isfinite(per_domain["car"][i])
                and np.isfinite(per_domain["motorcycle"][i]) else None)
            for i, c in enumerate(COCKPIT_CLASSES)},
        "hands": _hand_report(dataset.rows, predictions),
        "note": ("every number here is computed against reviewed human annotation; "
                 "no pseudo-label is used as a reference"),
    }
    atomic_write_json(args.output, report)
    print(json.dumps(report, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
