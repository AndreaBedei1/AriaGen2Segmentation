#!/usr/bin/env python3
"""Phase 13: prepare, but do not start, the shared cockpit training stage.

Emits the split plan, the augmentation policy, the metric plan and the future
fusion contract, and reports exactly what is still missing. Training is blocked
until reviewed car AND motorcycle annotations exist; this script refuses to
proceed past that gate and says so.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
import yaml

from aria_drive_seg.article1.cockpit_training import (AugmentationPolicy,
                                                      SAFE_SPLIT_UNITS,
                                                      assert_no_temporal_leakage,
                                                      future_fusion_contract,
                                                      grouped_split, planned_metrics,
                                                      training_readiness,
                                                      validate_split_unit)
from aria_drive_seg.io_utils import atomic_write_json
from aria_drive_seg.logging_utils import get_logger, setup_logging

log = get_logger("cockpit.prepare")


def build_split_plan(selection_csv: Path, unit: str,
                     validation_fraction: float = 0.34) -> Dict[str, Any]:
    """Propose a leakage-safe validation set over whole groups."""
    validate_split_unit(unit)
    df = pd.read_csv(selection_csv)
    if unit not in df.columns:
        return {"available": False,
                "reason": f"the selection has no {unit} column yet"}

    groups: Dict[str, List[str]] = {}
    for domain, sub in df.groupby("domain"):
        groups[domain] = sorted(str(g) for g in sub[unit].dropna().unique())

    # Hold out whole groups, and hold out groups from BOTH domains: a validation
    # set containing only one vehicle would measure nothing about domain shift.
    validation: List[str] = []
    for domain, gs in groups.items():
        take = max(1, int(round(len(gs) * validation_fraction)))
        validation.extend(gs[-take:])

    plan: Dict[str, Any] = {
        "available": True,
        "split_unit": unit,
        "groups_per_domain": groups,
        "validation_groups": sorted(validation),
        "rationale": [
            f"the split is over whole `{unit}` groups, never over frames",
            "consecutive frames of one drive are near duplicates; splitting per "
            "frame would measure memorisation instead of generalisation",
            "validation holds out groups from both domains, otherwise the "
            "car/motorcycle gap could not be measured",
        ],
    }

    try:
        train, valid = grouped_split(df, unit, validation)
        plan["train_frames"] = int(len(train))
        plan["validation_frames"] = int(len(valid))
        plan["train_per_domain"] = train["domain"].value_counts().to_dict()
        plan["validation_per_domain"] = valid["domain"].value_counts().to_dict()
        try:
            assert_no_temporal_leakage(train, valid, min_separation_s=5.0)
            plan["temporal_leakage_check"] = "passed at 5.0 s minimum separation"
        except ValueError as exc:
            plan["temporal_leakage_check"] = f"FAILED: {exc}"
            plan["available"] = False
    except (ValueError, RuntimeError) as exc:
        plan["available"] = False
        plan["error"] = str(exc)
    return plan


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--selection",
                    default="reports/article1_motorcycle_ingestion/annotation_selection.csv")
    ap.add_argument("--dataset-dir", default="datasets/article1_annotation_package")
    ap.add_argument("--config", default="configs/article1/segformer_cockpit.yaml")
    ap.add_argument("--output",
                    default="reports/article1_motorcycle_ingestion/cockpit_training_plan.json")
    ap.add_argument("--split-unit", default="recording_id",
                    choices=list(SAFE_SPLIT_UNITS))
    ap.add_argument("--log-level", default="INFO")
    args = ap.parse_args()
    setup_logging(args.log_level)

    cfg = yaml.safe_load(Path(args.config).read_text())["article1_cockpit"]
    readiness = training_readiness(args.dataset_dir)

    plan: Dict[str, Any] = {
        "status": "prepared_not_started",
        "training_started": False,
        "gate": readiness["gate"],
        "ready_to_train": readiness["ready_to_train"],
        "blockers": readiness["blockers"],
        "model": {
            "architecture": cfg["model"],
            "classes": cfg["classes"],
            "shared_between_domains": True,
            "domains": cfg["domains"],
            "note": ("one model for both vehicles: a car-only or motorcycle-only "
                     "cockpit model is not an acceptable final solution, because the "
                     "two domains must be measured with the same instrument"),
        },
        "hyperparameters": {k: cfg[k] for k in
                            ("image_size", "batch_size", "learning_rate",
                             "weight_decay", "epochs", "early_stopping_patience",
                             "class_weights", "domain_balanced_sampler", "seed",
                             "mixed_precision", "deterministic") if k in cfg},
        "augmentation": AugmentationPolicy().to_dict(),
        "split": build_split_plan(Path(args.selection), args.split_unit),
        "split_units_allowed": readiness["split_units_allowed"],
        "split_units_forbidden": readiness["split_units_forbidden"],
        "metrics": planned_metrics(),
        "future_fusion": future_fusion_contract(),
        "next_steps": [
            "1. a human reviews the CVAT package and delivers corrected masks",
            "2. the reviewer drops REVIEWED_ANNOTATIONS.json into the dataset "
            "directory, recording who reviewed what",
            "3. re-run this script: the gate opens only when that marker exists and "
            "both domains are present",
            "4. run scripts/train_article1_cockpit.py, then "
            "scripts/validate_article1_cockpit.py, then "
            "scripts/export_article1_cockpit.py",
            "5. recalibrate the fusion's internal_min_confidence against the trained "
            "model instead of inheriting the proxy's thresholds",
            "6. remove the geometric bottom-of-frame cockpit prior for the "
            "motorcycle once the trained model covers the handlebar",
        ],
    }

    atomic_write_json(args.output, plan)
    print(json.dumps({
        "ready_to_train": plan["ready_to_train"],
        "blockers": plan["blockers"],
        "split_available": plan["split"].get("available"),
        "train_frames": plan["split"].get("train_frames"),
        "validation_frames": plan["split"].get("validation_frames"),
        "temporal_leakage_check": plan["split"].get("temporal_leakage_check"),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
