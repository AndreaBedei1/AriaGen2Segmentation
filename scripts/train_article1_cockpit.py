#!/usr/bin/env python3
"""Train the shared car/motorcycle cockpit SegFormer-B2.

This script refuses to run until reviewed annotations for BOTH domains exist. That
is not a formality: pseudo-labels from Mask2Former, Grounding DINO, SAM 2.1 and the
geometric prior are the very thing this model is meant to replace, and training on
them would launder them into an apparent ground truth.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

from aria_drive_seg.article1.cockpit_segformer import (CockpitDataset,
                                                       require_reviewed_annotations,
                                                       train_cockpit)
from aria_drive_seg.article1.cockpit_training import (assert_no_temporal_leakage,
                                                      grouped_split,
                                                      validate_shared_domains,
                                                      validate_split_unit)
from aria_drive_seg.logging_utils import get_logger, setup_logging

log = get_logger("cockpit.train")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset-dir", required=True,
                    help="directory holding reviewed masks and REVIEWED_ANNOTATIONS.json")
    ap.add_argument("--manifest", required=True,
                    help="reviewed sample manifest CSV")
    ap.add_argument("--split-unit", default="recording_id")
    ap.add_argument("--validation-groups", required=True,
                    help="comma-separated group ids held out for validation")
    ap.add_argument("--config", default="configs/article1/segformer_cockpit.yaml")
    ap.add_argument("--min-separation-s", type=float, default=5.0)
    ap.add_argument("--log-level", default="INFO")
    args = ap.parse_args()
    setup_logging(args.log_level)

    import pandas as pd

    # --- gate 1: reviewed annotations must exist ----------------------------
    require_reviewed_annotations(args.dataset_dir)

    rows = pd.read_csv(args.manifest)
    # --- gate 2: the model is shared, so both domains must be present -------
    validate_shared_domains(rows)
    # --- gate 3: the split unit must be a group, never a frame --------------
    validate_split_unit(args.split_unit)

    train_rows, valid_rows = grouped_split(
        rows, args.split_unit, args.validation_groups.split(","))
    # --- gate 4: no near-in-time frame may straddle the split --------------
    assert_no_temporal_leakage(train_rows, valid_rows, args.min_separation_s)

    log.info("train %d frames, validation %d frames, split on %s",
             len(train_rows), len(valid_rows), args.split_unit)
    log.info("train domains: %s", train_rows["domain"].value_counts().to_dict())
    log.info("validation domains: %s", valid_rows["domain"].value_counts().to_dict())

    cfg = yaml.safe_load(Path(args.config).read_text())["article1_cockpit"]
    root = Path(args.dataset_dir)
    train_csv = root / ".split_train.csv"
    valid_csv = root / ".split_valid.csv"
    train_rows.to_csv(train_csv, index=False)
    valid_rows.to_csv(valid_csv, index=False)

    train_dataset = CockpitDataset(train_csv, root / "images", root / "masks")
    valid_dataset = CockpitDataset(valid_csv, root / "images", root / "masks")
    train_cockpit(train_dataset, valid_dataset, cfg)
    print(json.dumps({"trained": True,
                      "checkpoint_dir": cfg.get("checkpoint_dir")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
