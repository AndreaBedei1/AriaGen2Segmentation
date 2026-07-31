#!/usr/bin/env python3
"""Export a validated cockpit checkpoint to the location the fusion looks for.

Placing a checkpoint at `weights/segformer-b2-article1-cockpit` switches the
semantic camera from the explicitly-marked proxy to a real model, so the export
records provenance and refuses to publish an unvalidated checkpoint.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import shutil
from pathlib import Path

from aria_drive_seg.article1.cockpit_segformer import COCKPIT_CLASSES
from aria_drive_seg.hashing import sha256_file, stable_hash
from aria_drive_seg.logging_utils import get_logger, setup_logging

log = get_logger("cockpit.export")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--validation-report", required=True)
    ap.add_argument("--destination", default="weights/segformer-b2-article1-cockpit")
    ap.add_argument("--reviewed-marker", required=True,
                    help="the dataset's REVIEWED_ANNOTATIONS.json")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--log-level", default="INFO")
    args = ap.parse_args()
    setup_logging(args.log_level)

    source = Path(args.checkpoint)
    report_path = Path(args.validation_report)
    marker = Path(args.reviewed_marker)
    destination = Path(args.destination)

    if not source.exists():
        raise SystemExit(f"checkpoint {source} does not exist")
    if not marker.exists():
        raise SystemExit(
            f"{marker} is absent: this checkpoint was not trained on reviewed "
            "annotations and must not be published")
    if not report_path.exists():
        raise SystemExit(
            f"{report_path} is absent: an unvalidated checkpoint must not be "
            "published, because publishing it silently switches the fusion away "
            "from the explicitly-marked proxy")

    report = json.loads(report_path.read_text())
    per_domain = report.get("per_domain_iou", {})
    if set(per_domain) != {"car", "motorcycle"}:
        raise SystemExit(
            "the validation report must cover both domains; a cockpit model "
            "validated on one vehicle only is not acceptable")

    if destination.exists() and not args.force:
        raise SystemExit(f"{destination} already exists; pass --force to replace it")

    destination.parent.mkdir(parents=True, exist_ok=True)
    if source.is_dir():
        if destination.exists():
            shutil.rmtree(destination)
        shutil.copytree(source, destination)
    else:
        destination.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination / source.name)

    files = sorted(p for p in destination.rglob("*") if p.is_file())
    provenance = {
        "exported_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "source_checkpoint": str(source),
        "classes": list(COCKPIT_CLASSES),
        "trained_on_reviewed_annotations": True,
        "reviewed_marker": json.loads(marker.read_text()),
        "validation_report": report,
        "files": {str(p.relative_to(destination)): sha256_file(p) for p in files},
        "fingerprint": stable_hash([str(p.relative_to(destination)) for p in files]),
        "effect": ("the semantic camera's internal provider switches from "
                   "grounded_sam2_cockpit_proxy to SegFormerInternalProvider and "
                   "internal_is_fallback becomes false"),
        "required_follow_up": [
            "recalibrate fusion.internal_min_confidence against this model",
            "remove the geometric bottom-of-frame cockpit prior for the motorcycle",
            "re-run the frozen comparison so before/after is measured, not assumed",
        ],
    }
    (destination / "EXPORT_PROVENANCE.json").write_text(
        json.dumps(provenance, indent=2) + "\n")
    log.info("exported %d files to %s", len(files), destination)
    print(json.dumps({"destination": str(destination), "files": len(files)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
