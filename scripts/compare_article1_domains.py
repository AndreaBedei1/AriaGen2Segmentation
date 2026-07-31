#!/usr/bin/env python3
"""Phase 10: exploratory car vs motorcycle comparison of semantic-camera output.

Explicitly labelled `exploratory_preliminary`. It exists to expose failure modes
and domain shift before annotation, not to support any claim about how the vehicle
changes visual attention.

Temporal quantities are normalised per second, because the two recordings are
currently sampled at different rates.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

from aria_drive_seg.config import Config
from aria_drive_seg.ingestion.compare import collect_metrics, compare
from aria_drive_seg.io_utils import atomic_write_json
from aria_drive_seg.logging_utils import get_logger, setup_logging
from aria_drive_seg.taxonomy import Taxonomy

log = get_logger("compare")


def _gaze_summary(run_dir: Path) -> Dict[str, Any]:
    """Gaze validity and semantic-gaze distribution, if the stage produced them."""
    path = run_dir / "gaze" / "aligned_gaze.parquet"
    if not path.exists():
        return {"available": False, "reason": "no aligned gaze for this run"}
    df = pd.read_parquet(path)
    out: Dict[str, Any] = {"available": True, "frames": len(df)}
    for col, key in (("valid", "valid_fraction"),
                     ("combined_valid", "combined_valid_fraction"),
                     ("in_image", "in_image_fraction")):
        if col in df:
            out[key] = float(df[col].mean())
    for col in ("dt_ms", "nearest_dt_ms"):
        if col in df:
            out["abs_dt_ms_median"] = float(df[col].abs().median())
            break
    if "class_name" in df:
        counts = df["class_name"].value_counts(normalize=True)
        out["semantic_gaze_distribution"] = {str(k): float(v)
                                             for k, v in counts.items()}
    out["note"] = ("gaze is used only after segmentation and never as a "
                   "segmentation input")
    return out


def _hand_summary(path: Optional[str]) -> Dict[str, Any]:
    if not path or not Path(path).exists():
        return {"available": False}
    doc = json.loads(Path(path).read_text())
    return {
        "available": True,
        "status": doc.get("status"),
        "per_state": doc.get("per_state"),
        "evaluable_fraction": doc.get("evaluable_fraction"),
        "candidate_false_mask_count": doc.get("candidate_false_mask_count"),
        "note": "candidate states only; no hand ground truth exists",
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--car-run", required=True)
    ap.add_argument("--motorcycle-run", required=True)
    ap.add_argument("--car-recording-id", required=True)
    ap.add_argument("--motorcycle-recording-id", required=True)
    ap.add_argument("--semantic-subdir", default="semantic_camera")
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--hand-summary", default=None)
    ap.add_argument("--route-summary", default=None)
    ap.add_argument("--reports", default="reports/article1_motorcycle_ingestion")
    ap.add_argument("--config", default="configs/article1/semantic_camera.yaml")
    ap.add_argument("--log-level", default="INFO")
    args = ap.parse_args()
    setup_logging(args.log_level)

    cfg = Config.load(args.config)
    taxonomy = Taxonomy.load(cfg.resolve(cfg.get("semantic_camera.classes")))

    log.info("measuring car run %s", args.car_run)
    car = collect_metrics(args.car_run, args.car_recording_id, "car", taxonomy,
                          args.semantic_subdir, stride=args.stride,
                          gaze_summary=_gaze_summary(Path(args.car_run)))
    log.info("measuring motorcycle run %s", args.motorcycle_run)
    moto = collect_metrics(args.motorcycle_run, args.motorcycle_recording_id,
                           "motorcycle", taxonomy, args.semantic_subdir,
                           stride=args.stride,
                           gaze_summary=_gaze_summary(Path(args.motorcycle_run)),
                           hand_summary=_hand_summary(args.hand_summary))

    doc = compare(car, moto, taxonomy)
    if args.route_summary and Path(args.route_summary).exists():
        route = json.loads(Path(args.route_summary).read_text())
        doc["route_pairing_quality"] = {
            "status": route.get("status"),
            "accepted_count": route.get("accepted_count"),
            "accepted_fraction": route.get("accepted_fraction"),
            "median_pair_distance_m": route.get("median_pair_distance_m"),
            "median_pair_quality": route.get("median_pair_quality"),
            "common_route": route.get("common_route"),
        }

    reports = Path(args.reports)
    reports.mkdir(parents=True, exist_ok=True)
    atomic_write_json(reports / "auto_moto_comparison.json", doc)

    rows = []
    for name, v in doc["per_class"].items():
        rows.append({"class": name, **v})
    pd.DataFrame(rows).to_csv(reports / "auto_moto_class_comparison.csv", index=False)

    print(json.dumps({
        "car": {"frames": car.frame_count, "duration_s": car.duration_s,
                "fps": car.effective_fps, "cockpit_fraction": car.cockpit_fraction,
                "mean_confidence": car.mean_confidence,
                "mean_entropy": car.mean_entropy,
                "switch_per_second": car.class_switch_rate_per_second},
        "motorcycle": {"frames": moto.frame_count, "duration_s": moto.duration_s,
                       "fps": moto.effective_fps,
                       "cockpit_fraction": moto.cockpit_fraction,
                       "mean_confidence": moto.mean_confidence,
                       "mean_entropy": moto.mean_entropy,
                       "switch_per_second": moto.class_switch_rate_per_second},
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
