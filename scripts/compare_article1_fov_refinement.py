#!/usr/bin/env python3
"""Phase 7-8: compare the frozen baseline with the full-FOV refined run.

Same 450-frame window, same source frames, one changed thing at a time. These are
pre-ground-truth diagnostics: coverage, distribution, stability, fragmentation and
provenance. Nothing here is accuracy, because there is nothing to be accurate
against yet.

The refined run is in source geometry and the baseline in rectified geometry, so
per-pixel differencing between them is meaningless; every comparison is over
distributions and per-frame statistics, plus an explicit check of the peripheral
band the baseline could not see at all.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd

from aria_drive_seg.config import Config
from aria_drive_seg.io_utils import atomic_write_json, read_mask_u16
from aria_drive_seg.logging_utils import get_logger, setup_logging
from aria_drive_seg.taxonomy import Taxonomy

log = get_logger("fov.compare")

COCKPIT = ("mirror", "instrument_display", "control_and_ego_vehicle")
EXTERNAL = ("road_surface", "lane_marking", "regulatory_road_marking",
            "traffic_sign", "traffic_light", "vehicle", "two_wheeler",
            "pedestrian", "road_boundary_or_obstacle")


def _measure(run: Path, taxonomy: Taxonomy, valid: np.ndarray | None,
             label: str) -> Dict[str, Any]:
    import cv2

    root = run / "semantic_camera"
    metas = sorted((root / "metadata").glob("*.json"))
    names = taxonomy.names()
    fraction = {n: [] for n in names}
    components = {n: [] for n in names}
    per_frame: List[Dict[str, Any]] = []
    switches: List[float] = []
    previous = None
    invalid_ids = 0
    edge_class_fraction = {n: [] for n in names}

    for path in metas:
        meta = json.loads(path.read_text())
        mask = read_mask_u16(root / "final_masks" / f"{path.stem}.png")
        h, w = mask.shape
        region = valid if (valid is not None and valid.shape == mask.shape) \
            else np.ones_like(mask, bool)
        total = int(region.sum())
        invalid_ids += int(np.count_nonzero(((mask < 1) | (mask > taxonomy.max_id))
                                            & region))
        # the lateral bands the pinhole rectification could not represent
        band = np.zeros_like(region)
        band[:, :int(w * 0.14)] = True
        band[:, int(w * 0.86):] = True
        band &= region
        band_total = max(1, int(band.sum()))

        for cid, name in enumerate(names):
            hit = (mask == cid) & region
            n = int(hit.sum())
            fraction[name].append(n / max(1, total))
            edge_class_fraction[name].append(int((hit & band).sum()) / band_total)
            if n:
                count, _, _, _ = cv2.connectedComponentsWithStats(
                    hit.astype(np.uint8), connectivity=8)
                components[name].append(max(0, count - 1))

        per_frame.append({
            "frame_index": int(meta["frame_index"]),
            "mean_final_confidence": float(meta.get("mean_final_confidence", np.nan)),
            "mean_final_entropy": float(meta.get("mean_final_entropy", np.nan)),
            "external_selected_fraction": float(meta.get("external_selected_fraction", 0)),
            "internal_model_selected_fraction": float(
                meta.get("internal_model_selected_fraction", 0)),
            "geometric_proxy_selected_fraction": float(
                meta.get("geometric_proxy_selected_fraction", 0)),
            "dense_fill_fraction": float(meta.get("dense_fill_fraction", 0)),
            "conflict_fraction": float(meta.get("conflict_fraction", 0)),
            "dense_coverage": float(meta.get("dense_coverage", 1.0)),
        })
        if previous is not None and previous.shape == mask.shape:
            switches.append(float(((previous != mask) & region).sum() / max(1, total)))
        previous = mask

    df = pd.DataFrame(per_frame)
    duration = 450 / 15.0012
    return {
        "label": label,
        "run": str(run),
        "frames": len(metas),
        "geometry": [int(previous.shape[1]), int(previous.shape[0])] if previous is not None else None,
        "measured_over": "valid pixels only" if valid is not None else "whole frame",
        "invalid_class_ids": invalid_ids,
        "dense_coverage_mean": float(df["dense_coverage"].mean()),
        "mean_confidence": float(df["mean_final_confidence"].mean()),
        "mean_entropy": float(df["mean_final_entropy"].mean()),
        "provenance": {
            "external": float(df["external_selected_fraction"].mean()),
            "internal_model": float(df["internal_model_selected_fraction"].mean()),
            "geometric_prior": float(df["geometric_proxy_selected_fraction"].mean()),
            "dense_fill": float(df["dense_fill_fraction"].mean()),
        },
        "conflict_fraction": float(df["conflict_fraction"].mean()),
        "class_pixel_fraction": {n: float(np.mean(v)) for n, v in fraction.items()},
        "class_presence_fraction": {
            n: float(np.mean([x > 0 for x in v])) for n, v in fraction.items()},
        "class_components_when_present": {
            n: (float(np.mean(v)) if v else 0.0) for n, v in components.items()},
        "lateral_band_class_fraction": {
            n: float(np.mean(v)) for n, v in edge_class_fraction.items()},
        "switch_rate_per_second": (float(np.sum(switches) / duration)
                                   if switches else None),
        "cockpit_fraction": float(sum(np.mean(fraction[c]) for c in COCKPIT)),
        "external_fraction": float(sum(np.mean(fraction[c]) for c in EXTERNAL)),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline", required=True)
    ap.add_argument("--refined", required=True)
    ap.add_argument("--reports",
                    default="reports/article1_motorcycle_fov_mirror_refinement")
    ap.add_argument("--config", default="configs/article1/semantic_camera_full_fov.yaml")
    ap.add_argument("--log-level", default="INFO")
    args = ap.parse_args()
    setup_logging(args.log_level)

    import cv2

    cfg = Config.load(args.config)
    taxonomy = Taxonomy.load(cfg.resolve(cfg.get("semantic_camera.classes")))
    reports = Path(args.reports)
    reports.mkdir(parents=True, exist_ok=True)

    refined = Path(args.refined)
    mask_path = refined / "frames" / "valid_pixel_mask.png"
    valid = None
    if mask_path.exists():
        valid = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE) > 127
        log.info("using the valid-pixel mask: %.2f%% of the frame",
                 100 * valid.mean())

    log.info("measuring the frozen baseline")
    base = _measure(Path(args.baseline), taxonomy, None, "frozen_baseline_rectified")
    log.info("measuring the refined run")
    new = _measure(refined, taxonomy, valid, "full_fov_refined")

    per_class = {}
    for name in taxonomy.names():
        b = base["class_pixel_fraction"][name]
        r = new["class_pixel_fraction"][name]
        per_class[name] = {
            "baseline_pixel_fraction": b,
            "refined_pixel_fraction": r,
            "absolute_change": r - b,
            "relative_change": ((r - b) / b) if b > 0 else None,
            "baseline_presence": base["class_presence_fraction"][name],
            "refined_presence": new["class_presence_fraction"][name],
            "baseline_components": base["class_components_when_present"][name],
            "refined_components": new["class_components_when_present"][name],
            "baseline_lateral_band": base["lateral_band_class_fraction"][name],
            "refined_lateral_band": new["lateral_band_class_fraction"][name],
        }

    document = {
        "schema": "article1_fov_refinement_comparison_v1",
        "generated_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "status": "pre_ground_truth_diagnostics",
        "is_accuracy": False,
        "note": ("the two runs live in different geometries, so no per-pixel "
                 "difference is computed; the comparison is over distributions, "
                 "per-frame statistics and the lateral band the baseline could "
                 "not represent at all"),
        "baseline": base,
        "refined": new,
        "per_class": per_class,
        "headline": {
            "cockpit_fraction": [base["cockpit_fraction"], new["cockpit_fraction"]],
            "mirror_fraction": [base["class_pixel_fraction"]["mirror"],
                                new["class_pixel_fraction"]["mirror"]],
            "mirror_presence": [base["class_presence_fraction"]["mirror"],
                                new["class_presence_fraction"]["mirror"]],
            "instrument_presence": [
                base["class_presence_fraction"]["instrument_display"],
                new["class_presence_fraction"]["instrument_display"]],
            "geometric_prior_share": [base["provenance"]["geometric_prior"],
                                      new["provenance"]["geometric_prior"]],
            "external_fraction": [base["external_fraction"], new["external_fraction"]],
            "invalid_class_ids": [base["invalid_class_ids"], new["invalid_class_ids"]],
            "dense_coverage": [base["dense_coverage_mean"], new["dense_coverage_mean"]],
        },
    }
    atomic_write_json(reports / "fov_refinement_comparison.json", document)
    pd.DataFrame([{"class": k, **v} for k, v in per_class.items()]).to_csv(
        reports / "fov_refinement_class_comparison.csv", index=False)

    print(json.dumps(document["headline"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
