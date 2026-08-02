#!/usr/bin/env python3
"""Phase 5: semantic gaze behaviour over the frozen baseline's dense blocks.

Reads the frozen semantic camera's masks, confidence, entropy and provenance and
samples them through a Gaussian foveal window at every aligned gaze point.
Fixations come from the native 30 Hz gaze stream by I-VT, in degrees per second.

The frozen baseline is never modified and gaze is never fed back into it.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd

from aria_drive_seg.behavior import EXPLORATORY_MARKER
from aria_drive_seg.behavior.gaze_semantics import (SemanticBlock, foveal_weights,
                                                    identify_fixations,
                                                    pixels_per_degree,
                                                    sample_foveal_semantics,
                                                    scanpath_statistics,
                                                    semantic_gaze_metrics)
from aria_drive_seg.config import Config
from aria_drive_seg.io_utils import atomic_write_json
from aria_drive_seg.logging_utils import get_logger, setup_logging
from aria_drive_seg.taxonomy import Taxonomy

log = get_logger("behavior.semantic_gaze")


def analyse_block(block_root: Path, domain: str, recording_id: str,
                  class_names: List[str], cfg: Config) -> Dict[str, Any]:
    block = SemanticBlock(block_root)
    lo, hi = block.frame_range
    log.info("  block %s: frames %d..%d (%d)", block_root.name, lo, hi,
             len(block.frames))

    aligned = pd.read_parquet(block_root / "gaze" / "aligned_gaze.parquet")
    raw = pd.read_parquet(block_root / "gaze" / "raw_gaze.parquet")

    gcfg = cfg.get("gaze") or {}
    focal = float(cfg.get("rectify.focal", 879.0))
    ppd = pixels_per_degree(focal)
    sigma_px = float(gcfg.get("foveal_sigma_deg", 1.5)) * ppd
    radius_px = int(round(float(gcfg.get("foveal_radius_sigmas", 2.5)) * sigma_px))
    weights = foveal_weights(radius_px, sigma_px)
    log.info("  fovea: sigma %.1f px, radius %d px (%.2f px/deg)",
             sigma_px, radius_px, ppd)

    # --- fixations on the NATIVE gaze stream, restricted to the block window ---
    t0 = int(aligned["capture_timestamp_ns"].min())
    t1 = int(aligned["capture_timestamp_ns"].max())
    win = raw[(raw["gaze_timestamp_ns"] >= t0) & (raw["gaze_timestamp_ns"] <= t1)]
    win = win.sort_values("gaze_timestamp_ns")
    kin, fixations = identify_fixations(
        win["gaze_timestamp_ns"].values, win["yaw"].values, win["pitch"].values,
        valid=win["combined_valid"].values,
        velocity_threshold_deg_s=float(gcfg.get("ivt_velocity_threshold_deg_s", 30.0)),
        min_duration_s=float(gcfg.get("min_fixation_duration_s", 0.100)),
        max_gap_s=float(gcfg.get("max_gap_within_fixation_s", 0.075)))
    log.info("  %d native gaze samples -> %d fixations", len(win), len(fixations))

    # --- foveal sampling at every aligned frame -------------------------------
    n_classes = len(class_names)
    fix_ts = kin.timestamp_ns
    rows: List[Dict[str, Any]] = []
    for r in aligned.itertuples():
        layers = block.load(int(r.frame_index)) if int(r.frame_index) in block else None
        rec: Dict[str, Any] = {
            "recording_id": recording_id, "domain": domain,
            "frame_index": int(r.frame_index),
            "timestamp_ns": int(r.capture_timestamp_ns),
            "rel_time_s": float((int(r.capture_timestamp_ns) - t0) / 1e9),
            "gaze_valid": bool(r.valid),
            "gaze_dt_ms": float(r.dt_ms),
            "rect_u": float(r.rect_u), "rect_v": float(r.rect_v),
            "in_image": bool(r.in_image_rect),
            "semantic_valid": False, "invalid_reason": None,
            "top1_class_id": -1, "top2_class_id": -1,
            "top1_class": None, "top2_class": None,
            "top1_probability": np.nan, "top2_probability": np.nan,
            "foveal_entropy": np.nan, "model_entropy": np.nan,
            "confidence": np.nan, "provenance_code": -1,
            "distance_from_centre_px": np.nan,
            "fixation_id": -1, "is_fixation": False,
            "angular_velocity_deg_s": np.nan,
        }
        # Fixation state: the native gaze sample nearest this frame.
        if fix_ts.size:
            k = int(np.argmin(np.abs(fix_ts - int(r.capture_timestamp_ns))))
            rec["fixation_id"] = int(kin.fixation_id[k])
            rec["is_fixation"] = bool(kin.is_fixation[k])
            rec["angular_velocity_deg_s"] = float(kin.angular_velocity_deg_s[k])

        if not bool(r.valid):
            rec["invalid_reason"] = str(r.validity_reason)
        elif layers is None:
            rec["invalid_reason"] = "frame outside the dense semantic block"
        elif not bool(r.in_image_rect):
            rec["invalid_reason"] = "gaze falls outside the rectified image"
        else:
            s = sample_foveal_semantics(layers, r.rect_u, r.rect_v, weights, n_classes)
            if s is None:
                rec["invalid_reason"] = "foveal window has no usable pixels"
            else:
                probs = s.pop("foveal_probabilities")
                rec.update(s)
                rec["semantic_valid"] = True
                rec["top1_class"] = class_names[s["top1_class_id"]]
                rec["top2_class"] = (class_names[s["top2_class_id"]]
                                     if s["top2_class_id"] >= 0 else None)
                for ci, cn in enumerate(class_names):
                    rec[f"p_{cn}"] = float(probs[ci])
        rows.append(rec)

    df = pd.DataFrame(rows)
    valid_frac = float(df["semantic_valid"].mean()) if len(df) else 0.0
    log.info("  %d gaze samples, %.1f%% with a foveal semantic reading",
             len(df), 100 * valid_frac)

    interval_s = (float(np.median(np.diff(df["timestamp_ns"].values)) / 1e9)
                  if len(df) > 1 else 0.0)
    metrics = semantic_gaze_metrics(df, class_names, fixations, interval_s)
    scan = scanpath_statistics(df["rect_u"].values, df["rect_v"].values,
                               df["semantic_valid"].values, ppd, (1512, 2016))
    if scan.get("scanpath_length_deg") and metrics["total_valid_gaze_time_s"] > 0:
        scan["scanpath_length_deg_per_s"] = (
            scan["scanpath_length_deg"] / metrics["total_valid_gaze_time_s"])
    metrics["scanpath"] = scan
    metrics["block"] = {
        "root": str(block_root), "frame_range": [lo, hi],
        "frames": len(block.frames),
        "duration_s": float((t1 - t0) / 1e9),
        "semantic_valid_fraction": valid_frac,
        "foveal_sigma_deg": float(gcfg.get("foveal_sigma_deg", 1.5)),
        "pixels_per_degree": ppd,
    }
    return {"samples": df, "fixations": fixations, "metrics": metrics}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/article1/behavior_analysis.yaml")
    ap.add_argument("--output", default="output/article1/behavior_analysis")
    ap.add_argument("--reports", default="reports/article1_behavior_analysis")
    ap.add_argument("--log-level", default="INFO")
    args = ap.parse_args()
    setup_logging(args.log_level)

    cfg = Config.load(args.config)
    taxonomy = Taxonomy.load(cfg.resolve(
        cfg.get("article1.classes", "configs/article1/classes_article1.yaml")))
    class_names = taxonomy.names()
    out_root = Path(args.output)

    results: Dict[str, Any] = {}
    for domain, spec in (cfg.get("recordings") or {}).items():
        rec_id = spec["recording_id"]
        dest = out_root / rec_id
        dest.mkdir(parents=True, exist_ok=True)
        log.info("== %s", domain)
        blocks = spec.get("semantic_blocks") or []
        if not blocks:
            log.warning("  no dense semantic block configured; skipping")
            continue

        all_samples, all_metrics = [], []
        for b in blocks:
            root = cfg.resolve(b["root"])
            if not root.exists():
                log.warning("  missing block %s; skipping", root)
                continue
            res = analyse_block(root, domain, rec_id, class_names, cfg)
            all_samples.append(res["samples"])
            all_metrics.append(res["metrics"])
            pd.DataFrame([f.to_dict() for f in res["fixations"]]).to_parquet(
                dest / "gaze_fixations.parquet", index=False)

        if not all_samples:
            continue
        samples = pd.concat(all_samples, ignore_index=True)
        samples.to_parquet(dest / "semantic_gaze.parquet", index=False)
        results[domain] = {
            "recording_id": rec_id,
            "blocks": all_metrics,
            "semantic_coverage": {
                "recording_duration_s": None,
                "analysed_duration_s": float(sum(m["block"]["duration_s"]
                                                 for m in all_metrics)),
                "block_count": len(all_metrics),
            },
        }

    # Coverage against the whole recording: the honest denominator.
    for domain, spec in (cfg.get("recordings") or {}).items():
        if domain not in results:
            continue
        tl = out_root / spec["recording_id"] / "timeline_summary.json"
        if tl.exists():
            total = float(json.loads(tl.read_text())["duration_s"])
            cov = results[domain]["semantic_coverage"]
            cov["recording_duration_s"] = total
            cov["coverage_fraction"] = cov["analysed_duration_s"] / total

    report = {
        "schema": "article1_semantic_gaze_v1",
        "result_status": EXPLORATORY_MARKER,
        "method": {
            "foveal_window": ("Gaussian over the class-label field, weighted by "
                              "each pixel's own segmentation confidence"),
            "distribution_is_not_model_posterior": True,
            "fixation_algorithm": "I-VT on the native 30 Hz gaze stream",
            "gaze_used_for_segmentation": False,
            "segmentation_source": "frozen Article 1 semantic-camera baseline",
        },
        "domains": results,
    }
    reports = Path(args.reports)
    reports.mkdir(parents=True, exist_ok=True)
    atomic_write_json(reports / "semantic_gaze_summary.json", report)
    atomic_write_json(out_root / "semantic_gaze_summary.json", report)
    print(json.dumps({d: {
        "coverage_fraction": round(r["semantic_coverage"].get("coverage_fraction", 0), 4),
        "analysed_s": round(r["semantic_coverage"]["analysed_duration_s"], 1),
        "road_relevant_time_fraction": round(
            r["blocks"][0]["road_relevant_time_fraction"], 3),
        "switching_rate_per_s": round(
            r["blocks"][0]["gaze_switching_rate_per_s"] or 0, 3),
    } for d, r in results.items()}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
