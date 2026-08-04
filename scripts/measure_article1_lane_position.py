#!/usr/bin/env python3
"""Phase 7: visual lane-position proxy over the full recording.

Reads the frozen full-run segmentation masks and measures, per frame, where the
head-mounted camera sits between the two lane markings visible in the near field.
The primary quantity is normalised by the lane's own half-width in the same
image, so it does not inherit an assumed lane width, a focal length or a
perspective scale.

Nothing here is a measurement of the vehicle's position in its lane, and nothing
here decides that a line was crossed unlawfully. Every artefact is labelled
`visual_lane_position_proxy`.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd

from aria_drive_seg.behavior import EXPLORATORY_MARKER
from aria_drive_seg.behavior.lane_position import (PROXY_LABEL, crossing_candidates,
                                                   measure_run, road_row_profile,
                                                   select_measurement_band,
                                                   summarise)
from aria_drive_seg.config import Config
from aria_drive_seg.io_utils import atomic_write_json
from aria_drive_seg.logging_utils import get_logger, setup_logging
from aria_drive_seg.taxonomy import Taxonomy

log = get_logger("behavior.lane_position")

RUN_DIRS = {"car": "car", "motorcycle": "motorcycle"}


def _atomic_parquet(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp.parquet")
    frame.to_parquet(tmp, index=False)
    tmp.replace(path)


def main() -> int:
    import cv2

    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/article1/behavior_analysis.yaml")
    ap.add_argument("--gaze-config",
                    default="configs/article1/fast_semantic_gaze.yaml")
    ap.add_argument("--run-root", default="output/article1/fast_semantic_gaze")
    ap.add_argument("--output", default="output/article1/final_behavior_statistics")
    ap.add_argument("--log-level", default="INFO")
    args = ap.parse_args()
    setup_logging(args.log_level)

    cfg = Config.load(args.config)
    gaze_cfg = Config.load(args.gaze_config)
    lcfg = cfg.get("visual_lane_position") or {}
    taxonomy = Taxonomy.load(gaze_cfg.resolve(
        gaze_cfg.get("semantic_gaze_fast_external.classes")))
    lane_id = taxonomy.id_of("lane_marking")
    road_id = taxonomy.id_of("road_surface")
    run_root = Path(args.run_root)
    out_root = Path(args.output)

    summaries: Dict[str, Any] = {}
    for domain, spec in (cfg.get("recordings") or {}).items():
        rec_id = str(spec["recording_id"])
        root = run_root / RUN_DIRS[domain]
        index = pd.read_parquet(
            root / "segmentation" / "segmentation_index.parquet").sort_values(
            "capture_timestamp_ns").reset_index(drop=True)
        log.info("== %s: %d segmented frames", domain, len(index))

        def load(frame_index: int, _root=root):
            path = _root / "segmentation" / "masks" / f"frame_{frame_index:06d}.png"
            if not path.exists():
                return None
            return cv2.imread(str(path), cv2.IMREAD_UNCHANGED)

        # The near-field road band is measured from this recording's own
        # segmentation: the rows that are near-field road on a motorcycle are
        # dashboard in a car, and a fixed band would measure the dashboard.
        probe_count = max(1, int(lcfg.get("band_probe_frames", 120)))
        stride = max(1, len(index) // probe_count)
        profile = road_row_profile(load, index["frame_index"].tolist()[::stride],
                                   road_id)
        band = select_measurement_band(
            profile,
            min_road_row_fraction=float(lcfg.get("min_road_row_fraction", 0.15)),
            lower_portion=float(lcfg.get("band_lower_portion", 0.60)),
            min_band_fraction=float(lcfg.get("min_band_fraction", 0.05)))
        if not band["selected"]:
            raise SystemExit(f"{domain}: {band['reason']}")
        log.info("  measurement band rows %.3f-%.3f (%.0f%% road surface)",
                 band["top_fraction"], band["bottom_fraction"],
                 100 * band["median_road_fraction_in_band"])

        records = measure_run(
            load, index["frame_index"].tolist(),
            index["capture_timestamp_ns"].tolist(),
            lane_marking_id=lane_id, road_surface_id=road_id,
            assumed_lane_width_m=float(lcfg.get("assumed_lane_width_m", 3.5)),
            band_top_fraction=band["top_fraction"],
            band_bottom_fraction=band["bottom_fraction"],
            min_lane_width_fraction=float(lcfg.get("min_lane_width_fraction", 0.05)),
            max_lane_width_fraction=float(lcfg.get("max_lane_width_fraction", 0.90)),
            min_road_fraction_between=float(
                lcfg.get("min_road_fraction_between", 0.50)))

        timestamps = index["capture_timestamp_ns"].to_numpy(np.int64)
        interval_s = (float(np.median(np.diff(timestamps)) / 1e9)
                      if timestamps.size > 1 else 0.0)
        summary = summarise(
            records, interval_s,
            assumed_lane_width_m=float(lcfg.get("assumed_lane_width_m", 3.5)),
            min_crossing_duration_s=float(lcfg.get("min_crossing_duration_s", 0.5)))
        summary.update({"domain": domain, "recording_id": rec_id,
                        "result_status": EXPLORATORY_MARKER,
                        "mask_source": str(root / "segmentation" / "masks"),
                        "measurement_band": band})
        log.info("  coverage %.1f%%, p05-p95 band %.3f, %d crossing candidates",
                 100 * summary["coverage_fraction"],
                 (summary.get("normalized_offset") or {}).get("p05_p95_band_width")
                 or float("nan"),
                 summary["crossing_candidates"]["count"])

        frame = pd.DataFrame(records)
        frame["domain"] = domain
        frame["recording_id"] = rec_id
        frame["metric"] = PROXY_LABEL
        frame["is_vehicle_metric_position"] = False
        dest = out_root / rec_id
        _atomic_parquet(dest / "lane_position_frames.parquet", frame)

        measured = frame[frame["measured"].astype(bool)]
        candidates = crossing_candidates(
            measured["timestamp_ns"].to_numpy(np.int64),
            measured["normalized_offset"].to_numpy(float),
            min_duration_s=float(lcfg.get("min_crossing_duration_s", 0.5)))
        candidate_frame = pd.DataFrame(candidates)
        if not candidate_frame.empty:
            candidate_frame["domain"] = domain
            candidate_frame["recording_id"] = rec_id
        _atomic_parquet(dest / "lane_position_crossing_candidates.parquet",
                        candidate_frame)
        atomic_write_json(dest / "lane_position_summary.json", summary)
        summaries[domain] = summary

    combined = {
        "schema": "article1_visual_lane_position_v1",
        "result_status": EXPLORATORY_MARKER,
        "metric": PROXY_LABEL,
        "is_vehicle_metric_position": False,
        "domains": summaries,
    }
    atomic_write_json(out_root / "lane_position_summary.json", combined)
    print(json.dumps({d: {
        "coverage_fraction": s["coverage_fraction"],
        "frames_measured": s["frames_measured"],
        "median_normalized_offset": (s.get("normalized_offset") or {}).get("median"),
        "p05_p95_band_width": (s.get("normalized_offset") or {}).get("p05_p95_band_width"),
        "crossing_candidates": s["crossing_candidates"]["count"],
    } for d, s in summaries.items()}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
