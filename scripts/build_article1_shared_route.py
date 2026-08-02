#!/usr/bin/env python3
"""Phase 3: find the stretches of road both vehicles actually drove.

Bins are defined by the map — (way, offset along the way, travel direction) — so a
bin means the same place for both recordings and the pairing does not depend on
which vehicle is treated as the reference. Every rejected bin keeps its reason.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd

from aria_drive_seg.behavior import EXPLORATORY_MARKER
from aria_drive_seg.behavior.route import (PairingCriteria, assign_frames_to_bins,
                                           expected_fixes_per_bin,
                                           build_route_bins, pair_route_bins,
                                           positional_resolution_m,
                                           shared_route_summary)
from aria_drive_seg.config import Config
from aria_drive_seg.io_utils import atomic_write_json
from aria_drive_seg.logging_utils import get_logger, setup_logging

log = get_logger("behavior.route")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/article1/behavior_analysis.yaml")
    ap.add_argument("--output", default="output/article1/behavior_analysis")
    ap.add_argument("--reports", default="reports/article1_behavior_analysis")
    ap.add_argument("--log-level", default="INFO")
    args = ap.parse_args()
    setup_logging(args.log_level)

    cfg = Config.load(args.config)
    out_root = Path(args.output)
    reports = Path(args.reports) / "route"
    reports.mkdir(parents=True, exist_ok=True)

    recordings = cfg.get("recordings") or {}
    matched: Dict[str, pd.DataFrame] = {}
    rec_ids: Dict[str, str] = {}
    for domain, spec in recordings.items():
        rec_ids[domain] = spec["recording_id"]
        path = out_root / spec["recording_id"] / "map_matched.parquet"
        if not path.exists():
            raise SystemExit(f"missing {path}; run match_article1_routes first")
        matched[domain] = pd.read_parquet(path)

    if set(matched) != {"car", "motorcycle"}:
        raise SystemExit(f"expected both domains, got {sorted(matched)}")

    pc = cfg.get("route.pairing") or {}
    criteria = PairingCriteria(
        max_lateral_distance_m=float(pc.get("max_lateral_distance_m", 25.0)),
        max_heading_difference_deg=float(pc.get("max_heading_difference_deg", 45.0)),
        min_samples_per_bin=int(pc.get("min_samples_per_bin", 2)),
        require_same_direction=bool(pc.get("require_same_direction", True)),
        require_map_match=bool(pc.get("require_map_match", True)),
    )

    # Positional resolution of a nearest-real-fix assignment, from the measured
    # GPS cadence and the observed speed, not from an assumed value.
    resolutions, gps_hz, gps_speed = {}, {}, {}
    for domain, df in matched.items():
        ok = df[df["matched"].astype(bool)]
        dt = np.diff(np.sort(ok["timestamp_ns"].values)) / 1e9
        hz = 1.0 / float(np.median(dt)) if dt.size else float("nan")
        speed = (float(np.nanmedian(ok["gps_speed_mps"]))
                 if "gps_speed_mps" in ok else np.nan)
        gps_hz[domain], gps_speed[domain] = hz, speed
        resolutions[domain] = positional_resolution_m(hz, speed)
        log.info("%s: GPS %.2f Hz, median speed %.1f m/s -> %.1f m resolution",
                 domain, hz, speed, resolutions[domain])
    resolution = float(np.nanmax(list(resolutions.values())))

    all_bins, all_pairs, summaries = [], [], {}
    for bin_size in cfg.get("route.bin_sizes_m", [10.0, 20.0, 50.0]):
        bin_size = float(bin_size)
        bins = {d: build_route_bins(matched[d], d, rec_ids[d], bin_size)
                for d in ("car", "motorcycle")}
        for b in bins.values():
            all_bins.append(b.to_frame())
        pairs = pair_route_bins(bins["car"], bins["motorcycle"], criteria)
        if not pairs.empty:
            pairs.insert(2, "domain_a", "car")
            pairs.insert(3, "domain_b", "motorcycle")
        all_pairs.append(pairs)
        # The scarcer domain decides whether a bin can hold enough fixes.
        expected = float(np.nanmin([expected_fixes_per_bin(bin_size, gps_hz[d],
                                                           gps_speed[d])
                                    for d in matched]))
        s = shared_route_summary(pairs, bin_size, resolution, expected,
                                 criteria.min_samples_per_bin)
        s["car_bins_occupied"] = int(len(bins["car"].keys))
        s["motorcycle_bins_occupied"] = int(len(bins["motorcycle"].keys))
        summaries[f"{bin_size:.0f}m"] = s
        log.info("%.0f m bins: car %d, moto %d, shared %d, paired %d "
                 "(~%.1f fixes/bin)", bin_size, s["car_bins_occupied"],
                 s["motorcycle_bins_occupied"], s["shared_bins_considered"],
                 s["paired_bins"], expected)

    bins_df = pd.concat(all_bins, ignore_index=True)
    pairs_df = pd.concat(all_pairs, ignore_index=True)

    # Committed CSVs: no absolute coordinates. The projected metric centroid is
    # relative to the network origin, which is itself withheld from the report.
    public_bins = bins_df.drop(columns=[c for c in ("mean_x_m", "mean_y_m")
                                        if c in bins_df])
    public_bins.to_csv(reports / "shared_route_bins.csv", index=False)
    pairs_df.to_csv(reports / "paired_route_segments.csv", index=False)
    bins_df.to_parquet(out_root / "route_bins_full.parquet", index=False)

    # The analysis bin is the SMALLEST configured size the GPS cadence can
    # actually fill, not the smallest size available: a finer bin would report a
    # sampling limit as a route difference.
    usable = [float(k[:-1]) for k, s in summaries.items()
              if not s["gps_cadence_undersamples_this_bin"]]
    analysis_bin = min(usable) if usable else float(
        max(cfg.get("route.bin_sizes_m", [50.0])))
    log.info("analysis bin size: %.0f m (usable sizes: %s)", analysis_bin,
             sorted(usable) or "none")
    frame_assignments = {}
    for domain, spec in recordings.items():
        tl = pd.read_parquet(out_root / spec["recording_id"] /
                             "multimodal_timeline.parquet")
        assign = assign_frames_to_bins(tl["timestamp_ns"].values, matched[domain],
                                       analysis_bin)
        adf = pd.DataFrame({"frame_index": tl["frame_index"].values,
                            "timestamp_ns": tl["timestamp_ns"].values, **assign})
        adf.to_parquet(out_root / spec["recording_id"] / "frame_route_bins.parquet",
                       index=False)
        frame_assignments[domain] = {
            "bin_size_m": analysis_bin,
            "frames": int(len(adf)),
            "frames_with_valid_bin": int(adf["bin_valid"].sum()),
            "valid_fraction": float(adf["bin_valid"].mean()),
            "median_gps_dt_s": float(np.nanmedian(adf["gps_dt_s"])),
            "median_position_uncertainty_m": float(
                np.nanmedian(adf["position_uncertainty_m"])),
        }
        log.info("%s frames binned: %d/%d valid, median position uncertainty %.1f m",
                 domain, frame_assignments[domain]["frames_with_valid_bin"],
                 frame_assignments[domain]["frames"],
                 frame_assignments[domain]["median_position_uncertainty_m"])

    report: Dict[str, Any] = {
        "schema": "article1_route_alignment_v1",
        "result_status": EXPLORATORY_MARKER,
        "alignment": "spatial_only_never_temporal",
        "bin_definition": ("map-defined: (osm_way_id, floor(offset_along_way / "
                           "bin_size), travel_direction)"),
        "pairing_criteria": {
            "max_lateral_distance_m": criteria.max_lateral_distance_m,
            "max_heading_difference_deg": criteria.max_heading_difference_deg,
            "min_samples_per_bin": criteria.min_samples_per_bin,
            "require_same_direction": criteria.require_same_direction,
            "require_map_match": criteria.require_map_match,
        },
        "positional_resolution_m": resolution,
        "positional_resolution_per_domain_m": resolutions,
        "gps_rate_hz": gps_hz,
        "median_gps_speed_mps": gps_speed,
        "analysis_bin_size_m": analysis_bin,
        "analysis_bin_rationale": (
            "smallest configured bin size the 1 Hz GPS cadence can fill with the "
            f"{criteria.min_samples_per_bin} fixes pairing requires"),
        "bin_sizes": summaries,
        "frame_bin_assignment": frame_assignments,
        "privacy": "no absolute coordinates in any committed file",
    }
    atomic_write_json(reports / "route_alignment_summary.json", report)
    print(json.dumps(summaries, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
