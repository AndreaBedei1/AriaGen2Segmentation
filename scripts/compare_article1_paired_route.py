#!/usr/bin/env python3
"""Phases 10-12: paired spatial comparison, derived indices and statistics.

Car and motorcycle are compared only where they drove the same stretch of road in
the same direction. Every comparison is paired on the bin, the unit of analysis is
the bin (never the frame), intervals come from a block bootstrap and the family of
comparisons is FDR-controlled.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from aria_drive_seg.behavior import EXPLORATORY_MARKER
from aria_drive_seg.behavior.indices import (attentional_tunnelling_candidates,
                                             cross_correlation,
                                             head_eye_coordination,
                                             road_complexity_index,
                                             traffic_interaction_index,
                                             visual_demand_index)
from aria_drive_seg.behavior.stats import (PILOT_CAVEAT, benjamini_hochberg,
                                           block_permutation_test, cliffs_delta,
                                           describe, paired_block_bootstrap)
from aria_drive_seg.config import Config
from aria_drive_seg.io_utils import atomic_write_json
from aria_drive_seg.logging_utils import get_logger, setup_logging

log = get_logger("behavior.paired")

#: Metrics carried into the paired table, per domain and bin.
BIN_METRICS = (
    ("speed_mps", "vehicle"),
    ("acceleration_mps2", "vehicle"),
    ("lateral_acceleration_mps2", "vehicle"),
    ("head_angular_speed_rad_s", "head"),
    ("head_abs_yaw_rate_rad_s", "head"),
    ("head_vibration_rms_msec2", "head"),
    ("heart_rate_bpm", "ppg"),
    ("ppg_sqi", "ppg"),
    ("lux", "light"),
    ("gaze_foveal_entropy", "gaze"),
    ("gaze_road_relevant_mass", "gaze"),
)


def per_bin_metrics(dest: Path, bin_size_m: float) -> pd.DataFrame:
    """Aggregate every signal family onto the frame-level route bins."""
    bins = pd.read_parquet(dest / "frame_route_bins.parquet")
    bins = bins[bins["bin_valid"].astype(bool)]
    if bins.empty:
        return pd.DataFrame()
    tl = pd.read_parquet(dest / "multimodal_timeline.parquet")
    frame_ts = tl.set_index("frame_index")["timestamp_ns"]

    out = bins[["frame_index", "timestamp_ns", "bin_key",
                "position_uncertainty_m"]].copy()

    def attach(path: Path, ts_col: str, cols: Dict[str, str]) -> None:
        if not path.exists():
            return
        src = pd.read_parquet(path).sort_values(ts_col)
        if src.empty:
            return
        st = src[ts_col].values.astype(np.int64)
        pos = np.clip(np.searchsorted(st, out["timestamp_ns"].values), 0, st.size - 1)
        left = np.clip(pos - 1, 0, st.size - 1)
        take = np.abs(out["timestamp_ns"].values - st[left]) <= \
            np.abs(st[pos] - out["timestamp_ns"].values)
        idx = np.where(take, left, pos)
        for src_col, dst_col in cols.items():
            if src_col in src:
                out[dst_col] = src[src_col].values[idx]

    attach(dest / "vehicle_dynamics.parquet", "timestamp_ns", {
        "speed_mps": "speed_mps", "acceleration_mps2": "acceleration_mps2",
        "lateral_acceleration_mps2": "lateral_acceleration_mps2",
        "curvature_1_per_m": "curvature_1_per_m"})
    attach(dest / "head_dynamics_per_frame.parquet", "frame_timestamp_ns", {
        "angular_speed_rad_s": "head_angular_speed_rad_s",
        "vibration_rms_msec2": "head_vibration_rms_msec2",
        "yaw_rate_rad_s": "head_yaw_rate_rad_s"})
    if "head_yaw_rate_rad_s" in out:
        out["head_abs_yaw_rate_rad_s"] = out["head_yaw_rate_rad_s"].abs()

    beats = dest / "ppg_beats.parquet"
    if beats.exists():
        b = pd.read_parquet(beats)
        b = b[b["valid"].astype(bool)]
        if not b.empty:
            attach_src = dest / "_ppg_valid.parquet"
            b.to_parquet(attach_src, index=False)
            attach(attach_src, "timestamp_ns",
                   {"heart_rate_bpm": "heart_rate_bpm", "sqi_at_beat": "ppg_sqi"})
            attach_src.unlink(missing_ok=True)

    attach(dest / "sensors" / "als.parquet", "timestamp_ns", {"lux": "lux"})

    sg = dest / "semantic_gaze.parquet"
    if sg.exists():
        g = pd.read_parquet(sg)
        g = g[g["semantic_valid"].astype(bool)]
        if not g.empty:
            road = [c for c in g.columns if c.startswith("p_") and c[2:] in (
                "road_surface", "lane_marking", "regulatory_road_marking",
                "vehicle", "pedestrian", "traffic_sign", "traffic_light",
                "road_boundary_or_obstacle")]
            g = g.assign(gaze_road_relevant_mass=g[road].sum(axis=1),
                         gaze_foveal_entropy=g["foveal_entropy"])
            tmp = dest / "_gaze_tmp.parquet"
            g.to_parquet(tmp, index=False)
            attach(tmp, "timestamp_ns", {
                "gaze_foveal_entropy": "gaze_foveal_entropy",
                "gaze_road_relevant_mass": "gaze_road_relevant_mass"})
            tmp.unlink(missing_ok=True)
            # Only frames inside a dense semantic block have a real reading.
            covered = set(g["frame_index"].tolist())
            mask = ~out["frame_index"].isin(covered)
            for c in ("gaze_foveal_entropy", "gaze_road_relevant_mass"):
                if c in out:
                    out.loc[mask, c] = np.nan

    numeric = [c for c in out.columns if c not in
               ("frame_index", "timestamp_ns", "bin_key")]
    agg = out.groupby("bin_key")[numeric].median()
    agg["frames_in_bin"] = out.groupby("bin_key").size()
    return agg.reset_index()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/article1/behavior_analysis.yaml")
    ap.add_argument("--output", default="output/article1/behavior_analysis")
    ap.add_argument("--reports", default="reports/article1_behavior_analysis")
    ap.add_argument("--log-level", default="INFO")
    args = ap.parse_args()
    setup_logging(args.log_level)

    cfg = Config.load(args.config)
    scfg = cfg.get("statistics") or {}
    rng = np.random.default_rng(int(scfg.get("random_seed", 20260802)))
    out_root = Path(args.output)
    reports = Path(args.reports) / "paired"
    reports.mkdir(parents=True, exist_ok=True)

    align = json.loads((Path(args.reports) / "route" /
                        "route_alignment_summary.json").read_text())
    bin_size = float(align["analysis_bin_size_m"])
    log.info("analysis bin size %.0f m", bin_size)

    per_domain: Dict[str, pd.DataFrame] = {}
    for domain, spec in (cfg.get("recordings") or {}).items():
        dest = out_root / spec["recording_id"]
        df = per_bin_metrics(dest, bin_size)
        per_domain[domain] = df
        log.info("%s: %d occupied bins with metrics", domain, len(df))

    pairs = pd.read_csv(Path(args.reports) / "route" / "paired_route_segments.csv")
    pairs = pairs[(pairs["bin_size_m"] == bin_size) & pairs["paired"].astype(bool)]
    log.info("%d paired bins at %.0f m", len(pairs), bin_size)

    merged = pairs[["bin_key", "osm_way_id", "road_class", "road_is_roundabout",
                    "curvature_1_per_m", "distance_to_junction_m",
                    "distance_to_roundabout_m", "road_lanes", "road_maxspeed_kph",
                    "pair_quality", "centroid_distance_m",
                    "heading_difference_deg"]].copy()
    merged = merged.merge(per_domain["car"].add_suffix("_car"),
                          left_on="bin_key", right_on="bin_key_car", how="inner")
    merged = merged.merge(per_domain["motorcycle"].add_suffix("_moto"),
                          left_on="bin_key", right_on="bin_key_moto", how="inner")
    log.info("%d paired bins with metrics on both sides", len(merged))

    # ---------------- derived indices --------------------------------------
    indices: Dict[str, Any] = {}
    for domain, suffix in (("car", "_car"), ("motorcycle", "_moto")):
        rci = road_complexity_index(
            curvature=merged["curvature_1_per_m"].values,
            junction_density=-merged["distance_to_junction_m"].values,
            lanes=merged["road_lanes"].values)
        vdi = visual_demand_index(
            gaze_entropy=merged.get(f"gaze_foveal_entropy{suffix}"),
            head_motion=merged.get(f"head_angular_speed_rad_s{suffix}"))
        tii = traffic_interaction_index(
            speed_variation=merged.get(f"acceleration_mps2{suffix}"))
        merged[f"road_complexity_index{suffix}"] = rci.values
        merged[f"visual_demand_index{suffix}"] = vdi.values
        merged[f"traffic_interaction_index{suffix}"] = tii.values
        indices[domain] = {"road_complexity": rci.to_dict(),
                           "visual_demand": vdi.to_dict(),
                           "traffic_interaction": tii.to_dict()}

    # ---------------- paired comparisons -----------------------------------
    comparisons: List[Dict[str, Any]] = []
    block_len_s = float((scfg.get("block_bootstrap") or {}).get("block_length_s", 30.0))
    # A bin covers bin_size metres; at the observed speeds that is a few seconds.
    typical_speed = float(np.nanmedian(merged.get("speed_mps_moto", pd.Series([12.0]))))
    unit_s = bin_size / max(typical_speed, 1e-6)
    block_units = max(1, int(round(block_len_s / unit_s)))
    log.info("bootstrap block = %d bins (%.0f s at %.1f m/s)", block_units,
             block_len_s, typical_speed)

    metric_names = [m for m, _ in BIN_METRICS] + [
        "road_complexity_index", "visual_demand_index", "traffic_interaction_index"]
    for metric in metric_names:
        ca, mo = f"{metric}_car", f"{metric}_moto"
        if ca not in merged or mo not in merged:
            continue
        a = merged[mo].values          # motorcycle minus car, per the brief
        b = merged[ca].values
        ok = np.isfinite(a) & np.isfinite(b)
        if ok.sum() < 3:
            comparisons.append({
                "metric": metric, "n_paired_bins": int(ok.sum()),
                "skipped": True,
                "reason": f"only {int(ok.sum())} bins have the metric on both sides",
            })
            continue
        eff = paired_block_bootstrap(
            a[ok], b[ok], block_size=block_units,
            iterations=int((scfg.get("block_bootstrap") or {}).get("iterations", 2000)),
            rng=rng)
        perm = block_permutation_test(
            a[ok], b[ok],
            lambda x, y: float(np.median(x) - np.median(y)),
            block_size=block_units,
            iterations=int((scfg.get("permutation") or {}).get("iterations", 5000)),
            rng=rng)
        ratio = None
        if np.all(b[ok] > 0):
            ratio = float(np.median(a[ok] / b[ok]))
        # A bin's road attributes are a property of the PLACE, and a paired bin is
        # the same place for both vehicles. Their difference is therefore zero by
        # construction and the comparison carries no information about behaviour.
        place_property = metric in ("road_complexity_index",)
        comparisons.append({
            "metric": metric, "skipped": False,
            "n_paired_bins": int(ok.sum()),
            "effective_independent_blocks": int(np.ceil(ok.sum() / block_units)),
            "is_place_property": place_property,
            "place_property_note": (
                "built only from map attributes of the bin, which are identical "
                "for both vehicles because the bin is the same place; the zero "
                "difference is arithmetic, not a finding" if place_property
                else None),
            "unit_of_analysis": f"{bin_size:.0f} m route bin",
            "motorcycle_median": float(np.median(a[ok])),
            "car_median": float(np.median(b[ok])),
            "paired_median_difference_moto_minus_car": eff.estimate,
            "ci_low": eff.ci_low, "ci_high": eff.ci_high,
            "ci_method": eff.method, "ci_caveat": eff.caveat,
            "median_ratio_moto_over_car": ratio,
            "cliffs_delta": cliffs_delta(a[ok], b[ok]),
            "permutation_p": perm.get("p_value"),
            "permutation_iterations": perm.get("iterations"),
            "permutation_reason": perm.get("reason"),
        })

    fdr = benjamini_hochberg([c.get("permutation_p") for c in comparisons],
                             alpha=float(scfg.get("fdr_alpha", 0.05)))
    for c, adj in zip(comparisons, fdr["adjusted"]):
        c["permutation_p_fdr"] = adj
        c["significant_after_fdr"] = bool(adj is not None and
                                          adj <= fdr["alpha"])

    merged.to_csv(reports / "paired_route_comparison.csv", index=False)
    pd.DataFrame(comparisons).to_csv(reports / "paired_statistics.csv", index=False)

    report = {
        "schema": "article1_paired_comparison_v1",
        "result_status": EXPLORATORY_MARKER,
        "pilot_caveat": PILOT_CAVEAT,
        "unit_of_analysis": f"{bin_size:.0f} m map-defined route bin",
        "frames_are_not_independent_samples": True,
        "analysis_bin_size_m": bin_size,
        "paired_bins_available": int(len(pairs)),
        "paired_bins_with_metrics": int(len(merged)),
        "bootstrap_block_bins": block_units,
        "bootstrap_block_seconds": block_len_s,
        "effective_independent_blocks": int(np.ceil(len(merged) / block_units)),
        "effective_sample_size_note": (
            f"{len(merged)} paired bins resolve to about "
            f"{int(np.ceil(len(merged) / block_units))} independent "
            f"{block_len_s:.0f} s blocks. That block count, not the bin count, is "
            "the effective sample size behind every interval below."),
        "semantic_gaze_in_paired_comparison": {
            "available": False,
            "reason": ("neither 30 s frozen semantic block falls on a bin both "
                       "vehicles drove, so no paired bin has a semantic-gaze "
                       "reading on both sides"),
            "consequence": ("the paired comparison covers dynamics, physiology "
                            "and light only; car-motorcycle semantic gaze is "
                            "compared per recording, not per shared bin"),
            "fix": ("run the frozen pipeline on a block inside the shared route "
                    "identified in route/paired_route_segments.csv"),
        },
        "fdr": {k: v for k, v in fdr.items() if k != "adjusted"},
        "indices": indices,
        "comparisons": comparisons,
    }
    atomic_write_json(reports / "paired_comparison_summary.json", report)

    ok = [c for c in comparisons if not c.get("skipped")]
    print(json.dumps({
        "paired_bins": int(len(merged)),
        "metrics_compared": len(ok),
        "significant_after_fdr": sum(c["significant_after_fdr"] for c in ok),
        "top": [{"metric": c["metric"], "n": c["n_paired_bins"],
                 "moto_minus_car": round(c["paired_median_difference_moto_minus_car"], 4),
                 "p_fdr": (None if c["permutation_p_fdr"] is None
                           else round(c["permutation_p_fdr"], 4))}
                for c in sorted(ok, key=lambda x: x["permutation_p_fdr"] or 1)[:6]],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
