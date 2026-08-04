#!/usr/bin/env python3
"""Phase 8: paired car/motorcycle statistics over the 42 shared 50 m bins.

Three families of comparison, each with its own explicitly named unit:

* **bin** — the 42 map-defined 50 m bins both vehicles drove in the same
  direction. Paired on the bin, which is the same place for both, so speed,
  road class and curvature are held constant by construction. This is the primary
  family;
* **event** — one row per road event. The two drives do not share events, so this
  family is unpaired and compares the event-related *change from each event's own
  local baseline*, which removes the between-session level difference that an
  absolute value would carry;
* **time block** — 30 s blocks of the whole recording, for the metrics that have
  no spatial anchor at all.

The frame is never a unit. `tests/test_behavior_guards.py` pins that, and every
row of every output states its unit and its effective sample size — the number of
independent blocks, not the number of rows.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from aria_drive_seg.behavior import EXPLORATORY_MARKER
from aria_drive_seg.behavior.stats import (PILOT_CAVEAT, benjamini_hochberg,
                                           block_permutation_test, cliffs_delta,
                                           effective_sample_size,
                                           paired_block_bootstrap,
                                           two_sample_block_bootstrap)
from aria_drive_seg.config import Config
from aria_drive_seg.io_utils import atomic_write_json
from aria_drive_seg.logging_utils import get_logger, setup_logging

log = get_logger("behavior.final.paired")

NS_PER_S = 1_000_000_000

#: Bin-level metrics compared across the two vehicles, with the unit they are in.
BIN_METRICS = (
    ("road_relevant_mass_percent", "% foveal mass"),
    ("interior_cockpit_mass_percent", "% foveal mass"),
    ("vulnerable_road_user_mass_percent", "% foveal mass"),
    ("sign_and_signal_mass_percent", "% foveal mass"),
    ("road_relevant_top1_percent", "% of time"),
    ("foveal_entropy_mean", "normalised entropy"),
    ("top1_distribution_entropy", "normalised entropy"),
    ("semantic_transition_rate_per_s", "transitions/s"),
    ("fixation_time_percent", "% of time"),
    ("scanpath_length_deg_per_s", "deg/s"),
    ("blinks_per_minute", "blinks/min"),
    ("eye_closure_fraction", "fraction"),
    ("pupil_diameter_mm", "mm"),
    ("pupil_light_adjusted_residual_mm", "mm, light-adjusted"),
    ("lane_position_abs_offset", "lane half-widths"),
    ("head_angular_speed_rad_s", "rad/s"),
    ("head_abs_yaw_rate_rad_s", "rad/s"),
    ("head_vibration_rms_msec2", "m/s^2"),
    ("speed_mps", "m/s"),
    ("heart_rate_bpm", "bpm"),
    ("ibi_ms", "ms"),
)

#: Event-level metrics, compared as a change from each event's own baseline.
EVENT_DELTA_METRICS = (
    ("delta_heart_rate_bpm", "bpm"),
    ("delta_ibi_ms", "ms"),
    ("delta_blink_rate_per_minute", "blinks/min"),
    ("delta_mean_blink_duration_s", "s"),
    ("delta_eye_closure_fraction", "fraction"),
    ("delta_pupil_light_adjusted_residual_mm", "mm, light-adjusted"),
    ("delta_road_relevant_mass_percent", "% foveal mass"),
    ("delta_gaze_entropy", "normalised entropy"),
    ("delta_head_angular_speed_rad_s", "rad/s"),
    ("delta_speed_mps", "m/s"),
    ("delta_visual_lane_position_proxy_abs_offset", "lane half-widths"),
)

#: Time-block metrics, for signals with no spatial anchor.
BLOCK_METRICS = (
    ("blinks_per_minute", "blinks/min"),
    ("eye_closure_fraction", "fraction"),
    ("pupil_diameter_mm", "mm"),
    ("pupil_light_adjusted_residual_mm", "mm, light-adjusted"),
    ("road_relevant_mass_percent", "% foveal mass"),
    ("foveal_entropy_mean", "normalised entropy"),
    ("head_angular_speed_rad_s", "rad/s"),
    ("lane_position_abs_offset", "lane half-widths"),
)

#: Metrics that are a property of the PLACE, identical for both vehicles in a
#: paired bin. Their difference is arithmetic, not a finding.
PLACE_PROPERTIES = ("bin_size_m",)


def _atomic_parquet(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp.parquet")
    frame.to_parquet(tmp, index=False)
    tmp.replace(path)


def _median(values: np.ndarray) -> Optional[float]:
    finite = values[np.isfinite(values)]
    return float(np.median(finite)) if finite.size else None


def _compare(metric: str, unit: str, moto: np.ndarray, car: np.ndarray,
             block_size: int, paired: bool, family: str,
             iterations: int, permutations: int, rng) -> Dict[str, Any]:
    """One car/motorcycle comparison, with its interval, p-value and effect size."""
    if paired:
        ok = np.isfinite(moto) & np.isfinite(car)
        a, b = moto[ok], car[ok]
        n_units = int(ok.sum())
    else:
        a = moto[np.isfinite(moto)]
        b = car[np.isfinite(car)]
        n_units = int(min(a.size, b.size))

    row: Dict[str, Any] = {
        "family": family, "metric": metric, "unit": unit,
        "statistical_unit": {"bin": "50 m route bin", "event": "road event",
                             "block": "time block"}[family],
        "paired": paired,
        "n_units": n_units,
        "n_car": int(b.size), "n_motorcycle": int(a.size),
        "car_median": _median(b), "motorcycle_median": _median(a),
        "block_size_units": block_size,
        "effective_sample_size": effective_sample_size(n_units, block_size),
        "frames_are_independent_samples": False,
    }
    if n_units < 3:
        row.update({"skipped": True,
                    "reason": f"only {n_units} units have this metric on both "
                              "sides; too few for any interval or test"})
        return row

    effect = (paired_block_bootstrap(a, b, block_size=block_size,
                                     iterations=iterations, rng=rng)
              if paired else
              two_sample_block_bootstrap(a, b, block_size=block_size,
                                         iterations=iterations, rng=rng))
    permutation = block_permutation_test(
        a, b, lambda x, y: float(np.median(x) - np.median(y)),
        block_size=block_size, iterations=permutations, rng=rng)
    row.update({
        "skipped": False,
        "difference_moto_minus_car": effect.estimate,
        "ci_low": effect.ci_low, "ci_high": effect.ci_high,
        "ci_method": effect.method, "ci_caveat": effect.caveat,
        "cliffs_delta": cliffs_delta(a, b),
        "permutation_p": permutation.get("p_value"),
        "permutation_iterations": permutation.get("iterations"),
        "permutation_reason": permutation.get("reason"),
        "is_place_property": metric in PLACE_PROPERTIES,
    })
    return row


def _apply_fdr(rows: List[Dict[str, Any]], alpha: float) -> Dict[str, Any]:
    """Benjamini-Hochberg within one family; families are never pooled."""
    tested = [r for r in rows if not r.get("skipped")]
    fdr = benjamini_hochberg([r.get("permutation_p") for r in tested], alpha=alpha)
    for row, adjusted in zip(tested, fdr["adjusted"]):
        row["permutation_p_fdr"] = adjusted
        row["significant_after_fdr"] = bool(adjusted is not None
                                            and adjusted <= alpha)
    for row in rows:
        row.setdefault("permutation_p_fdr", None)
        row.setdefault("significant_after_fdr", False)
    return {k: v for k, v in fdr.items() if k != "adjusted"}


def _events_per_block(frame: pd.DataFrame, domain: str, block_s: float) -> int:
    """How many events a `block_s`-second decorrelation window contains."""
    starts = frame.loc[frame["domain"] == domain,
                       "event_start_ns"].to_numpy(np.int64)
    if starts.size < 2:
        return 1
    median_gap_s = float(np.median(np.diff(np.sort(starts))) / NS_PER_S)
    if median_gap_s <= 0:
        return 1
    return max(1, int(round(block_s / median_gap_s)))


def _time_blocks(frame: pd.DataFrame, timestamp_column: str, block_s: float,
                 columns: Sequence[str]) -> pd.DataFrame:
    """Median of each metric inside contiguous blocks of `block_s` seconds."""
    ts = frame[timestamp_column].to_numpy(np.int64)
    if ts.size == 0:
        return pd.DataFrame()
    block = ((ts - ts.min()) // int(round(block_s * NS_PER_S))).astype(int)
    present = [c for c in columns if c in frame.columns]
    grouped = frame.assign(_block=block).groupby("_block")[present].median()
    return grouped.reset_index()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/article1/behavior_analysis.yaml")
    ap.add_argument("--analysis",
                    default="output/article1/final_behavior_statistics")
    ap.add_argument("--behavior", default="output/article1/behavior_analysis")
    ap.add_argument("--reports",
                    default="reports/article1_final_behavior_statistics")
    ap.add_argument("--log-level", default="INFO")
    args = ap.parse_args()
    setup_logging(args.log_level)

    cfg = Config.load(args.config)
    scfg = cfg.get("statistics") or {}
    alpha = float(scfg.get("fdr_alpha", 0.05))
    iterations = int((scfg.get("block_bootstrap") or {}).get("iterations", 2000))
    permutations = int((scfg.get("permutation") or {}).get("iterations", 5000))
    block_s = float((scfg.get("block_bootstrap") or {}).get("block_length_s", 30.0))
    seed = int(scfg.get("random_seed", 20260802))
    # One independent generator per family. Sharing one would make a change in the
    # event family silently shift the block family's intervals, which would look
    # like a result and would only be a draw order.
    rngs = {name: np.random.default_rng([seed, index])
            for index, name in enumerate(("bin", "event", "block"))}

    analysis = Path(args.analysis)
    reports = Path(args.reports)
    reports.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ bins #
    bins = pd.read_parquet(analysis / "semantic_attention_bins.parquet")
    wide = bins.pivot(index="bin_key", columns="domain")
    order = bins[bins.domain == "car"].set_index("bin_key")["bin_order"]
    wide = wide.reindex(order.sort_values().index)
    # A bin covers 50 m; at the observed speeds that is a few seconds, so the
    # 30 s bootstrap block spans several bins.
    typical_speed = float(np.nanmedian(
        bins.loc[bins.domain == "motorcycle", "speed_mps"]))
    bin_duration_s = float(bins["bin_size_m"].iloc[0]) / max(typical_speed, 1e-6)
    bin_block = max(1, int(round(block_s / bin_duration_s)))
    log.info("bin family: %d bins, bootstrap block = %d bins (%.0f s at %.1f m/s)",
             len(wide), bin_block, block_s, typical_speed)

    bin_rows: List[Dict[str, Any]] = []
    for metric, unit in BIN_METRICS:
        if (metric, "motorcycle") not in wide.columns:
            continue
        bin_rows.append(_compare(
            metric, unit, wide[(metric, "motorcycle")].to_numpy(float),
            wide[(metric, "car")].to_numpy(float), bin_block, True, "bin",
            iterations, permutations, rngs["bin"]))
    bin_fdr = _apply_fdr(bin_rows, alpha)

    # ---------------------------------------------------------------- events #
    events = pd.read_parquet(analysis / "event_response.parquet")
    immediate = events[events["window"] == "phys_immediate"].sort_values(
        ["domain", "event_start_ns"])
    # Events are not independent either: two junctions 5 s apart on the same road
    # share their traffic, their light and their rider. The block spans the same
    # 30 s decorrelation length used everywhere else, expressed in events, and the
    # larger of the two domains' block sizes is used so the assumption is the
    # conservative one for both.
    event_block = max(_events_per_block(immediate, domain, block_s)
                      for domain in ("car", "motorcycle"))
    event_rows: List[Dict[str, Any]] = []
    for metric, unit in EVENT_DELTA_METRICS:
        if metric not in immediate.columns:
            continue
        event_rows.append(_compare(
            metric, unit,
            immediate.loc[immediate.domain == "motorcycle", metric].to_numpy(float),
            immediate.loc[immediate.domain == "car", metric].to_numpy(float),
            event_block, False, "event", iterations, permutations,
            rngs["event"]))
    event_fdr = _apply_fdr(event_rows, alpha)
    log.info("event family: %d car / %d motorcycle events, block = %d events",
             int((immediate.domain == "car").sum()),
             int((immediate.domain == "motorcycle").sum()), event_block)

    # Per event type, for the table the road-events report needs.
    per_type: List[Dict[str, Any]] = []
    for event_type, group in immediate.groupby("event_type"):
        for metric, unit in EVENT_DELTA_METRICS:
            if metric not in group.columns:
                continue
            car = group.loc[group.domain == "car", metric].to_numpy(float)
            moto = group.loc[group.domain == "motorcycle", metric].to_numpy(float)
            per_type.append({
                "event_type": event_type, "metric": metric, "unit": unit,
                "n_car": int(np.isfinite(car).sum()),
                "n_motorcycle": int(np.isfinite(moto).sum()),
                "car_median": _median(car), "motorcycle_median": _median(moto),
                "statistical_unit": "road event",
                "counts_are_single_digit": bool(
                    min(np.isfinite(car).sum(), np.isfinite(moto).sum()) < 10),
            })

    # ---------------------------------------------------------------- blocks #
    block_frames: Dict[str, pd.DataFrame] = {}
    for domain, spec in (cfg.get("recordings") or {}).items():
        rec_id = str(spec["recording_id"])
        dest = analysis / rec_id
        pupil = pd.read_parquet(dest / "pupil_samples.parquet")
        pupil_blocks = _time_blocks(
            pupil, "timestamp_ns", block_s,
            ["pupil_mean_m", "pupil_light_adjusted_residual_m"])
        pupil_blocks = pupil_blocks.rename(columns={
            "pupil_mean_m": "pupil_diameter_mm",
            "pupil_light_adjusted_residual_m": "pupil_light_adjusted_residual_mm"})
        for column in ("pupil_diameter_mm", "pupil_light_adjusted_residual_mm"):
            if column in pupil_blocks:
                pupil_blocks[column] = pupil_blocks[column] * 1000.0

        windows = pd.read_parquet(dest / "blink_rate_windows.parquet")
        blink_blocks = _time_blocks(windows.rename(
            columns={"window_start_ns": "timestamp_ns"}),
            "timestamp_ns", block_s, ["blinks_per_minute"])

        merged = pupil_blocks.merge(blink_blocks, on="_block", how="outer")
        lane_path = dest / "lane_position_frames.parquet"
        if lane_path.exists():
            lane = pd.read_parquet(lane_path)
            lane = lane.assign(
                lane_position_abs_offset=lane["normalized_offset"].abs())
            merged = merged.merge(
                _time_blocks(lane, "timestamp_ns", block_s,
                             ["lane_position_abs_offset"]),
                on="_block", how="outer")
        merged["domain"] = domain
        block_frames[domain] = merged

    block_rows: List[Dict[str, Any]] = []
    for metric, unit in BLOCK_METRICS:
        car = block_frames["car"]
        moto = block_frames["motorcycle"]
        if metric not in car.columns or metric not in moto.columns:
            continue
        block_rows.append(_compare(
            metric, unit, moto[metric].to_numpy(float), car[metric].to_numpy(float),
            1, False, "block", iterations, permutations, rngs["block"]))
    block_fdr = _apply_fdr(block_rows, alpha)
    log.info("block family: %d car / %d motorcycle %.0f s blocks",
             len(block_frames["car"]), len(block_frames["motorcycle"]), block_s)

    # ----------------------------------------------------------------- write #
    all_rows = pd.DataFrame(bin_rows + event_rows + block_rows)
    _atomic_parquet(analysis / "paired_statistics.parquet", all_rows)
    all_rows.to_csv(reports / "paired_statistics.csv", index=False)
    pd.DataFrame(per_type).to_csv(reports / "event_type_medians.csv", index=False)
    wide.to_csv(reports / "paired_bin_metrics.csv")

    surviving = {family: sorted(
        (r["metric"] for r in rows if r.get("significant_after_fdr")))
        for family, rows in (("bin", bin_rows), ("event", event_rows),
                             ("block", block_rows))}
    summary = {
        "schema": "article1_final_paired_statistics_v1",
        "result_status": EXPLORATORY_MARKER,
        "pilot_caveat": PILOT_CAVEAT,
        "random_seed": seed,
        "seeding": ("one generator per family, so a change in one family cannot "
                    "shift another's intervals through the draw order"),
        "families": {
            "bin": {
                "statistical_unit": "50 m map-defined route bin",
                "paired": True,
                "units": int(len(wide)),
                "block_size_units": bin_block,
                "block_length_s": block_s,
                "effective_sample_size": effective_sample_size(len(wide), bin_block),
                "metrics_compared": len([r for r in bin_rows if not r["skipped"]]),
                "surviving_fdr": len(surviving["bin"]),
                "fdr": bin_fdr,
            },
            "event": {
                "statistical_unit": "road event",
                "paired": False,
                "pairing_note": ("the two drives do not share events, so this "
                                 "family compares each event's change from its "
                                 "own local baseline rather than an absolute "
                                 "level"),
                "car_units": int((immediate.domain == "car").sum()),
                "motorcycle_units": int((immediate.domain == "motorcycle").sum()),
                "block_size_units": event_block,
                "block_length_s": block_s,
                "metrics_compared": len([r for r in event_rows if not r["skipped"]]),
                "surviving_fdr": len(surviving["event"]),
                "fdr": event_fdr,
            },
            "block": {
                "statistical_unit": f"{block_s:.0f} s time block",
                "paired": False,
                "car_units": int(len(block_frames["car"])),
                "motorcycle_units": int(len(block_frames["motorcycle"])),
                "block_size_units": 1,
                "block_size_note": (
                    f"one unit, because a unit here already *is* the {block_s:.0f} s "
                    "decorrelation window used everywhere else"),
                "metrics_compared": len([r for r in block_rows if not r["skipped"]]),
                "surviving_fdr": len(surviving["block"]),
                "fdr": block_fdr,
            },
        },
        "surviving_fdr": surviving,
        "gaze_in_paired_comparison": {
            "available": True,
            "bins_with_gaze_on_both_sides": int(len(wide)),
            "source": "full-recording fast semantic gaze at 5 Hz segmentation",
            "note": ("semantic gaze now enters the paired comparison on every "
                     "shared bin; the earlier limitation, where two 30 s frozen "
                     "blocks covered no shared bin, no longer applies"),
        },
        "frames_are_independent_samples": False,
        "unit_of_analysis_is_never_the_frame": True,
    }
    atomic_write_json(reports / "paired_comparison_summary.json", summary)
    atomic_write_json(analysis / "paired_comparison_summary.json", summary)

    print(json.dumps({
        "bins": int(len(wide)),
        "surviving_fdr": surviving,
        "bin_effective_sample_size": summary["families"]["bin"][
            "effective_sample_size"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
