#!/usr/bin/env python3
"""Phases 1, 4, 5 and 6: full-recording semantic attention, events, head-eye.

This is the stage that replaces the two 30 s frozen semantic blocks with the
**complete** semantic gaze produced by the fast full run. Every gaze number
downstream of here covers the whole recording, and the 42 shared 50 m bins carry
a gaze reading on both sides.

Three conventions hold throughout:

* the unit of analysis is a route bin, an event or a time block — never a frame,
  and never a gaze sample;
* every window is a duration in seconds, taken from the configuration, and is
  applied to each signal family on that family's own real samples;
* foveal probability mass is the primary gaze metric and top-1 share the
  secondary one, for the viewing-geometry reason set out in
  `behavior/semantic_attention.py`.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from aria_drive_seg.behavior import EXPLORATORY_MARKER
from aria_drive_seg.behavior.events import RoadEvent, build_windows
from aria_drive_seg.behavior.blink import (blink_rate_in_window, closed_series,
                                           closure_fraction_in_window,
                                           determine_blink_polarity)
from aria_drive_seg.behavior.head_eye import (attach_head_rate, gaze_yaw_rate,
                                              head_eye_metrics, head_motion_before)
from aria_drive_seg.behavior.pupil import pupil_in_window
from aria_drive_seg.behavior.semantic_attention import (attention_metrics,
                                                        flatten_unit,
                                                        per_class_rows)
from aria_drive_seg.config import Config
from aria_drive_seg.io_utils import atomic_write_json
from aria_drive_seg.logging_utils import get_logger, setup_logging
from aria_drive_seg.taxonomy import Taxonomy

log = get_logger("behavior.final")

NS_PER_S = 1_000_000_000
RUN_DIRS = {"car": "car", "motorcycle": "motorcycle"}

#: Map-derived event kinds carried into the Phase-5 table, with the short name
#: used in every report and figure.
EVENT_KINDS = {
    "roundabout_traverse": "roundabout",
    "junction_crossing": "junction",
    "curve": "curve",
    "straight": "straight",
    "pedestrian_crossing": "pedestrian_crossing",
    "traffic_signals": "traffic_signal",
}

#: The window whose value every delta is taken against.
BASELINE_WINDOW = "phys_baseline"


def _atomic_parquet(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp.parquet")
    frame.to_parquet(tmp, index=False)
    tmp.replace(path)


def _nearest_bin(sample_ns: np.ndarray, frame_ns: np.ndarray,
                 frame_bin: np.ndarray, max_dt_s: float) -> np.ndarray:
    """Bin key of the nearest real reference frame, or None beyond a tolerance."""
    if frame_ns.size == 0:
        return np.full(sample_ns.size, None, dtype=object)
    hi = np.clip(np.searchsorted(frame_ns, sample_ns, side="left"), 0,
                 frame_ns.size - 1)
    lo = np.clip(hi - 1, 0, frame_ns.size - 1)
    use_hi = np.abs(frame_ns[hi] - sample_ns) < np.abs(sample_ns - frame_ns[lo])
    idx = np.where(use_hi, hi, lo)
    out = frame_bin[idx].astype(object)
    out[np.abs(frame_ns[idx] - sample_ns) > max_dt_s * NS_PER_S] = None
    return out


def _median_in_window(ts: np.ndarray, values: np.ndarray, start_ns: int,
                      end_ns: int) -> Optional[float]:
    sel = (ts >= int(start_ns)) & (ts < int(end_ns))
    window = values[sel]
    window = window[np.isfinite(window)]
    return float(np.median(window)) if window.size else None


# --------------------------------------------------------------------------- #
# Phase 4 — semantic attention
# --------------------------------------------------------------------------- #
def _semantic_context(root: Path) -> Dict[str, Any]:
    """Sampling interval and pixel scale of one fast run, from its own artefacts."""
    calibration = json.loads((root / "frames" / "calibration.json").read_text())
    stored = calibration["stored_resolution"]
    focal = float(calibration["pinhole_focal_at_stored_resolution"])
    return {"px_per_deg": focal * np.tan(np.radians(1.0)),
            "image_shape": (int(stored["height"]), int(stored["width"]))}


def semantic_attention_for_run(root: Path, class_names: Sequence[str]
                               ) -> Dict[str, Any]:
    """Whole-recording semantic attention for one fast run."""
    gaze = pd.read_parquet(root / "gaze" / "semantic_gaze.parquet").sort_values(
        "timestamp_ns").reset_index(drop=True)
    fixations = pd.read_parquet(root / "gaze" / "fixations.parquet")
    valid = gaze[gaze["semantic_valid"].astype(bool)]
    interval_s = (float(np.median(np.diff(gaze["timestamp_ns"].to_numpy(np.int64)))
                        / NS_PER_S) if len(gaze) > 1 else 0.0)
    context = _semantic_context(root)
    metrics = attention_metrics(
        valid, class_names, interval_s, fixations=fixations,
        px_per_deg=context["px_per_deg"], image_shape=context["image_shape"])
    metrics["gaze_samples"] = int(len(gaze))
    metrics["semantic_valid_samples"] = int(len(valid))
    metrics["semantic_valid_fraction"] = (float(len(valid) / len(gaze))
                                          if len(gaze) else 0.0)
    metrics["recording_coverage"] = (
        "the whole recording: every real gaze sample is associated with the "
        "nearest real segmented frame, and no part of the drive is excluded by "
        "construction")
    return metrics


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/article1/behavior_analysis.yaml")
    ap.add_argument("--gaze-config",
                    default="configs/article1/fast_semantic_gaze.yaml")
    ap.add_argument("--run-root", default="output/article1/fast_semantic_gaze")
    ap.add_argument("--native-run-root",
                    default="output/article1/fast_semantic_gaze_native")
    ap.add_argument("--behavior", default="output/article1/behavior_analysis")
    ap.add_argument("--reports", default="reports/article1_behavior_analysis")
    ap.add_argument("--output", default="output/article1/final_behavior_statistics")
    ap.add_argument("--bin-tolerance-s", type=float, default=0.5)
    ap.add_argument("--log-level", default="INFO")
    args = ap.parse_args()
    setup_logging(args.log_level)

    cfg = Config.load(args.config)
    gaze_cfg = Config.load(args.gaze_config)
    taxonomy = Taxonomy.load(gaze_cfg.resolve(
        gaze_cfg.get("semantic_gaze_fast_external.classes")))
    class_names = [c.name for c in taxonomy.classes if c.name != "unknown"]
    hcfg = cfg.get("head_eye") or {}
    bcfg = (cfg.get("eye_state") or {}).get("blink") or {}
    windows_cfg: Dict[str, Sequence[float]] = {
        **(cfg.get("events.windows_s") or {}),
        **{f"phys_{k}": v for k, v in (cfg.get("events.physiology_windows_s")
                                       or {}).items()}}

    out_root = Path(args.output)
    behavior_root = Path(args.behavior)
    reports_root = Path(args.reports)
    run_root = Path(args.run_root)
    native_root = Path(args.native_run_root)

    align = json.loads((reports_root / "route" /
                        "route_alignment_summary.json").read_text())
    bin_size_m = float(align["analysis_bin_size_m"])
    pairs = pd.read_csv(reports_root / "route" / "paired_route_segments.csv")
    pairs = pairs[(pairs["bin_size_m"] == bin_size_m) &
                  pairs["paired"].astype(bool)].sort_values("first_timestamp_ns_a")
    bin_order = {key: i for i, key in enumerate(pairs["bin_key"])}
    log.info("%d paired bins at %.0f m", len(bin_order), bin_size_m)

    events = pd.read_csv(reports_root / "events" / "route_events.csv")
    solid = pd.read_csv(reports_root / "solid_line" / "solid_line_candidates.csv")

    recording_metrics: Dict[str, Any] = {}
    native_metrics: Dict[str, Any] = {}
    head_eye: Dict[str, Any] = {}
    bin_rows: List[Dict[str, Any]] = []
    bin_class_rows: List[Dict[str, Any]] = []
    event_rows: List[Dict[str, Any]] = []

    for domain, spec in (cfg.get("recordings") or {}).items():
        rec_id = str(spec["recording_id"])
        dest = out_root / rec_id
        root = run_root / RUN_DIRS[domain]
        log.info("== %s (%s)", domain, rec_id)

        # ---------------- sources ------------------------------------------ #
        gaze = pd.read_parquet(root / "gaze" / "semantic_gaze.parquet").sort_values(
            "timestamp_ns").reset_index(drop=True)
        fixations = pd.read_parquet(root / "gaze" / "fixations.parquet")
        gaze_valid = gaze[gaze["semantic_valid"].astype(bool)].reset_index(drop=True)
        gaze_interval_s = float(np.median(
            np.diff(gaze["timestamp_ns"].to_numpy(np.int64))) / NS_PER_S)
        context = _semantic_context(root)

        eye_state = pd.read_parquet(dest / "eye_state_samples.parquet")
        blink_events = pd.read_parquet(dest / "blink_events.parquet")
        pupil = pd.read_parquet(dest / "pupil_samples.parquet")
        polarity = determine_blink_polarity(
            eye_state["left_blink"].to_numpy(bool),
            eye_state["left_blink_valid"].to_numpy(bool),
            eye_state["timestamp_ns"].to_numpy(np.int64),
            max_closed_fraction=float(bcfg.get("max_closed_fraction", 0.35)))
        closed = closed_series(eye_state, polarity)

        head = pd.read_parquet(
            behavior_root / rec_id / "head_dynamics_native.parquet").sort_values(
            "timestamp_ns")
        vehicle = pd.read_parquet(
            behavior_root / rec_id / "vehicle_dynamics.parquet").sort_values(
            "timestamp_ns")
        beats = pd.read_parquet(behavior_root / rec_id / "ppg_beats.parquet")
        beats = beats[beats["valid"].astype(bool)].sort_values("timestamp_ns")
        lane_path = dest / "lane_position_frames.parquet"
        lane = (pd.read_parquet(lane_path).sort_values("timestamp_ns")
                if lane_path.exists() else None)
        if lane is None:
            log.warning("  no lane-position table yet; run "
                        "measure_article1_lane_position.py first")

        route_bins = pd.read_parquet(
            behavior_root / rec_id / "frame_route_bins.parquet")
        route_bins = route_bins[route_bins["bin_valid"].astype(bool)].sort_values(
            "timestamp_ns")
        frame_ns = route_bins["timestamp_ns"].to_numpy(np.int64)
        frame_bin = route_bins["bin_key"].to_numpy(object)

        # ---------------- Phase 4: whole recording ------------------------- #
        recording_metrics[domain] = semantic_attention_for_run(root, class_names)
        native_dir = native_root / RUN_DIRS[domain]
        if (native_dir / "gaze" / "semantic_gaze.parquet").exists():
            native_metrics[domain] = semantic_attention_for_run(
                native_dir, class_names)
        log.info("  semantic gaze: %d/%d samples valid (%.1f%%), road-relevant "
                 "mass %.1f%%",
                 recording_metrics[domain]["semantic_valid_samples"],
                 recording_metrics[domain]["gaze_samples"],
                 100 * recording_metrics[domain]["semantic_valid_fraction"],
                 recording_metrics[domain]["groups"]["road_relevant"][
                     "foveal_mass_percent"])

        # ---------------- Phase 4: per shared-route bin -------------------- #
        gaze_bins = _nearest_bin(
            gaze_valid["timestamp_ns"].to_numpy(np.int64), frame_ns, frame_bin,
            args.bin_tolerance_s)
        gaze_valid = gaze_valid.assign(bin_key=gaze_bins)
        eye_bins = _nearest_bin(
            eye_state["timestamp_ns"].to_numpy(np.int64), frame_ns, frame_bin,
            args.bin_tolerance_s)
        pupil_bins = eye_bins
        lane_bins = (_nearest_bin(lane["timestamp_ns"].to_numpy(np.int64), frame_ns,
                                  frame_bin, args.bin_tolerance_s)
                     if lane is not None else None)

        head_ts = head["timestamp_ns"].to_numpy(np.int64)
        vehicle_ts = vehicle["timestamp_ns"].to_numpy(np.int64)
        beat_ts = beats["timestamp_ns"].to_numpy(np.int64)

        covered = 0
        for bin_key, order in sorted(bin_order.items(), key=lambda kv: kv[1]):
            subset = gaze_valid[gaze_valid["bin_key"] == bin_key]
            metrics = attention_metrics(
                subset, class_names, gaze_interval_s, fixations=fixations,
                px_per_deg=context["px_per_deg"],
                image_shape=context["image_shape"])
            covered += int(metrics.get("usable", False))
            base = {"domain": domain, "recording_id": rec_id, "bin_key": bin_key,
                    "bin_order": order, "bin_size_m": bin_size_m}
            row = {**base, **flatten_unit(metrics)}

            in_bin = eye_bins == bin_key
            known = closed["left_known"] & closed["right_known"]
            both_closed = closed["left_closed"] & closed["right_closed"]
            bin_ts = eye_state["timestamp_ns"].to_numpy(np.int64)[in_bin]
            bin_duration_s = (float((bin_ts[-1] - bin_ts[0]) / NS_PER_S)
                              if bin_ts.size > 1 else 0.0)
            starts = blink_events[blink_events["is_identified_blink"].astype(bool)][
                "start_ns"].to_numpy(np.int64)
            blinks = (int(np.sum((starts >= bin_ts[0]) & (starts <= bin_ts[-1])))
                      if bin_ts.size else 0)
            row["blinks_per_minute"] = (float(blinks * 60.0 / bin_duration_s)
                                        if bin_duration_s > 0 else None)
            row["eye_closure_fraction"] = (
                float(both_closed[in_bin].sum() / known[in_bin].sum())
                if known[in_bin].sum() else None)

            row["pupil_diameter_mm"] = _finite_median(
                pupil.loc[pupil_bins == bin_key, "pupil_mean_m"], scale=1000.0)
            row["pupil_light_adjusted_residual_mm"] = _finite_median(
                pupil.loc[pupil_bins == bin_key,
                          "pupil_light_adjusted_residual_m"], scale=1000.0)

            if lane is not None:
                offsets = lane.loc[lane_bins == bin_key, "normalized_offset"]
                row["lane_position_abs_offset"] = _finite_median(offsets.abs())
                row["lane_position_offset"] = _finite_median(offsets)
            else:
                row["lane_position_abs_offset"] = None
                row["lane_position_offset"] = None

            if bin_ts.size:
                lo, hi = int(bin_ts[0]), int(bin_ts[-1]) + 1
                row["head_angular_speed_rad_s"] = _median_in_window(
                    head_ts, head["angular_speed_rad_s"].to_numpy(float), lo, hi)
                row["head_abs_yaw_rate_rad_s"] = _median_in_window(
                    head_ts, np.abs(head["yaw_rate_rad_s"].to_numpy(float)), lo, hi)
                row["head_vibration_rms_msec2"] = _median_in_window(
                    head_ts, head["vibration_rms_msec2"].to_numpy(float), lo, hi)
                row["speed_mps"] = _median_in_window(
                    vehicle_ts, vehicle["speed_mps"].to_numpy(float), lo, hi)
                row["heart_rate_bpm"] = _median_in_window(
                    beat_ts, beats["heart_rate_bpm"].to_numpy(float), lo, hi)
                row["ibi_ms"] = _median_in_window(
                    beat_ts, beats["ibi_ms"].to_numpy(float), lo, hi)
            bin_rows.append(row)
            bin_class_rows.extend(per_class_rows(metrics, class_names, **base))
        log.info("  %d/%d shared bins carry a semantic-gaze reading",
                 covered, len(bin_order))

        # ---------------- Phase 6: head-eye coordination ------------------- #
        gaze_rate = gaze_yaw_rate(gaze["timestamp_ns"].to_numpy(np.int64),
                                  gaze["yaw_rad"].to_numpy(float),
                                  gaze["combined_valid"].to_numpy(bool))
        head_at_gaze, _ = attach_head_rate(
            gaze["timestamp_ns"].to_numpy(np.int64), head_ts,
            head["yaw_rate_rad_s"].to_numpy(float))
        head_eye[domain] = head_eye_metrics(
            gaze["timestamp_ns"].to_numpy(np.int64), gaze_rate, head_at_gaze,
            head_ts, head["yaw_rate_rad_s"].to_numpy(float),
            head_threshold_rad_s=float(hcfg.get("head_yaw_rate_threshold_rad_s", .35)),
            head_min_duration_s=float(hcfg.get("head_min_duration_s", .15)),
            gaze_threshold_rad_s=float(hcfg.get("gaze_yaw_rate_threshold_rad_s", 1.)),
            gaze_min_duration_s=float(hcfg.get("gaze_min_duration_s", .066)),
            concurrency_tolerance_s=float(hcfg.get("concurrency_tolerance_s", .30)))
        head_eye[domain]["domain"] = domain
        head_eye[domain]["recording_id"] = rec_id
        log.info("  %.1f lateral head checks/min, %.1f gaze excursions/min, "
                 "%.0f%% head-assisted",
                 head_eye[domain]["lateral_head_checks"]["per_minute"] or 0.0,
                 head_eye[domain]["gaze_lateral_excursions"]["per_minute"] or 0.0,
                 100 * (head_eye[domain]["scan_split"]["head_assisted_fraction"] or 0))

        # ---------------- Phase 5: event windows --------------------------- #
        domain_events = _collect_events(events, solid, domain, rec_id)
        road_events = _road_events(domain_events)
        recording_start_ns = int(min(gaze["timestamp_ns"].min(), head_ts.min()))
        recording_end_ns = int(max(gaze["timestamp_ns"].max(), head_ts.max()))
        # The canonical windower, so these windows are the same ones the road-event
        # analysis used: one anchor per window chosen by the sign of its start
        # offset, clipped to the recording, and trimmed away from a neighbouring
        # manoeuvre that would otherwise contaminate a baseline.
        for event, road_event in zip(domain_events, road_events):
            windows = build_windows(road_event, windows_cfg, recording_start_ns,
                                    recording_end_ns, other_events=road_events)
            for window_name, window in windows.items():
                event_rows.append({
                    "event_id": event["event_id"], "domain": domain,
                    "recording_id": rec_id, "event_type": event["event_type"],
                    "event_start_ns": event["start_ns"],
                    "event_end_ns": event["end_ns"],
                    "event_duration_s": event["duration_s"],
                    "window": window_name,
                    "window_start_ns": window.start_ns,
                    "window_end_ns": window.end_ns,
                    "window_duration_s": window.duration_s,
                    "window_clipped": window.clipped,
                    "window_clip_reason": window.clip_reason,
                    **_window_metrics(
                        window.start_ns, window.end_ns, gaze_valid, class_names,
                        gaze_interval_s, fixations, context, eye_state,
                        blink_events, polarity, pupil, head, head_ts, vehicle,
                        vehicle_ts, beats, beat_ts, lane),
                })
        log.info("  %d events across %d windows", len(domain_events),
                 len(windows_cfg))

        # head motion in the seconds before each event
        lead_s = float(hcfg.get("head_motion_lead_s", 5.0))
        head_before = {
            event["event_id"]: head_motion_before(
                head_ts, head["angular_speed_rad_s"].to_numpy(float),
                event["start_ns"], lead_s=lead_s)
            for event in domain_events}
        for row in event_rows:
            if row["domain"] == domain and row["event_id"] in head_before:
                lead = head_before[row["event_id"]]
                row["head_motion_before_event_rad_s"] = \
                    lead["median_angular_speed_rad_s"]
                row["head_motion_lead_s"] = lead["lead_s"]

    # ---------------- write ------------------------------------------------ #
    bins = pd.DataFrame(bin_rows)
    _atomic_parquet(out_root / "semantic_attention_bins.parquet", bins)
    bins.to_csv(out_root / "semantic_attention_bins.csv", index=False)
    bin_classes = pd.DataFrame(bin_class_rows)
    _atomic_parquet(out_root / "semantic_attention_bin_classes.parquet", bin_classes)
    event_frame = _add_event_deltas(pd.DataFrame(event_rows))
    _atomic_parquet(out_root / "event_response.parquet", event_frame)
    event_frame.to_csv(out_root / "event_response.csv", index=False)
    atomic_write_json(out_root / "head_eye_summary.json", {
        "schema": "article1_head_eye_coordination_v1",
        "result_status": EXPLORATORY_MARKER, "domains": head_eye})
    atomic_write_json(out_root / "semantic_attention_recording.json", {
        "schema": "article1_semantic_attention_v1",
        "result_status": EXPLORATORY_MARKER,
        "primary_run": {"frequency": "5 Hz", "domains": recording_metrics},
        "robustness_check": {"frequency": "native", "domains": native_metrics},
        "unit_of_analysis": "whole recording",
        "frames_are_independent_samples": False})

    shared_with_gaze = bins.dropna(subset=["road_relevant_mass_percent"]).groupby(
        "bin_key").domain.nunique()
    both_sides = int((shared_with_gaze >= 2).sum())
    summary = {
        "schema": "article1_final_behavior_statistics_v1",
        "result_status": EXPLORATORY_MARKER,
        "analysis_bin_size_m": bin_size_m,
        "paired_bins": len(bin_order),
        "paired_bins_with_gaze_on_both_sides": both_sides,
        "semantic_gaze_source": str(run_root),
        "semantic_gaze_robustness_source": str(native_root),
        "semantic_gaze_covers_whole_recording": True,
        "bin_rows": int(len(bins)),
        "bin_class_rows": int(len(bin_classes)),
        "event_rows": int(len(event_frame)),
        "event_types": sorted(set(event_frame["event_type"])) if len(event_frame) else [],
        "windows": sorted(windows_cfg),
        "unit_of_analysis": ["route bin", "event", "time block"],
        "frames_are_independent_samples": False,
        "primary_gaze_metric": "foveal_mass_percent",
    }
    atomic_write_json(out_root / "final_behavior_summary.json", summary)
    print(json.dumps(summary, indent=2))
    return 0


def _finite_median(series, scale: float = 1.0) -> Optional[float]:
    values = np.asarray(series, float)
    values = values[np.isfinite(values)]
    return float(np.median(values) * scale) if values.size else None


def _collect_events(events: pd.DataFrame, solid: pd.DataFrame, domain: str,
                    recording_id: str) -> List[Dict[str, Any]]:
    """Map events plus solid-line candidates, in one shape."""
    out: List[Dict[str, Any]] = []
    subset = events[(events["domain"] == domain) & events["kind"].isin(EVENT_KINDS)]
    for row in subset.itertuples(index=False):
        out.append({"event_id": str(row.event_id), "kind": str(row.kind),
                    "event_type": EVENT_KINDS[str(row.kind)],
                    "domain": domain, "recording_id": recording_id,
                    "start_ns": int(row.start_ns), "end_ns": int(row.end_ns),
                    "duration_s": float(row.duration_s)})
    candidates = solid[solid["domain"] == domain]
    for row in candidates.itertuples(index=False):
        out.append({"event_id": str(row.candidate_id),
                    # Not a map-derived kind, so it can never trim another
                    # event's window: it is a state, not a manoeuvre.
                    "kind": "solid_line_candidate",
                    "event_type": "solid_line_candidate",
                    "domain": domain, "recording_id": recording_id,
                    "start_ns": int(row.start_ns), "end_ns": int(row.end_ns),
                    "duration_s": float(row.duration_s)})
    return out


def _road_events(events: Sequence[Dict[str, Any]]) -> List[RoadEvent]:
    """The collected events as `RoadEvent`s, so the canonical windower can run."""
    return [RoadEvent(event_id=e["event_id"], kind=e["kind"], domain=e["domain"],
                      recording_id=e["recording_id"], start_ns=e["start_ns"],
                      end_ns=e["end_ns"], peak_ns=e["start_ns"],
                      route_progress_m=float("nan"), osm_way_id=-1,
                      road_class=None)
            for e in events]


def _window_metrics(start_ns: int, end_ns: int, gaze_valid, class_names,
                    gaze_interval_s, fixations, context, eye_state, blink_events,
                    polarity, pupil, head, head_ts, vehicle, vehicle_ts, beats,
                    beat_ts, lane) -> Dict[str, Any]:
    """Every Phase-5 metric inside one window, each on its own real samples."""
    gaze_ts = gaze_valid["timestamp_ns"].to_numpy(np.int64)
    selected = gaze_valid[(gaze_ts >= start_ns) & (gaze_ts < end_ns)]
    attention = attention_metrics(
        selected, class_names, gaze_interval_s, fixations=fixations,
        px_per_deg=context["px_per_deg"], image_shape=context["image_shape"])
    flat = flatten_unit(attention)

    identified = blink_events[blink_events["is_identified_blink"].astype(bool)]
    blink = blink_rate_in_window(identified["start_ns"].to_numpy(np.int64),
                                 identified["duration_s"].to_numpy(float),
                                 start_ns, end_ns)

    pupil_window = pupil_in_window(pupil, start_ns, end_ns)
    lane_offsets = (lane.loc[(lane["timestamp_ns"] >= start_ns) &
                             (lane["timestamp_ns"] < end_ns), "normalized_offset"]
                    if lane is not None else None)

    return {
        "gaze_samples": flat["samples"],
        "road_relevant_mass_percent": flat["road_relevant_mass_percent"],
        "interior_cockpit_mass_percent": flat["interior_cockpit_mass_percent"],
        "vulnerable_road_user_mass_percent":
            flat["vulnerable_road_user_mass_percent"],
        "sign_and_signal_mass_percent": flat["sign_and_signal_mass_percent"],
        "gaze_entropy": flat["foveal_entropy_mean"],
        "semantic_transition_rate_per_s": flat["semantic_transition_rate_per_s"],
        "fixation_time_percent": flat["fixation_time_percent"],
        "blinks": blink["blinks"],
        "blink_rate_per_minute": blink["blinks_per_minute"],
        "mean_blink_duration_s": blink["mean_blink_duration_s"],
        "eye_closure_fraction": closure_fraction_in_window(
            eye_state, polarity, start_ns, end_ns),
        "pupil_diameter_mm": pupil_window["median_diameter_mm"],
        "pupil_light_adjusted_residual_mm":
            pupil_window["median_light_adjusted_residual_mm"],
        "head_angular_speed_rad_s": _median_in_window(
            head_ts, head["angular_speed_rad_s"].to_numpy(float), start_ns, end_ns),
        "head_abs_yaw_rate_rad_s": _median_in_window(
            head_ts, np.abs(head["yaw_rate_rad_s"].to_numpy(float)),
            start_ns, end_ns),
        "head_yaw_rate_rad_s": _median_in_window(
            head_ts, head["yaw_rate_rad_s"].to_numpy(float), start_ns, end_ns),
        "speed_mps": _median_in_window(
            vehicle_ts, vehicle["speed_mps"].to_numpy(float), start_ns, end_ns),
        "heart_rate_bpm": _median_in_window(
            beat_ts, beats["heart_rate_bpm"].to_numpy(float), start_ns, end_ns),
        "ibi_ms": _median_in_window(
            beat_ts, beats["ibi_ms"].to_numpy(float), start_ns, end_ns),
        "visual_lane_position_proxy_abs_offset": (
            _finite_median(lane_offsets.abs()) if lane_offsets is not None else None),
    }


#: Metrics for which a change from the physiological baseline window is reported.
DELTA_METRICS = (
    "heart_rate_bpm", "ibi_ms", "blink_rate_per_minute", "mean_blink_duration_s",
    "eye_closure_fraction", "pupil_light_adjusted_residual_mm",
    "pupil_diameter_mm", "road_relevant_mass_percent", "gaze_entropy",
    "head_angular_speed_rad_s", "head_yaw_rate_rad_s", "speed_mps",
    "visual_lane_position_proxy_abs_offset",
)


def _add_event_deltas(frame: pd.DataFrame) -> pd.DataFrame:
    """Change of each metric from the event's own local baseline window."""
    if frame.empty:
        return frame
    baseline = frame[frame["window"] == BASELINE_WINDOW].set_index("event_id")
    for metric in DELTA_METRICS:
        if metric not in frame.columns:
            continue
        reference = frame["event_id"].map(baseline[metric]) \
            if metric in baseline.columns else np.nan
        frame[f"delta_{metric}"] = frame[metric] - reference
    frame["delta_reference_window"] = BASELINE_WINDOW
    frame["unit_of_analysis"] = "event"
    frame["frames_are_independent_samples"] = False
    return frame


if __name__ == "__main__":
    raise SystemExit(main())
