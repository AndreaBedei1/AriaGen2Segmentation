#!/usr/bin/env python3
"""Phases 6 and 9: cartographic road events and their multimodal responses.

Events come from the map, so the response of a signal to an event is not
circular. Each signal family is windowed on its own real samples at its own rate.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd

from aria_drive_seg.behavior import EXPLORATORY_MARKER
from aria_drive_seg.behavior.events import (detect_events, event_response,
                                            event_summary)
from aria_drive_seg.config import Config
from aria_drive_seg.io_utils import atomic_write_json
from aria_drive_seg.logging_utils import get_logger, setup_logging

log = get_logger("behavior.events")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/article1/behavior_analysis.yaml")
    ap.add_argument("--output", default="output/article1/behavior_analysis")
    ap.add_argument("--reports", default="reports/article1_behavior_analysis")
    ap.add_argument("--log-level", default="INFO")
    args = ap.parse_args()
    setup_logging(args.log_level)

    cfg = Config.load(args.config)
    ecfg = cfg.get("events") or {}
    out_root = Path(args.output)
    reports = Path(args.reports) / "events"
    reports.mkdir(parents=True, exist_ok=True)

    # Windows: the cartographic set plus the physiological set, all in seconds.
    windows = {**{k: tuple(v) for k, v in (ecfg.get("windows_s") or {}).items()},
               **{f"phys_{k}": tuple(v)
                  for k, v in (ecfg.get("physiology_windows_s") or {}).items()}}
    windows["event"] = (0.0, 0.0)      # the event's own extent, filled below

    all_events, all_rows, summaries = [], [], {}
    for domain, spec in (cfg.get("recordings") or {}).items():
        rec_id = spec["recording_id"]
        dest = out_root / rec_id
        log.info("== %s", domain)

        matched = pd.read_parquet(dest / "map_matched.parquet")
        veh = pd.read_parquet(dest / "vehicle_dynamics.parquet")
        ok = matched[matched["matched"].astype(bool)].sort_values("timestamp_ns")
        speed = veh.set_index("timestamp_ns").reindex(
            ok["timestamp_ns"].values)["speed_mps"].values

        events = detect_events(
            matched, domain, rec_id, speed_mps=speed,
            approach_radius_m=ecfg.get("approach_radius_m"),
            curvature_threshold=float(
                ecfg.get("curve_curvature_threshold_1_per_m", 0.02)),
            stopped_speed_mps=float(cfg.get("dynamics.stopped_speed_mps", 0.5)))
        s = event_summary(events)
        log.info("  %d events: %s", s["total"],
                 {k: v["count"] for k, v in s["kinds"].items()})

        # --- signal families, each on its own clock -------------------------
        sources: Dict[str, Any] = {
            "vehicle": (veh["timestamp_ns"].values, {
                "speed_mps": veh["speed_mps"].values,
                "acceleration_mps2": veh["acceleration_mps2"].values,
                "lateral_acceleration_mps2": veh["lateral_acceleration_mps2"].values,
                "jerk_mps3": veh["jerk_mps3"].values,
            }),
        }
        head = pd.read_parquet(dest / "head_dynamics_per_frame.parquet")
        sources["head"] = (head["frame_timestamp_ns"].values, {
            "angular_speed_rad_s": head["angular_speed_rad_s"].values,
            "abs_yaw_rate_rad_s": np.abs(head["yaw_rate_rad_s"].values),
            "vibration_rms_msec2": head["vibration_rms_msec2"].values,
        })
        beats = pd.read_parquet(dest / "ppg_beats.parquet")
        gb = beats[beats["valid"].astype(bool)]
        sources["ppg"] = (gb["timestamp_ns"].values, {
            "heart_rate_bpm": gb["heart_rate_bpm"].values,
            "ibi_ms": gb["ibi_ms"].values,
            "pulse_amplitude": gb["amplitude"].values,
            "sqi": gb["sqi_at_beat"].values,
        })
        als_path = dest / "sensors" / "als.parquet"
        if als_path.exists():
            als = pd.read_parquet(als_path)
            sources["light"] = (als["timestamp_ns"].values,
                                {"lux": als["lux"].values, "cct": als["cct"].values})
        sg_path = dest / "semantic_gaze.parquet"
        if sg_path.exists():
            sg = pd.read_parquet(sg_path)
            v = sg[sg["semantic_valid"].astype(bool)]
            sources["gaze"] = (v["timestamp_ns"].values, {
                "foveal_entropy": v["foveal_entropy"].values,
                "top1_probability": v["top1_probability"].values,
                "confidence": v["confidence"].values,
                "is_fixation": v["is_fixation"].values.astype(float),
                "p_road_surface": v.get("p_road_surface", pd.Series(np.nan, index=v.index)).values,
                "p_vehicle": v.get("p_vehicle", pd.Series(np.nan, index=v.index)).values,
            })

        tl = json.loads((dest / "timeline_summary.json").read_text())
        start_ns = int(matched["timestamp_ns"].min())
        end_ns = int(matched["timestamp_ns"].max())

        rows = event_response(events, windows, sources, start_ns, end_ns,
                              baseline_window="phys_baseline")
        for r in rows:
            r["recording_duration_s"] = tl["duration_s"]
        all_rows.extend(rows)
        all_events.extend(e.to_dict() for e in events)
        summaries[domain] = {"recording_id": rec_id, **s}
        log.info("  %d event-response rows", len(rows))

    ev_df = pd.DataFrame(all_events)
    rs_df = pd.DataFrame(all_rows)
    ev_df.to_csv(reports / "route_events.csv", index=False)
    rs_df.to_parquet(out_root / "event_response_metrics.parquet", index=False)
    # The committed CSV keeps the columns a reader needs, not every window mean.
    keep = [c for c in rs_df.columns
            if not c.startswith(("straight_", "curve_")) or "delta_" in c]
    rs_df[keep].to_csv(reports / "event_response_metrics.csv", index=False)

    report = {
        "schema": "article1_road_events_v1",
        "result_status": EXPLORATORY_MARKER,
        "event_source": "OpenStreetMap geometry and tags, not the analysed signals",
        "windows_s": {k: list(v) for k, v in windows.items()},
        "window_policy": ("windows are clipped to the recording and trimmed away "
                          "from neighbouring events of a different kind; every "
                          "clip is recorded on the row"),
        "domains": summaries,
        "rows": int(len(rs_df)),
    }
    atomic_write_json(reports / "road_events_summary.json", report)
    print(json.dumps({d: {k: v["count"] for k, v in s["kinds"].items()}
                      for d, s in summaries.items()}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
