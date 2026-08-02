#!/usr/bin/env python3
"""Phase 8: PPG quality gate, beat detection and gated heart-rate variability.

Nothing here is medical. Every window passes a quality gate before it produces a
beat, and every variability statistic is refused, with a reason, on a window too
short or too noisy to support it.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd

from aria_drive_seg.behavior import EXPLORATORY_MARKER
from aria_drive_seg.behavior.ppg import (assess_quality, detect_beats, ppg_summary,
                                         rolling_hrv, tag_beats_with_quality)
from aria_drive_seg.config import Config
from aria_drive_seg.io_utils import atomic_write_json
from aria_drive_seg.logging_utils import get_logger, setup_logging

log = get_logger("behavior.ppg")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/article1/behavior_analysis.yaml")
    ap.add_argument("--output", default="output/article1/behavior_analysis")
    ap.add_argument("--reports", default="reports/article1_behavior_analysis")
    ap.add_argument("--quality-window-s", type=float, default=10.0)
    ap.add_argument("--log-level", default="INFO")
    args = ap.parse_args()
    setup_logging(args.log_level)

    cfg = Config.load(args.config)
    pcfg = cfg.get("ppg") or {}
    hcfg = pcfg.get("hrv") or {}
    out_root = Path(args.output)

    results: Dict[str, Any] = {}
    for domain, spec in (cfg.get("recordings") or {}).items():
        rec_id = spec["recording_id"]
        dest = out_root / rec_id
        log.info("== %s", domain)

        ppg = pd.read_parquet(dest / "sensors" / "ppg.parquet")
        imu = pd.read_parquet(dest / "sensors" / "imu_left.parquet")
        accel_mag = np.linalg.norm(
            imu[["accel_x_msec2", "accel_y_msec2", "accel_z_msec2"]].values, axis=1)

        beats, filtered, fs = detect_beats(
            ppg["timestamp_ns"].values, ppg["ppg_value"].values,
            bandpass_hz=tuple(pcfg.get("bandpass_hz", [0.5, 8.0])),
            min_hr_bpm=float(pcfg.get("min_heart_rate_bpm", 40.0)),
            max_hr_bpm=float(pcfg.get("max_heart_rate_bpm", 180.0)))
        log.info("  %.1f Hz PPG, %d candidate beats", fs, len(beats))

        quality = assess_quality(
            ppg["timestamp_ns"].values, ppg["ppg_value"].values, filtered, fs, beats,
            window_s=args.quality_window_s,
            bandpass_hz=tuple(pcfg.get("bandpass_hz", [0.5, 8.0])),
            min_sqi=float(pcfg.get("min_sqi", 0.5)),
            saturation_fraction_limit=float(
                pcfg.get("saturation_fraction_limit", 0.02)),
            imu_timestamp_ns=imu["timestamp_ns"].values,
            imu_accel_magnitude=accel_mag,
            motion_accel_std_threshold=float(
                pcfg.get("motion_artefact_accel_std_threshold_msec2", 2.0)))
        beats = tag_beats_with_quality(beats, quality)

        ts = ppg["timestamp_ns"].values
        duration_s = float((ts[-1] - ts[0]) / 1e9)
        windows = rolling_hrv(
            beats, int(ts[0]), int(ts[-1]),
            window_s=float(hcfg.get("sdnn_min_window_s", 60.0)), step_s=30.0,
            min_window_s=float(hcfg.get("min_window_s", 30.0)),
            sdnn_min_window_s=float(hcfg.get("sdnn_min_window_s", 60.0)),
            min_valid_beat_fraction=float(hcfg.get("min_valid_beat_fraction", 0.8)),
            pnn50_min_beats=int(hcfg.get("pnn50_min_beats", 50)))

        beats.to_frame().to_parquet(dest / "ppg_beats.parquet", index=False)
        quality.to_frame().to_parquet(dest / "ppg_quality.parquet", index=False)
        wdf = pd.DataFrame([{k: v for k, v in w.items() if k != "refusals"}
                            for w in windows])
        wdf["refusals"] = [json.dumps(w["refusals"]) for w in windows]
        wdf.to_parquet(dest / "ppg_windows.parquet", index=False)

        summary = ppg_summary(beats, quality, duration_s)
        summary["hrv_windows"] = {
            "count": len(windows),
            "with_heart_rate": int(sum(w["heart_rate_bpm"] is not None
                                       for w in windows)),
            "with_rmssd": int(sum(w["rmssd_ms"] is not None for w in windows)),
            "with_sdnn": int(sum(w["sdnn_ms"] is not None for w in windows)),
            "with_pnn50": int(sum(w["pnn50"] is not None for w in windows)),
            "window_s": float(hcfg.get("sdnn_min_window_s", 60.0)),
        }
        results[domain] = {"recording_id": rec_id, "sampling_rate_hz": fs, **summary}
        log.info("  usable windows %d/%d (%.1f%%), median SQI %s, "
                 "%d/%d HRV windows produced RMSSD",
                 summary["usable_windows"], summary["quality_windows"],
                 100 * summary["usable_fraction"],
                 (f"{summary['median_sqi']:.2f}" if summary["median_sqi"] else "n/a"),
                 summary["hrv_windows"]["with_rmssd"], len(windows))

    report = {
        "schema": "article1_ppg_v1",
        "result_status": EXPLORATORY_MARKER,
        "interpretation": "physiological_proxy_not_medical",
        "gate": {
            "order": ["window quality", "beat plausibility", "window length"],
            "motion_artefact_source": "head IMU accelerometer magnitude",
            **{k: pcfg.get(k) for k in ("bandpass_hz", "min_sqi",
                                        "min_heart_rate_bpm", "max_heart_rate_bpm",
                                        "saturation_fraction_limit")},
            "hrv": hcfg,
        },
        "domains": results,
    }
    reports = Path(args.reports)
    reports.mkdir(parents=True, exist_ok=True)
    atomic_write_json(reports / "ppg_summary.json", report)
    atomic_write_json(out_root / "ppg_summary.json", report)
    print(json.dumps({d: {
        "usable_fraction": round(r["usable_fraction"], 3),
        "median_sqi": r["median_sqi"],
        "beats_passing_gate": r["beats_passing_gate"],
        "hr_median_bpm": r["heart_rate_bpm"]["median"],
        "hrv_windows_with_sdnn": r["hrv_windows"]["with_sdnn"],
    } for d, r in results.items()}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
