#!/usr/bin/env python3
"""Phase 4: vehicle dynamics and head dynamics, computed and stored separately.

Vehicle motion comes from GPS (this device wrote no usable VIO/SLAM pose stream);
head motion comes from the glasses IMU with gravity removed. The two never meet
in the same table, and every row of each says which of the two it describes.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd

from aria_drive_seg.behavior import EXPLORATORY_MARKER
from aria_drive_seg.behavior.dynamics import (compute_head_dynamics,
                                              compute_vehicle_dynamics,
                                              head_dynamics_summary,
                                              vehicle_dynamics_summary)
from aria_drive_seg.config import Config
from aria_drive_seg.io_utils import atomic_write_json
from aria_drive_seg.logging_utils import get_logger, setup_logging

log = get_logger("behavior.dynamics")


def camera_axes(vrs_path: Path):
    """RGB optical and lateral axes expressed in device coordinates.

    They are the "forward" and "right" references for resolving head yaw, pitch
    and roll, so the rotation axes are physical rather than a device-axis naming
    convention. Roll in particular needs the lateral axis: it cannot be recovered
    from the optical axis and gravity alone.
    """
    from aria_drive_seg.vrs.provider import AriaProvider
    provider = AriaProvider(vrs_path)
    T = np.asarray(provider.rgb_calib().get_transform_device_camera().to_matrix())
    return T[:3, 2], T[:3, 0]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/article1/behavior_analysis.yaml")
    ap.add_argument("--output", default="output/article1/behavior_analysis")
    ap.add_argument("--reports", default="reports/article1_behavior_analysis")
    ap.add_argument("--imu", default="imu-left",
                    help="which head IMU to use for the primary head dynamics")
    ap.add_argument("--log-level", default="INFO")
    args = ap.parse_args()
    setup_logging(args.log_level)

    cfg = Config.load(args.config)
    out_root = Path(args.output)
    dyn_cfg = cfg.get("dynamics") or {}
    head_cfg = dyn_cfg.get("head") or {}

    results: Dict[str, Any] = {}
    for domain, spec in (cfg.get("recordings") or {}).items():
        rec_id = spec["recording_id"]
        dest = out_root / rec_id
        log.info("== %s (%s)", domain, rec_id)

        # ---------------- vehicle -------------------------------------------
        matched = pd.read_parquet(dest / "map_matched.parquet")
        ok = matched[matched["matched"].astype(bool)].sort_values("timestamp_ns")
        veh = compute_vehicle_dynamics(
            timestamp_ns=ok["timestamp_ns"].values,
            x_m=ok["matched_x_m"].values, y_m=ok["matched_y_m"].values,
            gps_speed_mps=ok["gps_speed_mps"].values,
            curvature_1_per_m=ok["curvature_1_per_m"].values,
            heading_deg=ok["road_heading_deg"].values,
            speed_priority=dyn_cfg.get("vehicle_speed_priority",
                                       ["vio_pose", "gps_speed_field",
                                        "gps_position_derivative"]),
            vio_speed_mps=None,      # no usable pose stream in these recordings
            smoothing_window_s=float(dyn_cfg.get("smoothing_window_s", 1.0)),
            stopped_speed_mps=float(dyn_cfg.get("stopped_speed_mps", 0.5)),
            max_plausible_speed_mps=float(cfg.get("gps.max_plausible_speed_mps", 60.0)),
        )
        veh.to_frame().to_parquet(dest / "vehicle_dynamics.parquet", index=False)
        veh_summary = vehicle_dynamics_summary(
            veh,
            hard_braking_mps2=float(dyn_cfg.get("hard_braking_threshold_mps2", -2.5)),
            strong_acceleration_mps2=float(
                dyn_cfg.get("strong_acceleration_threshold_mps2", 2.0)))
        log.info("  vehicle: source=%s, median speed %.1f m/s, %d hard-braking "
                 "episodes", veh_summary["speed_source"],
                 veh_summary["speed_mps"]["median"] or float("nan"),
                 veh_summary["hard_braking_episodes"])

        # ---------------- head ----------------------------------------------
        imu_path = dest / "sensors" / f"{args.imu.replace('-', '_')}.parquet"
        if not imu_path.exists():
            raise SystemExit(f"missing {imu_path}")
        imu = pd.read_parquet(imu_path)
        fwd, right = camera_axes(cfg.resolve(spec["vrs"]))
        head = compute_head_dynamics(
            timestamp_ns=imu["timestamp_ns"].values,
            accel_msec2=imu[["accel_x_msec2", "accel_y_msec2",
                             "accel_z_msec2"]].values,
            gyro_rad_s=imu[["gyro_x_radsec", "gyro_y_radsec",
                            "gyro_z_radsec"]].values,
            camera_forward_device=fwd, camera_right_device=right,
            gravity_cutoff_hz=float(head_cfg.get("gravity_cutoff_hz", 0.3)),
            vibration_band_hz=tuple(head_cfg.get("vibration_band_hz", [5.0, 20.0])),
            control_band_hz=tuple(head_cfg.get("control_band_hz", [0.2, 2.0])),
        )
        head_summary = head_dynamics_summary(
            head,
            lateral_check_threshold_rad_s=float(
                head_cfg.get("lateral_check_yaw_rate_threshold_radsec", 0.35)),
            lateral_check_min_duration_s=float(
                head_cfg.get("lateral_check_min_duration_s", 0.15)))
        log.info("  head: %.0f Hz, gravity %.2f m/s^2, %.1f lateral checks/min, "
                 "vibration RMS median %.3f", head.meta["sampling_rate_hz"],
                 head.meta["gravity_magnitude_median"],
                 head_summary["lateral_checks_per_minute"] or float("nan"),
                 head_summary["vibration_rms_msec2"]["median"] or float("nan"))

        # The full-rate head table is large; store it downsampled to the RGB grid
        # by NEAREST REAL SAMPLE, and keep the native-rate table too.
        head.to_frame().to_parquet(dest / "head_dynamics_native.parquet", index=False)
        timeline = pd.read_parquet(dest / "multimodal_timeline.parquet")
        idx = timeline[f"{args.imu.replace('-', '_')}_index"].values
        hf = head.to_frame()
        per_frame = hf.iloc[np.clip(idx, 0, len(hf) - 1)].reset_index(drop=True)
        per_frame.insert(0, "frame_index", timeline["frame_index"].values)
        per_frame.insert(1, "frame_timestamp_ns", timeline["timestamp_ns"].values)
        per_frame["imu_dt_ms"] = timeline[f"{args.imu.replace('-', '_')}_dt_ms"].values
        per_frame.to_parquet(dest / "head_dynamics_per_frame.parquet", index=False)

        results[domain] = {"recording_id": rec_id, "vehicle": veh_summary,
                           "head": head_summary, "head_imu": args.imu}

    report = {
        "schema": "article1_dynamics_v1",
        "result_status": EXPLORATORY_MARKER,
        "separation": ("vehicle dynamics and head dynamics are computed by "
                       "different functions from different sensors and stored in "
                       "different tables; neither is derived from the other"),
        "vehicle_source_priority": dyn_cfg.get("vehicle_speed_priority"),
        "domains": results,
    }
    reports = Path(args.reports)
    reports.mkdir(parents=True, exist_ok=True)
    atomic_write_json(reports / "dynamics_summary.json", report)
    atomic_write_json(out_root / "dynamics_summary.json", report)
    print(json.dumps({d: {
        "speed_source": r["vehicle"]["speed_source"],
        "median_speed_mps": r["vehicle"]["speed_mps"]["median"],
        "hard_braking_per_minute": r["vehicle"]["hard_braking_per_minute"],
        "lateral_checks_per_minute": r["head"]["lateral_checks_per_minute"],
    } for d, r in results.items()}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
