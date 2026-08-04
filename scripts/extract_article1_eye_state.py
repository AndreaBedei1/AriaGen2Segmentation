#!/usr/bin/env python3
"""Phase 2-3: per-sample blink and pupil state from the on-device gaze stream.

`projectaria_tools` binds the eye-gaze record without its pupil-diameter fields,
so the record payload is decoded directly against the `DataLayout` descriptor the
recording itself carries. Nothing is written until that decode has been verified
field-by-field against the official binding on every sample.

Nothing is interpolated. A blink event is a run of consecutive real samples, and
a sampling gap ends the run rather than being bridged.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd

from aria_drive_seg.behavior import EXPLORATORY_MARKER
from aria_drive_seg.behavior.blink import (blink_metrics, build_blink_events,
                                           closed_series,
                                           determine_blink_polarity)
from aria_drive_seg.behavior.eye_state import (extraction_summary, read_eye_state,
                                               verify_against_provider)
from aria_drive_seg.behavior.pupil import (add_light_residual, build_pupil_table,
                                           pupil_metrics)
from aria_drive_seg.config import Config
from aria_drive_seg.io_utils import atomic_write_json
from aria_drive_seg.logging_utils import get_logger, setup_logging

log = get_logger("behavior.eye_state")


def _atomic_parquet(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp.parquet")
    frame.to_parquet(tmp, index=False)
    tmp.replace(path)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/article1/behavior_analysis.yaml")
    ap.add_argument("--behavior", default="output/article1/behavior_analysis",
                    help="where the ALS sensor tables already live")
    ap.add_argument("--output", default="output/article1/final_behavior_statistics")
    ap.add_argument("--skip-verification", action="store_true",
                    help="diagnostic only; the summary records that it was skipped")
    ap.add_argument("--log-level", default="INFO")
    args = ap.parse_args()
    setup_logging(args.log_level)

    cfg = Config.load(args.config)
    ecfg = cfg.get("eye_state") or {}
    bcfg = ecfg.get("blink") or {}
    pcfg = ecfg.get("pupil") or {}
    out_root = Path(args.output)
    behavior_root = Path(args.behavior)

    results: Dict[str, Any] = {}
    for domain, spec in (cfg.get("recordings") or {}).items():
        rec_id = str(spec["recording_id"])
        vrs = cfg.resolve(spec["vrs"])
        dest = out_root / rec_id
        log.info("== %s (%s)", domain, rec_id)

        samples = read_eye_state(vrs, recording_id=rec_id, domain=domain)
        summary = extraction_summary(samples, vrs)
        log.info("  %d samples at %.2f Hz", len(samples),
                 summary["measured_rate_hz"] or 0.0)

        if args.skip_verification:
            verification = {"verified": False, "skipped": True,
                            "reason": "--skip-verification was passed"}
        else:
            verification = verify_against_provider(
                samples, vrs, label=str(ecfg.get("stream_label", "eyegaze")))
            if not verification["verified"]:
                raise SystemExit(
                    f"{domain}: the raw eye-state decode does not reproduce the "
                    f"projectaria_tools binding: {json.dumps(verification)}")
            log.info("  decode verified against projectaria_tools on %d samples",
                     verification["provider_samples"])

        # ------------------------------------------------------------------ #
        # Blink
        # ------------------------------------------------------------------ #
        polarity = determine_blink_polarity(
            samples["left_blink"].to_numpy(bool),
            samples["left_blink_valid"].to_numpy(bool),
            samples["timestamp_ns"].to_numpy(np.int64),
            max_closed_fraction=float(bcfg.get("max_closed_fraction", 0.35)))
        if not polarity["decided"]:
            raise SystemExit(f"{domain}: blink polarity undetermined: "
                             f"{polarity.get('reason')}")
        log.info("  blink polarity: closed when the raw flag is %s",
                 polarity["closed_value"])

        events = build_blink_events(
            samples, polarity,
            max_gap_s=float(bcfg.get("max_gap_s", 0.075)),
            max_blink_duration_s=float(bcfg.get("max_blink_duration_s", 1.0)))
        blink = blink_metrics(samples, events, polarity,
                              window_s=float(bcfg.get("rate_window_s", 60.0)),
                              step_s=float(bcfg.get("rate_step_s", 10.0)))
        log.info("  %d blink events, %.1f /min, closure %.1f%%",
                 blink["identified_blinks"],
                 blink["blinks_per_minute_whole_recording"] or 0.0,
                 100 * (blink["eye_closure"]["both_eyes"]["eye_closure_fraction"] or 0))

        # ------------------------------------------------------------------ #
        # Pupil
        # ------------------------------------------------------------------ #
        als_path = behavior_root / rec_id / "sensors" / "als.parquet"
        if als_path.exists():
            als = pd.read_parquet(als_path).sort_values("timestamp_ns")
            als_ts = als["timestamp_ns"].to_numpy(np.int64)
            als_lux = als["lux"].to_numpy(float)
        else:
            log.warning("  no ALS table at %s; the light correction is skipped",
                        als_path)
            als_ts = als_lux = None

        series = closed_series(samples, polarity)
        pupil_table = build_pupil_table(
            samples, als_ts, als_lux,
            closed=series["left_closed"] | series["right_closed"],
            max_light_dt_s=float(pcfg.get("max_light_dt_s", 0.25)),
            baseline_window_s=float(pcfg.get("baseline_window_s", 60.0)))
        model = add_light_residual(pupil_table)
        pupil = pupil_metrics(pupil_table, model)
        log.info("  pupil median %.3f mm (left) / %.3f mm (right); light model "
                 "R2=%s", pupil["per_eye"]["left"]["median_mm"] or float("nan"),
                 pupil["per_eye"]["right"]["median_mm"] or float("nan"),
                 model.get("r_squared"))

        # ------------------------------------------------------------------ #
        # Write
        # ------------------------------------------------------------------ #
        _atomic_parquet(dest / "eye_state_samples.parquet", samples)
        _atomic_parquet(dest / "blink_events.parquet",
                        pd.DataFrame([e.to_dict() for e in events]))
        _atomic_parquet(dest / "blink_rate_windows.parquet",
                        pd.DataFrame(blink["windows"]))
        _atomic_parquet(dest / "pupil_samples.parquet", pupil_table)

        record = {
            "schema": "article1_eye_state_summary_v1",
            "result_status": EXPLORATORY_MARKER,
            "domain": domain, "recording_id": rec_id,
            "extraction": summary,
            "decode_verification": verification,
            "blink": {k: v for k, v in blink.items() if k != "windows"},
            "blink_rate_windows": len(blink["windows"]),
            "pupil": pupil,
        }
        atomic_write_json(dest / "eye_state_summary.json", record)
        results[domain] = record

    combined = {
        "schema": "article1_eye_state_v1",
        "result_status": EXPLORATORY_MARKER,
        "domains": {d: {
            "samples": r["extraction"]["samples"],
            "measured_rate_hz": r["extraction"]["measured_rate_hz"],
            "decode_verified": r["decode_verification"]["verified"],
            "blinks_per_minute": r["blink"]["blinks_per_minute_whole_recording"],
            "median_blink_duration_s": r["blink"]["duration"]["median_s"],
            "eye_closure_fraction":
                r["blink"]["eye_closure"]["both_eyes"]["eye_closure_fraction"],
            "pupil_median_mm_left": r["pupil"]["per_eye"]["left"]["median_mm"],
            "pupil_median_mm_right": r["pupil"]["per_eye"]["right"]["median_mm"],
            "light_model_r_squared": (r["pupil"]["light_model"] or {}).get("r_squared"),
        } for d, r in results.items()},
        "blink_interpolated": False,
        "eye_closure_is_not_a_drowsiness_measure": True,
        "pupil_clinical_interpretation": "not attempted and not supportable",
    }
    atomic_write_json(out_root / "eye_state_summary.json", combined)
    print(json.dumps(combined["domains"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
