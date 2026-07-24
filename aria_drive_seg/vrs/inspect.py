"""`inspect` command (§4): open VRS, enumerate streams, validate RGB + eyegaze,
check timestamp monotonicity/gaps, verify calibration, emit inspection_report.json
and correctly-oriented sample images."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

import numpy as np

from ..config import Config
from ..io_utils import atomic_write_json
from ..logging_utils import get_logger
from .provider import AriaProvider, RectifyParams

log = get_logger("inspect")


def _timestamp_health(ts_ns: np.ndarray) -> Dict[str, Any]:
    if ts_ns.size < 2:
        return {"count": int(ts_ns.size), "monotonic": True, "gaps": 0}
    dt = np.diff(ts_ns.astype(np.int64))
    med = float(np.median(dt))
    non_mono = int(np.sum(dt <= 0))
    # a "gap" = interval > 1.5x median (a likely dropped frame)
    gap_mask = dt > 1.5 * med if med > 0 else np.zeros_like(dt, dtype=bool)
    n_gaps = int(np.sum(gap_mask))
    est_dropped = int(np.sum(np.round(dt[gap_mask] / med) - 1)) if med > 0 else 0
    return {
        "count": int(ts_ns.size),
        "first_ns": int(ts_ns[0]),
        "last_ns": int(ts_ns[-1]),
        "duration_s": float((ts_ns[-1] - ts_ns[0]) / 1e9),
        "median_dt_ms": med / 1e6,
        "rate_hz": float(1e9 / med) if med > 0 else None,
        "monotonic": non_mono == 0,
        "non_monotonic_count": non_mono,
        "num_gaps": n_gaps,
        "estimated_dropped_frames": est_dropped,
    }


def run_inspect(vrs_path: str, out_dir: str, cfg: Config,
                num_samples: int = 6) -> Dict[str, Any]:
    out = Path(out_dir)
    (out / "samples").mkdir(parents=True, exist_ok=True)

    prov = AriaProvider(vrs_path, cfg.get("vrs.time_domain", "DEVICE_TIME"))
    report: Dict[str, Any] = {"vrs_path": str(vrs_path), "readable": True}

    # streams
    streams = prov.list_streams()
    report["streams"] = [s.__dict__ for s in streams]
    report["num_streams"] = len(streams)
    log.info("found %d streams", len(streams))

    rgb_label = cfg.get("vrs.rgb_label", "camera-rgb")
    eg_label = cfg.get("vrs.eyegaze_label", "eyegaze")
    report["rgb_label"] = rgb_label if prov.has_label(rgb_label) else None
    report["eyegaze_label"] = eg_label if prov.has_eyegaze(eg_label) else None

    if report["rgb_label"] is None:
        report["readable"] = False
        report["error"] = f"RGB stream {rgb_label!r} not found"
        atomic_write_json(out / "inspection_report.json", report)
        return report

    # RGB
    report["rgb_config"] = prov.rgb_config(rgb_label)
    rgb_ts = prov.rgb_timestamps_ns(rgb_label)
    report["rgb_timestamps"] = _timestamp_health(rgb_ts)
    log.info("RGB frames=%d rate=%.2fHz dur=%.1fs",
             report["rgb_timestamps"]["count"],
             report["rgb_timestamps"].get("rate_hz") or 0,
             report["rgb_timestamps"].get("duration_s") or 0)

    # Eyegaze
    if report["eyegaze_label"]:
        n_eg = prov.num_eyegaze(eg_label)
        eg_valid = _eyegaze_validity(prov, eg_label, n_eg)
        report["eyegaze"] = {"num_samples": n_eg, **eg_valid}
        log.info("eyegaze samples=%d combined_valid_frac=%.2f",
                 n_eg, eg_valid.get("combined_valid_frac", 0))

    # Calibration
    try:
        report["calibration"] = prov.calib_summary(rgb_label)
        report["calibration"]["available"] = True
        report["T_device_cpf"] = prov.T_device_cpf.tolist()
    except Exception as e:
        report["calibration"] = {"available": False, "error": str(e)}

    # Sample images (raw + rectified), evenly spread, correctly oriented
    rp = RectifyParams(
        out_width=int(cfg.get("rectify.out_width", 2016)),
        out_height=int(cfg.get("rectify.out_height", 1512)),
        focal=float(cfg.get("rectify.focal", 879.0)),
        rotate_ccw90=int(cfg.get("rectify.rotate_ccw90", 0)),
    )
    rect = prov.rectifier(rp, rgb_label) if cfg.get("rectify.enabled", True) else None
    n = report["rgb_timestamps"]["count"]
    idxs = np.linspace(0, n - 1, num_samples, dtype=int).tolist()
    from PIL import Image
    samples = []
    for i in idxs:
        raw, ts = prov.rgb_by_index(i, rgb_label)
        Image.fromarray(raw).save(out / "samples" / f"frame_{i:06d}_raw.jpg", quality=90)
        entry = {"index": i, "timestamp_ns": ts,
                 "raw": f"samples/frame_{i:06d}_raw.jpg"}
        if rect is not None:
            r = rect.rectify(raw)
            Image.fromarray(r).save(out / "samples" / f"frame_{i:06d}_rect.jpg", quality=90)
            entry["rectified"] = f"samples/frame_{i:06d}_rect.jpg"
        samples.append(entry)
    report["samples"] = samples

    atomic_write_json(out / "inspection_report.json", report)
    log.info("wrote %s", out / "inspection_report.json")
    return report


def _eyegaze_validity(prov: AriaProvider, label: str, n: int, stride: int = 1) -> Dict[str, Any]:
    comb = spat = 0
    checked = 0
    for i in range(0, n, stride):
        gz = prov.eyegaze_by_index(i, label)
        checked += 1
        if bool(getattr(gz, "combined_gaze_valid", False)):
            comb += 1
        if bool(getattr(gz, "spatial_gaze_point_valid", False)):
            spat += 1
    return {
        "checked": checked,
        "combined_valid_frac": comb / checked if checked else 0.0,
        "spatial_point_valid_frac": spat / checked if checked else 0.0,
    }
