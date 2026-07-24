"""`align-gaze` (§9): temporally align on-device eyegaze to each RGB frame,
project into original + rectified geometry, and persist raw + aligned parquet."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from ..config import Config
from ..io_utils import atomic_write_json
from ..logging_utils import get_logger
from ..vrs.provider import AriaProvider, RectifyParams
from .project import GazeProjector

log = get_logger("gaze")


def _lerp(a: float, b: float, w: float) -> float:
    return float(a * (1 - w) + b * w)


def run_align_gaze(vrs_path: str, input_dir: str, cfg: Config) -> Dict[str, Any]:
    import pandas as pd

    out = Path(input_dir)
    gaze_dir = out / "gaze"
    gaze_dir.mkdir(parents=True, exist_ok=True)

    prov = AriaProvider(vrs_path, cfg.get("vrs.time_domain", "DEVICE_TIME"))
    rgb_label = cfg.get("vrs.rgb_label", "camera-rgb")
    eg_label = cfg.get("vrs.eyegaze_label", "eyegaze")
    if not prov.has_eyegaze(eg_label):
        log.warning("no eyegaze stream; skipping gaze alignment")
        atomic_write_json(gaze_dir / "summary.json", {"eyegaze": False})
        return {"eyegaze": False}

    # geometry
    rp = RectifyParams(int(cfg.get("rectify.out_width", 2016)),
                       int(cfg.get("rectify.out_height", 1512)),
                       float(cfg.get("rectify.focal", 879.0)),
                       int(cfg.get("rectify.rotate_ccw90", 0)))
    fisheye = prov.rgb_calib(rgb_label)
    pinhole = prov.make_pinhole(rp, rgb_label)
    projector = GazeProjector(prov.device_calib, fisheye, pinhole, rgb_label,
                              fallback_depth_m=float(cfg.get("gaze.fallback_depth_m", 8.0)))

    max_dt_ns = float(cfg.get("gaze.max_dt_ms", 20.0)) * 1e6
    do_interp = bool(cfg.get("gaze.interp", True))

    # gaze sample timeline
    eg_ts = prov.timestamps_ns(eg_label)
    n_eg = eg_ts.size

    # RGB frames to align: prefer the extracted frame index (matches segmentation)
    frames_pq = out / "frames" / "frames.parquet"
    if frames_pq.exists():
        fdf = pd.read_parquet(frames_pq)
        frame_items = list(zip(fdf["frame_index"].astype(int), fdf["capture_timestamp_ns"].astype(np.int64)))
    else:
        rgb_ts = prov.rgb_timestamps_ns(rgb_label)
        frame_items = [(i, int(rgb_ts[i])) for i in range(rgb_ts.size)]

    # ---- raw gaze dump (all samples) ----
    raw_rows = []
    eg_cache: Dict[int, Any] = {}

    def get_eg(i: int):
        if i not in eg_cache:
            eg_cache[i] = prov.eyegaze_by_index(i, eg_label)
        return eg_cache[i]

    for i in range(n_eg):
        gz = get_eg(i)
        origin = np.asarray(gz.combined_gaze_origin_in_cpf, dtype=float).tolist()
        raw_rows.append({
            "gaze_index": i, "gaze_timestamp_ns": int(eg_ts[i]),
            "yaw": float(gz.yaw), "pitch": float(gz.pitch), "depth": float(gz.depth),
            "combined_valid": bool(gz.combined_gaze_valid),
            "spatial_point_valid": bool(gz.spatial_gaze_point_valid),
            "origin_x": origin[0], "origin_y": origin[1], "origin_z": origin[2],
        })
    _to_parquet(gaze_dir / "raw_gaze.parquet", raw_rows)

    # ---- aligned per-frame ----
    aligned: List[Dict[str, Any]] = []
    n_valid = 0
    for frame_index, rgb_ts_ns in frame_items:
        rec = _align_one(frame_index, rgb_ts_ns, eg_ts, get_eg, projector,
                         max_dt_ns, do_interp)
        aligned.append(rec)
        if rec["valid"]:
            n_valid += 1
    _to_parquet(gaze_dir / "aligned_gaze.parquet", aligned)

    frac = n_valid / len(frame_items) if frame_items else 0.0
    summary = {"eyegaze": True, "num_frames": len(frame_items),
               "num_valid": n_valid, "valid_fraction": frac,
               "num_raw_samples": n_eg, "max_dt_ms": cfg.get("gaze.max_dt_ms"),
               "fallback_depth_m": cfg.get("gaze.fallback_depth_m")}
    atomic_write_json(gaze_dir / "summary.json", summary)
    log.info("gaze aligned: %d/%d frames valid (%.1f%%)", n_valid, len(frame_items), 100 * frac)
    return summary


def _align_one(frame_index, rgb_ts_ns, eg_ts, get_eg, projector,
               max_dt_ns, do_interp) -> Dict[str, Any]:
    rec: Dict[str, Any] = {
        "frame_index": int(frame_index), "capture_timestamp_ns": int(rgb_ts_ns),
        "valid": False, "validity_reason": "", "interpolated": False,
        "gaze_timestamp_ns": None, "dt_ms": None,
        "yaw": None, "pitch": None, "depth": None, "depth_source": None,
        "orig_u": None, "orig_v": None, "rect_u": None, "rect_v": None,
        "in_image_orig": None, "in_image_rect": None,
    }
    hi = int(np.searchsorted(eg_ts, rgb_ts_ns, side="left"))
    lo = hi - 1
    cands = [c for c in (lo, hi) if 0 <= c < eg_ts.size]
    if not cands:
        rec["validity_reason"] = "no_gaze_samples"
        return rec
    nearest = min(cands, key=lambda c: abs(int(eg_ts[c]) - rgb_ts_ns))
    dt_ns = int(eg_ts[nearest]) - rgb_ts_ns
    rec["gaze_timestamp_ns"] = int(eg_ts[nearest])
    rec["dt_ms"] = dt_ns / 1e6

    gz_near = get_eg(nearest)
    if abs(dt_ns) > max_dt_ns:
        rec["validity_reason"] = "far_dt"
    if not bool(gz_near.combined_gaze_valid):
        rec["validity_reason"] = (rec["validity_reason"] + ";invalid_flag").strip(";")

    # decide interpolation
    use_interp = False
    if do_interp and 0 <= lo < eg_ts.size and 0 <= hi < eg_ts.size and lo != hi:
        gl, gh = get_eg(lo), get_eg(hi)
        if (bool(gl.combined_gaze_valid) and bool(gh.combined_gaze_valid)
                and abs(int(eg_ts[lo]) - rgb_ts_ns) <= max_dt_ns
                and abs(int(eg_ts[hi]) - rgb_ts_ns) <= max_dt_ns):
            use_interp = True

    if use_interp:
        gl, gh = get_eg(lo), get_eg(hi)
        span = int(eg_ts[hi]) - int(eg_ts[lo])
        w = (rgb_ts_ns - int(eg_ts[lo])) / span if span else 0.0
        yaw = _lerp(gl.yaw, gh.yaw, w); pitch = _lerp(gl.pitch, gh.pitch, w)
        if gl.depth > 0 and gh.depth > 0:
            depth, dsrc = _lerp(gl.depth, gh.depth, w), "device"
        else:
            depth, dsrc = projector.fallback_depth_m, "fallback"
        rec.update(yaw=yaw, pitch=pitch, depth=depth, depth_source=dsrc, interpolated=True)
        po = projector.reproject_yawpitch(yaw, pitch, depth, "original")
        pr = projector.reproject_yawpitch(yaw, pitch, depth, "rectified")
    else:
        depth, dsrc = projector.depth_of(gz_near)
        rec.update(yaw=float(gz_near.yaw), pitch=float(gz_near.pitch),
                   depth=depth, depth_source=dsrc)
        po = projector.reproject_official(gz_near, depth, "original")
        pr = projector.reproject_official(gz_near, depth, "rectified")

    if po is not None:
        rec.update(orig_u=po.u, orig_v=po.v, in_image_orig=po.in_image)
    if pr is not None:
        rec.update(rect_u=pr.u, rect_v=pr.v, in_image_rect=pr.in_image)
    if pr is None and po is None:
        rec["validity_reason"] = (rec["validity_reason"] + ";projection_failed").strip(";")

    rec["valid"] = (rec["validity_reason"] == "" and pr is not None and pr.in_image)
    if rec["valid"]:
        rec["validity_reason"] = "ok"
    elif rec["validity_reason"] == "":
        rec["validity_reason"] = "out_of_image"
    return rec


def _to_parquet(path: Path, rows: List[Dict[str, Any]]) -> None:
    import pandas as pd
    df = pd.DataFrame(rows)
    tmp = path.with_suffix(".tmp.parquet")
    df.to_parquet(tmp, index=False)
    tmp.replace(path)
