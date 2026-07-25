#!/usr/bin/env python3
"""Freeze a reproducible 60-frame validation set for GT + ablation (§1).

Three buckets (20 each, deduplicated, no near-duplicate consecutives unless useful):
  * uniform      — evenly spaced along the whole recording;
  * hard         — high image complexity / difficult lighting (clutter, shadow, glare)
                   from a decoded stride sample;
  * gaze         — diverse/extreme gaze (cockpit-down, mirror region, wide yaw, invalid)
                   which stress the gaze-target resolution and the max-recall errors.

Writes validation/grounded_sam2_gt60_manifest.csv (frame_index, capture_timestamp_ns,
selection_reason, scene_category, gaze_valid, hint_classes[hint only]), extracts the 60
rectified frames to validation/frames/, and builds validation/contact_sheet.jpg.

    ARIA_VRS_PYTHON python scripts/build_validation_set.py --vrs rec.vrs
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

from aria_drive_seg.config import Config
from aria_drive_seg.logging_utils import get_logger, setup_logging
from aria_drive_seg.vrs.provider import AriaProvider, RectifyParams
from aria_drive_seg.gaze.project import GazeProjector

log = get_logger("valset")
N_PER_BUCKET = 20
MIN_GAP = 12  # min frame gap to avoid near-duplicate consecutives


def _rect_params(cfg) -> RectifyParams:
    return RectifyParams(int(cfg.get("rectify.out_width", 2016)),
                         int(cfg.get("rectify.out_height", 1512)),
                         float(cfg.get("rectify.focal", 879.0)),
                         int(cfg.get("rectify.rotate_ccw90", 0)))


def _spread(cands: List[int], k: int, min_gap: int, taken: set) -> List[int]:
    out = []
    for i in cands:
        if len(out) >= k:
            break
        if i in taken:
            continue
        if all(abs(i - j) >= min_gap for j in out) and all(abs(i - j) >= min_gap for j in taken):
            out.append(i); taken.add(i)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--vrs", required=True)
    ap.add_argument("--out", default="validation")
    ap.add_argument("--hard-stride", type=int, default=30)
    args = ap.parse_args()
    setup_logging("INFO")
    import cv2

    cfg = Config.load()
    out = Path(args.out); (out / "frames").mkdir(parents=True, exist_ok=True)
    prov = AriaProvider(args.vrs, cfg.get("vrs.time_domain", "DEVICE_TIME"))
    rgb_label = cfg.get("vrs.rgb_label", "camera-rgb")
    rgb_ts = prov.rgb_timestamps_ns(rgb_label)
    n = rgb_ts.size
    rp = _rect_params(cfg)
    rect = prov.rectifier(rp, rgb_label)

    # ---- gaze per frame (nearest sample) ----
    eg_label = cfg.get("vrs.eyegaze_label", "eyegaze")
    projector = None
    gaze_info: Dict[int, Dict] = {}
    if prov.has_eyegaze(eg_label):
        projector = GazeProjector(prov.device_calib, prov.rgb_calib(rgb_label),
                                  prov.make_pinhole(rp, rgb_label), rgb_label,
                                  float(cfg.get("gaze.fallback_depth_m", 8.0)))
        eg_ts = prov.timestamps_ns(eg_label)
        for i in range(n):
            j = int(np.searchsorted(eg_ts, rgb_ts[i]))
            j = min(max(j, 0), eg_ts.size - 1)
            gz = prov.eyegaze_by_index(j, eg_label)
            valid = bool(gz.combined_gaze_valid) and abs(int(eg_ts[j]) - int(rgb_ts[i])) < 20e6
            gaze_info[i] = {"valid": valid, "yaw": float(gz.yaw), "pitch": float(gz.pitch)}

    taken: set = set()
    rows: Dict[int, Dict] = {}

    # ---- bucket 1: uniform ----
    for i in _spread([int(x) for x in np.linspace(0, n - 1, N_PER_BUCKET * 2).astype(int)],
                     N_PER_BUCKET, MIN_GAP, taken):
        g = gaze_info.get(i, {})
        rows[i] = {"selection_reason": "uniform", "scene_category": "uniform",
                   "gaze_valid": g.get("valid", False)}

    # ---- bucket 2: hard (image complexity / difficult lighting) ----
    log.info("scoring difficulty on a stride sample (this decodes frames)...")
    diff: List[Tuple[float, int]] = []
    for i in range(0, n, args.hard_stride):
        try:
            raw, _ = prov.rgb_by_index(i, rgb_label)
        except Exception:
            continue
        r = rect.rectify(raw)
        g = cv2.cvtColor(r, cv2.COLOR_RGB2GRAY)
        lap = float(cv2.Laplacian(g, cv2.CV_32F).var())          # clutter / detail
        dark = float((g < 30).mean()); bright = float((g > 225).mean())  # shadow / glare
        score = lap / 1000.0 + 3.0 * dark + 3.0 * bright
        diff.append((score, i))
    diff.sort(reverse=True)
    for i in _spread([i for _, i in diff], N_PER_BUCKET, MIN_GAP, taken):
        g = gaze_info.get(i, {})
        rows[i] = {"selection_reason": "hard_scene", "scene_category": "high_complexity_or_lighting",
                   "gaze_valid": g.get("valid", False)}

    # ---- bucket 3: gaze-based (diverse/extreme gaze) ----
    if projector is not None:
        def cat(i):
            g = gaze_info[i]
            if not g["valid"]:
                return "gaze_invalid"
            if g["pitch"] < -0.45:
                return "gaze_cockpit_down"
            if abs(g["yaw"]) > 0.30:
                return "gaze_wide_yaw"
            return "gaze_ahead"
        # rank frames to maximise category diversity: interleave categories
        by_cat: Dict[str, List[int]] = {}
        for i in range(n):
            if i in taken:
                continue
            by_cat.setdefault(cat(i), []).append(i)
        picks: List[int] = []
        cats = ["gaze_cockpit_down", "gaze_wide_yaw", "gaze_invalid", "gaze_ahead"]
        ci = 0
        while len(picks) < N_PER_BUCKET and any(by_cat.get(c) for c in cats):
            c = cats[ci % len(cats)]; ci += 1
            pool = by_cat.get(c, [])
            for i in pool:
                if all(abs(i - j) >= MIN_GAP for j in picks) and all(abs(i - j) >= MIN_GAP for j in taken):
                    picks.append(i); taken.add(i); pool.remove(i); break
        for i in picks:
            rows[i] = {"selection_reason": "gaze", "scene_category": cat(i),
                       "gaze_valid": gaze_info[i]["valid"]}

    # ---- extract frames + write manifest + contact sheet ----
    order = sorted(rows)
    manifest = []
    thumbs = []
    for i in order:
        raw, ts = prov.rgb_by_index(i, rgb_label)
        r = rect.rectify(raw)
        cv2.imwrite(str(out / "frames" / f"frame_{i:06d}.jpg"), r[:, :, ::-1],
                    [cv2.IMWRITE_JPEG_QUALITY, 92])
        th = cv2.resize(r[:, :, ::-1], (320, 240))
        cv2.putText(th, f"{i} {rows[i]['selection_reason']}", (6, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
        thumbs.append(th)
        manifest.append({"frame_index": i, "capture_timestamp_ns": int(ts),
                         "selection_reason": rows[i]["selection_reason"],
                         "scene_category": rows[i]["scene_category"],
                         "gaze_valid": rows[i]["gaze_valid"], "hint_classes": ""})

    with open(out / "grounded_sam2_gt60_manifest.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["frame_index", "capture_timestamp_ns",
                                          "selection_reason", "scene_category",
                                          "gaze_valid", "hint_classes"])
        w.writeheader(); w.writerows(manifest)

    cols = 6
    rows_n = (len(thumbs) + cols - 1) // cols
    grid = np.zeros((rows_n * 240, cols * 320, 3), np.uint8)
    for k, th in enumerate(thumbs):
        rr, cc = divmod(k, cols)
        grid[rr * 240:(rr + 1) * 240, cc * 320:(cc + 1) * 320] = th
    cv2.imwrite(str(out / "contact_sheet.jpg"), grid, [cv2.IMWRITE_JPEG_QUALITY, 85])

    log.info("validation set: %d frames -> %s", len(manifest), out)
    from collections import Counter
    log.info("by reason: %s", dict(Counter(m["selection_reason"] for m in manifest)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
