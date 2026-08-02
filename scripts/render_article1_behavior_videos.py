#!/usr/bin/env python3
"""Phase 14: the three multimodal analysis videos.

Two per-domain videos over the dense frozen semantic blocks, and one paired video
that puts car and motorcycle side by side **by position along the road**, not by
timestamp. The two drives happened weeks apart; matching them on time would
compare unrelated moments.

The MP4s are local artefacts and are not committed.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import pandas as pd
import yaml

from aria_drive_seg.behavior import EXPLORATORY_MARKER
from aria_drive_seg.behavior.gaze_semantics import (SemanticBlock,
                                                    pixels_per_degree)
from aria_drive_seg.behavior.video import (PROXY_CLASSES, VideoWriter, compose,
                                           draw_gaze, semantic_overlay,
                                           side_by_side, telemetry_panel)
from aria_drive_seg.config import Config
from aria_drive_seg.io_utils import atomic_write_json
from aria_drive_seg.logging_utils import get_logger, setup_logging

log = get_logger("behavior.video")


def class_colours(path: Path) -> Tuple[Dict[int, List[int]], Dict[int, str]]:
    tax = yaml.safe_load(path.read_text())
    colours = {int(c["id"]): list(reversed(c["color"])) for c in tax["classes"]}
    names = {int(c["id"]): str(c["name"]) for c in tax["classes"]}
    return colours, names


def _nearest(series_ts: np.ndarray, t: int) -> Optional[int]:
    if series_ts.size == 0:
        return None
    k = int(np.argmin(np.abs(series_ts - t)))
    return k


def _fmt(v, spec="{:.2f}", missing="-"):
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        return missing
    return spec.format(v)


def render_domain(domain: str, spec: Dict[str, Any], cfg: Config, out_root: Path,
                  dest: Path, colours, names) -> Optional[Dict[str, Any]]:
    rec_id = spec["recording_id"]
    blocks = spec.get("semantic_blocks") or []
    if not blocks:
        return None
    root = cfg.resolve(blocks[0]["root"])
    if not root.exists():
        return None
    block = SemanticBlock(root)
    frames = pd.read_parquet(root / "frames" / "frames.parquet")
    d = out_root / rec_id

    gaze = pd.read_parquet(d / "semantic_gaze.parquet")
    gaze = gaze.set_index("frame_index")
    veh = pd.read_parquet(d / "vehicle_dynamics.parquet")
    head = pd.read_parquet(d / "head_dynamics_per_frame.parquet").set_index(
        "frame_index")
    beats = pd.read_parquet(d / "ppg_beats.parquet")
    beats = beats[beats["valid"].astype(bool)]
    quality = pd.read_parquet(d / "ppg_quality.parquet")
    matched = pd.read_parquet(d / "map_matched.parquet")
    matched = matched[matched["matched"].astype(bool)]
    cand = pd.read_csv(Path("reports/article1_behavior_analysis/solid_line") /
                       "solid_line_candidates.csv")
    cand = cand[cand.domain == domain] if len(cand) else cand

    ppd = pixels_per_degree(float(cfg.get("rectify.focal", 879.0)))
    radius = int(round(float(cfg.get("gaze.foveal_radius_sigmas", 2.5)) *
                       float(cfg.get("gaze.foveal_sigma_deg", 1.5)) * ppd))

    fi_col = "frame_index" if "frame_index" in frames else "source_frame_index"
    frames = frames.set_index(fi_col)
    order = [f for f in block.frames if f in frames.index]
    if not order:
        return None

    ts_all = frames.loc[order, "capture_timestamp_ns"].values.astype(np.int64)
    fps = float(1e9 / np.median(np.diff(ts_all))) if len(ts_all) > 1 else 10.0

    first = block.load(order[0])
    h, w = first["mask"].shape[:2]
    scale = 1280.0 / w
    size = (1280, int(round(h * scale)) + 190)

    path = dest / f"{domain}_multimodal_analysis.mp4"
    log.info("  rendering %s (%d frames at %.1f fps)", path.name, len(order), fps)
    with VideoWriter(path, fps, size) as writer:
        for fi in order:
            layers = block.load(fi)
            if layers is None:
                continue
            ts = int(frames.loc[fi, "capture_timestamp_ns"])
            rp = frames.loc[fi].get("rectified_path")
            img = cv2.imread(str(root / rp)) if isinstance(rp, str) else None
            if img is None:
                img = np.zeros((h, w, 3), np.uint8)
            img = cv2.resize(img, (w, h))
            img = semantic_overlay(img, layers["mask"], colours)

            g = gaze.loc[fi] if fi in gaze.index else None
            if g is not None:
                draw_gaze(img, float(g.rect_u), float(g.rect_v), radius,
                          bool(g.is_fixation), bool(g.semantic_valid))
            img = cv2.resize(img, (1280, int(round(h * scale))))

            kv = _nearest(veh["timestamp_ns"].values, ts)
            kb = _nearest(beats["timestamp_ns"].values, ts) if len(beats) else None
            kq = _nearest(quality["start_ns"].values, ts)
            km = _nearest(matched["timestamp_ns"].values, ts)
            hd = head.loc[fi] if fi in head.index else None

            top1 = (str(g.top1_class) if g is not None and g.semantic_valid
                    else "-")
            proxy = " [proxy]" if top1 in PROXY_CLASSES else ""
            in_cand = "no"
            if len(cand):
                hit = cand[(cand.start_ns <= ts) & (cand.end_ns >= ts)]
                if len(hit):
                    in_cand = f"{hit.iloc[0].state}"

            fields = [
                ("gaze class", f"{top1}{proxy}", "frozen semantic camera"),
                ("foveal top-1 p", _fmt(getattr(g, "top1_probability", None)),
                 "Gaussian foveal window"),
                ("seg confidence", _fmt(getattr(g, "confidence", None)),
                 "frozen baseline"),
                ("fixation", ("yes" if g is not None and g.is_fixation else "no"),
                 "I-VT, 30 Hz gaze"),
                ("speed", _fmt(veh["speed_mps"].values[kv] if kv is not None else None,
                               "{:.1f} m/s"), "GPS (no pose stream)"),
                ("acceleration",
                 _fmt(veh["acceleration_mps2"].values[kv] if kv is not None else None,
                      "{:+.2f} m/s2"), "d(GPS speed)/dt"),
                ("head angular speed",
                 _fmt(hd["angular_speed_rad_s"] if hd is not None else None,
                      "{:.2f} rad/s"), "head IMU - NOT vehicle"),
                ("heart rate",
                 _fmt(beats["heart_rate_bpm"].values[kb] if kb is not None else None,
                      "{:.0f} bpm"), "PPG proxy, not medical"),
                ("PPG quality",
                 _fmt(quality["sqi"].values[kq] if kq is not None else None),
                 ("usable" if kq is not None and quality["usable"].values[kq]
                  else "window rejected")),
                ("dist to junction",
                 _fmt(matched["distance_to_junction_m"].values[km]
                      if km is not None else None, "{:.0f} m"), "OpenStreetMap"),
                ("dist to roundabout",
                 _fmt(matched["distance_to_roundabout_m"].values[km]
                      if km is not None else None, "{:.0f} m"), "OpenStreetMap"),
                ("line-crossing candidate", in_cand, "awaiting human review"),
            ]
            panel = telemetry_panel(
                1280, fields,
                f"{domain}  |  {rec_id}  |  frame {fi}  |  t={ts / 1e9:.2f} s  "
                f"|  {EXPLORATORY_MARKER}")
            writer.write(compose(img, panel))
    return {"path": str(path), "frames": len(order), "fps": fps}


def render_paired(cfg: Config, out_root: Path, dest: Path,
                  reports: Path) -> Optional[Dict[str, Any]]:
    """Side by side along the shared road, matched on position not on time."""
    align = json.loads((reports / "route" / "route_alignment_summary.json").read_text())
    bin_size = float(align["analysis_bin_size_m"])
    pairs = pd.read_csv(reports / "route" / "paired_route_segments.csv")
    pairs = pairs[(pairs.bin_size_m == bin_size) & pairs.paired.astype(bool)]
    if pairs.empty:
        return None

    providers, bins, matched, veh = {}, {}, {}, {}
    from aria_drive_seg.vrs.provider import AriaProvider, RectifyParams
    for domain, spec in (cfg.get("recordings") or {}).items():
        d = out_root / spec["recording_id"]
        bins[domain] = pd.read_parquet(d / "frame_route_bins.parquet")
        matched[domain] = pd.read_parquet(d / "map_matched.parquet")
        veh[domain] = pd.read_parquet(d / "vehicle_dynamics.parquet")
        p = AriaProvider(cfg.resolve(spec["vrs"]))
        providers[domain] = (p, p.rectifier(RectifyParams(
            int(cfg.get("rectify.width", 2016)),
            int(cfg.get("rectify.height", 1512)),
            float(cfg.get("rectify.focal", 879.0)))))

    # One frame per shared bin, per domain: the frame closest to the bin centre.
    picks: Dict[str, List[int]] = {}
    for domain in bins:
        b = bins[domain]
        chosen = []
        for key in pairs["bin_key"]:
            sel = b[(b["bin_key"] == key) & b["bin_valid"].astype(bool)]
            chosen.append(int(sel.iloc[len(sel) // 2]["frame_index"])
                          if len(sel) else -1)
        picks[domain] = chosen

    usable = [i for i in range(len(pairs))
              if all(picks[d][i] >= 0 for d in picks)]
    if not usable:
        return None
    log.info("  paired video over %d shared bins", len(usable))

    size = (1920, 840)
    path = dest / "paired_route_comparison.mp4"
    with VideoWriter(path, 2.0, size) as writer:
        for i in usable:
            row = pairs.iloc[i]
            panels = {}
            info = {}
            for domain, (prov, rect) in providers.items():
                fi = picks[domain][i]
                raw, ts = prov.rgb_by_index(fi)
                img = rect.rectify(raw)
                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                panels[domain] = cv2.resize(img, (940, 705))
                m = matched[domain]
                km = _nearest(m["timestamp_ns"].values, ts)
                v = veh[domain]
                kv = _nearest(v["timestamp_ns"].values, ts)
                info[domain] = {
                    "speed": v["speed_mps"].values[kv] if kv is not None else None,
                    "frame": fi, "t": ts / 1e9,
                }
            frame = side_by_side(
                panels["car"], panels["motorcycle"],
                f"car  frame {info['car']['frame']}  "
                f"{_fmt(info['car']['speed'], '{:.1f} m/s')}",
                f"motorcycle  frame {info['motorcycle']['frame']}  "
                f"{_fmt(info['motorcycle']['speed'], '{:.1f} m/s')}")
            panel = telemetry_panel(
                frame.shape[1],
                [("shared bin", str(row.bin_key), f"{bin_size:.0f} m map bin"),
                 ("road class", str(row.road_class), "OpenStreetMap"),
                 ("centroid distance", _fmt(row.centroid_distance_m, "{:.1f} m"),
                  "between the two vehicles' means"),
                 ("heading difference", _fmt(row.heading_difference_deg, "{:.0f} deg"),
                  "same-direction check"),
                 ("dist to junction", _fmt(row.distance_to_junction_m, "{:.0f} m"),
                  "OpenStreetMap"),
                 ("pair quality", _fmt(row.pair_quality), "map-match quality")],
                f"matched by POSITION along the road, not by time  |  "
                f"{EXPLORATORY_MARKER}", height=120)
            writer.write(compose(frame, panel))
    return {"path": str(path), "bins": len(usable)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/article1/behavior_analysis.yaml")
    ap.add_argument("--output", default="output/article1/behavior_analysis")
    ap.add_argument("--reports", default="reports/article1_behavior_analysis")
    ap.add_argument("--skip-paired", action="store_true")
    ap.add_argument("--log-level", default="INFO")
    args = ap.parse_args()
    setup_logging(args.log_level)

    cfg = Config.load(args.config)
    out_root = Path(args.output)
    dest = out_root
    dest.mkdir(parents=True, exist_ok=True)
    colours, names = class_colours(cfg.resolve(
        cfg.get("article1.classes", "configs/article1/classes_article1.yaml")))

    produced = {}
    for domain, spec in (cfg.get("recordings") or {}).items():
        log.info("== %s", domain)
        r = render_domain(domain, spec, cfg, out_root, dest, colours, names)
        if r:
            produced[domain] = r

    if not args.skip_paired:
        log.info("== paired")
        r = render_paired(cfg, out_root, dest, Path(args.reports))
        if r:
            produced["paired"] = r

    atomic_write_json(dest / "video_manifest.json", {
        "schema": "article1_behavior_videos_v1",
        "result_status": EXPLORATORY_MARKER,
        "videos": produced,
        "committed": False,
        "note": ("MP4s are local artefacts. The per-domain videos cover the dense "
                 "frozen semantic blocks; the paired video is matched on position "
                 "along the road and carries no semantic overlay, because neither "
                 "block sits on the shared route."),
    })
    print(json.dumps(produced, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
