#!/usr/bin/env python3
"""Render checkpoint-1/checkpoint-2 external diagnostic videos."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import imageio.v2 as imageio
import numpy as np
import pandas as pd

from aria_drive_seg.io_utils import read_mask_u16
from aria_drive_seg.render.overlays import draw_gaze
from aria_drive_seg.taxonomy import Taxonomy


UNKNOWN_COLORS = np.array([
    [0, 0, 0], [255, 180, 0], [255, 0, 180], [255, 90, 0],
    [180, 0, 255], [255, 120, 80], [255, 60, 220], [255, 255, 0],
    [255, 0, 0], [255, 80, 0], [255, 0, 100], [255, 150, 0],
    [200, 0, 100], [255, 100, 100], [255, 80, 180], [255, 255, 255],
], np.uint8)


def overlay(rgb, mask, palette, alpha=.48, zero_transparent=True):
    out = rgb.copy()
    color = palette[np.clip(mask.astype(int), 0, len(palette)-1)]
    selected = mask > 0 if zero_transparent else np.ones(mask.shape, bool)
    out[selected] = (out[selected] * (1-alpha) + color[selected] * alpha).astype(np.uint8)
    return out


def panel(rgb, mask, palette, title, fi, clip_s, meta, gaze, zero_transparent=True):
    view = overlay(rgb, mask, palette, zero_transparent=zero_transparent)
    edges = cv2.Canny((mask.astype(np.uint16) % 256).astype(np.uint8), 0, 1)
    view[edges > 0] = 255
    if gaze is not None:
        draw_gaze(view, gaze.get("rect_u"), gaze.get("rect_v"), bool(gaze.get("valid", False)), 18)
    h, w = view.shape[:2]
    canvas = np.zeros((h, w + 416, 3), np.uint8)
    canvas[:, :w] = view
    canvas[:, w:] = (20, 23, 29)
    cv2.rectangle(canvas, (0, 0), (w, 84), (12, 15, 20), -1)
    cv2.putText(canvas, title, (18, 32), cv2.FONT_HERSHEY_SIMPLEX, .72,
                (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(canvas, f"frame {fi}  clip {clip_s:.3f}s", (18, 65),
                cv2.FONT_HERSHEY_SIMPLEX, .55, (210, 230, 255), 1, cv2.LINE_AA)
    lines = [
        f"unknown: {meta.get('unknown_rate', 0):.2%}",
        f"confidence: {meta.get('mean_confidence', 0):.3f}",
        f"norm entropy: {meta.get('mean_normalized_entropy', 0):.3f}",
        f"thin raw: {meta.get('thin_raw_pixels', 0)}",
        f"thin filtered: {meta.get('thin_filtered_pixels', 0)}",
        f"thin accepted: {meta.get('thin_reason_counts', {}).get('accepted', 0)}",
        f"thin rejected: {sum(v for k,v in meta.get('thin_reason_counts', {}).items() if k not in ('none','accepted'))}",
        f"ego native flag: {meta.get('mapillary_ego_region_pixels', 0)} px",
    ]
    for i, line in enumerate(lines):
        cv2.putText(canvas, line, (w+18, 38+i*27), cv2.FONT_HERSHEY_SIMPLEX,
                    .48, (180, 225, 255), 1, cv2.LINE_AA)
    return canvas


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", default="output/article1/checkpoint2_videos_30s")
    ap.add_argument("--taxonomy", default="configs/article1/classes_article1.yaml")
    ap.add_argument("--fps", type=float, default=2)
    args = ap.parse_args()
    root, out = Path(args.input), Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    frames = pd.read_parquet(root / "frames/frames.parquet").sort_values("frame_index")
    t0 = int(frames.capture_timestamp_ns.iloc[0])
    gaze_path = root / "gaze/aligned_gaze.parquet"
    gaze = pd.read_parquet(gaze_path).set_index("frame_index") if gaze_path.exists() else None
    tax = Taxonomy.load(args.taxonomy)
    writers = {
        name: imageio.get_writer(out / name, fps=args.fps, codec="libx264", quality=8,
                                 macro_block_size=8)
        for name in (
            "01_external_checkpoint1.mp4", "02_external_checkpoint2.mp4",
            "03_thin_raw.mp4", "04_thin_filtered.mp4", "05_unknown_reason.mp4",
            "06_checkpoint1_vs_checkpoint2.mp4")
    }
    previews = out / "previews"; previews.mkdir(exist_ok=True)
    qa_rows = []
    for pos, (_, row) in enumerate(frames.iterrows()):
        fi, stem = int(row.frame_index), f"frame_{int(row.frame_index):06d}"
        rgb = cv2.cvtColor(cv2.imread(str(root / row.rectified_path)), cv2.COLOR_BGR2RGB)
        cp1 = read_mask_u16(root / "article1_external/masks" / f"{stem}.png")
        base = root / "article1_external_checkpoint2"
        cp2 = read_mask_u16(base / "masks" / f"{stem}.png")
        raw = read_mask_u16(base / "raw_thin_masks" / f"{stem}.png")
        filt = read_mask_u16(base / "filtered_thin_masks" / f"{stem}.png")
        reason = cv2.imread(str(base / "unknown_reason" / f"{stem}.png"), cv2.IMREAD_UNCHANGED)
        meta = json.loads((base / "metadata" / f"{stem}.json").read_text())
        g = gaze.loc[fi] if gaze is not None and fi in gaze.index else None
        clip_s = (int(row.capture_timestamp_ns)-t0)/1e9
        panels = [
            panel(rgb, cp1, tax.palette(), "Checkpoint 1 external", fi, clip_s, meta, g),
            panel(rgb, cp2, tax.palette(), "Checkpoint 2 external", fi, clip_s, meta, g),
            panel(rgb, raw, tax.palette(), "Raw thin candidates", fi, clip_s, meta, g),
            panel(rgb, filt, tax.palette(), "Filtered thin markings", fi, clip_s, meta, g),
            panel(rgb, reason, UNKNOWN_COLORS, "Unknown reason bitmask", fi, clip_s, meta, g),
        ]
        for writer, image in zip(list(writers.values())[:5], panels):
            writer.append_data(image)
        thirds = [cv2.resize(x[:, :2016], (1008, 752), interpolation=cv2.INTER_AREA)
                  for x in [rgb, panels[0], panels[1]]]
        comparison = np.hstack(thirds)
        cv2.putText(comparison, "RGB", (12, 30), cv2.FONT_HERSHEY_SIMPLEX, .7, (255,255,255), 2)
        cv2.putText(comparison, "CHECKPOINT 1", (1020, 30), cv2.FONT_HERSHEY_SIMPLEX, .7, (255,255,255), 2)
        cv2.putText(comparison, "CHECKPOINT 2", (2028, 30), cv2.FONT_HERSHEY_SIMPLEX, .7, (255,255,255), 2)
        writers["06_checkpoint1_vs_checkpoint2.mp4"].append_data(comparison)
        qa_rows.append({"frame_index": fi, **meta})
        if pos in {0, 10, 20, 30, 40, 50, 59}:
            cv2.imwrite(str(previews / f"checkpoint2_{stem}.png"),
                        cv2.cvtColor(panels[1], cv2.COLOR_RGB2BGR))
    for writer in writers.values():
        writer.close()
    Path(out / "render_manifest.json").write_text(json.dumps({
        "frames": len(frames), "fps": args.fps, "videos": list(writers),
        "qa_metadata": qa_rows}, indent=2))


if __name__ == "__main__":
    main()
