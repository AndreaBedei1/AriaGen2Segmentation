#!/usr/bin/env python3
"""Render readable, metadata-rich A/B/C/D Grounded-SAM2 comparison videos."""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from aria_drive_seg.config import Config
from aria_drive_seg.io_utils import read_mask_u16
from aria_drive_seg.render.overlays import blend_mask, draw_gaze
from aria_drive_seg.taxonomy import Taxonomy


CONFIGS = [
    ("A", "A_baseline_original"),
    ("B", "B_phase1_clean"),
    ("C", "C_extended_hierarchical"),
    ("D", "D_selective_roi_multiscale"),
]
PROV_COLORS = {
    "full_frame": (255, 255, 255),
    "roi": (255, 190, 0),
    "tile": (255, 80, 220),
    "hierarchy": (40, 220, 255),
}


def text(img, value, xy, scale=.55, color=(255, 255, 255), thick=1):
    cv2.putText(img, str(value), xy, cv2.FONT_HERSHEY_SIMPLEX, scale, color,
                thick, cv2.LINE_AA)


def provenance_color(value):
    if str(value).startswith("roi:"):
        return PROV_COLORS["roi"]
    if str(value).startswith("tile:"):
        return PROV_COLORS["tile"]
    if "hier" in str(value) or "parent" in str(value):
        return PROV_COLORS["hierarchy"]
    return PROV_COLORS["full_frame"]


def clean_value(value, fallback="n/a"):
    if value is None:
        return fallback
    try:
        if pd.isna(value):
            return fallback
    except (TypeError, ValueError):
        pass
    return str(value)


def load_gaze(run):
    aligned = run / "gaze" / "aligned_gaze.parquet"
    labels = run / "comparison" / "gaze" / "gaze_labels_grounded_sam2.parquet"
    a = pd.read_parquet(aligned).set_index("frame_index") if aligned.exists() else None
    lab = pd.read_parquet(labels).set_index("frame_index") if labels.exists() else None
    return a, lab


def draw_boxes(img, detections, limit=12):
    # One strongest detection per class keeps the image readable.
    strongest = {}
    for d in detections:
        name = d.get("canonical_name", "?")
        if name not in strongest or d.get("combined_score", 0) > strongest[name].get("combined_score", 0):
            strongest[name] = d
    chosen = sorted(strongest.values(), key=lambda x: x.get("combined_score", 0), reverse=True)[:limit]
    for d in chosen:
        x0, y0, x1, y1 = [int(round(v)) for v in d.get("box", (0, 0, 0, 0))]
        col = (PROV_COLORS["hierarchy"] if d.get("parent_class")
               else provenance_color(d.get("provenance", "full_frame")))
        cv2.rectangle(img, (x0, y0), (x1, y1), col, 2, cv2.LINE_AA)
        label = f"{d.get('canonical_name', '?')} {d.get('combined_score', 0):.2f}"
        text(img, label, (max(0, x0), max(18, y0 - 5)), .45, col, 1)


def draw_roi_boxes(img, rois):
    for name, box in (rois or {}).items():
        x0, y0, x1, y1 = map(int, box)
        cv2.rectangle(img, (x0, y0), (x1, y1), PROV_COLORS["roi"], 3, cv2.LINE_AA)
        text(img, f"ROI {name}", (x0 + 4, min(img.shape[0] - 8, y0 + 22)),
             .52, PROV_COLORS["roi"], 2)


def make_panel(rgb, mask, meta, tax, letter, name, frame_row, first_rgb_ns,
               gaze_row=None, label_row=None, mean_ms=None):
    view = blend_mask(rgb, mask, tax, .48)
    # Thin semantic boundaries improve mask readability without obscuring RGB.
    edges = cv2.Canny((mask.astype(np.uint16) % 256).astype(np.uint8), 0, 1)
    view[edges > 0] = (255, 255, 255)
    draw_boxes(view, meta.get("detections", []))
    if letter in ("C", "D"):
        draw_roi_boxes(view, meta.get("rois", {}))
    if gaze_row is not None:
        view = draw_gaze(view, gaze_row.get("rect_u"), gaze_row.get("rect_v"),
                         bool(gaze_row.get("valid", False)), radius=18)

    h, w = view.shape[:2]
    side_w = 520
    canvas = np.zeros((h, w + side_w, 3), np.uint8)
    canvas[:, :w] = view
    canvas[:, w:] = (20, 23, 29)
    cv2.rectangle(canvas, (0, 0), (w, 108), (12, 15, 20), -1)
    ts_ns = int(frame_row["capture_timestamp_ns"])
    rec_s = (ts_ns - first_rgb_ns) / 1e9
    clip_s = (ts_ns - int(frame_row["_clip_t0_ns"])) / 1e9
    total_ms = meta.get("timings_ms", {}).get("total")
    coverage = meta.get("coverage")
    dets = meta.get("detections", [])

    text(canvas, f"{letter}  {name}", (22, 36), .82, (255, 255, 255), 2)
    text(canvas, f"source frame {int(frame_row['frame_index'])}   recording {rec_s:07.3f}s",
         (22, 72), .62)
    text(canvas, f"clip {clip_s:05.2f}s / 30.00s", (22, 99), .55, (190, 220, 255))
    right = [
        f"detections  {meta.get('num_detections', len(dets))}",
        f"coverage    {coverage:.1%}" if coverage is not None else "coverage    n/a",
        f"frame time  {total_ms/1000:.2f}s" if total_ms is not None else "frame time  n/a",
        f"mean time   {mean_ms/1000:.2f}s" if mean_ms is not None else "mean time   n/a",
    ]
    for i, line in enumerate(right):
        text(canvas, line, (w + 24, 42 + i * 31), .58,
             (120, 225, 255) if i < 2 else (255, 210, 120), 1)

    present = sorted({d.get("canonical_name", "?") for d in dets})
    counts = Counter(d.get("provenance", "full_frame") for d in dets)
    y = 190
    text(canvas, "CLASSES PRESENT", (w + 24, y), .62, (255, 255, 255), 2)
    y += 32
    lut = tax.palette()
    for cname in present[:35]:
        cid = tax.id_of(cname) or 0
        col = tuple(int(x) for x in lut[cid])
        cv2.rectangle(canvas, (w + 25, y - 14), (w + 42, y + 3), col, -1)
        text(canvas, cname, (w + 52, y + 2), .50)
        y += 25
    if len(present) > 35:
        text(canvas, f"+ {len(present)-35} more", (w + 25, y), .48, (180, 180, 180))
        y += 28

    y = max(y + 20, 760)
    text(canvas, "GAZE", (w + 24, y), .62, (120, 255, 150), 2)
    y += 29
    if label_row is not None:
        gaze_lines = [
            f"primary: {clean_value(label_row.get('primary_target'), clean_value(label_row.get('class_at_pixel')))}",
            f"reason: {clean_value(label_row.get('resolution_reason'))}",
            f"pixel/disc: {clean_value(label_row.get('class_at_pixel'))} / {clean_value(label_row.get('dominant_disc'))}",
            "layers: " + ", ".join(v for v in [
                clean_value(label_row.get("exterior_content"), ""),
                clean_value(label_row.get("cockpit_object"), ""),
                clean_value(label_row.get("transparent_surface"), ""),
                clean_value(label_row.get("mirror_region"), "")
            ] if v)[:52],
        ]
    else:
        gaze_lines = ["labels unavailable"]
    for line in gaze_lines:
        text(canvas, line, (w + 24, y), .45, (205, 245, 210))
        y += 25

    if letter in ("C", "D"):
        y += 15
        text(canvas, "HIERARCHY / ROI / PROVENANCE", (w + 24, y), .54,
             (255, 190, 80), 2)
        y += 29
        summary = Counter()
        for key, value in counts.items():
            group = key.split(":", 1)[0]
            summary[group] += value
        for key, value in sorted(summary.items()):
            text(canvas, f"{key}: {value}", (w + 24, y), .48, provenance_color(key))
            y += 24
        hierarchical = [d for d in dets if d.get("parent_class")]
        statuses = Counter(d.get("subclass_status") or "unspecified" for d in hierarchical)
        if hierarchical:
            text(canvas, "hierarchy: " + ", ".join(f"{k}={v}" for k, v in sorted(statuses.items())),
                 (w + 24, y), .43, PROV_COLORS["hierarchy"])
            y += 23
            accepted = [f"{d.get('parent_class')}->{d.get('canonical_name')}"
                        for d in hierarchical if d.get("subclass_status") == "accepted"]
            if accepted:
                text(canvas, "accepted: " + ", ".join(accepted[:4]), (w + 24, y),
                     .42, PROV_COLORS["hierarchy"])
                y += 22
        rois = list((meta.get("rois") or {}).keys())
        text(canvas, "ROIs: " + (", ".join(rois) if rois else "none this frame"),
             (w + 24, y), .43, PROV_COLORS["roi"])
    return canvas


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="output/videos_abcd_30s")
    ap.add_argument("--fps", type=float, default=2.0)
    ap.add_argument("--first-rgb-ns", type=int, default=1055278217953)
    ap.add_argument("--grid", action="store_true")
    args = ap.parse_args()
    root = Path(args.root)
    cfg = Config.load()
    tax = Taxonomy.load(cfg.resolve(cfg.get("project.classes")))
    frames = pd.read_parquet(root / "_shared/frames/frames.parquet").sort_values("frame_index")
    frames["_clip_t0_ns"] = int(frames.capture_timestamp_ns.iloc[0])
    panels_by_config = {}
    report = {"fps": args.fps, "frames": len(frames), "videos": {}}

    import imageio.v2 as imageio
    for letter, name in CONFIGS:
        run = root / "runs" / name
        gaze, labels = load_gaze(run)
        metas = {}
        for p in sorted((run / "grounded_sam2/metadata").glob("frame_*.json")):
            m = json.loads(p.read_text())
            metas[int(m["frame_index"])] = m
        mean_ms = np.mean([m.get("timings_ms", {}).get("total") for m in metas.values()
                           if m.get("timings_ms", {}).get("total") is not None])
        out = root / f"{name}.mp4"
        writer = imageio.get_writer(out, fps=args.fps, codec="libx264", quality=8,
                                   macro_block_size=8)
        panels = []
        for _, row in frames.iterrows():
            fi = int(row.frame_index)
            rgb = cv2.cvtColor(cv2.imread(str(run / row.rectified_path)), cv2.COLOR_BGR2RGB)
            mask = read_mask_u16(run / "grounded_sam2/canonical_masks" / f"frame_{fi:06d}.png")
            panel = make_panel(rgb, mask, metas[fi], tax, letter, name, row,
                               args.first_rgb_ns,
                               gaze.loc[fi] if gaze is not None and fi in gaze.index else None,
                               labels.loc[fi] if labels is not None and fi in labels.index else None,
                               mean_ms)
            writer.append_data(panel)
            panels.append(panel)
        writer.close()
        panels_by_config[name] = panels
        report["videos"][name] = {"path": str(out), "mean_ms": float(mean_ms)}
        preview_dir = root / "previews"
        preview_dir.mkdir(exist_ok=True)
        for pos in (0, len(panels)//2, len(panels)-1):
            cv2.imwrite(str(preview_dir / f"{letter}_frame_{int(frames.iloc[pos].frame_index):06d}.png"),
                        cv2.cvtColor(panels[pos], cv2.COLOR_RGB2BGR))

    if args.grid:
        out = root / "comparison_ABCD_grid.mp4"
        writer = imageio.get_writer(out, fps=args.fps, codec="libx264", quality=7,
                                   macro_block_size=8)
        names = [name for _, name in CONFIGS]
        for i in range(len(frames)):
            small = [cv2.resize(panels_by_config[n][i], (1268, 756),
                                interpolation=cv2.INTER_AREA) for n in names]
            grid = np.vstack([np.hstack(small[:2]), np.hstack(small[2:])])
            writer.append_data(grid)
        writer.close()
        report["grid"] = str(out)
    (root / "render_report.json").write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
