#!/usr/bin/env python3
"""Render eight readable Article 1 static-vs-temporal diagnostic videos."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import imageio.v2 as imageio
import numpy as np
import pandas as pd

from aria_drive_seg.io_utils import read_mask_u16
from aria_drive_seg.taxonomy import Taxonomy


PROVENANCE_COLORS = np.array([
    [0, 0, 0], [70, 210, 90], [255, 150, 30], [60, 180, 255],
    [255, 255, 60], [255, 70, 230], [170, 100, 255], [220, 220, 220],
], np.uint8)


def overlay(rgb, mask, palette, alpha=.42):
    color = palette[np.clip(mask.astype(int), 0, len(palette) - 1)]
    result = rgb.copy()
    selected = mask > 0
    result[selected] = (
        result[selected] * (1 - alpha) + color[selected] * alpha).astype(np.uint8)
    edges = cv2.Canny(
        (mask.astype(np.uint16) % 256).astype(np.uint8), 0, 1)
    result[edges > 0] = 255
    return result


def gaze_point(image, row):
    if row is None or not bool(row.get("valid", False)):
        return
    x, y = int(row.get("rect_u", -1)), int(row.get("rect_v", -1))
    if 0 <= x < image.shape[1] and 0 <= y < image.shape[0]:
        cv2.circle(image, (x, y), 18, (255, 255, 255), 4, cv2.LINE_AA)
        cv2.circle(image, (x, y), 7, (255, 40, 40), -1, cv2.LINE_AA)


def annotate(image, title, metadata, extra=""):
    result = image.copy()
    cv2.rectangle(result, (0, 0), (result.shape[1], 82), (12, 15, 20), -1)
    cv2.putText(result, title, (16, 29), cv2.FONT_HERSHEY_SIMPLEX,
                .68, (255, 255, 255), 2, cv2.LINE_AA)
    line = (
        f"frame {metadata['frame_index']}  t={metadata['capture_timestamp_ns']}  "
        f"flow={metadata['flow_valid_fraction']:.1%}  "
        f"prop={metadata['propagated_pixels']}  "
        f"age={metadata['mean_propagation_age']:.2f}  "
        f"reset={metadata['reset_reason']} {extra}")
    cv2.putText(result, line, (16, 61), cv2.FONT_HERSHEY_SIMPLEX,
                .43, (210, 230, 255), 1, cv2.LINE_AA)
    return result


def half(image):
    return cv2.resize(image, (1008, 756), interpolation=cv2.INTER_AREA)


def build_frames(rgb, raw, temporal, raw_thin, temporal_thin,
                 validity, occlusion, provenance, unknown_raw,
                 unknown_temporal, metadata, gaze, taxonomy):
    raw_view = annotate(overlay(rgb, raw, taxonomy.palette()),
                        "STATIC RAW", metadata)
    temporal_view = annotate(overlay(rgb, temporal, taxonomy.palette()),
                             "TEMPORAL STABILIZED", metadata)
    gaze_point(raw_view, gaze)
    gaze_point(temporal_view, gaze)
    static_single = np.hstack([half(raw_view), np.full((756, 280, 3), 22, np.uint8)])
    temporal_single = np.hstack([half(temporal_view), np.full((756, 280, 3), 22, np.uint8)])
    compare = np.hstack([half(raw_view), half(temporal_view)])
    thin_compare = np.hstack([
        half(annotate(overlay(rgb, raw_thin, taxonomy.palette()),
                      "STATIC THIN", metadata)),
        half(annotate(overlay(rgb, temporal_thin, taxonomy.palette()),
                      "TEMPORAL THIN", metadata)),
    ])
    flow_color = np.zeros_like(rgb)
    flow_color[validity > 0] = (30, 210, 80)
    flow_color[occlusion > 0] = (240, 60, 60)
    flow_view = annotate(
        (rgb * .45 + flow_color * .55).astype(np.uint8),
        "FLOW VALIDITY (green) / OCCLUSION (red)", metadata)
    provenance_view = annotate(
        overlay(rgb, provenance, PROVENANCE_COLORS, .55),
        "TEMPORAL PROVENANCE", metadata)
    unknown_palette = np.array([[0, 0, 0], [255, 60, 60]], np.uint8)
    unknown_compare = np.hstack([
        half(annotate(overlay(rgb, (unknown_raw > 0).astype(np.uint16),
                              unknown_palette, .60), "RAW UNKNOWN", metadata)),
        half(annotate(overlay(rgb, (unknown_temporal > 0).astype(np.uint16),
                              unknown_palette, .60), "TEMPORAL UNKNOWN", metadata)),
    ])
    raw_gaze_class = int(raw[int(gaze.rect_v), int(gaze.rect_u)]) \
        if gaze is not None and bool(gaze.get("valid", False)) and \
        0 <= int(gaze.rect_u) < raw.shape[1] and 0 <= int(gaze.rect_v) < raw.shape[0] else -1
    temporal_gaze_class = int(temporal[int(gaze.rect_v), int(gaze.rect_u)]) \
        if gaze is not None and bool(gaze.get("valid", False)) and \
        0 <= int(gaze.rect_u) < raw.shape[1] and 0 <= int(gaze.rect_v) < raw.shape[0] else -1
    gaze_compare = np.hstack([
        half(annotate(raw_view, "RAW AT GAZE", metadata,
                      f"class={taxonomy.name_of(raw_gaze_class)}")),
        half(annotate(temporal_view, "TEMPORAL AT GAZE", metadata,
                      f"class={taxonomy.name_of(temporal_gaze_class)}")),
    ])
    return (
        static_single, temporal_single, compare, thin_compare,
        np.hstack([half(flow_view), half(annotate(rgb, "RGB", metadata))]),
        np.hstack([half(provenance_view), half(temporal_view)]),
        unknown_compare, gaze_compare,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--frames", required=True)
    parser.add_argument("--temporal", required=True)
    parser.add_argument("--output", default="output/article1/temporal_videos_30s")
    parser.add_argument("--taxonomy", default="configs/article1/classes_article1.yaml")
    parser.add_argument("--fps", type=float, default=10)
    args = parser.parse_args()
    frames_root, temporal_root, output = (
        Path(args.frames), Path(args.temporal), Path(args.output))
    output.mkdir(parents=True, exist_ok=True)
    frames = pd.read_parquet(
        frames_root / "frames" / "frames.parquet").sort_values("frame_index")
    gaze_path = frames_root / "gaze" / "aligned_gaze.parquet"
    gaze = pd.read_parquet(gaze_path).set_index("frame_index") \
        if gaze_path.exists() else None
    taxonomy = Taxonomy.load(args.taxonomy)
    names = (
        "01_static_raw.mp4", "02_temporal_stabilized.mp4",
        "03_static_vs_temporal.mp4", "04_thin_static_vs_temporal.mp4",
        "05_flow_validity_and_occlusion.mp4", "06_temporal_provenance.mp4",
        "07_unknown_raw_vs_temporal.mp4", "08_gaze_raw_vs_temporal.mp4",
    )
    writers = {
        name: imageio.get_writer(
            output / name, fps=args.fps, codec="libx264", quality=8,
            macro_block_size=8)
        for name in names}
    contact_images, qa = [], []
    best = {"difference": (-1, None), "propagated": (-1, None),
            "reset": (-1, None), "lane_recovery": (-1, None)}
    for position, row in enumerate(frames.itertuples()):
        frame_id = int(row.frame_index)
        stem = f"frame_{frame_id:06d}"
        rgb = cv2.cvtColor(
            cv2.imread(str(frames_root / row.rectified_path)),
            cv2.COLOR_BGR2RGB)
        raw = read_mask_u16(temporal_root / "static_masks" / f"{stem}.png")
        temporal = read_mask_u16(temporal_root / "temporal_masks" / f"{stem}.png")
        raw_thin = read_mask_u16(temporal_root / "static_thin" / f"{stem}.png")
        temporal_thin = read_mask_u16(temporal_root / "temporal_thin" / f"{stem}.png")
        validity = cv2.imread(
            str(temporal_root / "flow_validity" / f"{stem}.png"), 0)
        occlusion = cv2.imread(
            str(temporal_root / "occlusion" / f"{stem}.png"), 0)
        provenance = cv2.imread(
            str(temporal_root / "provenance" / f"{stem}.png"), 0)
        unknown_raw = cv2.imread(
            str(temporal_root / "unknown_raw" / f"{stem}.png"), 0)
        unknown_temporal = cv2.imread(
            str(temporal_root / "unknown_temporal" / f"{stem}.png"), 0)
        metadata = json.loads(
            (temporal_root / "metadata" / f"{stem}.json").read_text())
        sample = gaze.loc[frame_id] if gaze is not None and frame_id in gaze.index else None
        rendered = build_frames(
            rgb, raw, temporal, raw_thin, temporal_thin, validity, occlusion,
            provenance, unknown_raw, unknown_temporal, metadata, sample, taxonomy)
        for writer, image in zip(writers.values(), rendered):
            writer.append_data(image)
        if position % max(1, len(frames) // 20) == 0:
            contact_images.append(cv2.resize(rendered[2], (672, 252)))
        criteria = {
            "difference": metadata["raw_temporal_difference"],
            "propagated": metadata["propagated_pixels"],
            "reset": int(metadata["reset_reason_code"] != 0),
            "lane_recovery": int(((raw_thin == 0) & (temporal_thin == 2)).sum()),
        }
        for key, value in criteria.items():
            if value > best[key][0]:
                best[key] = (value, rendered[2].copy())
        qa.append(metadata)
    for writer in writers.values():
        writer.close()
    if contact_images:
        rows = [np.hstack(contact_images[i:i + 4])
                for i in range(0, len(contact_images), 4)]
        width = max(row.shape[1] for row in rows)
        rows = [cv2.copyMakeBorder(
            row, 0, 0, 0, width - row.shape[1], cv2.BORDER_CONSTANT)
            for row in rows]
        cv2.imwrite(str(output / "contact_sheet.jpg"),
                    cv2.cvtColor(np.vstack(rows), cv2.COLOR_RGB2BGR))
    for key, (_, image) in best.items():
        if image is not None:
            cv2.imwrite(str(output / f"preview_{key}.png"),
                        cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
    (output / "render_manifest.json").write_text(json.dumps({
        "frames": len(frames), "fps": args.fps, "videos": list(names),
        "qa_metadata": qa}, indent=2))


if __name__ == "__main__":
    main()
