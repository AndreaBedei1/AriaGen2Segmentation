#!/usr/bin/env python3
"""Render current, improved and offline-stabilized semantic-camera videos."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import imageio.v2 as imageio
import numpy as np
import pandas as pd

from aria_drive_seg.article1.semantic_camera_video import (
    INTERNAL_CLASS_IDS, THIN_CLASS_IDS,
)
from aria_drive_seg.io_utils import read_mask_u16
from aria_drive_seg.taxonomy import Taxonomy


VIDEO_NAMES = (
    "01_current_fusion.mp4",
    "02_improved_fusion.mp4",
    "03_raw_vs_stabilized.mp4",
    "04_internal_regions_focus.mp4",
    "05_lane_and_marking_focus.mp4",
)


def _text(image: np.ndarray, value: str, xy: tuple[int, int],
          scale=.52, color=(245, 248, 252), thickness=1) -> None:
    cv2.putText(
        image, value, xy, cv2.FONT_HERSHEY_SIMPLEX, scale,
        color, thickness, cv2.LINE_AA)


def overlay(
        rgb: np.ndarray, mask: np.ndarray, taxonomy: Taxonomy,
        class_ids: np.ndarray | list[int] | None = None,
        alpha: float = .45) -> np.ndarray:
    color = taxonomy.colorize(mask)
    result = rgb.copy()
    selected = mask > 0
    if class_ids is not None:
        selected &= np.isin(mask, class_ids)
    result[selected] = (
        result[selected] * (1.0 - alpha)
        + color[selected] * alpha).astype(np.uint8)
    edge_source = np.where(selected, mask, 0).astype(np.uint8)
    edges = cv2.Canny(edge_source, 0, 1)
    result[edges > 0] = 255
    return result


def _legend(image: np.ndarray, taxonomy: Taxonomy,
            mask: np.ndarray) -> None:
    present = [
        class_id for class_id in range(1, taxonomy.max_id + 1)
        if np.any(mask == class_id)]
    y0 = image.shape[0] - 70
    cv2.rectangle(
        image, (0, y0), (image.shape[1], image.shape[0]),
        (12, 15, 18), -1)
    for index, class_id in enumerate(present[:13]):
        column, row = index % 7, index // 7
        x, y = 12 + column * 180, y0 + 22 + row * 30
        color = tuple(
            int(value) for value in taxonomy.by_id[class_id].color)
        cv2.rectangle(image, (x, y - 13), (x + 18, y + 5), color, -1)
        _text(
            image, taxonomy.name_of(class_id), (x + 24, y + 3),
            .40, (235, 240, 245))


def _dense_panel(
        rgb: np.ndarray, mask: np.ndarray, taxonomy: Taxonomy,
        title: str, subtitle: str) -> np.ndarray:
    panel = cv2.resize(
        overlay(rgb, mask, taxonomy), (1280, 960),
        interpolation=cv2.INTER_AREA)
    cv2.rectangle(panel, (0, 0), (1280, 82), (12, 15, 18), -1)
    _text(panel, title, (14, 31), .70, thickness=2)
    _text(panel, subtitle, (14, 63), .46, (210, 230, 255))
    _legend(panel, taxonomy, mask)
    return panel


def _comparison(
        rgb: np.ndarray, masks: tuple[np.ndarray, ...],
        names: tuple[str, ...], taxonomy: Taxonomy,
        subtitle: str, class_ids=None) -> np.ndarray:
    panels = [
        cv2.resize(
            overlay(rgb, mask, taxonomy, class_ids=class_ids),
            (960, 720), interpolation=cv2.INTER_AREA)
        for mask in masks
    ]
    result = np.zeros((800, 1920, 3), np.uint8)
    result[80:] = np.hstack(panels)
    for index, name in enumerate(names):
        _text(result, name, (index * 960 + 14, 31), .64, thickness=2)
    _text(result, subtitle, (14, 64), .47, (210, 230, 255))
    return cv2.resize(result, (1920, 560), interpolation=cv2.INTER_AREA)


def _focus(
        rgb: np.ndarray, masks: tuple[np.ndarray, ...],
        names: tuple[str, ...], taxonomy: Taxonomy,
        title: str, subtitle: str, class_ids, crop_start: float,
        crop_end: float = 1.0) -> np.ndarray:
    full_panels = [
        cv2.resize(
            overlay(rgb, mask, taxonomy, class_ids=class_ids),
            (640, 480), interpolation=cv2.INTER_AREA)
        for mask in masks
    ]
    h = rgb.shape[0]
    y0 = max(0, min(h - 1, int(round(h * crop_start))))
    y1 = max(y0 + 1, min(h, int(round(h * crop_end))))
    cropped_rgb = rgb[y0:y1]
    crop_panels = [
        cv2.resize(
            overlay(
                cropped_rgb, mask[y0:y1], taxonomy,
                class_ids=class_ids),
            (640, 400), interpolation=cv2.INTER_AREA)
        for mask in masks
    ]
    result = np.zeros((960, 1920, 3), np.uint8)
    result[80:560] = np.hstack(full_panels)
    result[560:] = np.hstack(crop_panels)
    _text(result, title, (14, 30), .67, thickness=2)
    _text(result, subtitle, (14, 63), .45, (210, 230, 255))
    for index, name in enumerate(names):
        _text(
            result, name, (index * 640 + 14, 105),
            .54, thickness=2)
    return result


def build_video_frames(
        rgb: np.ndarray,
        current: np.ndarray,
        improved: np.ndarray,
        presentation: np.ndarray,
        metadata: dict,
        taxonomy: Taxonomy,
        metrics: dict,
) -> tuple[np.ndarray, ...]:
    return tuple(
        build_video_frame(
            index, rgb, current, improved, presentation,
            metadata, taxonomy, metrics)
        for index in range(len(VIDEO_NAMES)))


def build_video_frame(
        product_index: int,
        rgb: np.ndarray,
        current: np.ndarray,
        improved: np.ndarray,
        presentation: np.ndarray,
        metadata: dict,
        taxonomy: Taxonomy,
        metrics: dict,
) -> np.ndarray:
    frame_id = int(metadata["frame_index"])
    timestamp = int(metadata["capture_timestamp_ns"])
    presentation_stats = metadata.get("presentation", {})
    comparison = metrics.get("comparison", {}).get(
        "presentation_vs_improved", {})
    reduction = 100 * comparison.get("flicker_reduction", 0.0)
    common = f"frame {frame_id} | ts {timestamp}"
    if product_index == 0:
        return _dense_panel(
            rgb, current, taxonomy,
            f"CURRENT SCIENTIFIC FUSION | {common}",
            "raw per-frame baseline | preserved read-only")
    if product_index == 1:
        return _dense_panel(
            rgb, improved, taxonomy,
            f"IMPROVED RAW FUSION | {common}",
            "stronger cockpit priority | still frame-independent")
    if product_index == 2:
        return _comparison(
            rgb, (improved, presentation),
            ("IMPROVED RAW", "STABILIZED PRESENTATION"),
            taxonomy,
            f"{common} | non-causal +/-5 frames | "
            f"global flicker reduction={reduction:.2f}%")
    if product_index == 3:
        return _focus(
            rgb, (current, improved, presentation),
            ("CURRENT", "IMPROVED RAW", "PRESENTATION"),
            taxonomy, "INTERNAL REGIONS FOCUS",
            f"{common} | mirror / display / controls "
            f"(driver hand rolls into control) | "
            f"internal={100 * presentation_stats.get('full_resolution_internal_fraction', 0):.2f}%",
            INTERNAL_CLASS_IDS, .52)
    if product_index == 4:
        return _focus(
            rgb, (current, improved, presentation),
            ("CURRENT", "IMPROVED RAW", "PRESENTATION"),
            taxonomy, "LANE AND ROAD-MARKING FOCUS",
            f"{common} | protected external evidence + "
            f"bidirectional continuity",
            np.asarray([1, *THIN_CLASS_IDS], np.uint16), .30, .88)
    raise IndexError(product_index)


def _qa_candidates(
        semantic_dir: Path, frames: pd.DataFrame,
        class_ids: np.ndarray, limit: int = 4) -> list[dict]:
    masks = []
    frame_ids = [int(value) for value in frames.frame_index]
    for frame_id in frame_ids:
        mask = read_mask_u16(
            semantic_dir / "final_masks" /
            f"frame_{frame_id:06d}.png")
        masks.append(cv2.resize(
            mask, (504, 378), interpolation=cv2.INTER_NEAREST))
    scores = []
    for index in range(1, len(masks) - 1):
        relevant = np.isin(masks[index - 1], class_ids) | \
            np.isin(masks[index + 1], class_ids)
        flicker = (
            relevant
            & (masks[index - 1] == masks[index + 1])
            & (masks[index] != masks[index - 1]))
        scores.append({
            "frame_index": frame_ids[index],
            "score_pixels_at_quarter_scale": int(flicker.sum()),
        })
    return sorted(
        scores, key=lambda row: (
            row["score_pixels_at_quarter_scale"],
            -row["frame_index"]), reverse=True)[:limit]


def render(
        frames_dir: str | Path,
        semantic_dir: str | Path,
        presentation_dir: str | Path,
        output_dir: str | Path,
        taxonomy_path: str | Path,
        fps: float = 10.0,
        selected_videos: list[int] | None = None,
) -> dict:
    frames_dir = Path(frames_dir)
    semantic_dir = Path(semantic_dir)
    presentation_dir = Path(presentation_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    frames = pd.read_parquet(
        frames_dir / "frames/frames.parquet").sort_values("frame_index")
    manifest = json.loads(
        (presentation_dir / "manifest.json").read_text())
    done = {int(key) for key in manifest.get("presentation_done", {})}
    frames = frames[frames.frame_index.isin(done)]
    if frames.empty:
        raise RuntimeError("presentation renderer found no completed frames")
    metrics = json.loads((presentation_dir / "metrics.json").read_text())
    taxonomy = Taxonomy.load(taxonomy_path)
    selected = (
        list(range(len(VIDEO_NAMES)))
        if selected_videos is None else sorted(set(selected_videos)))
    if not selected or any(
            index < 0 or index >= len(VIDEO_NAMES) for index in selected):
        raise ValueError("selected videos must be zero-based indices 0..4")
    internal_qa = _qa_candidates(
        semantic_dir, frames, INTERNAL_CLASS_IDS)
    lane_qa = _qa_candidates(
        semantic_dir, frames, THIN_CLASS_IDS)
    internal_ids = {row["frame_index"] for row in internal_qa}
    lane_ids = {row["frame_index"] for row in lane_qa}
    writers = {
        index: imageio.get_writer(
            output_dir / VIDEO_NAMES[index], fps=fps, codec="libx264",
            quality=8, macro_block_size=8)
        for index in selected}
    contacts = []
    try:
        for position, row in enumerate(frames.itertuples()):
            frame_id = int(row.frame_index)
            stem = f"frame_{frame_id:06d}"
            bgr = cv2.imread(str(frames_dir / row.rectified_path))
            if bgr is None:
                raise FileNotFoundError(frames_dir / row.rectified_path)
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            current = read_mask_u16(
                semantic_dir / "final_masks" / f"{stem}.png")
            improved = read_mask_u16(
                presentation_dir / "improved_masks" / f"{stem}.png")
            presentation = read_mask_u16(
                presentation_dir / "presentation_masks" / f"{stem}.png")
            metadata = json.loads(
                (presentation_dir / "metadata" / f"{stem}.json").read_text())
            products = {
                index: build_video_frame(
                    index, rgb, current, improved, presentation,
                    metadata, taxonomy, metrics)
                for index in selected}
            for index, writer in writers.items():
                writer.append_data(products[index])
            if 2 in products and \
                    position % max(1, len(frames) // 12) == 0:
                contacts.append(cv2.resize(products[2], (960, 280)))
            if 3 in products and frame_id in internal_ids:
                cv2.imwrite(
                    str(output_dir /
                        f"preview_internal_flicker_{frame_id:06d}.png"),
                    cv2.cvtColor(products[3], cv2.COLOR_RGB2BGR))
            if 4 in products and frame_id in lane_ids:
                cv2.imwrite(
                    str(output_dir /
                        f"preview_lane_flicker_{frame_id:06d}.png"),
                    cv2.cvtColor(products[4], cv2.COLOR_RGB2BGR))
    finally:
        for writer in writers.values():
            writer.close()
    if contacts:
        rows = [
            np.hstack(contacts[index:index + 2])
            for index in range(0, len(contacts), 2)]
        width = max(row.shape[1] for row in rows)
        rows = [
            cv2.copyMakeBorder(
                row, 0, 0, 0, width - row.shape[1],
                cv2.BORDER_CONSTANT)
            for row in rows]
        cv2.imwrite(
            str(output_dir / "contact_sheet.jpg"),
            cv2.cvtColor(np.vstack(rows), cv2.COLOR_RGB2BGR))
    render_manifest_path = output_dir / "render_manifest.json"
    previous_result = (
        json.loads(render_manifest_path.read_text())
        if render_manifest_path.exists() else {})
    completed = set(previous_result.get("videos", []))
    completed.update(VIDEO_NAMES[index] for index in selected)
    result = {
        "frames": len(frames),
        "fps": float(fps),
        "videos": [
            name for name in VIDEO_NAMES if name in completed],
        "presentation_fingerprint": manifest["fingerprint"],
        "metrics": metrics,
        "internal_flicker_qa": internal_qa,
        "lane_flicker_qa": lane_qa,
    }
    render_manifest_path.write_text(json.dumps(result, indent=2))
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--frames", required=True)
    parser.add_argument("--semantic-camera", required=True)
    parser.add_argument("--presentation", required=True)
    parser.add_argument(
        "--output",
        default="output/article1/semantic_camera_video_stabilized_30s")
    parser.add_argument(
        "--taxonomy", default="configs/article1/classes_article1.yaml")
    parser.add_argument("--fps", type=float, default=10)
    parser.add_argument(
        "--videos", default="1,2,3,4,5",
        help="comma-separated one-based video numbers to render")
    args = parser.parse_args()
    selected = [int(value.strip()) - 1
                for value in args.videos.split(",") if value.strip()]
    render(
        args.frames, args.semantic_camera, args.presentation,
        args.output, args.taxonomy, args.fps, selected)


if __name__ == "__main__":
    main()
