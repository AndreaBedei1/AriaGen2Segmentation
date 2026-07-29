#!/usr/bin/env python3
"""Render dense semantic-camera and external/internal/fusion comparison videos."""
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


def overlay(rgb: np.ndarray, mask: np.ndarray, taxonomy: Taxonomy,
            alpha: float = .45) -> np.ndarray:
    color = taxonomy.colorize(mask)
    result = rgb.copy()
    selected = mask > 0
    result[selected] = (
        result[selected] * (1 - alpha)
        + color[selected] * alpha).astype(np.uint8)
    edges = cv2.Canny(
        (mask.astype(np.uint16) % 256).astype(np.uint8), 0, 1)
    result[edges > 0] = 255
    return result


def _text(image: np.ndarray, value: str, xy: tuple[int, int],
          scale=.55, color=(255, 255, 255), thickness=1) -> None:
    cv2.putText(
        image, value, xy, cv2.FONT_HERSHEY_SIMPLEX, scale, color,
        thickness, cv2.LINE_AA)


def _legend(image: np.ndarray, taxonomy: Taxonomy,
            mask: np.ndarray) -> None:
    present = [
        cid for cid in range(1, taxonomy.max_id + 1)
        if np.any(mask == cid)]
    y0 = image.shape[0] - 70
    cv2.rectangle(
        image, (0, y0), (image.shape[1], image.shape[0]),
        (12, 15, 18), -1)
    for index, cid in enumerate(present[:13]):
        column, row = index % 7, index // 7
        x, y = 12 + column * 180, y0 + 22 + row * 30
        color = tuple(int(x) for x in taxonomy.by_id[cid].color)
        cv2.rectangle(image, (x, y - 13), (x + 18, y + 5), color, -1)
        _text(
            image, taxonomy.name_of(cid), (x + 24, y + 3),
            .40, (235, 240, 245))


def build_video_frames(
        rgb: np.ndarray, external: np.ndarray, internal: np.ndarray,
        final: np.ndarray, metadata: dict, taxonomy: Taxonomy
        ) -> tuple[np.ndarray, np.ndarray]:
    dense = cv2.resize(
        overlay(rgb, final, taxonomy), (1280, 960),
        interpolation=cv2.INTER_AREA)
    cv2.rectangle(dense, (0, 0), (1280, 82), (12, 15, 18), -1)
    _text(
        dense,
        f"SEMANTIC CAMERA DENSE | frame {metadata['frame_index']} | "
        f"ts {metadata['capture_timestamp_ns']}",
        (14, 30), .67, thickness=2)
    _text(
        dense,
        f"coverage=100.00%  external={100 * metadata['external_selected_fraction']:.2f}%  "
        f"internal={100 * metadata['internal_model_selected_fraction']:.2f}%  "
        f"geometry={100 * metadata['geometric_proxy_selected_fraction']:.2f}%  "
        f"fill={100 * metadata['dense_fill_fraction']:.2f}%",
        (14, 62), .46, (210, 230, 255))
    _legend(dense, taxonomy, final)

    names = ("EXTERNAL MASK2FORMER", "INTERNAL / PROXY", "FINAL FUSION")
    masks = (external, internal, final)
    panels = []
    for name, mask in zip(names, masks):
        view = cv2.resize(
            overlay(rgb, mask, taxonomy), (640, 480),
            interpolation=cv2.INTER_AREA)
        panels.append(view)
    comparison = np.zeros((560, 1920, 3), np.uint8)
    comparison[80:] = np.hstack(panels)
    for index, name in enumerate(names):
        _text(comparison, name, (index * 640 + 14, 31), .64, thickness=2)
    _text(
        comparison,
        f"frame {metadata['frame_index']} | internal={metadata['internal_source']} "
        f"(fallback={metadata['internal_is_fallback']}) | "
        f"conflicts={100 * metadata['conflict_fraction']:.2f}%",
        (14, 64), .47, (210, 230, 255))
    return dense, comparison


def render(
        frames_dir: str | Path, semantic_dir: str | Path,
        output_dir: str | Path, taxonomy_path: str | Path,
        fps: float = 10.0) -> dict:
    frames_dir, semantic_dir, output_dir = (
        Path(frames_dir), Path(semantic_dir), Path(output_dir))
    output_dir.mkdir(parents=True, exist_ok=True)
    frames = pd.read_parquet(
        frames_dir / "frames/frames.parquet").sort_values("frame_index")
    manifest = json.loads((semantic_dir / "manifest.json").read_text())
    done = {int(key) for key in manifest.get("done", {})}
    frames = frames[frames.frame_index.isin(done)]
    if frames.empty:
        raise RuntimeError("semantic-camera renderer found no completed frames")
    taxonomy = Taxonomy.load(taxonomy_path)
    names = (
        "01_semantic_camera_dense.mp4",
        "02_external_internal_fusion.mp4",
    )
    writers = [
        imageio.get_writer(
            output_dir / name, fps=fps, codec="libx264", quality=8,
            macro_block_size=8)
        for name in names
    ]
    contacts = []
    best = {
        "internal": (-1.0, None),
        "fill": (-1.0, None),
        "conflict": (-1.0, None),
    }
    qa = []
    try:
        for position, row in enumerate(frames.itertuples()):
            frame_id = int(row.frame_index)
            stem = f"frame_{frame_id:06d}"
            bgr = cv2.imread(str(frames_dir / row.rectified_path))
            if bgr is None:
                raise FileNotFoundError(frames_dir / row.rectified_path)
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            external = read_mask_u16(
                semantic_dir / "external_masks" / f"{stem}.png")
            internal = read_mask_u16(
                semantic_dir / "internal_masks" / f"{stem}.png")
            final = read_mask_u16(
                semantic_dir / "final_masks" / f"{stem}.png")
            metadata = json.loads(
                (semantic_dir / "metadata" / f"{stem}.json").read_text())
            dense, comparison = build_video_frames(
                rgb, external, internal, final, metadata, taxonomy)
            writers[0].append_data(dense)
            writers[1].append_data(comparison)
            if position % max(1, len(frames) // 12) == 0:
                contacts.append(cv2.resize(comparison, (960, 280)))
            criteria = {
                "internal": (
                    metadata["internal_model_selected_fraction"]
                    + metadata["geometric_proxy_selected_fraction"]),
                "fill": metadata["dense_fill_fraction"],
                "conflict": metadata["conflict_fraction"],
            }
            for key, score in criteria.items():
                if score > best[key][0]:
                    best[key] = (float(score), comparison.copy())
            qa.append({
                "frame_index": frame_id,
                "capture_timestamp_ns": int(row.capture_timestamp_ns),
                **criteria,
            })
    finally:
        for writer in writers:
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
    for key, (_, image) in best.items():
        if image is not None:
            cv2.imwrite(
                str(output_dir / f"preview_max_{key}.png"),
                cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
    result = {
        "frames": len(frames),
        "fps": float(fps),
        "videos": list(names),
        "source_manifest_fingerprint": manifest["fingerprint"],
        "qa": qa,
    }
    (output_dir / "render_manifest.json").write_text(
        json.dumps(result, indent=2))
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--frames", required=True)
    parser.add_argument("--semantic-camera", required=True)
    parser.add_argument(
        "--output", default="output/article1/semantic_camera_30s")
    parser.add_argument(
        "--taxonomy", default="configs/article1/classes_article1.yaml")
    parser.add_argument("--fps", type=float, default=10)
    args = parser.parse_args()
    render(
        args.frames, args.semantic_camera, args.output,
        args.taxonomy, args.fps)


if __name__ == "__main__":
    main()
