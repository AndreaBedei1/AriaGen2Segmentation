#!/usr/bin/env python3
"""Render the single final video and a lightweight tracked QA image package."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

import cv2
import imageio.v2 as imageio
import numpy as np
import pandas as pd

from aria_drive_seg.article1.semantic_camera_final import (
    BOUNDARY_ID,
    FINAL_PROVENANCE,
    INTERNAL_CLASS_IDS,
    LINE_CLASS_IDS,
)
from aria_drive_seg.io_utils import read_mask_u16
from aria_drive_seg.taxonomy import Taxonomy


FINAL_VIDEO_NAME = "semantic_camera_final_30s.mp4"
QA_CATEGORIES = (
    (
        "01_lane_dropout",
        "Linee stradali che scompaiono e ricompaiono",
        "La finestra flow-aligned ±10 recupera evidenza lane nei frame "
        "intermedi e il filtro geometrico la limita al supporto stradale.",
    ),
    (
        "02_broken_lane_link",
        "Linee spezzate con gap compatibili",
        "Closing multi-orientamento e collegamento controllato degli estremi "
        "riducono piccoli gap senza attraversare barriere semantiche.",
    ),
    (
        "03_intermittent_boundary",
        "Cordolo o bordo strada intermittente",
        "Persistenza temporale, prossimità al bordo strada e pulizia delle "
        "componenti rendono più continuo road_boundary_or_obstacle.",
    ),
    (
        "04_internal_flicker",
        "Cockpit, mani, specchi o display con flicker",
        "Persistenza lunga e isteresi per le classi interne recuperano i "
        "dropout; le mani restano nella classe control_and_ego_vehicle.",
    ),
    (
        "05_internal_external_conflict",
        "Conflitto tra interno ed esterno",
        "La fusione conserva le regioni esterne forti e vieta ai collegamenti "
        "di attraversare cockpit, veicoli o pedoni.",
    ),
)


def _text(
        image: np.ndarray,
        value: str,
        xy: tuple[int, int],
        scale: float = .48,
        color: tuple[int, int, int] = (245, 248, 252),
        thickness: int = 1,
) -> None:
    cv2.putText(
        image, value, xy, cv2.FONT_HERSHEY_SIMPLEX,
        scale, color, thickness, cv2.LINE_AA)


def final_overlay(
        rgb: np.ndarray,
        mask: np.ndarray,
        taxonomy: Taxonomy,
        alpha: float = .44,
) -> np.ndarray:
    color = taxonomy.colorize(mask)
    result = rgb.copy()
    selected = mask > 0
    result[selected] = (
        result[selected] * (1.0 - alpha)
        + color[selected] * alpha).astype(np.uint8)
    edges = cv2.Canny(mask.astype(np.uint8), 0, 1)
    result[edges > 0] = np.asarray([235, 240, 245], np.uint8)
    return result


def _legend(
        image: np.ndarray,
        mask: np.ndarray,
        taxonomy: Taxonomy,
) -> None:
    present = [
        class_id for class_id in range(1, taxonomy.max_id + 1)
        if np.any(mask == class_id)]
    rows = 2 if len(present) > 7 else 1
    height = 28 * rows + 12
    y0 = image.shape[0] - height
    overlay = image.copy()
    cv2.rectangle(
        overlay, (0, y0), (image.shape[1], image.shape[0]),
        (12, 15, 18), -1)
    cv2.addWeighted(overlay, .76, image, .24, 0, image)
    columns = 7
    column_width = image.shape[1] // columns
    for index, class_id in enumerate(present[:14]):
        column, row = index % columns, index // columns
        x = 10 + column * column_width
        y = y0 + 24 + row * 28
        color = tuple(
            int(value) for value in taxonomy.by_id[class_id].color)
        cv2.rectangle(image, (x, y - 12), (x + 15, y + 3), color, -1)
        _text(
            image, taxonomy.name_of(class_id), (x + 21, y + 2),
            .32, (236, 240, 244))


def build_final_video_frame(
        rgb: np.ndarray,
        mask: np.ndarray,
        taxonomy: Taxonomy,
        frame_index: int,
) -> np.ndarray:
    result = cv2.resize(
        final_overlay(rgb, mask, taxonomy),
        (1280, 960), interpolation=cv2.INTER_AREA)
    banner = result.copy()
    cv2.rectangle(banner, (0, 0), (215, 42), (12, 15, 18), -1)
    cv2.addWeighted(banner, .78, result, .22, 0, result)
    _text(
        result, f"frame {frame_index:06d}", (12, 28),
        .58, thickness=2)
    _legend(result, mask, taxonomy)
    return result


def render_final_video(
        frames_dir: str | Path,
        final_dir: str | Path,
        output_path: str | Path,
        taxonomy_path: str | Path,
        fps: float = 10.0,
) -> dict[str, Any]:
    frames_dir = Path(frames_dir)
    final_dir = Path(final_dir)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frames = pd.read_parquet(
        frames_dir / "frames/frames.parquet").sort_values("frame_index")
    manifest = json.loads((final_dir / "manifest.json").read_text())
    done = {int(value) for value in manifest.get("done", {})}
    frames = frames[frames.frame_index.isin(done)]
    if frames.empty:
        raise RuntimeError("final video renderer found no completed frames")
    taxonomy = Taxonomy.load(taxonomy_path)
    writer = imageio.get_writer(
        output_path, fps=fps, codec="libx264",
        quality=8, macro_block_size=8)
    try:
        for row in frames.itertuples():
            frame_id = int(row.frame_index)
            stem = f"frame_{frame_id:06d}"
            bgr = cv2.imread(str(frames_dir / row.rectified_path))
            if bgr is None:
                raise FileNotFoundError(frames_dir / row.rectified_path)
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            mask = read_mask_u16(
                final_dir / "final_masks" / f"{stem}.png")
            writer.append_data(build_final_video_frame(
                rgb, mask, taxonomy, frame_id))
    finally:
        writer.close()
    capture = cv2.VideoCapture(str(output_path))
    decoded = 0
    while True:
        ok, _ = capture.read()
        if not ok:
            break
        decoded += 1
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    capture.release()
    if decoded != len(frames):
        raise RuntimeError(
            f"final video decoded {decoded}/{len(frames)} frames")
    return {
        "path": str(output_path),
        "frames": decoded,
        "fps": float(fps),
        "duration_seconds": decoded / fps,
        "geometry": [height, width],
        "bytes": output_path.stat().st_size,
    }


def _flicker_scores(
        masks: Sequence[np.ndarray],
        class_ids: np.ndarray,
) -> np.ndarray:
    scores = np.zeros(len(masks), np.int64)
    for index in range(1, len(masks) - 1):
        previous = masks[index - 1]
        current = masks[index]
        following = masks[index + 1]
        scores[index] = int(
            ((previous == following)
             & np.isin(previous, class_ids)
             & (current != previous)).sum())
    return scores


def _sequence_bounds(
        center: int,
        frame_count: int,
        length: int = 10,
) -> tuple[int, int]:
    start = max(0, min(center - length // 2, frame_count - length))
    return start, min(frame_count, start + length)


def select_qa_sequences(
        previous_masks: Sequence[np.ndarray],
        final_masks: Sequence[np.ndarray],
        external_masks: Sequence[np.ndarray],
        final_provenance: Sequence[np.ndarray],
) -> list[dict[str, Any]]:
    if not previous_masks or any(
            len(values) != len(previous_masks) for values in (
                final_masks, external_masks, final_provenance)):
        raise ValueError("QA selection sequences must be non-empty and aligned")
    lane_dropout = _flicker_scores(previous_masks, LINE_CLASS_IDS)
    morphology = np.asarray([
        int((value == 3).sum()) for value in final_provenance])
    boundary = _flicker_scores(
        previous_masks, np.asarray([BOUNDARY_ID], np.uint16))
    internal = _flicker_scores(previous_masks, INTERNAL_CLASS_IDS)
    conflict = np.asarray([
        int((
            np.isin(previous_mask, INTERNAL_CLASS_IDS)
            & (external_mask != previous_mask)
            & (external_mask > 0)).sum())
        for previous_mask, external_mask
        in zip(previous_masks, external_masks)])
    score_sets = (
        lane_dropout, morphology, boundary, internal, conflict)
    result = []
    used_centers: list[int] = []
    for category, scores in zip(QA_CATEGORIES, score_sets):
        order = np.argsort(-scores, kind="stable")
        center = int(order[0])
        for candidate in order:
            candidate = int(candidate)
            if all(abs(candidate - used) >= 5 for used in used_centers):
                center = candidate
                break
        used_centers.append(center)
        start, end = _sequence_bounds(center, len(previous_masks), 10)
        result.append({
            "slug": category[0],
            "problem": category[1],
            "correction": category[2],
            "center_position": center,
            "start_position": start,
            "end_position": end,
            "score_pixels_at_qa_scale": int(scores[center]),
        })
    return result


def _save_jpg(path: Path, rgb: np.ndarray, quality: int = 86) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(
            str(path), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR),
            [cv2.IMWRITE_JPEG_QUALITY, quality]):
        raise RuntimeError(f"could not write {path}")


def _save_png(path: Path, rgb: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(
            str(path), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR),
            [cv2.IMWRITE_PNG_COMPRESSION, 7]):
        raise RuntimeError(f"could not write {path}")


def _contact_row(
        rgb: np.ndarray,
        previous_mask: np.ndarray,
        final_mask: np.ndarray,
        taxonomy: Taxonomy,
        frame_id: int,
) -> np.ndarray:
    size = (360, 270)
    panels = [
        cv2.resize(rgb, size, interpolation=cv2.INTER_AREA),
        cv2.resize(
            final_overlay(rgb, previous_mask, taxonomy),
            size, interpolation=cv2.INTER_AREA),
        cv2.resize(
            final_overlay(rgb, final_mask, taxonomy),
            size, interpolation=cv2.INTER_AREA),
    ]
    row = np.hstack(panels)
    cv2.rectangle(row, (0, 0), (row.shape[1], 30), (12, 15, 18), -1)
    for index, label in enumerate(("RGB", "precedente", "finale")):
        _text(
            row, label, (index * size[0] + 8, 21),
            .43, thickness=1)
    _text(
        row, f"{frame_id:06d}", (row.shape[1] - 92, 21),
        .39, (210, 230, 255))
    return row


def build_qa_package(
        frames_dir: str | Path,
        semantic_dir: str | Path,
        previous_dir: str | Path,
        final_dir: str | Path,
        output_dir: str | Path,
        taxonomy_path: str | Path,
        source_start_seconds: float = 180.0,
) -> dict[str, Any]:
    frames_dir = Path(frames_dir)
    semantic_dir = Path(semantic_dir)
    previous_dir = Path(previous_dir)
    final_dir = Path(final_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    frames = pd.read_parquet(
        frames_dir / "frames/frames.parquet").sort_values(
            "frame_index").reset_index(drop=True)
    taxonomy = Taxonomy.load(taxonomy_path)
    qa_size = (504, 378)
    previous_small = []
    final_small = []
    external_small = []
    provenance_small = []
    for frame_id in frames.frame_index:
        stem = f"frame_{int(frame_id):06d}"
        previous_small.append(cv2.resize(
            read_mask_u16(
                previous_dir / "presentation_masks" / f"{stem}.png"),
            qa_size, interpolation=cv2.INTER_NEAREST))
        final_small.append(cv2.resize(
            read_mask_u16(final_dir / "final_masks" / f"{stem}.png"),
            qa_size, interpolation=cv2.INTER_NEAREST))
        external_small.append(cv2.resize(
            read_mask_u16(
                semantic_dir / "external_masks" / f"{stem}.png"),
            qa_size, interpolation=cv2.INTER_NEAREST))
        provenance = cv2.imread(
            str(final_dir / "final_provenance" / f"{stem}.png"),
            cv2.IMREAD_UNCHANGED)
        if provenance is None:
            raise FileNotFoundError(
                final_dir / "final_provenance" / f"{stem}.png")
        provenance_small.append(cv2.resize(
            provenance, qa_size, interpolation=cv2.INTER_NEAREST))
    selections = select_qa_sequences(
        previous_small, final_small, external_small, provenance_small)
    root_lines = [
        "# Article 1 semantic-camera final QA",
        "",
        "Preview leggere della sola modalità finale offline/presentation. "
        "Il risultato precedente è la presentazione stabilizzata del branch "
        "di partenza; gli output scientifici raw sono rimasti invariati.",
        "",
        "| Sequenza | Frame | Timestamp clip | Motivo | Contact sheet |",
        "|---|---:|---:|---|---|",
    ]
    package_rows = []
    first_timestamp = int(frames.capture_timestamp_ns.iloc[0])
    for selection in selections:
        sequence_dir = output_dir / selection["slug"]
        sequence_dir.mkdir(parents=True, exist_ok=True)
        positions = list(range(
            selection["start_position"], selection["end_position"]))
        rows = []
        frame_rows = []
        for position in positions:
            row = frames.iloc[position]
            frame_id = int(row.frame_index)
            stem = f"frame_{frame_id:06d}"
            bgr = cv2.imread(str(frames_dir / row.rectified_path))
            if bgr is None:
                raise FileNotFoundError(frames_dir / row.rectified_path)
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            previous_mask = read_mask_u16(
                previous_dir / "presentation_masks" / f"{stem}.png")
            final_mask = read_mask_u16(
                final_dir / "final_masks" / f"{stem}.png")
            rows.append(_contact_row(
                rgb, previous_mask, final_mask, taxonomy, frame_id))
            frame_rows.append((
                frame_id, int(row.capture_timestamp_ns), rgb,
                previous_mask, final_mask))
        contact = np.vstack(rows)
        _save_jpg(sequence_dir / "contact_sheet.jpg", contact, 84)
        selected_positions = sorted(set(
            [0, len(frame_rows) // 2, len(frame_rows) - 1]))
        individual_links = []
        for selected_position in selected_positions:
            frame_id, timestamp, rgb, _, final_mask = \
                frame_rows[selected_position]
            preview_size = (960, 720)
            rgb_preview = cv2.resize(
                rgb, preview_size, interpolation=cv2.INTER_AREA)
            mask_preview = cv2.resize(
                taxonomy.colorize(final_mask), preview_size,
                interpolation=cv2.INTER_NEAREST)
            overlay_preview = cv2.resize(
                final_overlay(rgb, final_mask, taxonomy),
                preview_size, interpolation=cv2.INTER_AREA)
            base = f"frame_{frame_id:06d}"
            _save_jpg(sequence_dir / f"{base}_rgb.jpg", rgb_preview)
            _save_png(sequence_dir / f"{base}_mask.png", mask_preview)
            _save_jpg(
                sequence_dir / f"{base}_overlay.jpg", overlay_preview)
            individual_links.append({
                "frame_index": frame_id,
                "capture_timestamp_ns": timestamp,
                "rgb": f"{base}_rgb.jpg",
                "mask": f"{base}_mask.png",
                "overlay": f"{base}_overlay.jpg",
            })
        start_frame = frame_rows[0][0]
        end_frame = frame_rows[-1][0]
        center_row = frame_rows[len(frame_rows) // 2]
        clip_time = source_start_seconds + (
            center_row[1] - first_timestamp) / 1e9
        readme_lines = [
            f"# {selection['problem']}",
            "",
            f"- Frame: {start_frame}–{end_frame} (10 consecutivi)",
            f"- Centro: {center_row[0]}",
            f"- Timestamp sorgente: {center_row[1]} ns",
            f"- Timestamp clip: {clip_time:.3f} s",
            f"- Score di selezione: "
            f"{selection['score_pixels_at_qa_scale']} pixel alla scala QA",
            "",
            selection["correction"],
            "",
            "## Contact sheet",
            "",
            "[contact_sheet.jpg](contact_sheet.jpg)",
            "",
            "## Frame individuali",
            "",
        ]
        for item in individual_links:
            readme_lines.extend([
                f"### Frame {item['frame_index']} "
                f"({item['capture_timestamp_ns']} ns)",
                "",
                f"- [RGB]({item['rgb']})",
                f"- [Maschera finale colorata]({item['mask']})",
                f"- [Overlay finale]({item['overlay']})",
                "",
            ])
        (sequence_dir / "README.md").write_text(
            "\n".join(readme_lines), encoding="utf-8")
        relative = selection["slug"]
        root_lines.append(
            f"| [{selection['problem']}]({relative}/README.md) | "
            f"{start_frame}–{end_frame} | {clip_time:.3f} s | "
            f"{selection['correction']} | "
            f"[JPG]({relative}/contact_sheet.jpg) |")
        package_rows.append({
            **selection,
            "frame_range": [start_frame, end_frame],
            "center_frame": center_row[0],
            "capture_timestamp_ns": center_row[1],
            "source_timestamp_seconds": clip_time,
            "contact_sheet": f"{relative}/contact_sheet.jpg",
            "individuals": individual_links,
        })
    root_lines.extend([
        "",
        "Le maschere sono PNG; RGB, overlay e contact sheet sono JPG. "
        "Le preview individuali sono 960×720 e non includono l’intero output.",
        "",
        "La selezione è diagnostica e non implica un miglioramento di "
        "accuratezza in assenza di ground truth.",
        "",
        "Codici provenance del final pass: " + ", ".join(
            f"`{code}` {name}" for code, name
            in FINAL_PROVENANCE.items() if code),
        "",
    ])
    (output_dir / "README.md").write_text(
        "\n".join(root_lines), encoding="utf-8")
    result = {
        "sequences": package_rows,
        "sequence_count": len(package_rows),
        "output_dir": str(output_dir),
    }
    (output_dir / "selection.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--frames", required=True)
    parser.add_argument("--semantic-camera", required=True)
    parser.add_argument("--previous-presentation", required=True)
    parser.add_argument("--final", required=True)
    parser.add_argument(
        "--video",
        default="output/article1/semantic_camera_final_30s.mp4")
    parser.add_argument(
        "--qa",
        default="reports/article1_semantic_camera_final_qa")
    parser.add_argument(
        "--taxonomy",
        default="configs/article1/classes_article1.yaml")
    parser.add_argument("--fps", type=float, default=10.0)
    parser.add_argument(
        "--source-start-seconds", type=float, default=180.0)
    args = parser.parse_args()
    video = render_final_video(
        args.frames, args.final, args.video, args.taxonomy, args.fps)
    qa = build_qa_package(
        args.frames, args.semantic_camera, args.previous_presentation,
        args.final, args.qa, args.taxonomy, args.source_start_seconds)
    print(json.dumps({"video": video, "qa": qa}, indent=2))


if __name__ == "__main__":
    main()
