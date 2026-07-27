#!/usr/bin/env python3
"""Prepare unannotated Article 1 CVAT packages; never generates ground truth."""
from __future__ import annotations

import argparse
import csv
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

from aria_drive_seg.io_utils import atomic_write_json, atomic_write_text


EXTERNAL = [
    "road_surface", "lane_marking", "regulatory_road_marking", "vehicle",
    "two_wheeler", "pedestrian", "traffic_light", "traffic_sign",
    "road_boundary_or_obstacle", "other_environment", "unknown_void",
]
COCKPIT = ["mirror", "instrument_display", "control_and_ego_vehicle",
           "background_internal", "ignore"]


def write_csv(path, rows, fields):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    with tmp.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    tmp.replace(path)


def copy_images(root, frame_df, positions, sample_ids, package):
    image_dir = package / "images"
    image_dir.mkdir(parents=True, exist_ok=True)
    for pos, sample_id in zip(positions, sample_ids):
        row = frame_df.iloc[int(pos)]
        src = root / row.rectified_path
        shutil.copy2(src, image_dir / f"{sample_id}.jpg")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--session-id", required=True)
    ap.add_argument("--participant-id", default="unknown")
    ap.add_argument("--validation-root", default="validation/article1_auto_moto")
    args = ap.parse_args()
    root, val = Path(args.input), Path(args.validation_root)
    frames = pd.read_parquet(root / "frames" / "frames.parquet").sort_values("frame_index")
    gaze_path = root / "gaze" / "aligned_gaze.parquet"
    gaze = pd.read_parquet(gaze_path).set_index("frame_index") if gaze_path.exists() else None

    ext_pos = np.linspace(0, len(frames) - 1, min(20, len(frames)), dtype=int)
    ext_rows = []
    for number, pos in enumerate(ext_pos):
        row = frames.iloc[int(pos)]
        fi = int(row.frame_index)
        ext_rows.append({
            "sample_id": f"car_ext_{number:03d}", "vehicle_type": "car",
            "session_id": args.session_id, "participant_id": args.participant_id,
            "frame_index": fi, "capture_timestamp_ns": int(row.capture_timestamp_ns),
            "route_segment_id": "", "route_progress": "", "matched_pair_id": "",
            "scene_category": "to_review",
            "gaze_valid": (bool(gaze.loc[fi].valid) if gaze is not None and fi in gaze.index else ""),
            "lighting": "to_review", "traffic_density": "to_review",
            "annotation_split": "development_seeded",
            "selection_reason": "uniform temporal coverage; manual stratification pending",
        })
    ext_fields = list(ext_rows[0])
    write_csv(val / "external_dev40_manifest.csv", ext_rows, ext_fields)
    ext_pkg = val / "external_dev40" / "cvat_package"
    copy_images(root, frames, ext_pos, [row["sample_id"] for row in ext_rows], ext_pkg)
    atomic_write_json(ext_pkg / "labels.json", {"labels": EXTERNAL, "vehicle_frames": {"car": 20, "motorcycle": 0}})
    atomic_write_text(ext_pkg / "README.md",
        "# External dev40 CVAT package\n\n20 real car frames are populated; 20 motorcycle "
        "slots remain blocked. Import `images/` as an image task and create masks/polygons "
        "using `labels.json`. No generated mask is ground truth. Export CVAT for images 1.1 "
        "with masks, retain void/ignore, then run the conversion validator.\n")

    cock_pos = np.linspace(0, len(frames) - 1, min(60, len(frames)), dtype=int)
    cock_rows = []
    for number, pos in enumerate(cock_pos):
        row = frames.iloc[int(pos)]
        cock_rows.append({
            "sample_id": f"car_cockpit_{number:03d}", "vehicle_type": "car",
            "session_id": args.session_id, "participant_id": args.participant_id,
            "frame_index": int(row.frame_index),
            "capture_timestamp_ns": int(row.capture_timestamp_ns),
            "route_segment_id": "", "matched_pair_id": "",
            "lighting": "to_review", "road_geometry": "to_review",
            "hands_visible": "to_review", "mirrors_visible": "to_review",
            "gaze_context": "to_review", "annotation_split": "unassigned",
            "review_status": "not_annotated",
        })
    write_csv(val / "cockpit120_manifest.csv", cock_rows, list(cock_rows[0]))
    cock_pkg = val / "cockpit120" / "cvat_package"
    copy_images(root, frames, cock_pos, [row["sample_id"] for row in cock_rows], cock_pkg)
    atomic_write_json(cock_pkg / "labels.json", {
        "labels": COCKPIT,
        "attributes": {
            "mirror_side": ["left", "right", "central", "unknown"],
            "control_type": ["steering_wheel", "handlebar", "hand", "console", "ego_body", "other"],
            "display_type": ["instrument_cluster", "navigation", "central_display",
                             "motorcycle_display", "other"],
        },
        "vehicle_frames": {"car": len(cock_pos), "motorcycle": 0},
    })
    atomic_write_text(cock_pkg / "README.md",
        "# Cockpit120 CVAT package\n\n60 real car frames are populated. Motorcycle frames "
        "are absent and must be added before training. Attributes are metadata, not model "
        "classes. Grounded-SAM2 B/C proposals, if generated, belong under ignored `seeds/` "
        "and require manual review.\n")


if __name__ == "__main__":
    main()
