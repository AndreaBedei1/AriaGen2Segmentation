#!/usr/bin/env python3
"""Select reproducible visual-QA candidates; selections are not ground truth."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from aria_drive_seg.io_utils import read_mask_u16
from aria_drive_seg.taxonomy import Taxonomy


def top(rows, key, count):
    return [row["frame_index"] for row in
            sorted(rows, key=lambda row: row[key], reverse=True)[:count]]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--frames", required=True)
    parser.add_argument("--temporal", required=True)
    parser.add_argument("--output", default="reports/article1_temporal_qa_candidates.json")
    parser.add_argument("--contact-sheet", default="reports/article1_temporal_qa_contact.jpg")
    parser.add_argument("--taxonomy", default="configs/article1/classes_article1.yaml")
    args = parser.parse_args()
    frames_root, temporal = Path(args.frames), Path(args.temporal)
    frame_table = pd.read_parquet(
        frames_root / "frames/frames.parquet").sort_values("frame_index")
    records = []
    for row in frame_table.itertuples():
        frame_id, stem = int(row.frame_index), f"frame_{int(row.frame_index):06d}"
        raw = read_mask_u16(temporal / "static_masks" / f"{stem}.png")
        result = read_mask_u16(temporal / "temporal_masks" / f"{stem}.png")
        raw_thin = read_mask_u16(temporal / "static_thin" / f"{stem}.png")
        result_thin = read_mask_u16(temporal / "temporal_thin" / f"{stem}.png")
        metadata = json.loads(
            (temporal / "metadata" / f"{stem}.json").read_text())
        records.append({
            "frame_index": frame_id,
            "lane_pixels": int((raw_thin == 2).sum()),
            "lane_recovered": int(((raw_thin != 2) & (result_thin == 2)).sum()),
            "regulatory_pixels": int((raw_thin == 3).sum()),
            "vehicle_pixels": int((raw == 4).sum()),
            "pedestrian_two_wheeler_pixels": int(np.isin(raw, [5, 6]).sum()),
            "traffic_sign_pixels": int((raw == 8).sum()),
            "traffic_light_pixels": int((raw == 7).sum()),
            "raw_temporal_difference": float((raw != result).mean()),
            "flow_invalid_fraction": 1 - float(metadata["flow_valid_fraction"]),
            "reset": metadata["reset_reason_code"] != 0,
        })
    stable_lane = [
        row for row in records if row["lane_pixels"] > 0]
    stable_lane.sort(key=lambda row: (
        row["raw_temporal_difference"], -row["lane_pixels"]))
    selections = {
        "lane_stable": [row["frame_index"] for row in stable_lane[:10]],
        "lane_one_frame_recovery_candidates": top(records, "lane_recovered", 10),
        "regulatory_markings": top(records, "regulatory_pixels", 5),
        "vehicles": top(records, "vehicle_pixels", 10),
        "pedestrians_or_two_wheelers": top(
            records, "pedestrian_two_wheeler_pixels", 5),
        "traffic_signs": top(records, "traffic_sign_pixels", 5),
        "traffic_lights": top(records, "traffic_light_pixels", 5),
        "strong_motion_or_low_flow_validity": top(
            records, "flow_invalid_fraction", 10),
        "reflections_manual_candidates": [1840, 1865, 1880, 1905, 1920],
        "all_resets": [row["frame_index"] for row in records if row["reset"]],
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps({
        "disclaimer": "Candidate selection for manual visual QA; not GT.",
        "selections": selections, "per_frame": records}, indent=2))
    taxonomy = Taxonomy.load(args.taxonomy)
    selected = []
    for category, frame_ids in selections.items():
        for frame_id in frame_ids:
            if (category, frame_id) not in selected:
                selected.append((category, frame_id))
    thumbnails = []
    frame_indexed = frame_table.set_index("frame_index")
    for category, frame_id in selected:
        if frame_id not in frame_indexed.index:
            continue
        row = frame_indexed.loc[frame_id]
        stem = f"frame_{frame_id:06d}"
        rgb = cv2.cvtColor(
            cv2.imread(str(frames_root / row.rectified_path)), cv2.COLOR_BGR2RGB)
        raw = read_mask_u16(temporal / "static_masks" / f"{stem}.png")
        result = read_mask_u16(temporal / "temporal_masks" / f"{stem}.png")
        raw_color, result_color = taxonomy.colorize(raw), taxonomy.colorize(result)
        left = cv2.addWeighted(rgb, .58, raw_color, .42, 0)
        right = cv2.addWeighted(rgb, .58, result_color, .42, 0)
        panel = np.hstack([
            cv2.resize(left, (336, 252)), cv2.resize(right, (336, 252))])
        cv2.rectangle(panel, (0, 0), (panel.shape[1], 38), (10, 10, 10), -1)
        cv2.putText(panel, f"{category} | frame {frame_id} | raw / temporal",
                    (8, 25), cv2.FONT_HERSHEY_SIMPLEX, .42,
                    (255, 255, 255), 1, cv2.LINE_AA)
        thumbnails.append(panel)
    rows = [np.hstack(thumbnails[i:i + 3])
            for i in range(0, len(thumbnails), 3)]
    if rows:
        width = max(row.shape[1] for row in rows)
        rows = [cv2.copyMakeBorder(
            row, 0, 0, 0, width - row.shape[1], cv2.BORDER_CONSTANT)
            for row in rows]
        cv2.imwrite(args.contact_sheet, cv2.cvtColor(
            np.vstack(rows), cv2.COLOR_RGB2BGR))


if __name__ == "__main__":
    main()
