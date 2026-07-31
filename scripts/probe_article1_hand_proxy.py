#!/usr/bin/env python3
"""Diagnostic open-vocabulary hand probe over a segment's frames.

This is **not** part of the frozen Article 1 pipeline and does not modify it. The
frozen cockpit proxy carries a single, car-specific hand prompt ("human hand on
steering wheel"); this probe adds domain-appropriate phrasings so that the hand
audit can distinguish "the proxy has no suitable prompt" from "there is no hand in
the image".

The output is a proxy signal for the hand audit. It is not ground truth and no
accuracy is derived from it.

Runs in the ML environment.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from aria_drive_seg.config import Config
from aria_drive_seg.logging_utils import get_logger, setup_logging
from aria_drive_seg.taxonomy import Taxonomy

log = get_logger("hand.probe")

# Domain-appropriate phrasings, deliberately broader than the frozen prompt.
HAND_CAPTIONS = {
    "car": "a human hand. a hand on the steering wheel. a forearm.",
    "motorcycle": ("a human hand. a hand on the handlebar. a gloved hand. "
                   "a hand gripping a motorcycle grip. a forearm."),
}
BOX_THRESHOLD = 0.25
TEXT_THRESHOLD = 0.20


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", required=True, help="extracted run containing frames/")
    ap.add_argument("--output", required=True)
    ap.add_argument("--domain", required=True, choices=["car", "motorcycle"])
    ap.add_argument("--config", default="configs/article1/semantic_camera.yaml")
    ap.add_argument("--box-threshold", type=float, default=BOX_THRESHOLD)
    ap.add_argument("--text-threshold", type=float, default=TEXT_THRESHOLD)
    ap.add_argument("--save-masks", action="store_true")
    ap.add_argument("--log-level", default="INFO")
    args = ap.parse_args()
    setup_logging(args.log_level)

    import cv2
    from PIL import Image

    cfg = Config.load(args.config)
    taxonomy = Taxonomy.load(cfg.resolve(cfg.get("project.classes")))
    from aria_drive_seg.segmentation.grounded_sam2 import GroundedSAM2Segmenter
    segmenter = GroundedSAM2Segmenter(cfg, taxonomy)
    segmenter.load()

    root = Path(args.frames)
    frames = pd.read_parquet(root / "frames" / "frames.parquet") \
        .sort_values("source_frame_index")
    frames = frames[frames["valid"]]
    caption = HAND_CAPTIONS[args.domain]
    log.info("probing %d frames with caption %r", len(frames), caption)

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    mask_dir = out / "masks"
    if args.save_masks:
        mask_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for n, (_, row) in enumerate(frames.iterrows()):
        bgr = cv2.imread(str(root / row["rectified_path"]), cv2.IMREAD_COLOR)
        if bgr is None:
            continue
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        h, w = rgb.shape[:2]
        boxes, scores, labels = segmenter._run_gdino(
            Image.fromarray(rgb), caption, args.box_threshold, args.text_threshold, h, w)

        detections = []
        union = np.zeros((h, w), dtype=bool)
        if len(boxes):
            segmenter._sam.set_image(rgb)
            masks, ious, _ = segmenter._sam.predict(
                box=np.asarray(boxes, dtype=np.float32), multimask_output=False)
            masks = np.asarray(masks).reshape(-1, h, w) > 0.5
            for b, s, lab, m, iou in zip(boxes, scores, labels, masks,
                                         np.asarray(ious).reshape(-1)):
                area = int(m.sum())
                detections.append({
                    "box": [float(x) for x in b], "score": float(s),
                    "phrase": str(lab), "sam_iou": float(iou),
                    "area_px": area, "area_fraction": area / (h * w),
                    "centroid_x": float(np.mean(np.where(m)[1])) if area else None,
                    "centroid_y": float(np.mean(np.where(m)[0])) if area else None,
                })
                union |= m

        if args.save_masks and union.any():
            cv2.imwrite(str(mask_dir / f"frame_{int(row['source_frame_index']):06d}.png"),
                        (union.astype(np.uint8) * 255))

        rows.append({
            "recording_id": row["recording_id"], "domain": row["domain"],
            "source_frame_index": int(row["source_frame_index"]),
            "timestamp_ns": int(row["timestamp_ns"]),
            "proxy_available": True,
            "hand_detections": len(detections),
            "hand_mask_present": bool(union.any()),
            "hand_area_fraction": float(union.mean()),
            "best_score": max((d["score"] for d in detections), default=0.0),
            "detections_json": json.dumps(detections),
        })
        if (n + 1) % 50 == 0:
            log.info("  %d/%d", n + 1, len(frames))

    df = pd.DataFrame(rows)
    df.to_parquet(out / "hand_proxy.parquet", index=False)
    df.to_csv(out / "hand_proxy.csv", index=False)
    summary = {
        "status": "diagnostic_proxy_not_ground_truth",
        "part_of_frozen_pipeline": False,
        "domain": args.domain,
        "caption": caption,
        "box_threshold": args.box_threshold,
        "text_threshold": args.text_threshold,
        "frames": len(df),
        "frames_with_hand_region": int(df["hand_mask_present"].sum()) if len(df) else 0,
        "mean_hand_area_fraction": float(df["hand_area_fraction"].mean()) if len(df) else 0.0,
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
