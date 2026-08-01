#!/usr/bin/env python3
"""Phase 1: audit every geometric transform from the VRS to the final video.

Nothing is assumed. The retained region is computed by pushing every source pixel
through the real transform and asking where it lands, so this script can equally
well conclude "no field of view is lost".

Runs in the VRS I/O environment.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd

from aria_drive_seg.config import Config
from aria_drive_seg.geometry.audit import (angular_coverage, detect_centre_crop,
                                           edge_losses, model_valid_mask,
                                           retained_mask, summarise_stage)
from aria_drive_seg.io_utils import atomic_write_json
from aria_drive_seg.logging_utils import get_logger, setup_logging
from aria_drive_seg.vrs.provider import AriaProvider, RectifyParams

log = get_logger("geometry.audit")


def _overlay(source_rgb: np.ndarray, valid: np.ndarray, retained: np.ndarray,
             out: Path) -> Dict[str, Any]:
    """Draw the valid region, the retained region and what was discarded."""
    import cv2

    image = source_rgb.copy()
    lost = valid & ~retained
    outside = ~valid

    tint = image.astype(np.float32)
    # discarded but real scene content: red
    tint[lost] = 0.45 * tint[lost] + 0.55 * np.array([255, 40, 40], np.float32)
    # outside the camera model at all: grey
    tint[outside] = 0.55 * tint[outside] + 0.45 * np.array([110, 110, 110], np.float32)
    image = tint.astype(np.uint8)

    for mask, colour in ((valid, (255, 255, 0)), (retained, (0, 255, 0))):
        edges = cv2.Canny(mask.astype(np.uint8) * 255, 0, 1)
        image[edges > 0] = colour

    cv2.putText(image, "yellow = camera-model valid region", (20, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 0), 3)
    cv2.putText(image, "green = region the pipeline keeps", (20, 100),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
    cv2.putText(image, "red = valid scene content discarded", (20, 150),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 40, 40), 3)

    out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out), image[:, :, ::-1], [cv2.IMWRITE_JPEG_QUALITY, 90])
    return {"path": str(out),
            "lost_pixels": int(np.count_nonzero(lost)),
            "outside_model_pixels": int(np.count_nonzero(outside))}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--vrs", required=True)
    ap.add_argument("--frame-index", type=int, default=13950)
    ap.add_argument("--reports",
                    default="reports/article1_motorcycle_fov_mirror_refinement")
    ap.add_argument("--baseline-run",
                    default="output/article1/motorcycle_baseline_30s")
    ap.add_argument("--video",
                    default="output/article1/motorcycle_baseline_30s/"
                            "semantic_camera_moto_final.mp4")
    ap.add_argument("--config", default=None)
    ap.add_argument("--log-level", default="INFO")
    args = ap.parse_args()
    setup_logging(args.log_level)

    import cv2

    reports = Path(args.reports)
    geometry_dir = reports / "geometry"
    geometry_dir.mkdir(parents=True, exist_ok=True)

    cfg = Config.load(args.config)
    provider = AriaProvider(args.vrs, cfg.get("vrs.time_domain", "DEVICE_TIME"))
    rgb_label = cfg.get("vrs.rgb_label", "camera-rgb")

    source_calib = provider.rgb_calib(rgb_label)
    sw, sh = (int(x) for x in source_calib.get_image_size())
    log.info("source geometry %dx%d, model %s", sw, sh, source_calib.get_model_name())

    rp = RectifyParams(
        out_width=int(cfg.get("rectify.out_width", 2016)),
        out_height=int(cfg.get("rectify.out_height", 1512)),
        focal=float(cfg.get("rectify.focal", 879.0)),
        rotate_ccw90=int(cfg.get("rectify.rotate_ccw90", 0)),
    )
    pinhole = provider.make_pinhole(rp, rgb_label)

    log.info("computing the camera-model valid region")
    valid = model_valid_mask(source_calib, sw, sh)
    log.info("computing the region the rectifier keeps")
    retained = retained_mask(source_calib, pinhole, (sw, sh), (rp.out_width, rp.out_height))

    raw, timestamp = provider.rgb_by_index(args.frame_index, rgb_label)
    cv2.imwrite(str(geometry_dir / "source_frame.jpg"), raw[:, :, ::-1],
                [cv2.IMWRITE_JPEG_QUALITY, 92])
    overlay_info = _overlay(raw, valid, retained,
                            geometry_dir / "retained_region_overlay.jpg")

    rectified = provider.rectifier(rp, rgb_label).rectify(raw)
    cv2.imwrite(str(geometry_dir / "rectified_frame.jpg"), rectified[:, :, ::-1],
                [cv2.IMWRITE_JPEG_QUALITY, 92])

    # ------------------------------------------------------------------ #
    # the chain
    # ------------------------------------------------------------------ #
    stages: List[Any] = []
    stages.append(summarise_stage(
        "01_vrs_rgb", "raw RGB frame decoded from the VRS",
        "identity", "identity", (sw, sh), (sw, sh), valid, valid))
    stages[-1].notes.append(
        f"camera model {source_calib.get_model_name()}, focal "
        f"{source_calib.get_focal_lengths()[0]:.2f}, principal point "
        f"{tuple(round(float(x), 1) for x in source_calib.get_principal_point())}")

    stages.append(summarise_stage(
        "02_rectify_pinhole",
        f"fisheye624 -> linear pinhole at focal {rp.focal}",
        "unproject(fisheye) then project(pinhole)",
        "unproject(pinhole) then project(fisheye)",
        (rp.out_width, rp.out_height), (sw, sh), valid, retained))

    processing_scale = float(cfg.get("semantic_camera_final.processing_scale", 0.5)) \
        if cfg.get("semantic_camera_final") else 0.5
    for name, description, size in (
        ("03_mask2former_input",
         "Mask2Former image processor resize to a fixed square",
         (384, 384)),
        ("04_grounding_dino_input",
         "Grounding DINO processor resize (shortest edge 800, longest 1333)",
         (1333, 1000)),
        ("05_sam2_input", "SAM 2.1 operates on the rectified frame directly",
         (rp.out_width, rp.out_height)),
    ):
        stage = summarise_stage(name, description, "resize", "inverse resize",
                                size, (rp.out_width, rp.out_height))
        stage.notes.append(
            "model-space geometry only; the produced mask is mapped back to the "
            "rectified frame before fusion")
        stages.append(stage)

    stages.append(summarise_stage(
        "06_semantic_camera_masks", "dense fused mask in rectified geometry",
        "identity", "identity", (rp.out_width, rp.out_height),
        (rp.out_width, rp.out_height)))

    processing = (int(rp.out_width * processing_scale),
                  int(rp.out_height * processing_scale))
    stage = summarise_stage(
        "07_temporal_processing_scale",
        f"temporal stages operate at processing_scale={processing_scale}",
        "resize", "resize back", processing, (rp.out_width, rp.out_height))
    stage.notes.append("aspect ratio preserved; no crop")
    stages.append(stage)

    video_geometry = _probe_video(args.video)
    stage = summarise_stage(
        "08_final_video", "rendered presentation video",
        "resize", "n/a (presentation only)",
        (video_geometry["width"], video_geometry["height"]),
        (rp.out_width, rp.out_height))
    stage.notes.append(
        f"{video_geometry['frames']} frames at {video_geometry['fps']:.4f} fps")
    stages.append(stage)

    # ------------------------------------------------------------------ #
    # verdict on cropping
    # ------------------------------------------------------------------ #
    crop_checks = {
        "rectification": detect_centre_crop((sw, sh), (rp.out_width, rp.out_height), None),
        "final_video": detect_centre_crop(
            (rp.out_width, rp.out_height),
            (video_geometry["width"], video_geometry["height"]), None),
    }
    losses = edge_losses(valid, retained)
    coverage_source = angular_coverage(source_calib, sw, sh)
    coverage_pinhole = angular_coverage(pinhole, rp.out_width, rp.out_height)

    document = {
        "schema": "article1_geometry_audit_v1",
        "generated_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "source": {
            "vrs": str(Path(args.vrs).resolve()),
            "frame_index": args.frame_index,
            "timestamp_ns": int(timestamp),
            "width": sw, "height": sh, "aspect_ratio": sw / sh,
            "camera_model": str(source_calib.get_model_name()),
            "focal": [float(x) for x in source_calib.get_focal_lengths()],
            "principal_point": [float(x) for x in source_calib.get_principal_point()],
            "angular_coverage": coverage_source,
            "model_valid_pixels": int(np.count_nonzero(valid)),
            "model_valid_fraction": float(np.count_nonzero(valid) / valid.size),
        },
        "rectification": {
            "target": [rp.out_width, rp.out_height],
            "focal": rp.focal,
            "rotate_ccw90": rp.rotate_ccw90,
            "angular_coverage": coverage_pinhole,
            "retained_source_fraction": float(np.count_nonzero(retained) / retained.size),
            "retained_valid_fraction": float(
                np.count_nonzero(retained & valid) / max(1, np.count_nonzero(valid))),
            "discarded_valid_pixels": int(np.count_nonzero(valid & ~retained)),
            **losses,
        },
        "crop_checks": crop_checks,
        "final_video": video_geometry,
        "overlay": overlay_info,
        "stages": [s.to_dict() for s in stages],
    }
    atomic_write_json(geometry_dir / "transform_chain.json", document)

    rows = []
    for s in stages:
        d = s.to_dict()
        d["notes"] = " | ".join(d.get("notes") or [])
        rows.append(d)
    pd.DataFrame(rows).to_csv(geometry_dir / "stage_geometry_table.csv", index=False)

    print(json.dumps({
        "source": f"{sw}x{sh}",
        "model_valid_fraction": document["source"]["model_valid_fraction"],
        "retained_valid_fraction": document["rectification"]["retained_valid_fraction"],
        "discarded_valid_pixels": document["rectification"]["discarded_valid_pixels"],
        "horizontal_fov_source_deg": coverage_source["horizontal_fov_deg"],
        "horizontal_fov_rectified_deg": coverage_pinhole["horizontal_fov_deg"],
        "vertical_fov_source_deg": coverage_source["vertical_fov_deg"],
        "vertical_fov_rectified_deg": coverage_pinhole["vertical_fov_deg"],
        "edge_losses": {k: v for k, v in losses.items() if k.startswith("pixels_lost")},
        "aspect_changed_anywhere": any(c["aspect_changed"] for c in crop_checks.values()),
        "centre_crop_detected": any(c["is_centre_crop"] for c in crop_checks.values()),
    }, indent=2))
    return 0


def _probe_video(path: str) -> Dict[str, Any]:
    import cv2
    p = Path(path)
    if not p.exists():
        return {"available": False, "width": 0, "height": 0, "frames": 0, "fps": 0.0}
    cap = cv2.VideoCapture(str(p))
    info = {
        "available": True,
        "path": str(p),
        "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        "frames": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        "fps": float(cap.get(cv2.CAP_PROP_FPS)),
    }
    cap.release()
    info["aspect_ratio"] = info["width"] / info["height"] if info["height"] else None
    return info


if __name__ == "__main__":
    raise SystemExit(main())
