#!/usr/bin/env python3
"""Phase 12: assemble the CVAT-importable annotation package.

Pre-annotations are copied into a directory whose name says what they are and are
marked as automatic output in the manifest, the guide and their own README. They
are never presented as ground truth.

Heavy content (full-resolution images, pre-annotation masks) stays local and is
excluded from git; the manifest, label map, palette, guide, frame list, QA
thumbnails and checksums are what get committed.
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from aria_drive_seg.config import Config
from aria_drive_seg.hashing import sha256_file
from aria_drive_seg.ingestion.cvat_package import (PackageItem, build_label_spec,
                                                   write_checksums, write_guide,
                                                   write_labelmap, write_manifest,
                                                   write_palette)
from aria_drive_seg.logging_utils import get_logger, setup_logging
from aria_drive_seg.taxonomy import Taxonomy

log = get_logger("cvat.package")

PREANNOTATION_README = """# PRE-ANNOTATIONS — NOT GROUND TRUTH

Every mask in `SegmentationClass/` is automatic model output:

- external scene: Mask2Former Swin-L trained on Mapillary Vistas, aggregated into
  the Article 1 macro taxonomy;
- cockpit: an unreviewed Grounding DINO + SAM 2.1 proxy plus a geometric
  bottom-of-frame prior.

Neither has been validated on this data, and the geometric cockpit prior in
particular was shaped around a car interior and does not describe a handlebar.

These masks exist to save the annotator time. They are wrong often enough that
they must be checked pixel by pixel, and where they disagree with the image the
image wins. Nothing here may be used as a reference, a metric target, or training
supervision until a human has reviewed it.
"""


def _load_selection(reports: Path) -> List[Dict[str, Any]]:
    doc = json.loads((reports / "annotation_selection.json").read_text())
    return doc["selected"]


def _find_preannotation(roots: Dict[str, Path], recording_id: str,
                        frame_index: int) -> Optional[Path]:
    root = roots.get(recording_id)
    if root is None:
        return None
    p = root / "final_masks" / f"frame_{frame_index:06d}.png"
    return p if p.exists() else None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reports", default="reports/article1_motorcycle_ingestion")
    ap.add_argument("--output", default="datasets/article1_annotation_package")
    ap.add_argument("--preannotation-root", action="append", default=[],
                    help="recording_id=path/to/semantic_camera (repeatable)")
    ap.add_argument("--thumbnail-width", type=int, default=384)
    ap.add_argument("--config", default="configs/article1/semantic_camera.yaml")
    ap.add_argument("--log-level", default="INFO")
    args = ap.parse_args()
    setup_logging(args.log_level)

    import cv2

    cfg = Config.load(args.config)
    taxonomy = Taxonomy.load(cfg.resolve(cfg.get("semantic_camera.classes")))
    reports, out = Path(args.reports), Path(args.output)

    roots: Dict[str, Path] = {}
    for spec in args.preannotation_root:
        rec, _, path = spec.partition("=")
        roots[rec] = Path(path)

    images_dir = out / "images"
    pre_dir = out / "preannotations_not_ground_truth" / "SegmentationClass"
    thumbs_dir = out / "thumbnails_qa"
    sets_dir = out / "preannotations_not_ground_truth" / "ImageSets" / "Segmentation"
    for d in (images_dir, pre_dir, thumbs_dir, sets_dir):
        d.mkdir(parents=True, exist_ok=True)

    selected = _load_selection(reports)
    log.info("assembling %d items", len(selected))

    items: List[PackageItem] = []
    names: List[str] = []
    for s in selected:
        stem = (f"{s['domain']}_{s['recording_id'][-12:]}_"
                f"{s['source_frame_index']:06d}")
        src = Path(s["image_path"])
        if not src.exists():
            log.warning("missing image %s", src)
            continue
        dst = images_dir / f"{stem}.jpg"
        if not dst.exists():
            shutil.copy2(src, dst)

        img = cv2.imread(str(dst), cv2.IMREAD_COLOR)
        if img is not None:
            h = int(img.shape[0] * args.thumbnail_width / img.shape[1])
            cv2.imwrite(str(thumbs_dir / f"{stem}.jpg"),
                        cv2.resize(img, (args.thumbnail_width, h),
                                   interpolation=cv2.INTER_AREA),
                        [cv2.IMWRITE_JPEG_QUALITY, 78])

        pre_rel = None
        pre_sha = None
        pre_src = _find_preannotation(roots, s["recording_id"],
                                      s["source_frame_index"])
        if pre_src is not None:
            mask = cv2.imread(str(pre_src), cv2.IMREAD_UNCHANGED)
            if mask is not None:
                colour = taxonomy.colorize(mask.astype(np.uint16))
                dst_mask = pre_dir / f"{stem}.png"
                cv2.imwrite(str(dst_mask), colour[:, :, ::-1])
                pre_rel = str(dst_mask.relative_to(out))
                pre_sha = sha256_file(dst_mask)

        names.append(stem)
        items.append(PackageItem(
            image_name=f"{stem}.jpg",
            image_relative_path=str(dst.relative_to(out)),
            preannotation_relative_path=pre_rel,
            domain=s["domain"], recording_id=s["recording_id"],
            source_frame_index=s["source_frame_index"],
            timestamp_ns=s["timestamp_ns"], split_group=s["split_group"],
            selection_reason=s["selection_reason"],
            expected_classes=s["expected_classes"],
            failure_mode_candidate=s["failure_mode_candidate"],
            hand_visibility_candidate=s["hand_visibility_candidate"],
            route_segment=s["route_segment"],
            route_progression=s["route_progression"],
            pairing_id=s["pairing_id"],
            image_sha256=sha256_file(dst),
            preannotation_sha256=pre_sha,
        ))

    write_labelmap(out / "preannotations_not_ground_truth" / "labelmap.txt", taxonomy)
    write_labelmap(out / "labelmap.txt", taxonomy)
    write_palette(out / "palette.json", taxonomy)
    (out / "labels.json").write_text(
        json.dumps(build_label_spec(taxonomy), indent=2) + "\n")
    (out / "preannotations_not_ground_truth" / "README.md").write_text(
        PREANNOTATION_README)
    (sets_dir / "default.txt").write_text("\n".join(names) + "\n")
    write_guide(out)

    doc = write_manifest(out, items, taxonomy, extra={
        "cvat_import": {
            "images": "images/",
            "task_label_specification": "labels.json",
            "preannotation_format": "CVAT Segmentation mask 1.1",
            "preannotation_root": "preannotations_not_ground_truth/",
            "labelmap": "preannotations_not_ground_truth/labelmap.txt",
            "note": ("import the images as the task and, optionally, the "
                     "pre-annotations as a starting point; the pre-annotations are "
                     "automatic output and must be reviewed"),
        },
        "source_selection": "reports/article1_motorcycle_ingestion/annotation_selection.csv",
        "git_policy": {
            "committed": ["manifest.json", "labels.json", "labelmap.txt",
                          "palette.json", "ANNOTATION_GUIDE.md", "frame_list.csv",
                          "thumbnails_qa/", "checksums.sha256", "examples/"],
            "local_only": ["images/", "preannotations_not_ground_truth/SegmentationClass/"],
        },
    })

    pd.DataFrame([i.to_dict() for i in items]).to_csv(
        out / "frame_list.csv", index=False)

    tracked = [p for p in out.rglob("*")
               if p.is_file()
               and "images" not in p.relative_to(out).parts
               and "SegmentationClass" not in p.relative_to(out).parts]
    write_checksums(out, tracked)

    summary = {
        "items": len(items),
        "with_preannotation": sum(1 for i in items if i.preannotation_relative_path),
        "per_domain": doc["counts"]["per_domain"],
        "per_group": doc["counts"]["per_group"],
        "output": str(out),
    }
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
