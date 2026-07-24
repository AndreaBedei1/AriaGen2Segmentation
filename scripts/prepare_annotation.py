#!/usr/bin/env python3
"""Prepare a stratified ground-truth annotation package (§17.2).

Samples N frames spread ACROSS the whole recording (not just the start), biased to
include rare/safety-critical classes, then exports:
  annotation/images/            rectified frames to label
  annotation/preannotation/     OneFormer canonical id-masks + colorized (editable seed)
  annotation/labels_cvat.json   CVAT segmentation label schema (canonical classes + colors)
  annotation/manifest.json      sampled frame list + provenance
  annotation/README.md          how to annotate (CVAT import or local correction)

It DOES NOT produce accuracy numbers — only the material to create ground truth.

    python scripts/prepare_annotation.py --input run200 --num 60
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

from aria_drive_seg.taxonomy import Taxonomy

RARE = ["person", "rider", "bicycle", "motorcycle", "traffic_light",
        "traffic_sign", "crosswalk", "lane_marking", "truck", "bus"]


def _hex(rgb) -> str:
    return "#{:02x}{:02x}{:02x}".format(*[int(c) for c in rgb])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="an extract output dir")
    ap.add_argument("--num", type=int, default=60)
    ap.add_argument("--config", default="configs/classes.yaml")
    ap.add_argument("--seed-method", default="oneformer_mapillary")
    args = ap.parse_args()

    import pandas as pd
    import cv2
    from aria_drive_seg.io_utils import read_mask_u16
    from aria_drive_seg.segmentation.base import SegLayout

    root = Path(args.input)
    tax = Taxonomy.load(args.config)
    fdf = pd.read_parquet(root / "frames" / "frames.parquet").sort_values("frame_index").reset_index(drop=True)
    if len(fdf) == 0:
        print("no frames"); return 1
    seed = SegLayout(args.input, args.seed_method)
    rare_ids = {tax.id_of(n) for n in RARE if n in tax.by_name}

    # stratify by time into `num` bins; pick the rare-class-richest frame per bin
    n = min(args.num, len(fdf))
    bins = np.array_split(np.arange(len(fdf)), n)
    chosen: List[int] = []
    for b in bins:
        best_fi, best_score = None, -1
        for pos in b:
            fi = int(fdf.iloc[pos]["frame_index"])
            score = 0
            mp = seed.canonical_path(fi)
            if mp.exists():
                m = read_mask_u16(mp)
                score = int(np.isin(m, list(rare_ids)).sum())
            if score > best_score:
                best_score, best_fi = score, fi
        if best_fi is not None:
            chosen.append(best_fi)

    ann = root / "annotation"
    (ann / "images").mkdir(parents=True, exist_ok=True)
    (ann / "preannotation").mkdir(parents=True, exist_ok=True)

    manifest: Dict[str, Any] = {"input": str(root), "num_frames": len(chosen),
                                "seed_method": args.seed_method, "frames": []}
    for fi in chosen:
        row = fdf[fdf["frame_index"] == fi].iloc[0]
        src = root / row["rectified_path"]
        if not src.exists():
            continue
        name = f"frame_{fi:06d}"
        shutil.copy(src, ann / "images" / f"{name}.jpg")
        entry = {"frame_index": int(fi), "image": f"images/{name}.jpg",
                 "capture_timestamp_ns": int(row["capture_timestamp_ns"])}
        mp = seed.canonical_path(fi)
        if mp.exists():
            m = read_mask_u16(mp)
            shutil.copy(mp, ann / "preannotation" / f"{name}_id.png")
            cv2.imwrite(str(ann / "preannotation" / f"{name}_color.png"),
                        tax.colorize(m)[:, :, ::-1])
            entry["preannotation"] = f"preannotation/{name}_id.png"
        manifest["frames"].append(entry)

    # CVAT label schema (canonical classes + colors)
    labels = [{"name": c.name, "color": _hex(c.color), "type": "mask", "attributes": []}
              for c in tax.classes if c.id != 0]
    (ann / "labels_cvat.json").write_text(json.dumps(labels, indent=2))
    (ann / "manifest.json").write_text(json.dumps(manifest, indent=2))
    (ann / "README.md").write_text(_readme(len(manifest["frames"]), tax))
    print(f"annotation package ready: {ann}  ({len(manifest['frames'])} frames)")
    print("Only the common EVAL taxonomy is used to score both methods; annotate those classes.")
    return 0


def _readme(n: int, tax: Taxonomy) -> str:
    evalnames = ", ".join(tax.name_of(c) for c in tax.eval_ids())
    return f"""# Ground-truth annotation package ({n} frames)

Frames are sampled STRATIFIED across the whole recording and biased toward rare /
safety-critical classes. Annotate semantic segmentation using the CANONICAL classes.

## Option A — CVAT (recommended)
1. Create a CVAT task, upload `images/`.
2. Import the label schema `labels_cvat.json` (segmentation masks, canonical colours).
3. (Optional) Seed each frame from `preannotation/<frame>_color.png` and correct it.
4. Export as *Segmentation mask 1.1* or *CVAT for images*; convert exported masks to
   canonical uint16 id-PNGs named `frame_XXXXXX.png` for scoring.

## Option B — local correction
`preannotation/<frame>_id.png` are uint16 canonical id-masks (editable in any tool).
Correct them and save corrected uint16 id-PNGs into a `gt/` directory.

## Score
    python scripts/evaluate_gt.py --input <dir> --gt <gt_dir>

Common eval taxonomy: {evalnames}
(unknown is scored as a class, so unknown predictions count as errors.)
"""


if __name__ == "__main__":
    raise SystemExit(main())
