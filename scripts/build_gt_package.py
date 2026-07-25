#!/usr/bin/env python3
"""Build a CVAT-ready, TIERED ground-truth package for the frozen 60-frame set (§2).

Tier A — segmentation masks (19 classes). Tier B — bounding boxes + presence (14
classes). Tier C — every other class: `not_observed` (absent → NOT a false negative).
Human annotation is required; this script only prepares the importable package and the
exact import/export instructions. It does NOT invent annotations.

    python scripts/build_gt_package.py --validation validation [--seed-from runs/<cfg>]
"""
from __future__ import annotations

import argparse
import csv
import json
import shutil
from pathlib import Path
from typing import List

from aria_drive_seg.taxonomy import Taxonomy

TIER_A = ["road_surface", "lane_marking", "sidewalk", "car", "person", "bicycle",
          "motorcycle", "traffic_sign", "traffic_light", "building", "vegetation", "sky",
          "steering_wheel", "instrument_cluster", "dashboard", "driver_hand",
          "rear_view_mirror", "left_side_mirror", "right_side_mirror"]
TIER_B = ["stop_sign", "yield_sign", "speed_limit_sign", "warning_sign", "stop_line",
          "direction_arrow", "crosswalk", "bollard", "traffic_cone", "child", "cyclist",
          "motorcyclist", "van", "emergency_vehicle"]


def _hex(rgb) -> str:
    return "#{:02x}{:02x}{:02x}".format(*[int(c) for c in rgb])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--validation", default="validation")
    ap.add_argument("--config", default="configs/classes.yaml")
    ap.add_argument("--seed-from", default=None, help="a seg output dir for Tier-A pre-annotation seeds")
    args = ap.parse_args()

    tax = Taxonomy.load(args.config)
    vdir = Path(args.validation)
    manifest = list(csv.DictReader(open(vdir / "grounded_sam2_gt60_manifest.csv")))
    pkg = vdir / "cvat_package"
    (pkg / "images").mkdir(parents=True, exist_ok=True)

    # copy frames referenced by the manifest
    for m in manifest:
        fi = int(m["frame_index"])
        src = vdir / "frames" / f"frame_{fi:06d}.jpg"
        if src.exists():
            shutil.copy(src, pkg / "images" / src.name)

    # CVAT label schema: Tier A = mask, Tier B = rectangle; a `tier` attribute tags each.
    def label(name: str, ltype: str, tier: str):
        return {"name": name, "color": _hex(tax.by_name[name].color), "type": ltype,
                "attributes": [{"name": "tier", "mutable": False, "input_type": "select",
                                "default_value": tier, "values": [tier]}]}
    labels = ([label(n, "mask", "A") for n in TIER_A if n in tax.by_name]
              + [label(n, "rectangle", "B") for n in TIER_B if n in tax.by_name])
    (pkg / "labels.json").write_text(json.dumps(labels, indent=2))

    # optional Tier-A pre-annotation seeds (editable starting masks), never GT
    if args.seed_from:
        seedroot = Path(args.seed_from)
        seeddir = pkg / "preannotation_tierA"; seeddir.mkdir(exist_ok=True)
        import numpy as np
        from aria_drive_seg.io_utils import read_mask_u16
        import cv2
        keep = {tax.id_of(n) for n in TIER_A if n in tax.by_name}
        for m in manifest:
            fi = int(m["frame_index"])
            mp = seedroot / "grounded_sam2" / "canonical_masks" / f"frame_{fi:06d}.png"
            if mp.exists():
                mask = read_mask_u16(mp)
                mask[~np.isin(mask, list(keep))] = 0  # keep only Tier-A classes
                cv2.imwrite(str(seeddir / f"frame_{fi:06d}_color.png"), tax.colorize(mask)[:, :, ::-1])
                cv2.imwrite(str(seeddir / f"frame_{fi:06d}_id.png"), mask)

    (pkg / "TIERS.json").write_text(json.dumps(
        {"tier_A_segmentation_masks": TIER_A, "tier_B_boxes_presence": TIER_B,
         "tier_C_rule": "every class NOT in Tier A or B and absent from a frame is "
                        "`not_observed` and must NOT be counted as a false negative"}, indent=2))
    (pkg / "README_import.md").write_text(_readme(len(manifest)))
    print(f"CVAT package ready: {pkg}  ({len(manifest)} frames, "
          f"{len(TIER_A)} Tier-A + {len(TIER_B)} Tier-B labels)")
    print("Human annotation required — see", pkg / "README_import.md")
    return 0


def _readme(n: int) -> str:
    return f"""# CVAT ground-truth package ({n} frames) — import / annotate / export

**Tiers**: A = segmentation masks ({len(TIER_A)} classes); B = bounding boxes + presence
({len(TIER_B)} classes); C = any other class -> `not_observed` (absent ≠ false negative).
Do NOT invent labels; annotate only what is visible.

## Import into CVAT
1. Create a project, then a task; **upload** `images/` (all {n} frames).
2. In the task's **Labels** tab choose *Raw* and paste `labels.json` (Tier-A labels are
   `mask`, Tier-B are `rectangle`; each carries a read-only `tier` attribute).
3. (Optional) seed Tier-A from `preannotation_tierA/*_color.png` for faster correction —
   these are model output, NOT ground truth; correct them.

## Annotate
- Tier A: draw/refine a mask per visible class (leave unlabelled pixels as background).
- Tier B: draw one rectangle per visible instance; a class with no rectangle in a frame
  is treated as `not_observed` for that frame.
- Ambiguous / occluded: skip rather than guess.

## Export + convert for scoring
1. Export the task as **Segmentation mask 1.1** (Tier A) and **CVAT for images 1.1**
   (Tier B boxes), or **Datumaro**.
2. Convert the exported Tier-A masks to canonical uint16 id-PNGs named
   `frame_XXXXXX.png` into a `validation/gt/masks/` dir (map CVAT label colours to the
   canonical ids in configs/classes.yaml). Put the Tier-B boxes as `validation/gt/boxes.json`
   ({{frame_index: [{{class, x0,y0,x1,y1}}]}}).
3. Score:
```bash
python scripts/evaluate_gt.py --input <run_dir> --gt validation/gt/masks
```
(Tier-B box metrics + Tier-C handling are computed by reports/ablation_gt60 tooling once
`validation/gt/` exists.)
"""


if __name__ == "__main__":
    raise SystemExit(main())
