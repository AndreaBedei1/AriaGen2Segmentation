# Ground-truth protocol (Grounded-SAM2 operating-point selection)

Ground truth is collected on a **frozen, reproducible 60-frame validation set** and used
to choose the static Grounded-SAM2 operating point on metrics + visual QA. No accuracy is
claimed until this GT exists; the pipeline never fabricates labels.

## Validation set (60 frames)
`validation/grounded_sam2_gt60_manifest.csv` — 60 unique, reproducible frames built by
`scripts/build_validation_set.py`:
- **20 uniform** — evenly spaced along the whole recording;
- **20 hard** — high image complexity / difficult lighting (clutter, shadow, glare);
- **20 gaze** — diverse/extreme gaze (cockpit-down, wide yaw, invalid, ahead) that stress
  the gaze-target resolution and the errors seen in the max-recall config.
Minimum 12-frame gap avoids near-duplicate consecutives. Fields: `frame_index`,
`capture_timestamp_ns`, `selection_reason`, `scene_category`, `gaze_valid`,
`hint_classes` (a hint only — NOT ground truth). Contact sheet: `validation/contact_sheet.jpg`.

## Three annotation tiers
Not all 100 classes are hand-segmented. `scripts/build_gt_package.py` builds a CVAT package
(`validation/cvat_package/`) with:

**Tier A — segmentation masks (19 classes):** road_surface, lane_marking, sidewalk, car,
person, bicycle, motorcycle, traffic_sign, traffic_light, building, vegetation, sky,
steering_wheel, instrument_cluster, dashboard, driver_hand, rear_view_mirror,
left_side_mirror, right_side_mirror.

**Tier B — bounding boxes + presence (14 classes):** stop_sign, yield_sign,
speed_limit_sign, warning_sign, stop_line, direction_arrow, crosswalk, bollard,
traffic_cone, child, cyclist, motorcyclist, van, emergency_vehicle.

**Tier C — not_observed:** every class not in A/B that is absent from a frame is
`not_observed` and is **NOT** counted as a false negative.

## Workflow
1. `python scripts/build_validation_set.py --vrs <rec>.vrs` → manifest + frames + contact sheet (visual check).
2. `python scripts/build_gt_package.py --validation validation [--seed-from <run_dir>]` → CVAT package.
3. **Human annotation in CVAT** (see `validation/cvat_package/README_import.md`): import
   `images/` + `labels.json`, annotate Tier A masks + Tier B boxes, export.
4. Convert exports to `validation/gt/masks/frame_XXXXXX.png` (canonical uint16) and
   `validation/gt/boxes.json`.
5. `python scripts/evaluate_gt.py --input <run_dir> --gt validation/gt/masks` → metrics.

## Metrics (computed only once GT exists)
- **Segmentation (Tier A):** mIoU, per-class IoU, boundary F1, pixel precision/recall, unknown rate.
- **Detection (Tier B):** precision, recall, F1, FP/frame, IoU@0.5 matching, small-object recall.
- **Gaze:** primary-target accuracy, top-2 accuracy, through-glass accuracy, unknown-at-gaze
  rate, mirror-target accuracy, confidence calibration.
- **Efficiency:** s/frame, FPS, peak VRAM, #GDINO calls, #prompts, #crops/tiles, dets kept/rejected.
- **Short-subsequence stability:** class flicker, mask temporal IoU, gaze-class changes, persistent FPs.
Global coverage is NOT a primary metric.

> If you cannot annotate now: the CVAT package is ready — follow
> `validation/cvat_package/README_import.md`. Do not invent annotations.
