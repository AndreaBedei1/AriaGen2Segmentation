# Validation set (60 frames) — Grounded-SAM2 operating-point selection

Frozen, reproducible frame set for ground truth + the A–E ablation. **Do not edit the
manifest** without regenerating everything downstream; `val10` and `run200` baselines are
untouched.

## Files
- `grounded_sam2_gt60_manifest.csv` — the 60 frames (frame_index, capture_timestamp_ns,
  selection_reason ∈ {uniform, hard_scene, gaze}, scene_category, gaze_valid, hint_classes).
  `hint_classes` is a HINT, never ground truth.
- `frames/frame_XXXXXX.jpg` — rectified frames to annotate.
- `contact_sheet.jpg` — 60-frame overview (visual check).
- `cvat_package/` — CVAT-importable package (Tier A masks + Tier B boxes). See its
  `README_import.md`.
- `gt/` — **you create this** from the CVAT export: `gt/masks/frame_XXXXXX.png` (canonical
  uint16 Tier-A masks) + `gt/boxes.json` (Tier-B boxes).

## Reproduce the set
```bash
python scripts/build_validation_set.py --vrs unknown_20260528_090955.vrs
python scripts/build_gt_package.py --validation validation
```
Selection is deterministic (uniform = evenly spaced; hard = image-complexity/lighting
ranking; gaze = category-diverse), so the same VRS yields the same 60 frames.

## Annotate then score
Follow `cvat_package/README_import.md` to import → annotate → export → convert to `gt/`.
Then:
```bash
python scripts/evaluate_gt.py --input <run_dir> --gt validation/gt/masks
```

Tier C rule: classes absent from a frame and not in Tier A/B are `not_observed`, not false
negatives. See `docs/ground_truth_protocol.md`.
