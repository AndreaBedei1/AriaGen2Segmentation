# Hierarchical ROI + multi-scale (Grounded-SAM2, Phase 4 & 5)

Opt-in passes that lift recall for small / through-glass / mirror objects by
re-detecting targeted classes on cropped regions at higher effective resolution.
Implemented in `aria_drive_seg/segmentation/roi.py` + `grounded_sam2.py`; a single
reusable `_finalize_pass` runs the full-frame pass and every crop pass, so filtering,
NMS, SAM and geometry are identical everywhere. All detections carry `provenance`
(`full_frame` | `roi:<name>` | `tile:<i>`).

## ROIs are derived, not hardcoded (Phase 4)
`derive_rois()` reads the structural detections of the full-frame pass and takes the
largest box per structural class to define each ROI: `windshield_view`,
`left/right_window_view`, `rear_view_mirror_zone`, `left/right_side_mirror_zone`,
`dashboard_zone`, `instrument_cluster_zone`, `center_console_zone`. No per-frame
coordinates are baked in. For the same driver+car the ROIs are temporally stable, so
`ROIState` optionally EMA-smooths them and **snaps** on large moves (camera motion) —
`roi_stable: true`. Each ROI re-runs only a relevant class subset
(`roi_class_subset`): exterior/small classes in windshield & windows, reflected
vehicles/people in mirror zones, interior sub-parts in dashboard/console/cluster zones.
Config: `roi_enabled`, `roi_pad_frac`, `roi_scale`, `roi_stable`.

## Multi-scale tiles (Phase 5)
`tiles_of()` splits the windshield crop into overlapping horizontal tiles, each upscaled,
and re-runs the small-object class subset (traffic lights/signs, pedestrians, cyclists,
bollards, cones, distant vehicles, lane markings). Config: `multiscale_enabled`,
`ms_tiles`, `ms_tile_overlap`, `ms_scale`. Crop detections are remapped to full-frame
coordinates (`remap_box`, `paste_mask` — masks downscaled back and pasted at the offset).

## Cross-pass fusion
After all passes, `_fuse()` applies class-aware **box NMS** (per-class IoU), then
same-class **mask-IoU dedup** (`overlap_min_iou_dedup`) to merge full-frame ↔ tile
duplicates, then `max_instances` per class. Per-frame metadata records `rois`,
`provenance_counts`, and all rejection reasons.

## Parent→subclass hierarchy (fine-class control)
Fine subclasses (person/vehicle subtypes) must not compete on the full frame — that turns
a normal car into `emergency_vehicle` or an adult into `child` on a marginal score. With
`grounded_sam2.hierarchical_subclasses: true` (default in the standard config), the fine
subclasses (`configs/grounded_prompts.yaml` `hierarchy.parents`) are **excluded from global
prompting**; instead the parent (person / rider / car / truck) is detected first with its
mask, and the subtype is classified ONLY on the parent crop (`_classify_subclasses` +
`hierarchy.decide_subclass`). A subclass relabels the parent's mask only when its score
≥ `subclass_min_score` AND beats the second candidate by ≥ `subclass_min_margin`; otherwise
the parent label is kept. Mask geometry always comes from the parent. Each detection stores
`parent_class`, `subclass_status` (accepted|rejected|ambiguous), `subclass_confidence`,
`parent_confidence`. The old aggressive global behaviour stays available via the diagnostic
`global_fine_subclasses: true`. Tested in `tests/test_hierarchy.py` (a normal car stays car;
an adult stays person on a marginal child margin; rejected/ambiguous keep the parent).

## Validation (3 frames, ROI+multiscale on)
No errors; the crop passes roughly doubled the detection count (frame 0: 83 full-frame
+ ~125 from ROI/tile passes → 208 accepted after fusion). All 8–9 ROIs derived. Cost:
~47–50 s/frame (full-frame ~15 s + crop passes) — this is the **maximum-recall** config;
the ablation (`docs/ablation_grounded_sam2.md`) chooses the speed/recall operating point.
`roi.py` pure functions covered in `tests/test_roi.py`.
