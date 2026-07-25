# CVAT ground-truth package (60 frames) — import / annotate / export

**Tiers**: A = segmentation masks (19 classes); B = bounding boxes + presence
(14 classes); C = any other class -> `not_observed` (absent ≠ false negative).
Do NOT invent labels; annotate only what is visible.

## Import into CVAT
1. Create a project, then a task; **upload** `images/` (all 60 frames).
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
   ({frame_index: [{class, x0,y0,x1,y1}]}).
3. Score:
```bash
python scripts/evaluate_gt.py --input <run_dir> --gt validation/gt/masks
```
(Tier-B box metrics + Tier-C handling are computed by reports/ablation_gt60 tooling once
`validation/gt/` exists.)
