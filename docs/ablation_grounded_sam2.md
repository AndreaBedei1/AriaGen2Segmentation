# Grounded-SAM2 ablation & prompt calibration (Phase 6)

`scripts/calibrate_prompts.py` runs one or more configs over a stratified frame set and
reports per-class recall proxies + cost, so a config is chosen on **metrics + visual QA**,
not on coverage alone. Nothing overwrites `run200`/`val10`; output goes to
`runs/calibration/<config>/`.

## Run
```bash
ARIA_ML_PYTHON=~/aria_seg_ml_env/bin/python \
ARIA_VRS_PYTHON=~/projectaria_gen2_python_env/bin/python \
~/projectaria_gen2_python_env/bin/python scripts/calibrate_prompts.py \
    --vrs unknown_20260528_090955.vrs --num 80 \
    --configs configs/ablation/baseline_grouped.yaml,configs/ablation/per_class_variants.yaml,configs/ablation/hierarchical_roi.yaml,configs/ablation/hierarchical_roi_multiscale.yaml
```
`--num` frames are sampled evenly across the whole drive (day/traffic/junction variety
from spanning the recording). Use `--input <extract_dir>` to reuse a fixed frame set.

## Outputs
- `runs/calibration/calibration.json` — per config: per-class {detections, frames_present,
  mean_area_px, mean_score}, rejection reasons, mean ms/frame, mean coverage.
- `runs/calibration/comparison.md` — per-class frames-present across configs + totals.
- per-config `grounded_sam2/` outputs (masks/metadata) for visual QA / contact sheets.

## Ablation configs (`configs/ablation/`)
| config | what |
|---|---|
| `baseline_grouped` | ROI/multiscale off (reference) |
| `per_class_variants` | Phase-1 per-class prompts + thresholds + filters |
| `hierarchical_roi` | + Phase-4 ROI crop passes |
| `hierarchical_roi_multiscale` | + Phase-5 windshield tiles (max recall) |

## Choosing (per the user's directive)
The operating point is chosen **from the ablation**, balancing recall (per-class
frames-present, small-object recall, gaze unknown-rate) against cost (ms/frame) and
false positives seen in the contact sheets — not fixed a priori. Measured cost points on
this hardware: per-class full-frame ≈ 5–6.5 s/frame (extended taxonomy ≈ 15 s/frame);
+ROI ≈ +; +multiscale ≈ 47–50 s/frame. The chosen config is recorded here once the full
calibration run completes, and drives the final full-recording run (which stays gated).

> Accuracy is NOT claimed from these proxies — recall/precision require the Phase-7
> ground truth (`scripts/prepare_annotation.py` + `scripts/evaluate_gt.py`).
