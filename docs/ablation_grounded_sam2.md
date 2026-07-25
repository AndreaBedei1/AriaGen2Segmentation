# Grounded-SAM2 ablation & operating-point selection

The static Grounded-SAM2 operating point is chosen on **ground truth + metrics + visual
QA**, never on coverage or recall alone. Ground truth: `docs/ground_truth_protocol.md`.
Nothing here overwrites `run200`/`val10`; ablation output goes under `runs/`.

## The five configurations (`configs/ablation/`)
| id | file | what |
|---|---|---|
| **A** | `A_baseline_original` | original taxonomy, combined grouped caption, no ROI/multiscale/hierarchy — reference ≈ `run200` |
| **B** | `B_phase1_clean` | Phase-1 fixes (variants, synonyms, per-class thresholds, robust mapping, geometric filters) on the ORIGINAL taxonomy |
| **C** | `C_extended_hierarchical` | B + extended taxonomy + parent→subclass hierarchy (child/emergency_vehicle etc. classified on parent crops, never global) |
| **D** | `D_selective_roi_multiscale` | C + ROI + multiscale ONLY for small/safety classes (never road/building/sky/interior; never the crop-only subclasses) |
| **E** | `E_gaze_conditioned_multiscale` | C + high-res crops centred on the valid gaze — **experimental, gaze-assisted, NOT comparable** to the gaze-independent A–D |

The near-dense max-recall profile (extended taxonomy + full ROI + full multiscale, ~47–50
s/frame) is **not** in the main comparison; keep it only as a diagnostic `max_recall` profile.

## Run
Smoke test (before GT) — 10 frames per config, check for crashes / giant masks / coord errors,
**do not pick a winner**:
```bash
ARIA_ML_PYTHON=~/aria_seg_ml_env/bin/python \
python scripts/calibrate_prompts.py --input <10-frame dir> \
  --configs configs/ablation/A_baseline_original.yaml,...,configs/ablation/D_selective_roi_multiscale.yaml \
  --out runs/smoke
```
Full ablation (after GT) — A,B,C,D on the SAME 60 validation frames, identical geometry +
checkpoint, no incompatible cache, config hashes saved. E is a separate experiment.

## Metrics (families)
Segmentation (Tier A): mIoU, per-class IoU, boundary F1, pixel precision/recall, unknown rate.
Detection (Tier B): precision, recall, F1, FP/frame, IoU@0.5, small-object recall.
Gaze: primary + top-2 target accuracy, through-glass accuracy, unknown-at-gaze, mirror-target
accuracy, calibration. Efficiency: s/frame, FPS, peak VRAM, #GDINO calls/prompts/crops/tiles,
dets kept/rejected. Short-subsequence stability: flicker, mask temporal IoU, gaze-class changes.

## Operating-point selection — priority + gates (§8)
Priority order: (1) F1 of safety-critical classes; (2) gaze-target accuracy; (3) fine-subclass
precision; (4) mIoU of main classes; (5) small-object recall; (6) compute cost.

Hard gates the chosen config must pass:
- no macroscopic wrong masks;
- no material regression of person, car, traffic_light, traffic_sign vs **B**;
- fine-subclass false positives under control;
- unknown-at-gaze ≤ **B**;
- zero processing errors; offline behaviour verified.

A ~47–50 s/frame config is **not** eligible as the full-run default unless exceptionally and
documentedly better; it may remain the `max_recall` profile. Target for the main config:
**≈ 20–25 s/frame**.

> Results and the chosen config are recorded in `reports/ablation_gt60.md` once GT exists.
> SAM2 temporal (Phase 8) stays blocked until GT is ready, the static ablation is done, the
> operating point is chosen, and it has passed 200 consecutive frames.
