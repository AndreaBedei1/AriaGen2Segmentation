# Deliverable report — Aria Gen2 driving segmentation + gaze pipeline

Fully-local pipeline that extracts RGB from a Meta Project Aria Gen2 `.vrs` driving
recording, segments every frame with **two independent methods**, aligns the on-device
eye-gaze, and studies which semantic classes the driver looks at. Everything runs
locally; nothing is uploaded.

Test recording: `unknown_20260528_090955.vrs` (~434 MB). Date of this run: 2026-07-24.
All numbers below are **measured**, not estimated. Where a metric needs human ground
truth it is explicitly reported as *not computed* (no GT exists yet).

---

## 1. Implemented structure

```
aria_drive_seg/            cli, config, taxonomy, io_utils, hashing, hwinfo, netguard, sharding, pipeline
  vrs/        provider (VrsDataProvider wrapper), inspect (§4), extract (§5)
  gaze/       project (CPF→Device→Camera, official reprojection), align (§9)
  segmentation/ base (shared schema), overlap (priority resolution), grounded_sam2 (§7),
                oneformer (§8, Mask2Former/OneFormer backends), run (resumable runner)
  analyze/    gaze_labels (§10), compare (no-GT metrics §17.1), metrics_gt (§17.2), run
  render/     overlays (HUD/legend/gaze), run (7 videos + diagnostics §11)
configs/      classes.yaml, grounded_prompts.yaml, mapillary_to_canonical.yaml, default.yaml, sam2.1 cfg
scripts/      download_weights.py, run_offline.sh, check_offline.py, prepare_annotation.py, evaluate_gt.py
docs/         README + architecture, model_selection, privacy, vrs_and_calibration,
              class_taxonomy, evaluation, troubleshooting, reproducibility
tests/        41 tests (unit + real-VRS e2e)
```
CLI (all implemented + tested): `inspect · extract · segment · align-gaze · analyze · render · run-all · hwinfo`.

## 2. Model actually used per method

| Method | Model actually run | Checkpoint | Nature |
|---|---|---|---|
| **1 — Grounded-SAM2** | Grounding DINO base + **SAM 2.1 Hiera-Large** | `IDEA-Research/grounding-dino-base` + `sam2.1_hiera_large.pt` | open-vocabulary prompted (sparse) |
| **2 — Mapillary semantic** | **Mask2Former Swin-L (Mapillary Vistas v1.2, 65 cls)** | `facebook/mask2former-swin-large-mapillary-vistas-semantic` | dense semantic partition |

## 3. Why this choice

The task's preferred Method 2 is **OneFormer DiNAT-L Mapillary**. I kept that as the
configured intent (`source: hf_oneformer` + `oneformer_id` slot) but, after a real test
(§13 below), it is **not runnable on this exact stack**. I therefore ship **Mask2Former
Swin-L Mapillary**, which has the **identical Mapillary Vistas v1.2 65-class taxonomy**,
is native in `transformers` with **zero custom ops**, and differs by only ~0.8 mIoU
(63.2 vs 64.0 single-scale). Because the taxonomy is identical, the canonical mapping,
the gaze-on-class study, and the method-vs-method comparison are unaffected. Full survey +
verified finding: [docs/model_selection.md](docs/model_selection.md). This is a
reproducibility fallback, **not** a "newer/better model" substitution.

## 4. Versions & checkpoints

| Env | Purpose | Key versions |
|---|---|---|
| A `~/projectaria_gen2_python_env` | VRS I/O, gaze, analyze, render | Python 3.10.20, projectaria-tools 2.1.2, numpy 2.2.6, opencv 5.0 |
| B `~/aria_seg_ml_env` | Method 1 + Method 2 | Python 3.10.20, torch 2.13.0+cu130, transformers 5.14.1, sam2 (git) |
| C `~/aria_oneformer_env` | OneFormer-DiNAT attempt | torch 2.8.0+cu126, natten 0.21.1 |

Checkpoints (recorded with SHA-256 in `weights/manifest.json`):
- `sam2.1_hiera_large.pt` — 898 MB, sha256 `2647878d5dfa5098…`, from `dl.fbaipublicfiles.com/segment_anything_2/092824/`
- `grounding-dino-base` — sha256 (safetensors) `5548f844c928c4b6…`
- `mask2former-swin-large-mapillary-vistas-semantic` — sha256 (safetensors) `72721afb6894c0b1…`

## 5. No data left the machine

- One-time weights download only (§2-allowed). After that, both methods were run with
  `HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1` under **`scripts/check_offline.py`** — a strict
  socket **netguard** that blocks + records any non-loopback connection.
- **Result for both methods: `✓ no non-loopback connection attempts detected`.**
- No Machine-Perception cloud; the eye-gaze is read from the **on-device** `eyegaze`
  stream inside the VRS. No telemetry (disabled in `run_offline.sh`).

## 6. Streams found in the VRS (17)

`camera-rgb` (214-1, **3762** frames), `eyegaze` (373-1, **11283** on-device samples),
`camera-et-left/right` (211-1/2), `handtracking` (371-1), 4× SLAM (1201-1..4),
`imu-left/right` (1202), `mag0`, `baro0`, `ppg`, `als`, `temperature`, `gps-app` (281-1),
UTC. RGB is **2016×1512, FISHEYE624, 10.00 Hz, 376.07 s**, timestamps monotonic, **0 gaps,
0 dropped frames**. (`output/inspection/inspection_report.json`.)

## 7. Frames processed

| Run | Frames | Purpose |
|---|--:|---|
| `val10` | 10 (distributed every 376) | validation of orientation, rectification, gaze, both methods, comparison |
| `run200` | 200 (consecutive, 180–200 s) | stability / memory / resumability at scale |
Total recording is 3762 RGB frames; see §17 for the command to process all of them.

## 8. Valid gaze

- On-device eyegaze overall: **93.2% combined-valid**, 88.1% spatial-point-valid (whole recording).
- After temporal alignment + projection into the rectified camera: **run200 = 190/200 valid (95.0%)**;
  val10 = 9/10 (the one drop is frame 0, whose nearest gaze sample is 34 ms away > 20 ms gate — correctly rejected).

## 9. Performance per method (run200, 200 frames, 1× RTX 6000 Ada)

| Method | ms/frame | FPS | coverage | unknown | classes/frame |
|---|--:|--:|--:|--:|--:|
| grounded_sam2 | 1804 | 0.55 | 32.0% | 68.0% | 10.3 |
| oneformer_mapillary (Mask2Former) | 166 | 6.03 | 99.9% | 0.1% | 16.0 |

Between-method (consistency, **not** accuracy): pixelwise agreement **45.3%**, mean IoU
**0.06**, gaze-class agreement **26.8%** / disagreement 73.2% (190 valid frames),
gaze-class temporal stability **72.4%**.

## 10. Peak VRAM

grounded_sam2 **3.2 GB**, oneformer_mapillary **4.2 GB** (of 46 GB) — large headroom;
CUDA-OOM guard (retry after `empty_cache`) never triggered.

## 11. Outputs generated (per run dir)

`inspection/` (report + samples) · `frames/{original,rectified,frames.parquet,frames.csv,calibration.json}` ·
`gaze/{raw_gaze,aligned_gaze}.parquet` · `grounded_sam2/{canonical_masks(uint16 PNG),confidence,metadata,metadata.parquet,summary.json}` ·
`oneformer_mapillary/{native,canonical_masks,confidence,metadata,…}` ·
`comparison/{gaze/*.parquet, metrics/metrics_no_gt.json, figures/}` · **`videos/` (7 mp4)** · `reports/analysis_summary.md` · `logs/`.
Masks are lossless uint16 id-maps (colour is derived from the deterministic palette, never the only output).

## 12. Comparative images

`run200/comparison/figures/diag_frame_*.png` and the 7 videos (rgb_rectified, seg_×2,
seg_×2_gaze, comparison_sidebyside, comparison_gaze_agreement). Visually verified:
both methods track cars/road/cockpit correctly; the ego cabin is `unknown` in Method 1
(honest) and `other_cockpit` (Mapillary "Ego Vehicle") in Method 2; through-glass pixels
are labelled `windshield` by Method 1 vs the actual scene (road/vegetation) by Method 2 —
the informative disagreement the study is designed to surface.

## 13. Problems encountered (and fixes)

1. **Timestamp read timed out** decoding all frames → switched to `get_timestamps_ns` (no decode): 3762 ts in 0.01 s.
2. **Grounded-SAM "car" over-captured the ego cabin** (one mask = 87.8% of the frame) → added per-class `max_area_frac` (car 0.30) and lowered `windshield` priority so through-glass objects win. Re-verified.
3. **bfloat16 → numpy** crash in Method 2 → moved the mask-class einsum to float32 outside autocast.
4. **OneFormer DiNAT-L not runnable on this stack** (the §8 legacy conflict, tested): `transformers` 5.14 DiNAT imports the old natten API (`natten2dav`, `natten2dqkrpb`); the only natten with an Ada/torch-2.x wheel is 0.21.1, which exposes only the fused `na2d`; natten 0.17 exposes `na2d_qk`/`na2d_av` — matching neither; the original repo needs detectron2 (no numpy-2, breaks on CUDA 13). → shipped Mask2Former-Mapillary (same taxonomy); remediation documented.

## 14. Residual limitations

- **Grounded-SAM is not an exhaustive partition** (unassigned pixels stay `unknown`; coverage ~32%).
- **Mask2Former/OneFormer cannot label classes outside its Mapillary taxonomy** (e.g. no `steering_wheel`/`driver_hand`; those are Method-1-only).
- **In-cabin viewpoint is a domain shift** vs Mapillary/road datasets.
- **Model agreement ≠ accuracy** — no accuracy is claimed without ground truth.
- **A single gaze point** does not capture all eye-tracker uncertainty; depth uses a fallback (8 m) on the ~12% of samples without a valid device depth, flagged as approximation.
- OneFormer-DiNAT-L not run (see §13); Mask2Former-Mapillary is the reproducible equivalent.

## 15. Tests executed

`41 passed in ~18 s` (env A). Coverage: config parsing, class mapping, unknown handling,
overlap resolution + lane/crosswalk-over-road priority, monotonic-timestamp health,
RGB↔gaze association + interpolation + dt gating + out-of-image, uint16 masks, atomic
writes, manifest resumability + cache invalidation, multi-GPU shard disjointness +
ordered recompose, netguard blocking, and a real-VRS e2e (open → extract 2 frames →
align gaze → resume). Both methods additionally ran **0-error** over 200 consecutive frames.

## 16. Exact commands to reproduce

```bash
# --- one-time setup ---
uv venv --python 3.10 ~/aria_seg_ml_env
uv pip install --python ~/aria_seg_ml_env/bin/python torch torchvision transformers accelerate \
    safetensors opencv-python-headless pillow numpy pandas pyarrow pyyaml tqdm scipy timm
SAM2_BUILD_CUDA=0 uv pip install --python ~/aria_seg_ml_env/bin/python "git+https://github.com/facebookresearch/sam2.git"
~/projectaria_gen2_python_env/bin/pip install -e . && ~/aria_seg_ml_env/bin/pip install -e .
~/aria_seg_ml_env/bin/python scripts/download_weights.py --methods grounded_sam2,oneformer_mapillary

# --- per-stage (envs: A=~/projectaria_gen2_python_env, B=~/aria_seg_ml_env) ---
A=~/projectaria_gen2_python_env/bin/python ; B=~/aria_seg_ml_env/bin/python ; VRS=unknown_20260528_090955.vrs
$A -m aria_drive_seg inspect  --vrs $VRS --output out/inspection
$A -m aria_drive_seg extract  --vrs $VRS --output out --start-time 180 --max-frames 200
$B -m aria_drive_seg segment  --method oneformer_mapillary --input out --offline
$B -m aria_drive_seg segment  --method grounded_sam2       --input out --offline
$A -m aria_drive_seg align-gaze --vrs $VRS --input out
$A -m aria_drive_seg analyze   --input out
$A -m aria_drive_seg render    --input out

# --- or one shot ---
ARIA_VRS_PYTHON=$A ARIA_ML_PYTHON=$B $A -m aria_drive_seg run-all --vrs $VRS --output out

# --- prove offline / run tests / prep ground-truth ---
$B scripts/check_offline.py -- $B -m aria_drive_seg segment --method grounded_sam2 --input out --offline
$A -m pytest tests/ -q
$A scripts/prepare_annotation.py --input out --num 60   # then annotate -> gt/ ; then:
$A scripts/evaluate_gt.py --input out --gt out/gt
```

## 17. To process the entire recording

Drop the window flags to run all 3762 frames:
```bash
$A -m aria_drive_seg extract --vrs $VRS --output full
$B -m aria_drive_seg segment --method oneformer_mapillary --input full --offline   # ~10–11 min
$B -m aria_drive_seg segment --method grounded_sam2       --input full --offline   # ~1.9 h (0.55 fps)
$A -m aria_drive_seg align-gaze --vrs $VRS --input full && $A -m aria_drive_seg analyze --input full && $A -m aria_drive_seg render --input full
```
Every stage is resumable (manifest + config fingerprint), writes atomically, isolates
per-frame errors, and retries after CUDA-OOM — a kill mid-run is safe to re-launch.
Expected disk for the full run: order of a few tens of GB (uint16 masks + jpgs + videos);
462 GB free is ample. Grounded-SAM is the long pole; a single RTX 6000 Ada handles both
methods sequentially. To obtain the *exact* OneFormer-DiNAT-L Mapillary default, follow the
remediation in [docs/model_selection.md](docs/model_selection.md) (port the DiNAT attention
to natten 0.21's `na2d`, then set `source: hf_oneformer` + `oneformer_id`).
