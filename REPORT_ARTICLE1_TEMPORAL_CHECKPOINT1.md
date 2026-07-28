# Article 1 temporal stabilization — checkpoint 1

Date: 2026-07-28  
Branch: `feature/article1-temporal-stabilization`  
Initial SHA: `be717176c7aa9ccdb397c9e4c89517fe42402e46`  
Final SHA: reported in the pushed-branch handoff (a tracked report cannot contain
the hash of the commit that contains itself).

## Scientific status and scope

This checkpoint implements a cautious causal temporal baseline while preserving the
checkpoint-2 static result independently. Frame `t` uses only current and previous
frames. Gaze coordinates are not an input to flow, fusion, TTL, hysteresis, thin
processing or reset logic. Gaze is sampled only after segmentation.

There is no reviewed ground truth. Switch, unknown, continuity and persistence
statistics are pre-GT behavior diagnostics, not accuracy. Reduced flicker/unknown can
also preserve errors.

Not implemented here: future-frame or bidirectional smoothing, a complete VRS run,
cockpit fusion, SegFormer training, auto-vs-motorcycle analysis, or threshold selection
from GT.

## Git

- Protected base: `feature/auto-moto-semantic-gaze` at `be71717`.
- Work branch: `feature/article1-temporal-stabilization`.
- History was not rewritten; protected branches and prior outputs were not modified.
- The thematic commits before the final report are:
  - `3da2a43` — local OpenCV DIS flow and validity checks;
  - `f25bcaf` — causal temporal state, fusion, TTL, hysteresis and thin layer;
  - `bb95b7c` — streaming runner, provenance, resume and diagnostics;
  - `56230d4` — cache/render/performance hardening;
  - `e0fff7f` — temporal design documentation;
  - `fa69810` — separate T0–T4 ablations and QA diagnostics.
- Final source/test/report commits and push status are recorded in the terminal handoff.

## Temporal design

- Temporal grid: 1008×756 (`processing_scale=0.5`) while static masks remain
  2016×1512.
- Flow: local OpenCV DIS on rectified RGB-derived grayscale; forward and backward flow,
  forward/backward consistency, occlusion and photometric validation.
- Fusion: normalized current Article 1 probabilities plus valid backward-warped
  temporal probabilities. High-confidence current evidence wins.
- T1: hysteresis only, no flow or probability propagation.
- T2: generic flow, warp and probabilistic fusion.
- T3: T2 plus class-specific TTL, weights and dynamic current-support gates.
- T4: T3 plus a separate road-supported thin-marking layer.
- Reset gates: missing/non-contiguous frame, geometry change, non-monotonic or excessive
  timestamp gap, incompatible fingerprints/cache, excessive flow, insufficient valid
  flow, excessive photometric error/scene cut.
- Checkpoints: portable NPZ states for T0–T4 every 50 frames. No Python object pickle.
- Primary output: T4; full T0–T4 metrics are retained in the manifest.

The provenance codes distinguish current, warped, fused, thin, hysteresis and reset
evidence. Propagation age grows only when the output class is the warped previous class;
cross-class probability fusion is provenance `3` with age zero.

## Clip and staged runs

- Source: `unknown_20260528_090955.vrs`.
- Requested VRS interval: 180–210 s.
- New dedicated extraction:
  `outputs/article1/auto_temporal_180_210/`.
- Final frames: 300 consecutive frames, indices 1801–2100.
- Capture timestamps: 1235364468985–1265262269475 ns.
- Median frame interval: 0.099992397 s (nominal 10 fps).
- Rectified geometry: 2016×1512; rotation 0.
- Missing indices: none.
- Gaze: 285/300 post-hoc valid samples (95%); gaze is not used by segmentation.
- Staged gates completed in dedicated directories: 10 frames, 50 frames, then 300
  frames. The 300-frame data below supersede smoke-run metrics.

## T0–T4 diagnostics

All modes used the same 300 frames in one streaming experiment. `ms/frame` below is the
mode-specific temporal stage; shared static inference, flow and writing are reported in
the performance section. Thin continuity for T0–T3 is the static-thin continuity because
their thin temporal layer is disabled. “State MB” is only the six periodic NPZ
checkpoints for that mode, not a standalone full-output run.

| mode | switch | unknown | propagated px/frame | mean age | thin continuity | resets | temporal ms/frame | state MB |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| T0 static | 5.5868% | 1.1427% | 0 | 0.000 | 13.8426% | 7 | 67.4 | 39.54 |
| T1 hysteresis | 4.2742% | 0.3179% | 7,830 | 0.000 | 13.8426% | 7 | 328.6 | 38.44 |
| T2 flow+fusion | 4.7036% | 0.7880% | 38,069 | 0.233 | 13.8426% | 7 | 325.2 | 39.02 |
| T3 class policy | 4.9211% | 0.7142% | 37,978 | 0.184 | 13.8426% | 7 | 327.1 | 38.98 |
| T4 thin layer | 4.9269% | 0.7145% | 37,464 | 0.192 | 17.2582% | 7 | 347.8 | 38.99 |

The full-resolution T0/T4 post-run comparison gives raw switch 5.6043%, temporal switch
4.9434%, raw unknown 1.1428% and temporal unknown 0.7145%. The small difference from the
table is due to full-resolution versus common half-resolution metric grids.

No per-mode isolated VRAM measurement is available because T0–T4 were evaluated in one
shared process. The measured shared peak is 3708.13 MB. Reporting that value as five
independent per-mode measurements would be misleading. No scientific winner is selected
without GT.

## Class policy and observed pixel transitions

Appearances/disappearances are pixel transitions, not object counts. “Without current”
is a diagnostic proxy: propagated output pixels whose raw current class differs. It is
not automatically an error.

| class | TTL | previous weight | requires current support | appearances | disappearances | propagated pixels | without current |
|---|---:|---:|:---:|---:|---:|---:|---:|
| unknown | 0 | 0.00 | no | 5,687,008 | 5,797,916 | 9,556 | 2,455 (25.7%) |
| road_surface | 3 | 0.55 | no | 7,213,764 | 7,241,396 | 3,075,380 | 1,032,521 (33.6%) |
| lane_marking | 3 | 0.75 | no | 1,286,280 | 1,265,020 | 542,624 | 309,038 (57.0%) |
| regulatory_road_marking | 2 | 0.65 | no | 447,200 | 446,016 | 185,480 | 118,305 (63.8%) |
| vehicle | 1 | 0.35 | yes | 8,664,032 | 8,803,304 | 4,661,036 | 807,379 (17.3%) |
| two_wheeler | 1 | 0.25 | yes | 179,012 | 179,012 | 17,616 | 1,959 (11.1%) |
| pedestrian | 1 | 0.20 | yes | 807,752 | 840,584 | 853,380 | 233,566 (27.4%) |
| traffic_light | 2 | 0.45 | no | 1,164 | 1,164 | 848 | 85 (10.0%) |
| traffic_sign | 2 | 0.45 | no | 648,908 | 646,696 | 190,288 | 52,771 (27.7%) |
| road_boundary_or_obstacle | 2 | 0.45 | no | 4,105,752 | 4,154,700 | 1,489,728 | 440,888 (29.6%) |
| mirror | 1 | 0.15 | no | 0 | 0 | 0 | 0 (n/a) |
| instrument_display | 1 | 0.15 | no | 0 | 0 | 0 | 0 (n/a) |
| control_and_ego_vehicle | 1 | 0.15 | no | 0 | 0 | 0 | 0 (n/a) |
| other_environment | 1 | 0.15 | no | 16,013,944 | 15,679,008 | 33,930,780 | 2,954,620 (8.7%) |

Observed maximum diagnostic propagation ages obey the configured gates: road/lane 3,
regulatory/sign/boundary 2, dynamic classes and other_environment 1. Traffic light
reached 1 of its configured maximum 2 in this clip.

## Performance and storage

Measured per input frame on the 300-frame run:

| stage | mean ms | median ms | p95 ms |
|---|---:|---:|---:|
| static Mask2Former + Article 1 policy | 8382.50 | 8270.18 | 10872.78 |
| optical flow total | 97.48 | 97.60 | 106.35 |
| flow computation | 56.97 | 56.93 | 63.90 |
| flow validation | 39.75 | 39.89 | 42.35 |
| T4 warping | 65.55 | 66.83 | 75.36 |
| T4 probability fusion | 200.79 | 205.50 | 215.55 |
| T4 hysteresis | 13.83 | 13.94 | 16.59 |
| T4 thin processing | 25.54 | 26.04 | 29.40 |
| output writing | 367.43 | 238.91 | 952.06 |
| total shared T0–T4 experiment | 10346.75 | 10187.88 | 12739.75 |

- Shared experiment throughput: 0.09665 input frames/s; real time was not required.
- Peak RAM: 6405.78 MB.
- Peak VRAM: 3708.13 MB.
- Temporal run storage: 806,681,070 bytes after final reports/metadata refresh,
  approximately 2.689 MB per input frame (summary at run completion:
  806,669,720 bytes).
- Video/preview directory: 389,812,665 bytes.
- Eight-video rendering/encoding observed wall time: approximately 408 s, or 1.36 s
  per input frame for all eight products together.
- Full 65-channel native maps were not saved per frame. Only selected temporal/static
  diagnostic probability maps and reset/first/middle/last cases were retained.

## Resets and resume

Seven T4 resets occurred:

- 1801: `initial_frame`;
- 1804, 1826, 1836, 1838, 1839 and 2054:
  `insufficient_valid_flow`.

The synthetic end-to-end resume test loads compatible state, verifies contiguous frames
and fingerprints, and does not rewrite completed masks. A real resume attempt against
the completed 300-frame artifact was also made after the final fingerprint hardening:
it failed closed before model execution because the artifact contains the legacy static
fingerprint `c50103e4777dad8f`, while final code fingerprints model config, Article 1
policy and taxonomy/mapping contents together (`3120b71056c9b041`). The last mask mtime
remained unchanged. The artifact is complete and readable, but its legacy checkpoints
must not be resumed by final code without a full compatible rerun. This is an explicit
checkpoint limitation, not a silently accepted cache.

## Video deliverables and validation

Directory: `output/article1/temporal_videos_30s/`

| file | geometry | bytes |
|---|---:|---:|
| `01_static_raw.mp4` | 1288×760 | 25,555,811 |
| `02_temporal_stabilized.mp4` | 1288×760 | 25,224,729 |
| `03_static_vs_temporal.mp4` | 2016×760 | 50,706,553 |
| `04_thin_static_vs_temporal.mp4` | 2016×760 | 52,871,222 |
| `05_flow_validity_and_occlusion.mp4` | 2016×760 | 62,620,994 |
| `06_temporal_provenance.mp4` | 2016×760 | 53,656,950 |
| `07_unknown_raw_vs_temporal.mp4` | 2016×760 | 59,348,401 |
| `08_gaze_raw_vs_temporal.mp4` | 2016×760 | 50,664,912 |

Every file is H.264, 10 fps, 30.000 s and 300 frames. `ffprobe` verified the headers;
OpenCV decoded all 300 frames of every file sequentially with one constant geometry.
The encoder padded the requested 756-pixel height to 760 for macroblock compatibility,
consistently across like-for-like comparisons.

Additional visual artifacts:

- `contact_sheet.jpg`;
- `qa_contact_categories.jpg`;
- `preview_difference.png`;
- `preview_propagated.png`;
- `preview_reset.png`;
- `preview_lane_recovery.png`;
- `qa_candidates.json`;
- `render_manifest.json`.

## Visual QA

The complete frame-by-frame review is in
`reports/article1_temporal_visual_qa.md`. Main observations:

- useful-looking short lane/regulatory recoveries occur, especially at 1938, 1916,
  1887, 1910 and 1921;
- no long displaced vehicle trail was apparent, consistent with TTL=1 and current
  support gates;
- the dominant failure is inherited from the static model: cockpit/mirrors/hands are
  sometimes labeled as vehicle or pedestrian-like classes, and temporal processing
  cannot correct the semantic mistake;
- road borders/reflections can receive uncertain thin extensions;
- all resets visibly return to current-only provenance;
- no clearly reviewable external pedestrian/two-wheeler was present in the five
  metric-selected candidates; the selected frames expose hand/ego-body confusion;
- traffic-light candidates are very small and remain visually ambiguous.

These judgments are not GT.

## Gaze diagnostics

`outputs/article1/auto_temporal_180_210/article1_temporal/gaze/`
`article1_temporal_gaze_diagnostics.parquet` contains post-hoc raw/temporal gaze class,
confidence, unknown status, switch, propagation age and provenance. Gaze alignment was
valid on 285/300 frames. The segmentation modules have no gaze input.

## Tests and gates

- Full suite: **150 passed**.
- Existing baseline: 124 tests retained.
- New temporal/video tests: 26 (25 temporal module/pipeline tests plus renderer test).
- Covered gates include identity/known translations, vector scaling,
  forward/backward inconsistency, occlusion/invalid flow, all reset classes,
  normalization/determinism, current-evidence dominance, unknown recovery, TTL,
  propagation/class age, hysteresis, thin and dynamic policies, provenance, raw
  preservation, portable resume, incompatible cache, missing frames/static inputs,
  manifest, rendering geometry, and cross-class propagation-age semantics.
- Source-level and interface gates fail if the primary temporal segmentation reads gaze
  coordinates, indexes frame `t+1`, exposes a future-frame input, or introduces
  bidirectional processing.
- All eight MP4 files passed header and full sequential decode checks.

## Reproduction commands

Representative commands (all local/offline):

```bash
python -m aria_drive_seg extract \
  --vrs unknown_20260528_090955.vrs \
  --output outputs/article1/auto_temporal_180_210 \
  --start-time 180 --end-time 210 --frame-step 1 --offline

python -m aria_drive_seg article1 segment-temporal \
  --input outputs/article1/auto_temporal_180_210 \
  --vehicle-type car --session-id auto_temporal_180_210 \
  --participant-id checkpoint1 \
  --config configs/article1/temporal_segmentation.yaml --offline

python scripts/analyze_article1_temporal.py \
  --frames outputs/article1/auto_temporal_180_210 \
  --temporal outputs/article1/auto_temporal_180_210/article1_temporal

python scripts/render_article1_temporal.py \
  --frames outputs/article1/auto_temporal_180_210 \
  --temporal outputs/article1/auto_temporal_180_210/article1_temporal \
  --output output/article1/temporal_videos_30s --fps 10

python -m pytest -q
```

## Final limits

- No reviewed GT; no accuracy, precision, recall or IoU claim.
- No definitive T0–T4 winner and no claim that lower unknown/flicker is better.
- One car clip only; no motorcycle or auto-vs-moto result.
- No training, cockpit pipeline/fusion or complete-VRS run.
- Causal only; non-causal/bidirectional smoothing is not implemented.
- Thresholds are initial engineering values, not GT-selected optima.
- The completed run's legacy checkpoints intentionally fail the strengthened final
  static-policy fingerprint gate; outputs remain valid as a completed, non-resumable
  checkpoint artifact.

The delivered result is a conservative, gaze-independent temporal baseline that remains
directly comparable with the preserved static result.
