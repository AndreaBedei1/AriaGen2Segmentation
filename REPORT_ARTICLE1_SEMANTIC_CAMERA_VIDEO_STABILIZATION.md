# Article 1 semantic-camera video stabilization

Date: 2026-07-29

Base branch: `feature/article1-semantic-camera-fusion` at `11f1fe3`

Work branch: `feature/article1-semantic-camera-video-stabilization`

## Scope and separation

The existing semantic-camera result remains the scientific raw baseline and was used
read-only. The new stage adds:

- an improved per-frame fusion with stronger cockpit priority;
- a separate offline, non-causal presentation result using past and future frames.

The models were not changed or rerun. External evidence remains Mask2Former/Mapillary.
Internal evidence remains the explicitly marked local Grounding DINO + SAM2.1 cockpit
fallback plus geometric prior because the reviewed Cockpit SegFormer checkpoint is
still absent.

## Fusion changes

- Existing internal wins can no longer disappear during improved raw fusion.
- Mirror, instrument-display and control evidence has lower activation thresholds,
  higher class priority and a cockpit-position bonus.
- Strong road, lane, regulatory, vehicle, pedestrian, two-wheeler, sign and light
  pixels are protected by class-specific confidence thresholds.
- The protection gate is repeated after upsampling the presentation decision.
- Hands remain `control_and_ego_vehicle`, because Article 1 has no separate hand class.

An exhaustive audit found 48,414,523 strong external pixels. Both improved raw and
presentation output preserved 100% of them.

## Bidirectional presentation stabilization

- local OpenCV DIS flow at 1008×756;
- 299 adjacent pairs, with forward and backward flow and validity in both directions;
- symmetric window of ±5 frames;
- confidence-weighted voting, class priorities and temporal decay;
- activation/deactivation hysteresis;
- TTL 5 for mirror, display and control;
- TTL 3 for lane and regulatory road marking;
- TTL 1 for dynamic/external classes;
- no propagation for `other_environment`;
- per-pixel provenance for past, future, bidirectional and protected decisions.

This mode is deliberately recall-oriented. It can retain an incorrect proxy
detection, so it remains a presentation product rather than a scientific replacement.

## Clip and execution

- VRS: `unknown_20260528_090955.vrs`
- interval: 180.0–210.0 s
- frame range: 1801–2100
- frames: 300 consecutive frames at 10 fps
- full geometry: 2016×1512
- processing geometry: 1008×756
- flow-valid fraction: 54.66% forward, 55.31% backward on average
- initial presentation stage: 761.32 ms/frame after flow preparation
- resume invocation with all caches: 21.09 ms/frame
- peak RAM: 9254.35 MB
- presentation data and flow cache: 1,702,555,226 bytes

The ten-frame smoke run completed before the full run. Against current fusion it
showed switch-rate reduction 23.26%, flicker reduction 26.73% and internal persistence
gain 34.36 percentage points. Full-run metrics supersede those values.

## Full-run metrics

All values are pre-GT temporal diagnostics.

| metric | current scientific | improved raw | presentation |
|---|---:|---:|---:|
| image-coordinate switch rate | 6.4793% | 7.8201% | 7.7783% |
| image-coordinate consistency | 93.5207% | 92.1799% | 92.2217% |
| isolated one-frame flicker | 2.0284% | 2.6305% | 1.9081% |
| image-coordinate internal persistence | 23.2704% | 29.1717% | 74.7320% |
| image-coordinate lane/marking continuity | 20.8194% | 20.3314% | 25.9603% |
| flow-aligned switch rate | 5.5580% | 7.4743% | 2.5356% |
| flow-aligned consistency | 94.4420% | 92.5257% | 97.4644% |
| flow-aligned internal persistence | 27.1774% | 34.7112% | 95.1288% |
| flow-aligned lane/marking continuity | 26.5675% | 25.5725% | 51.3781% |

Presentation versus improved raw:

- image-coordinate switch rate: −0.53%;
- isolated one-frame flicker: −27.46%;
- flow-aligned switch rate: −66.08%;
- internal persistence: +45.56 image-coordinate points and +60.42
  flow-aligned points;
- lane/marking continuity: +5.63 image-coordinate points and +25.81
  flow-aligned points.

Presentation versus the original current fusion:

- isolated flicker: −5.93%;
- flow-aligned switch rate: −54.38%;
- flow-aligned internal persistence: +67.95 points;
- flow-aligned lane/marking continuity: +24.81 points.

The image-coordinate switch rate is higher than the original sparse baseline because
internal coverage grows from 1.8034% to 9.1798% and those boundaries move in image
coordinates. This is why both raw and flow-aligned measures are retained.

## Density, provenance and safety gates

Across 914,457,600 pixels in each mode:

- holes: 0;
- IDs outside 1–13: 0;
- all 13 final classes present;
- internal coverage: current 1.8034%, improved raw 3.0679%, presentation 9.1798%;
- lane/regulatory coverage: current 0.5525%, improved 0.5157%,
  presentation 0.5969%.

Presentation provenance:

- improved current: 93.1276%;
- forward from past: 2.7219%;
- backward from future: 2.4934%;
- bidirectional vote: 1.6519%;
- protected external: 0.0052%.

## Outputs

Presentation data:

`outputs/article1/auto_temporal_180_210/semantic_camera_video_stabilized/`

Video directory:

`output/article1/semantic_camera_video_stabilized_30s/`

| file | geometry | bytes |
|---|---:|---:|
| `01_current_fusion.mp4` | 1280×960 | 34,038,029 |
| `02_improved_fusion.mp4` | 1280×960 | 35,085,777 |
| `03_raw_vs_stabilized.mp4` | 1920×560 | 39,606,672 |
| `04_internal_regions_focus.mp4` | 1920×960 | 89,909,278 |
| `05_lane_and_marking_focus.mp4` | 1920×960 | 75,919,565 |

All five are H.264/yuv420p, 10 fps, 30.000 seconds and 300 frames. Header probing
and full sequential OpenCV decode passed.

The initial attempt to encode all five simultaneously was terminated before
completion. The renderer was hardened with `--videos`; each file was then encoded
sequentially and fully validated. Partial files were overwritten.

## Source changes and commits

- `aria_drive_seg/article1/semantic_camera_video.py`
- `configs/article1/semantic_camera_video.yaml`
- `aria_drive_seg/cli.py`
- `scripts/render_article1_semantic_camera_video.py`
- `tests/test_article1_semantic_camera_video.py`
- reproducibility, QA and final reports

Thematic implementation commits:

- `1238639` — improved video-fusion core;
- `34c4285` — resumable bidirectional presentation pipeline;
- `e39234d` — five-product renderer;
- `90ba96e` — flow-aligned diagnostics;
- `7b84f48` — sequential/resumable rendering.

## Verification and limits

- full automated suite: 174 passed;
- compile and whitespace gates: passed;
- real ten-frame and 300-frame runs: passed;
- real smoke resume: 9/9 flow cache hits, no mask rewrite;
- full resume: 299/299 flow cache hits;
- all masks: dense and valid;
- all videos: 300-frame full decode.

There is no reviewed GT, no motorcycle experiment and no trained cockpit SegFormer.
The presentation can retain fallback-model false positives, enlarge uncertain cockpit
regions and add small lane fragments. The next recommended step is a held-out,
annotated car/motorcycle cockpit set: train the existing Cockpit SegFormer, then tune
TTL and thresholds against per-class IoU, boundary quality and persistence before
using presentation output beyond visualization.
