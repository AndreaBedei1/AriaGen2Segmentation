# Article 1 semantic camera fusion

Date: 2026-07-29

Base branch: `feature/article1-temporal-stabilization` at `06f08ca`

Work branch: `feature/article1-semantic-camera-fusion`

## Delivered scope

The new stage creates one dense, frame-independent Article 1 semantic mask by fusing
the preserved external Mask2Former result with a cockpit evidence provider. It does
not modify the static or temporal source output and does not use gaze, optical flow,
tracking, propagation, previous frames or future frames.

The 13 final classes are `road_surface`, `lane_marking`,
`regulatory_road_marking`, `vehicle`, `two_wheeler`, `pedestrian`,
`traffic_light`, `traffic_sign`, `road_boundary_or_obstacle`, `mirror`,
`instrument_display`, `control_and_ego_vehicle` and `other_environment`.
Internal `unknown` is retained only as diagnostic evidence; final masks contain IDs
1–13 exclusively.

## Real models and temporary fallback

- External, real: the raw static Mask2Former/Mapillary Article 1 evidence preserved
  in `article1_temporal/static_masks` and `static_confidence`. Reusing it preserves
  exact comparability and avoids a redundant external inference.
- Internal, temporary fallback: local Grounding DINO
  (`IDEA-Research/grounding-dino-base`) plus local SAM2.1 Hiera Large, restricted to
  cockpit prompts, plus a low-confidence geometric bottom-band prior.
- Not available: `weights/segformer-b2-article1-cockpit`. The proxy is explicitly
  recorded as `fallback=true` in every manifest/metadata file and is not reported as
  a trained cockpit model.

When a reviewed checkpoint becomes available, `mode: auto` selects the existing
fail-closed `CockpitSegFormer` provider without changing the fusion/output contract.

## Fusion

The two streams compete pixel by pixel using confidence, normalized entropy and
class-priority scores. Dedicated internal thresholds apply to mirror, display and
control classes. The geometric prior may replace only weak `unknown`, `vehicle`,
`road_boundary_or_obstacle` or `other_environment` external evidence. Strong road
users, signs, lights and markings are protected by external priority.

The last unresolved pixels are assigned to `other_environment` with confidence zero,
entropy one and `dense_other_environment_fill` provenance. This is what makes the
contract dense without inventing high-confidence evidence.

Outputs are atomic, resumable and fingerprinted. An incompatible manifest fails
closed. A real resume on the completed 300-frame artifact did not rewrite completed
frame masks and retained the original peak RAM/VRAM measurements.

## Clip and experiments

- VRS: `unknown_20260528_090955.vrs`
- requested interval: 180.0–210.0 s
- consecutive frame range: 1801–2100
- capture timestamps: 1235364468985–1265262269475 ns
- frames: 300, nominal 10 fps
- geometry: 2016×1512, no rotation, no missing frame indices
- staged runs: first 10 frames, then all 300 frames

### Ten-frame smoke

- frames: 1801–1810
- dense coverage: 100%
- holes/invalid IDs: 0
- external selected: 81.9449%
- Grounded-SAM2 proxy selected: 2.0527%
- geometric proxy selected: 9.1317%
- dense fill: 6.8707%
- mean total: 1780.16 ms/frame, including model load warm-up
- peak RAM/VRAM: 2871.53/3274.34 MB

### Thirty-second run

The 300 masks contain 914,457,600 pixels. Exhaustive validation found zero class-0
pixels, zero IDs outside 1–13 and one constant 1512×2016 geometry.

- dense coverage: 100%
- external selected: 97.0538%
- Grounded-SAM2 proxy selected: 0.3189%
- geometric proxy selected: 1.4845%
- diagnostic dense fill: 1.1428%
- internal conflict wins: 1.8034%
- mean final confidence/entropy: 0.91065/0.16302
- mean total: 916.89 ms/frame
- mean internal inference: 574.09 ms/frame after warm-up
- peak RAM/VRAM: 2928.62/3274.34 MB
- semantic-camera data: 303,488,042 bytes

All 13 final classes occur in the clip. This is a coverage and execution result, not
an accuracy result.

## Artifacts

Semantic data:

`outputs/article1/auto_temporal_180_210/semantic_camera/`

Ten-frame smoke data:

`outputs/article1/auto_temporal_180_210/semantic_camera_smoke10/`

Videos and visual QA:

`output/article1/semantic_camera_30s/`

| file | geometry | bytes |
|---|---:|---:|
| `01_semantic_camera_dense.mp4` | 1280×960 | 34,159,802 |
| `02_external_internal_fusion.mp4` | 1920×560 | 36,615,428 |

Both videos are H.264/yuv420p, 10 fps, 30.000 seconds and 300 frames. `ffprobe`
validated the headers and OpenCV decoded every frame with constant geometry.
The output directory also contains a contact sheet and previews for maximum internal
selection, dense fill, conflict and the midpoint of the dense video.

## Source changes

- `aria_drive_seg/article1/semantic_camera.py`: evidence readers/providers, dense
  fusion, provenance, atomic output, manifest and resume.
- `configs/article1/semantic_camera.yaml`: model selection and fusion policy.
- `aria_drive_seg/cli.py`: `article1 semantic-camera` command.
- `scripts/render_article1_semantic_camera.py`: dense and external/internal/final
  comparison videos plus reproducible QA selection.
- `tests/test_article1_semantic_camera.py`: fusion, density, fail-closed inputs,
  resume/source preservation, CLI and renderer tests.
- `docs/article1_semantic_camera.md`: interface and reproduction.
- `reports/article1_semantic_camera_visual_qa.md`: non-GT visual review.

Thematic source commits before this report:

- `ae46aae` — dense semantic-camera fusion core;
- `bea92d4` — CLI and renderer;
- `80bd11f` — preserve GPU resource metrics on resume.

## Verification

- full automated suite: 161 passed
- Python bytecode compilation: passed
- Git whitespace check: passed
- 300 source frames/masks: contiguous and geometry-consistent
- 914,457,600 final pixels: no hole and no invalid class
- real completed-run resume: no mask rewrite
- visual QA: contact sheet and four targeted previews inspected
- MP4 validation: header probe and full sequential decode passed

## Limits and recommended next step

There is no reviewed cockpit model, no GT and no motorcycle run. The entropy for this
artifact is a conservative top-1 proxy because full external macro probabilities were
not uniformly available. Open-vocabulary cockpit masks can be fragmented; the
geometric prior is deliberately coarse. A dense map is not automatically a correct
map.

The next step is to annotate a balanced car/motorcycle cockpit set, train and review
the existing four-class cockpit SegFormer, then rerun the same fusion contract with
the proxy disabled and evaluate per-class IoU plus boundary quality against held-out
ground truth.
