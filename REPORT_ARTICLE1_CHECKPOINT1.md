# Article 1 auto–moto — checkpoint 1

Date: 2026-07-27  
Branch: `feature/auto-moto-semantic-gaze`

## Implemented and tested

- Reduced 14-class functional taxonomy (IDs 0–13).
- Complete 65-class Mapillary → Article 1 mapping with optional subtype attributes.
- Probability-space native-to-macro aggregation; no native-argmax-first shortcut.
- Explicit confidence/entropy unknown handling, distinct from other_environment.
- Native masks/probabilities, Article 1 probabilities, base masks, composite masks,
  confidence, entropy, top-2 and metadata outputs.
- Separate lane/regulatory thin layer with conservative component filtering and priority
  over road_surface.
- Resumable, fingerprinted stage manifest including taxonomy/mapping hashes.
- Article 1 external CLI and native/reduced diagnostic renderer.
- Stable cockpit SegFormer output contract that blocks training without reviewed annotations.
- Separate Article 1 annotation area and leakage-resistant manifest schema.

All repository tests pass: **87 passed**. The mapping covers all 65 native labels; zero are
unmapped.

## Real auto runs

### Ten-frame smoke test

Source: the ten existing extracted automobile frames copied from `val10` into a new ignored
output directory; `val10` itself was not modified.

- Frames: 10
- Mean inference/export time: 17.342 s/frame
- Mean unknown rate: 0.170%
- Class present in frames: road_surface 10, lane_marking 10, vehicle 10,
  road_boundary_or_obstacle 10, control_and_ego_vehicle 10, other_environment 10,
  pedestrian 7, traffic_sign 7, regulatory_road_marking 5, two_wheeler 1,
  traffic_light 1.

These are model-output presence counts, not accuracy measurements.

### Thirty-second external diagnostic

- Requested source interval: 180–210 s
- Source frames: 1805–2100, every fifth RGB frame
- 60 frames at 2 fps; output duration 30.0 s
- Mean inference/export time: 17.156 s/frame
- Peak PyTorch allocated VRAM: 5340.2 MB
- Mean unknown rate: 0.879%
- Mean thin lane pixels/frame: 20,354
- Mean thin regulatory pixels/frame: 5,905
- Zero processing errors and zero unmapped native labels

Generated and reopened successfully:

- `output/article1/videos_30s/01_mapillary_native.mp4`
- `output/article1/videos_30s/02_article1_external.mp4`

Both are H.264-readable, 2432×1512, 60 frames, 2 fps and 30.0 s.

## Visual assessment (not GT accuracy)

Working qualitatively:

- road surface is coherent;
- lane markings remain visible above road_surface;
- cars are consistently collapsed into vehicle;
- road boundaries/sidewalk/curb are visually separated;
- traffic signs and pedestrians appear on relevant smoke frames;
- cockpit/visible ego vehicle is separated from the outside by Mapillary's Ego Vehicle class.

Problematic or pending calibration:

- unknown rate is likely too low; confidence and entropy thresholds require reviewed GT;
- regulatory road marking has occasional broad/incorrect activations and must be calibrated;
- thin lane masks can respond to reflections or high-contrast cockpit/road boundaries;
- Mapillary's Ego Vehicle produces a broad control_and_ego_vehicle region; the future
  cockpit SegFormer must refine mirror/instrument/control functions;
- the local checkpoint emits a Transformers load warning for two newly initialized Swin
  layernorm parameters; this must be resolved or explicitly frozen/validated before paper runs;
- no Article 1 GT exists yet, therefore no IoU/F1/accuracy claim is made.

## Blocked by missing inputs

- SegFormer-B2 training: needs manually reviewed, balanced car and motorcycle cockpit masks.
- Motorcycle results: no motorcycle VRS supplied.
- Fusion and six-video fused/gaze set: needs a valid cockpit checkpoint.
- Paired alignment: needs a motorcycle route/session and route evidence.
- Statistical and cross-domain results: need multiple participants/sessions and valid splits.

## Next required step

Prepare the balanced Article 1 annotation package (at least 60 car + 60 motorcycle frames),
manually review Grounded-SAM2 B/C cockpit proposals, then calibrate external/marking thresholds
and train the single shared SegFormer-B2. Do not run full sessions before those gates pass.
