# Article 1 semantic-camera final pass

Date: 2026-07-31

Base branch: `feature/article1-semantic-camera-video-stabilization` at
`207fefe`

Work branch: `feature/article1-semantic-camera-final-pass`

## Scope and scientific separation

This change adds a final offline/presentation stage. It does not replace or
silently alter the frame-independent scientific semantic-camera baseline.

Read-only inputs:

- scientific semantic camera:
  `outputs/article1/auto_temporal_180_210/semantic_camera/`;
- preceding offline presentation:
  `outputs/article1/auto_temporal_180_210/semantic_camera_video_stabilized/`;
- existing rectified RGB frames and 299 cached adjacent-pair flows.

New local presentation data:

`outputs/article1/auto_temporal_180_210/semantic_camera_final_pass/`

The complete scientific raw contract was SHA-256 hashed before and after the
300-frame run. Both hashes are:

`fa9f1bdff05c7827c95cb7127cae819d05db4c7edecad8f1d2e7c3302c94a5ff`

The contract covers the source manifest plus all final, external and internal
masks/confidences. The hashes match, and the manifest explicitly records that
the raw scientific output was preserved.

## Temporal final pass

The final mode is non-causal and uses a symmetric ±10-frame window, equivalent
to about ±1 second at 10 fps. Every neighboring prediction is first warped into
the target geometry through the cached bidirectional optical flow.

The temporal vote uses:

- class confidence;
- class-specific vote weight and temporal decay;
- temporal distance;
- forward/backward flow validity;
- a continuous local-validity weight derived from the flow-valid mask;
- class-specific persistence and confidence floors;
- strong-current external-class protection;
- activation/deactivation hysteresis.

Lane marking, regulatory road marking, road boundary, mirror, instrument
display and control/ego receive longer persistence. Hands remain mapped to
`control_and_ego_vehicle`, because the Article 1 taxonomy has no separate hand
class. Vehicles, pedestrians and two-wheelers use short, confidence-gated
persistence. A new internal region must have support from both past and future;
this suppresses one-sided cockpit propagation while still filling intermediate
dropouts.

## Road-line geometry

The lane post-processor is not a plain dilation. It uses:

- linear closing kernels at 0°, 30°, 60°, 90°, 120° and 150°;
- orientation-binned component processing;
- endpoint linking only for compatible component directions;
- a maximum gap of 30 pixels at 1008×756 processing geometry;
- bridge-direction and component-orientation gates;
- mandatory road support or class-probability proxy support;
- exact semantic barriers for vehicles, two-wheelers, pedestrians, mirrors,
  displays and control/ego;
- component length/aspect filtering for synthesized regions;
- Zhang-Suen skeletonization after linking;
- controlled expansion toward the estimated original thickness.

Existing road-supported model/temporal line pixels are the recall anchor.
Shape filtering is strict for synthesized regions and for evidence that fails
the road/probability gate. Regulatory markings use temporal recovery and a more
conservative morphology policy to avoid treating non-elongated symbols like
lane stripes.

Road boundaries/curbs use confidence, temporal continuity, a road-edge band,
connected components, oriented-independent closing, small-hole filling and
removal of isolated incompatible regions.

## Provenance and outputs

Each final frame has a dense final mask, confidence, support count, JSON
metadata and per-pixel provenance:

1. `current_model`;
2. `temporal_evidence`;
3. `morphological_link`;
4. `final_fill`.

The only final evaluation video generated is:

`output/article1/semantic_camera_final_30s.mp4`

- 300 frames;
- 10 fps;
- 30.000 seconds;
- 1280×960;
- 45,376,872 bytes;
- full sequential OpenCV decode passed.

It contains only RGB with the final segmentation overlay, a discrete legend
and frame index.

The tracked external-review package is:

`reports/article1_semantic_camera_final_qa/`

It contains five 10-frame sequences (57 files, 4.9 MB total), covering lane
dropout, compatible broken-line gaps, intermittent road boundary, internal
flicker and internal/external conflict. Each sequence has a JPG contact sheet,
three RGB JPGs, three colored-mask PNGs, three final-overlay JPGs and a Markdown
description. The root README links every sequence and records frame indices and
timestamps.

## Pre-GT diagnostics

All values below compare the preceding presentation with the final pass at
1008×756 processing geometry. Flow-aligned values are the primary temporal
diagnostics because the scene and cockpit move across image coordinates.

| Metric | Before | Final | Change |
|---|---:|---:|---:|
| flow-aligned switch rate | 2.5356% | 2.1572% | −14.93% relative |
| flow-aligned lane/marking continuity | 51.3781% | 55.2112% | +3.83 points |
| short/broken components per class-frame | 7.5217 | 6.7283 | −10.55% |
| mean line-component length | 18.5632 px | 21.2630 px | +14.54% |
| flow-aligned internal flicker | 0.1232% | 0.1013% | −17.83% |
| flow-aligned internal persistence | 95.1288% | 96.0760% | +0.95 points |
| flow-aligned road-boundary temporal IoU | 74.7994% | 84.2635% | +9.46 points |
| dense coverage | 100% | 100% | unchanged |
| invalid IDs | 0 | 0 | unchanged |

The final filter added 137,934 pixels at processing geometry:

- 121,752 from constrained morphological linking;
- 16,182 from final fill;
- 0 outside road support.

The wider temporal stage used 1,789,483 final temporal-evidence pixels at
processing geometry. The full run reused 299/299 flow pairs, averaged
1,921.25 ms/frame for final-pass decisions, and peaked at 8,904.39 MB RAM.

Image-coordinate diagnostics are retained rather than hidden:

- switch rate: 7.7783% → 7.8797% (+0.10 points);
- lane/marking continuity: 25.9603% → 25.1268% (−0.83 points);
- isolated internal flicker: 0.5485% → 0.5684% (+0.02 points).

These differences are consistent with moving boundaries and increased
flow-aligned persistence, but remain a limitation. No accuracy improvement is
claimed without ground truth.

## Verification

- focused final-pass/video tests: 22 passed;
- complete automated suite: 183 passed in 25.66 s;
- compatible line-gap link: passed;
- incompatible orientations remain unlinked: passed;
- vehicle/cockpit barriers remain uncrossed: passed;
- bidirectional temporal continuity and internal activation gate: passed;
- dense coverage, valid IDs, raw immutability and determinism: passed;
- `git diff --check`: passed;
- real 50-frame tuning smoke runs: passed;
- real 300-frame final run: passed;
- final MP4 full decode: 300/300 frames.

No complete VRS run was executed.

## Limits

- There is no reviewed segmentation ground truth, so these are continuity and
  geometry diagnostics, not accuracy results.
- The internal stream still uses the explicitly marked local Grounding DINO +
  SAM2.1/geometric fallback; there is no reviewed Cockpit SegFormer checkpoint.
- The clip is a single 30-second car interval. No motorcycle run was added.
- Flow validity is not universal; invalid regions are gated rather than
  propagated.
- Regulatory road markings gain flow-aligned continuity but can still split
  into more small components than the preceding presentation.
- Image-coordinate switch, lane continuity and internal flicker are slightly
  worse even while their flow-aligned diagnostics improve.
- Thin structures can still be missed when neither the current model,
  neighboring aligned evidence nor the road/probability gate supports them.
