# Article 1 — preliminary car vs motorcycle comparison

Generated: 2026-07-31T21:41:25.282282+00:00

## Status: `exploratory_preliminary`

**This is not a scientific result.** It exists to expose failure modes and domain shift before annotation. Nothing here supports a claim about how the vehicle changes visual attention.

- The car recording is a provisional 10 fps development baseline; the final protocol is 15 fps for both vehicles.
- No reviewed ground truth exists, so no accuracy, IoU, precision or recall is reported: these are coverage, stability and agreement diagnostics only.
- The sampling-rate difference is never used as a feature and no vehicle classifier is trained here.
- The definitive comparison must be repeated when both recordings are acquired at the final protocol rate.

## What is being compared

| | car | motorcycle |
|---|---:|---:|
| recording | `car_2e84f0c3e245` | `motorcycle_5ab8604a14df` |
| frames | 300 | 450 |
| duration | 29.90 s | 29.93 s |
| measured rate | 10.0008 Hz | 15.0012 Hz |

The two clips have different frame counts because they have different sampling rates, not because they cover different amounts of time. Every temporal quantity below is per second for exactly that reason.

## Semantic coverage

Coverage, not accuracy. There is no ground truth to be accurate against.

| class | car pixel share | motorcycle pixel share | difference | car presence | motorcycle presence |
|---|---:|---:|---:|---:|---:|
| `unknown` | 0.0000% | 0.0000% | +0.0000% | 0.0% | 0.0% |
| `road_surface` | 3.4215% | 28.5883% | +25.1668% | 100.0% | 96.9% |
| `lane_marking` | 0.4614% | 3.5202% | +3.0588% | 95.0% | 98.9% |
| `regulatory_road_marking` | 0.0911% | 1.6453% | +1.5542% | 22.7% | 44.9% |
| `vehicle` | 2.4639% | 0.6712% | -1.7927% | 100.0% | 83.6% |
| `two_wheeler` | 0.0203% | 0.0428% | +0.0225% | 11.3% | 20.0% |
| `pedestrian` | 0.1416% | 0.0073% | -0.1343% | 25.0% | 21.6% |
| `traffic_light` | 0.0002% | 0.0012% | +0.0011% | 2.3% | 5.1% |
| `traffic_sign` | 0.0855% | 0.2334% | +0.1480% | 65.3% | 78.4% |
| `road_boundary_or_obstacle` | 1.0751% | 7.1464% | +6.0713% | 99.7% | 100.0% |
| `mirror` | 0.0027% | 0.0195% | +0.0169% | 8.0% | 22.2% |
| `instrument_display` | 0.0459% | 0.0202% | -0.0257% | 30.0% | 48.9% |
| `control_and_ego_vehicle` | 1.7548% | 0.7700% | -0.9848% | 100.0% | 99.3% |
| `other_environment` | 90.4360% | 57.3341% | -33.1019% | 100.0% | 100.0% |

## Confidence, entropy, provenance

| quantity | car | motorcycle |
|---|---:|---:|
| mean final confidence | 0.9106 | 0.9615 |
| mean final entropy | 0.1612 | 0.0654 |
| dense coverage | 1.0000 | 1.0000 |
| invalid class ids | 0 | 0 |
| cockpit pixel share | 1.8034% | 0.8097% |
| unreviewed-fallback frames | 100.0% | 100.0% |

### Provenance shares

| source | car | motorcycle |
|---|---:|---:|
| `conflict_external_wins` | 31.57% | 33.46% |
| `conflict_internal_wins` | 1.80% | 0.81% |
| `dense_other_environment_fill` | 1.14% | 0.72% |
| `external_mask2former` | 65.48% | 65.01% |

## Temporal stability, per second

| quantity | car | motorcycle |
|---|---:|---:|
| class switching per second | 0.6478 | 1.1495 |

switch rates are normalised per second; raw per-frame counts of recordings sampled at different rates are never compared.

**Caveat:** the two recordings are currently sampled at different rates; this metric is rate sensitive and must be recomputed once both are at the final protocol rate.

## Fragmentation

| class | car components per class-frame | motorcycle components per class-frame |
|---|---:|---:|
| `road_surface` | 5.38 | 11.40 |
| `lane_marking` | 10.43 | 27.60 |
| `regulatory_road_marking` | 2.76 | 2.34 |
| `vehicle` | 6.85 | 2.26 |
| `two_wheeler` | 1.09 | 1.56 |
| `pedestrian` | 6.32 | 1.29 |
| `traffic_light` | 1.43 | 1.26 |
| `traffic_sign` | 2.11 | 3.02 |
| `road_boundary_or_obstacle` | 4.87 | 7.57 |
| `mirror` | 8.83 | 12.01 |
| `instrument_display` | 8.88 | 11.34 |
| `control_and_ego_vehicle` | 66.41 | 165.88 |
| `other_environment` | 240.57 | 961.47 |

## Gaze

- **car**:
  - `frames`: 300
  - `valid_fraction`: 0.95
  - `abs_dt_ms_median`: 15.403864500000001
- **motorcycle**:
  - `frames`: 450
  - `valid_fraction`: 0.94
  - `abs_dt_ms_median`: 15.567246

gaze is used only after segmentation and never as a segmentation input.

## Hands

- car: not available (no data)
- **motorcycle**:
  - `status`: review_candidates_only
  - `per_state`: {"visible": 0, "partially_visible": 0, "occluded": 0, "out_of_frame": 304, "motion_blurred": 8, "uncertain": 175, "not_visible": 413}
  - `evaluable_fraction`: 0.0
  - `candidate_false_mask_count`: 132

## Route pairing quality

- status: `exploratory_preliminary`
- accepted pairs: 181 (19.0%)
- median pair distance: 16.3 m
- median pair quality: 0.510


## Where the differences come from

### Problems common to both domains

- both recordings use the unreviewed Grounding DINO + SAM2.1 cockpit fallback: no trained cockpit model exists yet
- dense coverage is complete in both recordings
- no invalid class id in either recording

### Specific to the motorcycle

- nothing identified

### Specific to the car

- nothing identified

### Attributable to the external model

- road_surface: 28.59% on the motorcycle vs 3.42% on the car
- lane_marking: 3.52% on the motorcycle vs 0.46% on the car
- regulatory_road_marking: 1.65% on the motorcycle vs 0.09% on the car
- vehicle: 0.67% on the motorcycle vs 2.46% on the car
- road_boundary_or_obstacle: 7.15% on the motorcycle vs 1.08% on the car
- other_environment: 57.33% on the motorcycle vs 90.44% on the car

### Attributable to the cockpit proxy

- nothing identified

### Attributable to the fusion

- mean final entropy differs (0.065 vs 0.161): the fusion is less certain in one domain

### Temporal

- per-second class switching is 1.150 on the motorcycle vs 0.648 on the car

### Probably due to the different sampling rate

- per-second switching still depends on how densely the motion was sampled; part of this difference is the 10 vs 15 fps artefact rather than a property of the vehicle

### Probably due to domain shift

- lane_marking differs markedly: the rider's viewpoint is higher, less occluded and more roll-dynamic than the driver's
- other_environment differs markedly: the rider's viewpoint is higher, less occluded and more roll-dynamic than the driver's
- regulatory_road_marking differs markedly: the rider's viewpoint is higher, less occluded and more roll-dynamic than the driver's
- road_boundary_or_obstacle differs markedly: the rider's viewpoint is higher, less occluded and more roll-dynamic than the driver's
- road_surface differs markedly: the rider's viewpoint is higher, less occluded and more roll-dynamic than the driver's
- vibration, roll and the absence of a windscreen change the motorcycle's image statistics independently of any model quality difference

## What this comparison may not be used for

- It may not support any statement about how the vehicle changes visual attention. That requires the re-recorded car and reviewed ground truth.
- The sampling-rate difference is not a feature and no vehicle classifier is trained here. A classifier given this data could learn the rate instead of the vehicle.
- No accuracy, IoU, precision or recall appears anywhere above, by design.

## What it is genuinely useful for

- It shows where the car-tuned pipeline breaks on a motorcycle, which is what the frozen run was for.
- It sizes the cockpit problem: the motorcycle exposes far less ego structure to the camera, and the geometric prior built for a dashboard does not describe a handlebar.
- It tells the annotation stage which classes and conditions need the most human attention.

