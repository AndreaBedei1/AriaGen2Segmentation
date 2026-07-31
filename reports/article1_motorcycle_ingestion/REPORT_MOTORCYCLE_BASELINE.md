# Article 1 — frozen baseline on the motorcycle recording

Generated: 2026-07-31T21:41:24.738728+00:00

Branch: `feature/article1-motorcycle-ingestion`

## What was run, and what `frozen` means

The existing Article 1 pipeline was run on the motorcycle segment without changing a single checkpoint, prompt, threshold, class mapping, fusion rule, temporal parameter or fallback policy. The configuration bundle was hashed before and after the run and the two hashes are recorded in `logs/frozen_run.log`; a mismatch aborts the run.

This is deliberate. The point of a frozen first run is to see what a car-tuned pipeline actually does on a motorcycle, before anything is adapted to make it look better.

Stages, in order:

1. timestamp-driven extraction of the selected window
2. Mask2Former Swin-L (Mapillary Vistas) external segmentation plus the causal temporal stage
3. dense semantic-camera fusion with the Grounding DINO + SAM 2.1 cockpit proxy and the geometric fallback
4. causal presentation stabilisation
5. bidirectional presentation final pass (presentation only, kept separate from the causal scientific output)
6. semantic gaze alignment, strictly after segmentation

## Selected segment

| quantity | value |
|---|---|
| recording | `motorcycle_5ab8604a14df` |
| source frame indices | 13875 to 14324 |
| device timestamps (ns) | 3,099,822,119,217 to 3,129,751,762,010 |
| duration | 29.930 s |
| frames | 450 |
| estimated missing frames | 0 |
| largest interval | 66.81 ms |
| score | 0.7010 |
| windows evaluated | 195 |
| windows passing every quality gate | 175 |

Why this window:

- strongest contributions: visual_diversity 0.198, motion 0.188, semantic_diversity 0.131
- 450 frames over 29.93 s (0 estimated missing, largest interval 66.8 ms)
- external traffic proxy covers 1.02% of the sampled pixels
- cockpit proxy covers 2.90% of the sampled pixels
- hand tracking reports a hand in 69.3% of the window's frames (proxy signal, not visibility ground truth)

### Ranked alternatives

| rank | start frame | start s | score | motion | visual diversity | semantic diversity | traffic | cockpit proxy | hand-tracking share |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 13875 | 3099.8 | 0.7010 | 0.0396 | 0.344 | 0.502 | 0.0102 | 0.0290 | 69.33% |
| 6 | 11775 | 2959.8 | 0.5738 | 0.0274 | 0.240 | 0.526 | 0.0476 | 0.0228 | 63.78% |
| 7 | 13125 | 3049.8 | 0.5487 | 0.0300 | 0.262 | 0.502 | 0.0223 | 0.0126 | 74.22% |
| 8 | 3450 | 2404.9 | 0.5452 | 0.0286 | 0.252 | 0.489 | 0.0146 | 0.0401 | 75.33% |
| 15 | 9525 | 2809.8 | 0.5187 | 0.0262 | 0.302 | 0.464 | 0.0150 | 0.0227 | 47.78% |

The weighting favours motion, visual variety and semantic variety over easy footage, so the ranking cannot be won by a static, visually trivial window. Quality gates (missing frames, pauses, exposure, sharpness) are applied as hard constraints before scoring.

## Extraction

| quantity | value |
|---|---|
| recording | `motorcycle_5ab8604a14df` |
| source SHA-256 | `5ab8604a14dfafecb15fc58a86c8d99ac59e4c83b1cb6025e43d22efe264f6d7` |
| source stream | `214-1` |
| measured source rate | 15.00117 Hz |
| frames extracted | 450 |
| errors | 0 |
| window duration | 29.930 s |
| resampled | False |
| interpolated | False |
| synthetic frames | 0 |

Gaze: 30138 samples in the recording; 449 of 450 frames have at least one valid sample within +/-100 ms. Association is by timestamp; several samples per frame are preserved.

Hand tracking: 10046 samples; the left hand is tracked in 312 and the right in 0 of 450 frames. Tracked is not the same as visible, and an untracked hand is not evidence of an absent hand.

## Semantic camera

- external evidence: `mask2former_temporal_static_raw`
- internal stream: `grounded_sam2_cockpit_proxy` (fallback: True)
- frames: 450


## Presentation final pass

- `fingerprint`: 94ef6885f92f4106
- `flow_cache_hits`: 449
- `flow_pairs`: 449
- `frame_count`: 450
- `mean_ms_per_frame`: 2547.350655341967
- `peak_ram_mb`: 13114.99609375
- `presentation_is_non_causal`: True
- `raw_contract_sha256`: 4dfbddc44ba947738682e0f92c4660d8265bfb72748264ff047d8e127004b75f
- `raw_scientific_output_preserved`: True
- `stage`: article1_semantic_camera_final_pass_v1
- `window_radius_frames`: 10

### Final-pass diagnostics

```json
{
  "before_previous_presentation": {
    "frame_count": 450,
    "evaluation_geometry": [
      756,
      1008
    ],
    "class_switch_rate": 0.07817660458007608,
    "temporal_consistency": 0.921823395419924,
    "isolated_one_frame_flicker_rate": 0.017882207480931084,
    "isolated_one_frame_flicker_pixels": 6104941,
    "internal_class_persistence": 0.4633810195360585,
    "internal_mean_temporal_iou": 0.1245963834900875,
    "lane_and_marking_continuity": 0.31985848967487696,
    "class_temporal_iou": {
      "road_surface": 0.8289813201576581,
      "lane_marking": 0.3908369272244552,
      "regulatory_road_marking": 0.24888005212529873,
      "vehicle": 0.4269340386474244,
      "two_wheeler": 0.08972278871368959,
      "pedestrian": 0.118741800551267,
      "traffic_light": 0.015859587407421806,
      "traffic_sign": 0.1588226933779054,
      "road_boundary_or_obstacle": 0.6315985496380205,
      "mirror": 0.08332772395090311,
      "instrument_display": 0.121230148968792,
      "control_and_ego_vehicle": 0.1692312775505674,
      "other_environment": 0.9467118043062067
    },
    "disclaimer": "Pre-GT image-coordinate continuity diagnostics only; lower flicker can preserve false positives and is not accuracy.",
    "flow_aligned": {
      "class_switch_rate": 0.017675467949718658,
      "temporal_consistency": 0.9823245320502814,
      "internal_class_persistence": 0.8805512016893685,
      "internal_mean_temporal_iou": 0.5287544475840841,
      "lane_and_marking_continuity": 0.6674527668126063,
      "class_temporal_iou": {
        "road_surface": 0.9316826618587563,
        "lane_marking": 0.7891939353558372,
        "regulatory_road_marking": 0.5457115982693755,
        "vehicle": 0.7498691916952417,
        "two_wheeler": 0.3310591797527102,
        "pedestrian": 0.265689876164036,
        "traffic_light": 0.4275308740477941,
        "traffic_sign": 0.6995644598225449,
        "road_boundary_or_obstacle": 0.8746801482758094,
        "mirror": 0.4630018859108023,
        "instrument_display": 0.5416391724774362,
        "control_and_ego_vehicle": 0.5816222843640134,
        "other_environment": 0.9921443555735758
      },
      "validity": "backward-flow valid pixels only",
      "disclaimer": "Pre-GT flow-aligned continuity diagnostics only; lower switch rate can preserve false positives and is not accuracy."
    },
    "line_components": {
      "mean_broken_components_per_class_frame": 9.562222222222223,
      "mean_total_components_per_class_frame": 13.11,
      "mean_component_length_px": 53.11161989009484,
      "broken_component_definition": "PCA length < 25 px at processing geometry",
      "per_class": {
        "lane_marking": {
          "mean_total_components_per_frame": 23.18,
          "mean_broken_components_per_frame": 16.744444444444444,
          "mean_component_length_px": 53.626775735085836
        },
        "regulatory_road_marking": {
          "mean_total_components_per_frame": 3.04,
          "mean_broken_components_per_frame": 2.38,
          "mean_component_length_px": 49.183556572038526
        }
      }
    },
    "internal_flicker": {
      "isolated_internal_flicker_pixels": 502639,
      "isolated_internal_flicker_rate": 0.001472298403212696,
      "geometry": "image coordinates"
    },
    "flow_aligned_internal_flicker": {
      "isolated_internal_flicker_pixels": 80267,
      "isolated_internal_flicker_rate": 0.0008570246455845886,
      "valid_pixels": 93657750,
      "geometry": "flow-aligned neighbor evidence"
    }
  },
  "after_final_pass": {
    "frame_count": 450,
    "evaluation_geometry": [
      756,
      1008
    ],
    "class_switch_rate": 0.08036984745642878,
    "temporal_consistency": 0.9196301525435713,
    "isolated_one_frame_flicker_rate": 0.01857904327267724,
    "isolated_one_frame_flicker_pixels": 6342839,
    "internal_class_persistence": 0.463058790413662,
    "internal_mean_temporal_iou": 0.1250959468229154,
    "lane_and_marking_continuity": 0.322301673
```

### Causal stabilisation summary

```json
{
  "stage": "article1_semantic_camera_video_presentation_v1",
  "fingerprint": "4ca55dad8ddea68f",
  "source_fingerprint": "7a91d5c28022359d",
  "frame_count": 450,
  "presentation_is_non_causal": true,
  "uses_future_frames": true,
  "mean_presentation_ms_per_frame": 867.9874585800442,
  "initial_run_mean_presentation_ms_per_frame": 867.9874585800442,
  "last_invocation_mean_presentation_ms_per_frame": 867.9874585800442,
  "peak_ram_mb": 13638.38671875,
  "storage_bytes": 2541997570
}
```

## Per-class coverage on the motorcycle

Coverage, not accuracy: these numbers say how often and how much of the frame each class occupies, not whether it is right.

| class | frames present | mean pixel share | median share when present | mean components when present | max components |
|---|---:|---:|---:|---:|---:|
| `unknown` | 0.0% | 0.0000% | 0.0000% | 0.0 | 0 |
| `road_surface` | 96.9% | 28.5883% | 31.7034% | 11.4 | 33 |
| `lane_marking` | 98.9% | 3.5202% | 2.1984% | 27.6 | 340 |
| `regulatory_road_marking` | 44.9% | 1.6453% | 0.2159% | 2.3 | 17 |
| `vehicle` | 83.6% | 0.6712% | 0.3221% | 2.3 | 10 |
| `two_wheeler` | 20.0% | 0.0428% | 0.0433% | 1.6 | 9 |
| `pedestrian` | 21.6% | 0.0073% | 0.0124% | 1.3 | 4 |
| `traffic_light` | 5.1% | 0.0012% | 0.0196% | 1.3 | 4 |
| `traffic_sign` | 78.4% | 0.2334% | 0.1734% | 3.0 | 11 |
| `road_boundary_or_obstacle` | 100.0% | 7.1464% | 4.8830% | 7.6 | 28 |
| `mirror` | 22.2% | 0.0195% | 0.0060% | 12.0 | 82 |
| `instrument_display` | 48.9% | 0.0202% | 0.0040% | 11.3 | 97 |
| `control_and_ego_vehicle` | 99.3% | 0.7700% | 0.2085% | 165.9 | 2276 |
| `other_environment` | 100.0% | 57.3341% | 57.9566% | 961.5 | 1941 |

## Candidate failure modes

Detected over 450 frames (29.93 s at 15.001 Hz). Counts are also given per second so they stay comparable with a run sampled at a different rate.

These are **candidates**, not confirmed errors. Thresholds are relative to this run's own distribution, so a mode firing here means "unusual for this clip", not "wrong".

| candidate failure mode | events | per second |
|---|---:|---:|
| `mirror_not_detected` | 350 | 11.694 |
| `cockpit_absorbed_by_external` | 322 | 10.759 |
| `fragmentation` | 257 | 8.587 |
| `instrument_display_not_detected` | 230 | 7.685 |
| `false_hand` | 139 | 4.644 |
| `high_entropy` | 45 | 1.504 |
| `anomalous_confidence` | 45 | 1.504 |
| `hand_mask_propagated_after_disappearance` | 21 | 0.702 |
| `excessive_fallback` | 16 | 0.535 |
| `vibration_related_failure` | 11 | 0.368 |
| `blur_related_failure` | 5 | 0.167 |
| `internal_external_fusion_error` | 3 | 0.100 |
| `temporal_trail` | 1 | 0.033 |

Modes checked and **not** triggered: `missing_external_segmentation`, `wrong_class_candidate`, `flicker`, `false_cockpit_mask`, `handlebar_classified_as_other`, `visible_hand_not_proposed`.

Causality: this is an offline diagnostic over an already-produced run; it looks at neighbouring frames in both directions to describe flicker and trailing, and never feeds back into any mask

### QA sequences

Each directory holds consecutive frames with RGB, coloured mask, overlay, a contact sheet and a description.

| directory | mode | frames | severity |
|---|---|---:|---:|
| `qa/01_cockpit_absorbed_by_external` | `cockpit_absorbed_by_external` | 5 | 1.00 |
| `qa/02_mirror_not_detected` | `mirror_not_detected` | 3 | 0.50 |
| `qa/03_instrument_display_not_detected` | `instrument_display_not_detected` | 5 | 0.50 |
| `qa/04_excessive_fallback` | `excessive_fallback` | 5 | 0.18 |
| `qa/05_fragmentation` | `fragmentation` | 5 | 1.00 |
| `qa/06_high_entropy` | `high_entropy` | 5 | 1.00 |
| `qa/07_blur_related_failure` | `blur_related_failure` | 5 | 0.60 |
| `qa/08_vibration_related_failure` | `vibration_related_failure` | 5 | 0.60 |
| `qa/09_internal_external_fusion_error` | `internal_external_fusion_error` | 5 | 0.52 |

## Local video output

- `output/article1/motorcycle_baseline_30s/semantic_camera_moto_final.mp4` (not committed)

## Limits

- No reviewed ground truth exists for this data, so nothing here is an accuracy statement. The vocabulary is deliberately coverage, stability, fragmentation, persistence, agreement, fallback share, confidence and entropy.
- The cockpit stream is an unreviewed Grounding DINO + SAM 2.1 proxy plus a geometric bottom-of-frame prior. That prior was shaped around a car interior and does not describe a handlebar; its behaviour on the motorcycle is one of the things this run was meant to expose.
- The frozen prompt set contains exactly one hand phrasing, "human hand on steering wheel", which no motorcycle frame can satisfy. Any conclusion about hands from the frozen run alone would be an artefact of that prompt.
- A single 30-second window is not the recording. It was chosen to be representative and difficult, not to be exhaustive.

