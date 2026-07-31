# Article 1 — motorcycle ingestion: final summary

Generated: 2026-07-31T21:43:52.957106+00:00

## Git

- branch: `feature/article1-motorcycle-ingestion`
- base commit: `240d088b3436bc8b6e0c1cb76b67bb1dbbd66d53`
- final commit: `43081ad37e8fa57571e1e0ef2ee7b10acb1e10c5`

Intermediate commits:

```
43081ad Run the frozen Article 1 baseline on the motorcycle and analyse it
fd39626 Render the side-by-side external/internal/fusion comparison
892d43d Keep the full hand-tracking record, not just its summary
56cbba4 Fail the frozen run when a stage fails
102a995 Populate the example directories the annotator guide refers to
8b745a1 Bound the hand proxy to plausible hand geometry
fb64f48 Gate segment selection on evidence that the vehicle is being ridden
3e007be Document the motorcycle ingestion and multi-rate policy
9c18b21 Add artefact-level guarantees and the report generators
e7f81a9 Prepare the balanced annotation selection and the CVAT package
b02d370 Prepare the shared cockpit training stage without starting it
98cfc14 Select the motorcycle analysis segment and prepare the frozen run
4797c0b Handle the temporary 10/15 fps difference explicitly
9232c95 Add preliminary auto-motorcycle route alignment
b4ac274 Add motorcycle acquisition inventory and stream QA
```

## The motorcycle recording

| quantity | value |
|---|---|
| file | `unknown_20260731_162521.vrs` |
| local path | `/home/andreabedei/Scrivania/AndreaSegmentazione/unknown_20260731_162521.vrs` |
| SHA-256 | `5ab8604a14dfafecb15fc58a86c8d99ac59e4c83b1cb6025e43d22efe264f6d7` |
| size | 1578.5 MB |
| recording id | `motorcycle_5ab8604a14df` |
| duration | 1004.516 s |
| RGB frames | 15,069 |
| nominal rate | 15.0 Hz |
| **measured rate** | **15.00117 Hz** (span 15.00026 Hz) |
| resolution | 2016 x 1512 |
| streams present | 17 |

Streams: `als`, `baro0`, `camera-et-left`, `camera-et-right`, `camera-rgb`, `eyegaze`, `gps-app`, `handtracking`, `imu-left`, `imu-right`, `mag0`, `ppg`, `slam-front-left`, `slam-front-right`, `slam-side-left`, `slam-side-right`, `temperature`.

How it was identified: not by its file name. The project tree was scanned, each VRS opened, and the domain decided from the static ego structure visible in sampled frames. The evidence object carries no rate, count or duration field, so the temporary 10 vs 15 fps difference cannot influence the decision.

### Gaps and excluded intervals

Exactly one anomalous RGB interval in the whole recording: 133.317 ms between source frames 827 and 828 (2.00 nominal periods), at t = 55.13 s. One dropped frame out of 15,069.

**No interval is excluded.** There is no pause, no restart, no non-monotonic timestamp, and the recording is a single continuous segment of 1004.5 s.

## The 30-second segment

| quantity | value |
|---|---|
| source frames | 13875 to 14324 |
| device timestamps | 3,099,822,119,217 to 3,129,751,762,010 ns |
| duration | 29.930 s |
| frames | 450 |
| missing frames | 0 |
| windows evaluated | 195 |
| score | 0.7010 (rank 1 of 175 passing) |

Chosen by a ranking that weights motion, visual variety and semantic variety above easy footage, after hard quality gates. The full ranking is in `segment_candidates_motorcycle.csv`.

## Pipeline execution

- extracted 450 frames, 0 errors
- source rate measured at 15.00117 Hz
- `resampled: False`, `interpolated: False`, `synthetic_frames: 0`

The frozen configuration bundle was hashed before and after the run and did not change; the per-file checksums are in `logs/frozen_config_checksums.sha256`.

## Main candidate failure modes on the motorcycle

| candidate mode | events | per second |
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

These are candidates, not confirmed errors: there is no reviewed ground truth to confirm them against.

## Hands

900 candidates over 450 frames.

| state | candidates |
|---|---:|
| `visible` | 0 |
| `partially_visible` | 0 |
| `occluded` | 0 |
| `out_of_frame` | 304 |
| `motion_blurred` | 8 |
| `uncertain` | 175 |
| `not_visible` | 413 |

- evaluable (`visible` or `partially_visible`): 0 (0.0%)
- candidate false masks: 132
- candidate over-propagation: 43

The failure inventory above counts `false_hand` per **frame**, while the audit counts per **side candidate** (two per frame) and routes some of them to the over-propagation case instead, so the two figures describe the same situation from different units.

Across the whole recordings, the on-device tracker reports a hand in 100.0% of car samples (75.1% projecting into the RGB image) against 37.0% on the motorcycle (12.5% into the image). Intermittent hand visibility on a motorcycle is the normal case and is never treated as a model error.

## Route pairing quality

- status: `exploratory_preliminary`
- accepted pairs: 181 of 953 (19.0%)
- median pair distance: 16.3 m
- shared stretch: about 2222 m
- motorcycle progression covered: [0.8128802769948967, 0.9679823456868]
- car progression covered: [0.0, 0.9923835505256435]
- monotonic (same direction of travel): True

GPS quality is asymmetric and bounds the pairing: the car's fix rate and accuracy are far worse than the motorcycle's, because a metal roof degrades reception.

## Annotation dataset

- total frames: 144
- per domain: {'car': 70, 'motorcycle': 74}
- per group: {'external_validation': 60, 'cockpit_training': 80, 'failure_mode_review': 4}
- strata covered: 43

### CVAT package

- items: 144
- with pre-annotation: 9
- pre-annotation status: `PRE_ANNOTATION_NOT_GROUND_TRUTH`

Committed: manifest, label specification, label map, palette, annotator guide, frame list, QA thumbnails and checksums. Local only: full-resolution images and the automatic pre-annotation masks.

## Training gate

- status: `prepared_not_started`
- ready to train: False

- blocker: datasets/article1_annotation_package/REVIEWED_ANNOTATIONS.json is absent: no human-reviewed annotation has been delivered
- blocker: no reviewed mask directory

training is blocked until reviewed car AND motorcycle cockpit annotations exist; pseudo-labels are never promoted to ground truth

## Tests

```
385 passed, 1 skipped in 24.17s
```

## Outputs

### Committed

- `reports/article1_motorcycle_ingestion/REPORT_ACQUISITION_QA.md`
- `reports/article1_motorcycle_ingestion/REPORT_ANNOTATION_DATASET.md`
- `reports/article1_motorcycle_ingestion/REPORT_FINAL_SUMMARY.md`
- `reports/article1_motorcycle_ingestion/REPORT_HAND_VISIBILITY_AUDIT.md`
- `reports/article1_motorcycle_ingestion/REPORT_MOTORCYCLE_BASELINE.md`
- `reports/article1_motorcycle_ingestion/REPORT_PRELIMINARY_AUTO_MOTO_COMPARISON.md`
- `reports/article1_motorcycle_ingestion/acquisition_manifest.csv`
- `reports/article1_motorcycle_ingestion/acquisition_manifest.json`
- `reports/article1_motorcycle_ingestion/annotation_selection.csv`
- `reports/article1_motorcycle_ingestion/annotation_selection.json`
- `reports/article1_motorcycle_ingestion/auto_moto_class_comparison.csv`
- `reports/article1_motorcycle_ingestion/auto_moto_comparison.json`
- `reports/article1_motorcycle_ingestion/cockpit_training_plan.json`
- `reports/article1_motorcycle_ingestion/comparable_timeline_car.csv`
- `reports/article1_motorcycle_ingestion/comparable_timeline_motorcycle.csv`
- `reports/article1_motorcycle_ingestion/comparable_timeline_summary.json`
- `reports/article1_motorcycle_ingestion/final_pass_qa/01_lane_dropout/README.md`
- `reports/article1_motorcycle_ingestion/final_pass_qa/01_lane_dropout/contact_sheet.jpg`
- `reports/article1_motorcycle_ingestion/final_pass_qa/01_lane_dropout/frame_013944_mask.png`
- `reports/article1_motorcycle_ingestion/final_pass_qa/01_lane_dropout/frame_013944_overlay.jpg`
- `reports/article1_motorcycle_ingestion/final_pass_qa/01_lane_dropout/frame_013944_rgb.jpg`
- `reports/article1_motorcycle_ingestion/final_pass_qa/01_lane_dropout/frame_013949_mask.png`
- `reports/article1_motorcycle_ingestion/final_pass_qa/01_lane_dropout/frame_013949_overlay.jpg`
- `reports/article1_motorcycle_ingestion/final_pass_qa/01_lane_dropout/frame_013949_rgb.jpg`
- `reports/article1_motorcycle_ingestion/final_pass_qa/01_lane_dropout/frame_013953_mask.png`
- `reports/article1_motorcycle_ingestion/final_pass_qa/01_lane_dropout/frame_013953_overlay.jpg`
- `reports/article1_motorcycle_ingestion/final_pass_qa/01_lane_dropout/frame_013953_rgb.jpg`
- `reports/article1_motorcycle_ingestion/final_pass_qa/02_broken_lane_link/README.md`
- `reports/article1_motorcycle_ingestion/final_pass_qa/02_broken_lane_link/contact_sheet.jpg`
- `reports/article1_motorcycle_ingestion/final_pass_qa/02_broken_lane_link/frame_014177_mask.png`
- `reports/article1_motorcycle_ingestion/final_pass_qa/02_broken_lane_link/frame_014177_overlay.jpg`
- `reports/article1_motorcycle_ingestion/final_pass_qa/02_broken_lane_link/frame_014177_rgb.jpg`
- `reports/article1_motorcycle_ingestion/final_pass_qa/02_broken_lane_link/frame_014182_mask.png`
- `reports/article1_motorcycle_ingestion/final_pass_qa/02_broken_lane_link/frame_014182_overlay.jpg`
- `reports/article1_motorcycle_ingestion/final_pass_qa/02_broken_lane_link/frame_014182_rgb.jpg`
- `reports/article1_motorcycle_ingestion/final_pass_qa/02_broken_lane_link/frame_014186_mask.png`
- `reports/article1_motorcycle_ingestion/final_pass_qa/02_broken_lane_link/frame_014186_overlay.jpg`
- `reports/article1_motorcycle_ingestion/final_pass_qa/02_broken_lane_link/frame_014186_rgb.jpg`
- `reports/article1_motorcycle_ingestion/final_pass_qa/03_intermittent_boundary/README.md`
- `reports/article1_motorcycle_ingestion/final_pass_qa/03_intermittent_boundary/contact_sheet.jpg`
- `reports/article1_motorcycle_ingestion/final_pass_qa/03_intermittent_boundary/frame_014300_mask.png`
- `reports/article1_motorcycle_ingestion/final_pass_qa/03_intermittent_boundary/frame_014300_overlay.jpg`
- `reports/article1_motorcycle_ingestion/final_pass_qa/03_intermittent_boundary/frame_014300_rgb.jpg`
- `reports/article1_motorcycle_ingestion/final_pass_qa/03_intermittent_boundary/frame_014305_mask.png`
- `reports/article1_motorcycle_ingestion/final_pass_qa/03_intermittent_boundary/frame_014305_overlay.jpg`
- `reports/article1_motorcycle_ingestion/final_pass_qa/03_intermittent_boundary/frame_014305_rgb.jpg`
- `reports/article1_motorcycle_ingestion/final_pass_qa/03_intermittent_boundary/frame_014309_mask.png`
- `reports/article1_motorcycle_ingestion/final_pass_qa/03_intermittent_boundary/frame_014309_overlay.jpg`
- `reports/article1_motorcycle_ingestion/final_pass_qa/03_intermittent_boundary/frame_014309_rgb.jpg`
- `reports/article1_motorcycle_ingestion/final_pass_qa/04_internal_flicker/README.md`
- `reports/article1_motorcycle_ingestion/final_pass_qa/04_internal_flicker/contact_sheet.jpg`
- `reports/article1_motorcycle_ingestion/final_pass_qa/04_internal_flicker/frame_013923_mask.png`
- `reports/article1_motorcycle_ingestion/final_pass_qa/04_internal_flicker/frame_013923_overlay.jpg`
- `reports/article1_motorcycle_ingestion/final_pass_qa/04_internal_flicker/frame_013923_rgb.jpg`
- `reports/article1_motorcycle_ingestion/final_pass_qa/04_internal_flicker/frame_013928_mask.png`
- `reports/article1_motorcycle_ingestion/final_pass_qa/04_internal_flicker/frame_013928_overlay.jpg`
- `reports/article1_motorcycle_ingestion/final_pass_qa/04_internal_flicker/frame_013928_rgb.jpg`
- `reports/article1_motorcycle_ingestion/final_pass_qa/04_internal_flicker/frame_013932_mask.png`
- `reports/article1_motorcycle_ingestion/final_pass_qa/04_internal_flicker/frame_013932_overlay.jpg`
- `reports/article1_motorcycle_ingestion/final_pass_qa/04_internal_flicker/frame_013932_rgb.jpg`
- `reports/article1_motorcycle_ingestion/final_pass_qa/05_internal_external_conflict/README.md`
- `reports/article1_motorcycle_ingestion/final_pass_qa/05_internal_external_conflict/contact_sheet.jpg`
- `reports/article1_motorcycle_ingestion/final_pass_qa/05_internal_external_conflict/frame_013969_mask.png`
- `reports/article1_motorcycle_ingestion/final_pass_qa/05_internal_external_conflict/frame_013969_overlay.jpg`
- `reports/article1_motorcycle_ingestion/final_pass_qa/05_internal_external_conflict/frame_013969_rgb.jpg`
- `reports/article1_motorcycle_ingestion/final_pass_qa/05_internal_external_conflict/frame_013974_mask.png`
- `reports/article1_motorcycle_ingestion/final_pass_qa/05_internal_external_conflict/frame_013974_overlay.jpg`
- `reports/article1_motorcycle_ingestion/final_pass_qa/05_internal_external_conflict/frame_013974_rgb.jpg`
- `reports/article1_motorcycle_ingestion/final_pass_qa/05_internal_external_conflict/frame_013978_mask.png`
- `reports/article1_motorcycle_ingestion/final_pass_qa/05_internal_external_conflict/frame_013978_overlay.jpg`
- `reports/article1_motorcycle_ingestion/final_pass_qa/05_internal_external_conflict/frame_013978_rgb.jpg`
- `reports/article1_motorcycle_ingestion/final_pass_qa/README.md`
- `reports/article1_motorcycle_ingestion/final_pass_qa/selection.json`
- `reports/article1_motorcycle_ingestion/hand_failure_summary.json`
- `reports/article1_motorcycle_ingestion/hand_proxy_agreement.csv`
- `reports/article1_motorcycle_ingestion/hand_qa_sequences/01_both_hands_absent/README.md`
- `reports/article1_motorcycle_ingestion/hand_qa_sequences/01_both_hands_absent/frame_013875.jpg`
- `reports/article1_motorcycle_ingestion/hand_qa_sequences/01_both_hands_absent/frame_013876.jpg`
- `reports/article1_motorcycle_ingestion/hand_qa_sequences/01_both_hands_absent/frame_013877.jpg`
- `reports/article1_motorcycle_ingestion/hand_qa_sequences/01_both_hands_absent/frame_013878.jpg`
- `reports/article1_motorcycle_ingestion/hand_qa_sequences/01_both_hands_absent/frame_013879.jpg`
- `reports/article1_motorcycle_ingestion/hand_qa_sequences/02_tracked_but_out_of_frame/README.md`
- `reports/article1_motorcycle_ingestion/hand_qa_sequences/02_tracked_but_out_of_frame/frame_013901.jpg`
- `reports/article1_motorcycle_ingestion/hand_qa_sequences/02_tracked_but_out_of_frame/frame_013902.jpg`
- `reports/article1_motorcycle_ingestion/hand_qa_sequences/02_tracked_but_out_of_frame/frame_013903.jpg`
- `reports/article1_motorcycle_ingestion/hand_qa_sequences/02_tracked_but_out_of_frame/frame_013904.jpg`
- `reports/article1_motorcycle_ingestion/hand_qa_sequences/02_tracked_but_out_of_frame/frame_013905.jpg`
- `reports/article1_motorcycle_ingestion/hand_qa_sequences/04_uncertain/README.md`
- `reports/article1_motorcycle_ingestion/hand_qa_sequences/04_uncertain/frame_013899.jpg`
- `reports/article1_motorcycle_ingestion/hand_qa_sequences/04_uncertain/frame_013900.jpg`
- `reports/article1_motorcycle_ingestion/hand_qa_sequences/04_uncertain/frame_013901.jpg`
- `reports/article1_motorcycle_ingestion/hand_qa_sequences/04_uncertain/frame_013902.jpg`
- `reports/article1_motorcycle_ingestion/hand_qa_sequences/04_uncertain/frame_013903.jpg`
- `reports/article1_motorcycle_ingestion/hand_qa_sequences/05_motion_blurred/README.md`
- `reports/article1_motorcycle_ingestion/hand_qa_sequences/05_motion_blurred/frame_013965.jpg`
- `reports/article1_motorcycle_ingestion/hand_qa_sequences/05_motion_blurred/frame_013966.jpg`
- `reports/article1_motorcycle_ingestion/hand_qa_sequences/05_motion_blurred/frame_013967.jpg`
- `reports/article1_motorcycle_ingestion/hand_qa_sequences/05_motion_blurred/frame_013968.jpg`
- `reports/article1_motorcycle_ingestion/hand_qa_sequences/05_motion_blurred/frame_013969.jpg`
- `reports/article1_motorcycle_ingestion/hand_qa_sequences/index.json`
- `reports/article1_motorcycle_ingestion/hand_visibility_candidates.csv`
- `reports/article1_motorcycle_ingestion/motorcycle_class_coverage.csv`
- `reports/article1_motorcycle_ingestion/motorcycle_failure_modes.json`
- `reports/article1_motorcycle_ingestion/preliminary_route_alignment.csv`
- `reports/article1_motorcycle_ingestion/qa/01_cockpit_absorbed_by_external/README.md`
- `reports/article1_motorcycle_ingestion/qa/01_cockpit_absorbed_by_external/contact_sheet.jpg`
- `reports/article1_motorcycle_ingestion/qa/01_cockpit_absorbed_by_external/frame_014145_mask.png`
- `reports/article1_motorcycle_ingestion/qa/01_cockpit_absorbed_by_external/frame_014145_overlay.jpg`
- `reports/article1_motorcycle_ingestion/qa/01_cockpit_absorbed_by_external/frame_014145_rgb.jpg`
- `reports/article1_motorcycle_ingestion/qa/01_cockpit_absorbed_by_external/frame_014146_mask.png`
- `reports/article1_motorcycle_ingestion/qa/01_cockpit_absorbed_by_external/frame_014146_overlay.jpg`
- `reports/article1_motorcycle_ingestion/qa/01_cockpit_absorbed_by_external/frame_014146_rgb.jpg`
- `reports/article1_motorcycle_ingestion/qa/01_cockpit_absorbed_by_external/frame_014147_mask.png`
- `reports/article1_motorcycle_ingestion/qa/01_cockpit_absorbed_by_external/frame_014147_overlay.jpg`
- `reports/article1_motorcycle_ingestion/qa/01_cockpit_absorbed_by_external/frame_014147_rgb.jpg`
- `reports/article1_motorcycle_ingestion/qa/01_cockpit_absorbed_by_external/frame_014148_mask.png`
- `reports/article1_motorcycle_ingestion/qa/01_cockpit_absorbed_by_external/frame_014148_overlay.jpg`
- `reports/article1_motorcycle_ingestion/qa/01_cockpit_absorbed_by_external/frame_014148_rgb.jpg`
- `reports/article1_motorcycle_ingestion/qa/01_cockpit_absorbed_by_external/frame_014149_mask.png`
- `reports/article1_motorcycle_ingestion/qa/01_cockpit_absorbed_by_external/frame_014149_overlay.jpg`
- `reports/article1_motorcycle_ingestion/qa/01_cockpit_absorbed_by_external/frame_014149_rgb.jpg`
- `reports/article1_motorcycle_ingestion/qa/02_mirror_not_detected/README.md`
- `reports/article1_motorcycle_ingestion/qa/02_mirror_not_detected/contact_sheet.jpg`
- `reports/article1_motorcycle_ingestion/qa/02_mirror_not_detected/frame_013875_mask.png`
- `reports/article1_motorcycle_ingestion/qa/02_mirror_not_detected/frame_013875_overlay.jpg`
- `reports/article1_motorcycle_ingestion/qa/02_mirror_not_detected/frame_013875_rgb.jpg`
- `reports/article1_motorcycle_ingestion/qa/02_mirror_not_detected/frame_013876_mask.png`
- `reports/article1_motorcycle_ingestion/qa/02_mirror_not_detected/frame_013876_overlay.jpg`
- `reports/article1_motorcycle_ingestion/qa/02_mirror_not_detected/frame_013876_rgb.jpg`
- `reports/article1_motorcycle_ingestion/qa/02_mirror_not_detected/frame_013877_mask.png`
- `reports/article1_motorcycle_ingestion/qa/02_mirror_not_detected/frame_013877_overlay.jpg`
- `reports/article1_motorcycle_ingestion/qa/02_mirror_not_detected/frame_013877_rgb.jpg`
- `reports/article1_motorcycle_ingestion/qa/03_instrument_display_not_detected/README.md`
- `reports/article1_motorcycle_ingestion/qa/03_instrument_display_not_detected/contact_sheet.jpg`
- `reports/article1_motorcycle_ingestion/qa/03_instrument_display_not_detected/frame_013885_mask.png`
- `reports/article1_motorcycle_ingestion/qa/03_instrument_display_not_detected/frame_013885_overlay.jpg`
- `reports/article1_motorcycle_ingestion/qa/03_instrument_display_not_detected/frame_013885_rgb.jpg`
- `reports/article1_motorcycle_ingestion/qa/03_instrument_display_not_detected/frame_013886_mask.png`
- `reports/article1_motorcycle_ingestion/qa/03_instrument_display_not_detected/frame_013886_overlay.jpg`
- `reports/article1_motorcycle_ingestion/qa/03_instrument_display_not_detected/frame_013886_rgb.jpg`
- `reports/article1_motorcycle_ingestion/qa/03_instrument_display_not_detected/frame_013887_mask.png`
- `reports/article1_motorcycle_ingestion/qa/03_instrument_display_not_detected/frame_013887_overlay.jpg`
- `reports/article1_motorcycle_ingestion/qa/03_instrument_display_not_detected/frame_013887_rgb.jpg`
- `reports/article1_motorcycle_ingestion/qa/03_instrument_display_not_detected/frame_013888_mask.png`
- `reports/article1_motorcycle_ingestion/qa/03_instrument_display_not_detected/frame_013888_overlay.jpg`
- `reports/article1_motorcycle_ingestion/qa/03_instrument_display_not_detected/frame_013888_rgb.jpg`
- `reports/article1_motorcycle_ingestion/qa/03_instrument_display_not_detected/frame_013889_mask.png`
- `reports/article1_motorcycle_ingestion/qa/03_instrument_display_not_detected/frame_013889_overlay.jpg`
- `reports/article1_motorcycle_ingestion/qa/03_instrument_display_not_detected/frame_013889_rgb.jpg`
- `reports/article1_motorcycle_ingestion/qa/04_excessive_fallback/README.md`
- `reports/article1_motorcycle_ingestion/qa/04_excessive_fallback/contact_sheet.jpg`
- `reports/article1_motorcycle_ingestion/qa/04_excessive_fallback/frame_014303_mask.png`
- `reports/article1_motorcycle_ingestion/qa/04_excessive_fallback/frame_014303_overlay.jpg`
- `reports/article1_motorcycle_ingestion/qa/04_excessive_fallback/frame_014303_rgb.jpg`
- `reports/article1_motorcycle_ingestion/qa/04_excessive_fallback/frame_014304_mask.png`
- `reports/article1_motorcycle_ingestion/qa/04_excessive_fallback/frame_014304_overlay.jpg`
- `reports/article1_motorcycle_ingestion/qa/04_excessive_fallback/frame_014304_rgb.jpg`
- `reports/article1_motorcycle_ingestion/qa/04_excessive_fallback/frame_014305_mask.png`
- `reports/article1_motorcycle_ingestion/qa/04_excessive_fallback/frame_014305_overlay.jpg`
- `reports/article1_motorcycle_ingestion/qa/04_excessive_fallback/frame_014305_rgb.jpg`
- `reports/article1_motorcycle_ingestion/qa/04_excessive_fallback/frame_014306_mask.png`
- `reports/article1_motorcycle_ingestion/qa/04_excessive_fallback/frame_014306_overlay.jpg`
- `reports/article1_motorcycle_ingestion/qa/04_excessive_fallback/frame_014306_rgb.jpg`
- `reports/article1_motorcycle_ingestion/qa/04_excessive_fallback/frame_014307_mask.png`
- `reports/article1_motorcycle_ingestion/qa/04_excessive_fallback/frame_014307_overlay.jpg`
- `reports/article1_motorcycle_ingestion/qa/04_excessive_fallback/frame_014307_rgb.jpg`
- `reports/article1_motorcycle_ingestion/qa/05_fragmentation/README.md`
- `reports/article1_motorcycle_ingestion/qa/05_fragmentation/contact_sheet.jpg`
- `reports/article1_motorcycle_ingestion/qa/05_fragmentation/frame_013882_mask.png`
- `reports/article1_motorcycle_ingestion/qa/05_fragmentation/frame_013882_overlay.jpg`
- `reports/article1_motorcycle_ingestion/qa/05_fragmentation/frame_013882_rgb.jpg`
- `reports/article1_motorcycle_ingestion/qa/05_fragmentation/frame_013883_mask.png`
- `reports/article1_motorcycle_ingestion/qa/05_fragmentation/frame_013883_overlay.jpg`
- `reports/article1_motorcycle_ingestion/qa/05_fragmentation/frame_013883_rgb.jpg`
- `reports/article1_motorcycle_ingestion/qa/05_fragmentation/frame_013884_mask.png`
- `reports/article1_motorcycle_ingestion/qa/05_fragmentation/frame_013884_overlay.jpg`
- `reports/article1_motorcycle_ingestion/qa/05_fragmentation/frame_013884_rgb.jpg`
- `reports/article1_motorcycle_ingestion/qa/05_fragmentation/frame_013885_mask.png`
- `reports/article1_motorcycle_ingestion/qa/05_fragmentation/frame_013885_overlay.jpg`
- `reports/article1_motorcycle_ingestion/qa/05_fragmentation/frame_013885_rgb.jpg`
- `reports/article1_motorcycle_ingestion/qa/05_fragmentation/frame_013886_mask.png`
- `reports/article1_motorcycle_ingestion/qa/05_fragmentation/frame_013886_overlay.jpg`
- `reports/article1_motorcycle_ingestion/qa/05_fragmentation/frame_013886_rgb.jpg`
- `reports/article1_motorcycle_ingestion/qa/06_high_entropy/README.md`
- `reports/article1_motorcycle_ingestion/qa/06_high_entropy/contact_sheet.jpg`
- `reports/article1_motorcycle_ingestion/qa/06_high_entropy/frame_014303_mask.png`
- `reports/article1_motorcycle_ingestion/qa/06_high_entropy/frame_014303_overlay.jpg`
- `reports/article1_motorcycle_ingestion/qa/06_high_entropy/frame_014303_rgb.jpg`
- `reports/article1_motorcycle_ingestion/qa/06_high_entropy/frame_014304_mask.png`
- `reports/article1_motorcycle_ingestion/qa/06_high_entropy/frame_014304_overlay.jpg`
- `reports/article1_motorcycle_ingestion/qa/06_high_entropy/frame_014304_rgb.jpg`
- `reports/article1_motorcycle_ingestion/qa/06_high_entropy/frame_014305_mask.png`
- `reports/article1_motorcycle_ingestion/qa/06_high_entropy/frame_014305_overlay.jpg`
- `reports/article1_motorcycle_ingestion/qa/06_high_entropy/frame_014305_rgb.jpg`
- `reports/article1_motorcycle_ingestion/qa/06_high_entropy/frame_014306_mask.png`
- `reports/article1_motorcycle_ingestion/qa/06_high_entropy/frame_014306_overlay.jpg`
- `reports/article1_motorcycle_ingestion/qa/06_high_entropy/frame_014306_rgb.jpg`
- `reports/article1_motorcycle_ingestion/qa/06_high_entropy/frame_014307_mask.png`
- `reports/article1_motorcycle_ingestion/qa/06_high_entropy/frame_014307_overlay.jpg`
- `reports/article1_motorcycle_ingestion/qa/06_high_entropy/frame_014307_rgb.jpg`
- `reports/article1_motorcycle_ingestion/qa/07_blur_related_failure/README.md`
- `reports/article1_motorcycle_ingestion/qa/07_blur_related_failure/contact_sheet.jpg`
- `reports/article1_motorcycle_ingestion/qa/07_blur_related_failure/frame_014203_mask.png`
- `reports/article1_motorcycle_ingestion/qa/07_blur_related_failure/frame_014203_overlay.jpg`
- `reports/article1_motorcycle_ingestion/qa/07_blur_related_failure/frame_014203_rgb.jpg`
- `reports/article1_motorcycle_ingestion/qa/07_blur_related_failure/frame_014204_mask.png`
- `reports/article1_motorcycle_ingestion/qa/07_blur_related_failure/frame_014204_overlay.jpg`
- `reports/article1_motorcycle_ingestion/qa/07_blur_related_failure/frame_014204_rgb.jpg`
- `reports/article1_motorcycle_ingestion/qa/07_blur_related_failure/frame_014205_mask.png`
- `reports/article1_motorcycle_ingestion/qa/07_blur_related_failure/frame_014205_overlay.jpg`
- `reports/article1_motorcycle_ingestion/qa/07_blur_related_failure/frame_014205_rgb.jpg`
- `reports/article1_motorcycle_ingestion/qa/07_blur_related_failure/frame_014206_mask.png`
- `reports/article1_motorcycle_ingestion/qa/07_blur_related_failure/frame_014206_overlay.jpg`
- `reports/article1_motorcycle_ingestion/qa/07_blur_related_failure/frame_014206_rgb.jpg`
- `reports/article1_motorcycle_ingestion/qa/07_blur_related_failure/frame_014207_mask.png`
- `reports/article1_motorcycle_ingestion/qa/07_blur_related_failure/frame_014207_overlay.jpg`
- `reports/article1_motorcycle_ingestion/qa/07_blur_related_failure/frame_014207_rgb.jpg`
- `reports/article1_motorcycle_ingestion/qa/08_vibration_related_failure/README.md`
- `reports/article1_motorcycle_ingestion/qa/08_vibration_related_failure/contact_sheet.jpg`
- `reports/article1_motorcycle_ingestion/qa/08_vibration_related_failure/frame_013951_mask.png`
- `reports/article1_motorcycle_ingestion/qa/08_vibration_related_failure/frame_013951_overlay.jpg`
- `reports/article1_motorcycle_ingestion/qa/08_vibration_related_failure/frame_013951_rgb.jpg`
- `reports/article1_motorcycle_ingestion/qa/08_vibration_related_failure/frame_013952_mask.png`
- `reports/article1_motorcycle_ingestion/qa/08_vibration_related_failure/frame_013952_overlay.jpg`
- `reports/article1_motorcycle_ingestion/qa/08_vibration_related_failure/frame_013952_rgb.jpg`
- `reports/article1_motorcycle_ingestion/qa/08_vibration_related_failure/frame_013953_mask.png`
- `reports/article1_motorcycle_ingestion/qa/08_vibration_related_failure/frame_013953_overlay.jpg`
- `reports/article1_motorcycle_ingestion/qa/08_vibration_related_failure/frame_013953_rgb.jpg`
- `reports/article1_motorcycle_ingestion/qa/08_vibration_related_failure/frame_013954_mask.png`
- `reports/article1_motorcycle_ingestion/qa/08_vibration_related_failure/frame_013954_overlay.jpg`
- `reports/article1_motorcycle_ingestion/qa/08_vibration_related_failure/frame_013954_rgb.jpg`
- `reports/article1_motorcycle_ingestion/qa/08_vibration_related_failure/frame_013955_mask.png`
- `reports/article1_motorcycle_ingestion/qa/08_vibration_related_failure/frame_013955_overlay.jpg`
- `reports/article1_motorcycle_ingestion/qa/08_vibration_related_failure/frame_013955_rgb.jpg`
- `reports/article1_motorcycle_ingestion/qa/09_internal_external_fusion_error/README.md`
- `reports/article1_motorcycle_ingestion/qa/09_internal_external_fusion_error/contact_sheet.jpg`
- `reports/article1_motorcycle_ingestion/qa/09_internal_external_fusion_error/frame_013953_mask.png`
- `reports/article1_motorcycle_ingestion/qa/09_internal_external_fusion_error/frame_013953_overlay.jpg`
- `reports/article1_motorcycle_ingestion/qa/09_internal_external_fusion_error/frame_013953_rgb.jpg`
- `reports/article1_motorcycle_ingestion/qa/09_internal_external_fusion_error/frame_013954_mask.png`
- `reports/article1_motorcycle_ingestion/qa/09_internal_external_fusion_error/frame_013954_overlay.jpg`
- `reports/article1_motorcycle_ingestion/qa/09_internal_external_fusion_error/frame_013954_rgb.jpg`
- `reports/article1_motorcycle_ingestion/qa/09_internal_external_fusion_error/frame_013955_mask.png`
- `reports/article1_motorcycle_ingestion/qa/09_internal_external_fusion_error/frame_013955_overlay.jpg`
- `reports/article1_motorcycle_ingestion/qa/09_internal_external_fusion_error/frame_013955_rgb.jpg`
- `reports/article1_motorcycle_ingestion/qa/09_internal_external_fusion_error/frame_013956_mask.png`
- `reports/article1_motorcycle_ingestion/qa/09_internal_external_fusion_error/frame_013956_overlay.jpg`
- `reports/article1_motorcycle_ingestion/qa/09_internal_external_fusion_error/frame_013956_rgb.jpg`
- `reports/article1_motorcycle_ingestion/qa/09_internal_external_fusion_error/frame_013957_mask.png`
- `reports/article1_motorcycle_ingestion/qa/09_internal_external_fusion_error/frame_013957_overlay.jpg`
- `reports/article1_motorcycle_ingestion/qa/09_internal_external_fusion_error/frame_013957_rgb.jpg`
- `reports/article1_motorcycle_ingestion/qa/index.json`
- `reports/article1_motorcycle_ingestion/route_alignment_qa/route_tracks.png`
- `reports/article1_motorcycle_ingestion/route_alignment_summary.json`
- `reports/article1_motorcycle_ingestion/segment_candidates_motorcycle.csv`
- `reports/article1_motorcycle_ingestion/segment_selection_motorcycle.json`
- `reports/article1_motorcycle_ingestion/stream_qa_auto.json`
- `reports/article1_motorcycle_ingestion/stream_qa_moto.json`
- `reports/article1_motorcycle_ingestion/timestamp_gaps_auto.csv`
- `reports/article1_motorcycle_ingestion/timestamp_gaps_moto.csv`

### Local only, not committed

- `output/article1/motorcycle_baseline_30s/` — frames, masks, confidence, entropy, provenance, logs
- `output/article1/ingestion/` — RGB scans, scouting frames, stream exports
- `output/article1/motorcycle_baseline_30s/semantic_camera_moto_final.mp4` — the presentation video

- the two VRS recordings, unmodified

## Limitations

- No reviewed ground truth exists, so this work reports coverage, stability, fragmentation, persistence, agreement, fallback share, confidence and entropy — never accuracy, IoU, precision or recall.
- The car recording is a provisional 10 fps baseline. Every car/motorcycle number here is exploratory and must be recomputed once the car is re-recorded at 15 fps.
- The cockpit stream is an unreviewed Grounding DINO + SAM 2.1 proxy plus a geometric prior shaped around a car interior.
- The frozen prompt set has one hand phrasing, tied to a steering wheel, which no motorcycle frame can satisfy.
- The analysed segment is 30 s of a 1004 s recording.
- No MPS/VIO trajectory product exists in these files, so route alignment rests on GPS alone.

## What needs human review

1. The annotation package: every pre-annotated pixel.
2. The candidate hand visibility states, especially the `uncertain` ones and the candidate false masks.
3. The candidate failure modes, which are unusual-for-this-clip signals rather than confirmed errors.
4. The route pairing, which is GPS-only and bounded by the car's degraded reception.
5. The choice of the 30-second segment, whose full ranking is published so it can be overruled.

## Verdict

### A. Motorcycle ready for the dataset, annotation ready

the recording is continuous, complete, at the target protocol rate, fully multimodal, and the annotation package is prepared.

| check | result |
|---|---|
| `rgb_monotonic` | pass |
| `rgb_single_segment` | pass |
| `no_long_pause` | pass |
| `missing_frames_negligible` | pass |
| `rate_matches_target_protocol` | pass |
| `multimodal_complete` | pass |
| `file_decoded_end_to_end` | pass |
| `baseline_pipeline_ran` | pass |

Evidence:

```json
{
  "rgb_frames": 15069,
  "rgb_effective_fps": 15.001172266606774,
  "duration_s": 1004.516195844,
  "non_monotonic_intervals": 0,
  "continuous_segments": 1,
  "intervals_over_250ms": 0,
  "estimated_missing_frames": 1,
  "missing_frame_fraction": 6.636140420731303e-05,
  "streams_present": 17,
  "missing_multimodal_streams": [],
  "extracted_frames": 450,
  "extraction_errors": 0
}
```

The work stops here, at the human-annotation gate, exactly as required: the motorcycle is validated, the frozen baseline has been run and analysed, the preliminary comparison exists, the annotation package and its pre-annotations are ready, and the shared cockpit training stage is configured but not started.

