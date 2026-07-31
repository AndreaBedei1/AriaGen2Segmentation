# Article 1 — motorcycle hand visibility audit

Generated: 2026-07-31T21:41:25.192914+00:00

## What this is, and what it is not

Every state below is a **candidate proposed for human review**. There is no reviewed hand ground truth in this project yet, so this report contains no recall, no precision and no accuracy, and it must not be read as a measurement of how well anything detects hands.

Intermittent hand visibility is **normal** on a motorcycle. A hand can be below the camera, hidden by the handlebar or the rider's own arm, smeared by vibration, or simply outside the field of view when the rider looks away. None of that is a model error, and the audit is built so that "no mask because no hand is visible" is a correct outcome.

The failures that *are* worth counting are the opposite ones: a hand region proposed where no hand is visible, and a hand region that keeps living after the hand has gone.

Hands are not a taxonomy class. They stay inside `control_and_ego_vehicle` (12) and are described through separate attributes.

## Recording-level hand tracking

The on-device tracker gives the first, cheapest signal. It is a proxy: it can track a hand that is outside the RGB field of view, and it can fail on a perfectly visible one.

| recording | domain | samples | left tracked | right tracked | either tracked | any landmark inside the RGB image |
|---|---|---:|---:|---:|---:|---:|
| `car_2e84f0c3e245` | car | 3,761 | 96.5% | 99.0% | 100.0% | 75.1% |
| `motorcycle_5ab8604a14df` | motorcycle | 10,046 | 31.7% | 12.1% | 37.0% | 12.5% |

The contrast between the two domains is the central observation of this audit, and it is a property of the acquisition geometry rather than of any model: in the car the wheel sits inside the camera's view, on the motorcycle the grips sit mostly below it.

## Candidate states over the analysed segment

900 candidates over 450 frames (29.93 s at 15.001 Hz), two per frame (left and right).

| state | candidates | share | per second |
|---|---:|---:|---:|
| `visible` | 0 | 0.0% | 0.000 |
| `partially_visible` | 0 | 0.0% | 0.000 |
| `occluded` | 0 | 0.0% | 0.000 |
| `out_of_frame` | 304 | 33.8% | 10.157 |
| `motion_blurred` | 8 | 0.9% | 0.267 |
| `uncertain` | 175 | 19.4% | 5.847 |
| `not_visible` | 413 | 45.9% | 13.799 |

Evaluable candidates (`visible` or `partially_visible`): 0 (0.0%). **A model may be scored on hands only on frames a reviewer confirms as one of those two states.** In the other states a missing mask is not an error and must not be penalised; a present mask still deserves inspection.

### By side

| state | left | right |
|---|---:|---:|
| `visible` | 0 | 0 |
| `partially_visible` | 0 | 0 |
| `occluded` | 0 | 0 |
| `out_of_frame` | 304 | 0 |
| `motion_blurred` | 8 | 0 |
| `uncertain` | 36 | 139 |
| `not_visible` | 102 | 311 |

## Agreement between the signals

The ten cases the audit is required to separate:

| case | meaning | candidates |
|---:|---|---:|
| 1 | hand visible and proxy present | 0 |
| 2 | hand visible and proxy absent | 0 |
| 3 | hand not visible and proxy absent | 413 |
| 4 | hand not visible but a false mask is present | 132 |
| 5 | tracking present but the hand is not visible in the RGB frame | 304 |
| 6 | hand visible but tracking absent | 0 |
| 7 | blurred hand | 8 |
| 8 | partially visible hand | 0 |
| 9 | mask propagated for too long | 43 |
| 10 | confusion with the handlebar or a mirror | 0 |

- candidate false masks (case 4): **132**
- candidate over-propagation (case 9): **43**

## The segmentation proxy used here

The frozen cockpit proxy carries exactly one hand phrasing, "human hand on steering wheel", which no motorcycle frame can satisfy. Reading the frozen run alone would therefore confuse "the prompt cannot match" with "there is no hand".

This audit adds a separate diagnostic probe with domain-appropriate phrasings. It is not part of the frozen pipeline and does not modify it.

- caption: `a human hand. a hand on the handlebar. a gloved hand. a hand gripping a motorcycle grip. a forearm.`
- box / text thresholds: 0.25 / 0.2
- frames probed: 450
- frames with a proposed hand region: 150
- mean proposed area share: 0.4482%
- status: `diagnostic_proxy_not_ground_truth`

## QA sequences

| directory | found | centre frame |
|---|---|---:|
| `hand_qa_sequences/01_both_hands_absent` | yes | 13875 |
| `hand_qa_sequences/02_tracked_but_out_of_frame` | yes | 13903 |
| `hand_qa_sequences/03_hand_visible_candidate` | no | — |
| `hand_qa_sequences/04_uncertain` | yes | 13901 |
| `hand_qa_sequences/05_motion_blurred` | yes | 13967 |

## Attributes carried into annotation

Per side: `*_hand_visible`, `*_hand_partially_visible`, `*_hand_occluded`, `*_hand_out_of_frame`, `*_hand_motion_blurred`, `*_arm_visible`. Shared: `hand_tracking_available`, `hand_tracking_valid`, `hand_segmentation_proxy_available`.

`*_arm_visible` is deliberately left empty by the automatic pass: it is not inferable from the available signals and guessing it would give the reviewer a wrong default to accept.

## Outputs

- `hand_visibility_candidates.csv` — one row per frame and side, with the proposed state, its rationale and the annotation attributes
- `hand_proxy_agreement.csv` — the raw signals behind each candidate
- `hand_failure_summary.json` — aggregate counts
- `hand_qa_sequences/` — consecutive-frame examples of the main situations

## What a reviewer must decide

1. Confirm or overturn each proposed state, especially every `uncertain`.
2. Confirm the candidate false masks: those are the only cases where the cockpit stream is doing something actively wrong about hands.
3. Mark the frames where a hand is visible but neither signal found it; those are the cases that will teach the shared cockpit model most.

