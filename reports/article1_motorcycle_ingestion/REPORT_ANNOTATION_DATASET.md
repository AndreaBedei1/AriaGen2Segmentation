# Article 1 — annotation dataset and CVAT package

Generated: 2026-07-31T21:41:25.661413+00:00

## Purpose

This is the first reviewed ground truth the project will have. It is built to support five things at once: validating the external scene model, training the shared car/motorcycle cockpit model, evaluating mirrors, instrument displays and controls, evaluating hands where they are actually visible, and measuring the domain shift between the two vehicles.

Nothing in the package is ground truth yet. The pre-annotations are automatic model output, marked as such in the directory name, the manifest and the guide.

## How the frames were chosen

Not randomly. Selection is a greedy maximisation of stratum coverage: each pick is the candidate that adds the most conditions not yet represented, so rare situations survive instead of being drowned by ordinary driving.

### Balancing

- equal per-domain quotas: the annotation budget follows scientific need, not the number of frames a recording happens to contain
- the motorcycle is sampled at 15 fps and the car at 10 fps; selecting per frame would have given the motorcycle 1.5x the budget for no reason
- candidates are drawn from a 1 Hz scouting subsample of each recording, so the two pools are already time-uniform rather than frame-uniform
- diversity constraints are expressed in seconds, so they mean the same thing in both recordings

| domain | pool | selected |
|---|---:|---:|
| car | 377 | 70 |
| motorcycle | 1005 | 74 |

**Total: 144 frames, 70 car, 74 motorcycle.** The two domains receive the same budget even though the motorcycle recording is 2.7x longer and sampled 1.5x more densely.

### Groups

| domain | group | requested | selected | note |
|---|---|---:|---:|---|
| car | `external_validation` | 30 | 30 |  |
| car | `cockpit_training` | 40 | 40 |  |
| motorcycle | `external_validation` | 30 | 30 |  |
| motorcycle | `cockpit_training` | 40 | 40 |  |
| motorcycle | `failure_mode_review` | 15 | 4 | the pool ran out of frames separated enough in time and appearance to remain non-duplicated |

- `external_validation` — validates the Mask2Former external scene against human annotation
- `cockpit_training` — trains the shared cockpit model; restricted to frames where ego structure is actually present, because a frame with no cockpit teaches that model nothing
- `failure_mode_review` — frames where the frozen baseline behaved unusually; only available where the baseline actually ran

### Avoiding near duplicates

- minimum temporal separation: 2.0 s
- minimum perceptual-hash distance: 8 of 64 bits
- visual deduplication window: 30.0 s

a hard minimum temporal separation always applies; the perceptual-hash test applies only within the visual window, because a route revisited minutes later is a distinct sample rather than a duplicate

The temporal separation is the binding constraint on the car, whose recording is only 376 s long. Quotas were sized so that the shorter recording can meet them; asking for more would have quietly unbalanced the dataset in the motorcycle's favour.

## What the selection covers

43 distinct strata are represented.

| stratum | frames |
|---|---:|
| `scene:boundary` | 129 |
| `scene:markings` | 109 |
| `geometry:straight` | 96 |
| `light:normal` | 91 |
| `scene:signage` | 85 |
| `motion:medium` | 81 |
| `scene:traffic` | 78 |
| `domain:motorcycle` | 74 |
| `cockpit:prominent` | 71 |
| `domain:car` | 70 |
| `scene:clear` | 66 |
| `hand:right:not_visible` | 64 |
| `hand:right:out_of_frame` | 56 |
| `hand:left:visible` | 55 |
| `scene:pedestrian` | 55 |
| `route:q3` | 50 |
| `geometry:turning` | 48 |
| `motion:high` | 48 |
| `cockpit:partial` | 45 |
| `hand:left:not_visible` | 45 |
| `hand:left:out_of_frame` | 36 |
| `route:q0` | 34 |
| `light:dark` | 32 |
| `route:q2` | 32 |
| `cockpit:minimal` | 28 |
| `route:q1` | 28 |
| `light:bright` | 21 |
| `hand:right:visible` | 15 |
| `motion:low` | 15 |
| `scene:two_wheeler` | 15 |
| `hand:right:partially_visible` | 9 |
| `provenance:external_mask2former` | 9 |
| `provenance:fallback` | 9 |
| `uncertainty:model_conflict` | 9 |
| `hand:left:partially_visible` | 8 |
| `quality:low_sharpness` | 8 |
| `failure:cockpit_absorbed_by_external` | 4 |
| `failure:fragmentation` | 3 |
| `uncertainty:high_entropy` | 3 |
| `uncertainty:low_confidence` | 3 |
| `failure:blur_related_failure` | 1 |
| `failure:mirror_not_detected` | 1 |
| `provenance:geometric_cockpit_proxy` | 1 |

a stratum with a low count is a genuine scarcity in the source recordings, not a sampling bug

## Package contents

- items: 144
- per domain: {'car': 70, 'motorcycle': 74}
- per group: {'cockpit_training': 80, 'external_validation': 60, 'failure_mode_review': 4}

### Pre-annotations

- present: True
- status: `PRE_ANNOTATION_NOT_GROUND_TRUTH`
- source: frozen Article 1 semantic camera (Mask2Former external + Grounding DINO/SAM2.1 cockpit proxy + geometric fallback)
- instruction: every pre-annotated pixel must be reviewed; nothing here is validated

They live in `preannotations_not_ground_truth/`, in CVAT `Segmentation mask 1.1` layout, with their own README repeating that they are automatic output.

### What is committed

Committed: `manifest.json`, `labels.json`, `labelmap.txt`, `palette.json`, `ANNOTATION_GUIDE.md`, `frame_list.csv`, `thumbnails_qa/`, `checksums.sha256`, `examples/`

Local only: `images/`, `preannotations_not_ground_truth/SegmentationClass/`. Full-resolution images and mask PNGs stay out of git; the manifest carries a SHA-256 for each so the local copy can be verified.

## Label specification

14 labels, ids 0-13, exported as CVAT mask labels.

| id | label | attributes |
|---:|---|---:|
| 0 | `unknown` | 5 |
| 1 | `road_surface` | 5 |
| 2 | `lane_marking` | 5 |
| 3 | `regulatory_road_marking` | 5 |
| 4 | `vehicle` | 5 |
| 5 | `two_wheeler` | 5 |
| 6 | `pedestrian` | 5 |
| 7 | `traffic_light` | 5 |
| 8 | `traffic_sign` | 5 |
| 9 | `road_boundary_or_obstacle` | 5 |
| 10 | `mirror` | 20 |
| 11 | `instrument_display` | 20 |
| 12 | `control_and_ego_vehicle` | 20 |
| 13 | `other_environment` | 5 |

The three cockpit labels carry the full hand attribute vocabulary (20 attributes each): per-side `visible`, `partially_visible`, `occluded`, `out_of_frame`, `motion_blurred` and `arm_visible`, plus `hand_tracking_available`, `hand_tracking_valid` and `hand_segmentation_proxy_available`.

Hands are **not** a class. They are painted as `control_and_ego_vehicle` (12) and described by those attributes.

## Rules the guide fixes for the annotators

The full text is `ANNOTATION_GUIDE.md` inside the package. The decisions that most affect the resulting ground truth:

- **`unknown` may not survive.** It exists in the tool as a parking place for an unresolved region during the session; a delivered frame must contain none. Unresolvable regions become `other_environment` with `ambiguity_flag` set.
- **`mirror`** is the reflecting surface plus housing and stalk when they read as one object. What is *inside* the mirror is mirror, not the class of the reflected object.
- **`instrument_display`** is the readable or emissive area and its inseparable bezel; the surrounding binnacle is `control_and_ego_vehicle`.
- **`control_and_ego_vehicle`** covers the wheel, the handlebar, levers, grips, switchgear, the non-display dashboard, fairing, tank and interior structure — and the visible hands and forearms of the driver or rider.
- **`other_environment`** is for pixels genuinely not assignable to 1-12, not a dumping ground for difficult ones.
- **Thin markings** follow the painted extent; a dashed line is reconnected only where paint is visible, and a line is never thickened to make it easier to see.
- **Hands** are never invented and never propagated from a neighbouring frame. On the motorcycle an absent hand is the normal case.

## Limits

- The car recording is a provisional 10 fps baseline. Frames drawn from it are still useful cockpit and external training material, but a definitive paired comparison needs the re-recorded car.
- The candidate pool is a 1 Hz scouting subsample, so the finest temporal structure of either recording is not represented in the annotation set. That is deliberate: annotating near-adjacent frames buys little and costs reviewer time.
- Class expectations attached to each frame come from the external model and are a hint for the annotator, not a target to reproduce.

## Next step

Human review. When it is delivered, the reviewer drops `REVIEWED_ANNOTATIONS.json` into the dataset directory and the cockpit training gate opens. Nothing downstream may proceed before that.

