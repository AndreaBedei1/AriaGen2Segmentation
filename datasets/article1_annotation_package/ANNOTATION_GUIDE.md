# Article 1 annotation guide — car and motorcycle

## What you are producing

The first reviewed ground truth of this project. Everything currently in
`preannotations/` is **automatic model output**: Mask2Former for the external
scene, and an unreviewed Grounding DINO + SAM 2.1 proxy plus a geometric prior for
the cockpit. Treat it as a time-saving starting point that is frequently wrong,
never as a reference. Where it disagrees with the image, the image wins.

## Label set (macro taxonomy, ids 1-13)

| id | label | what belongs to it |
|---|---|---|
| 1 | `road_surface` | the drivable carriageway surface |
| 2 | `lane_marking` | painted lane lines, edge lines, dashes |
| 3 | `regulatory_road_marking` | arrows, stop lines, crossings, painted symbols and text |
| 4 | `vehicle` | cars, vans, trucks, buses |
| 5 | `two_wheeler` | motorcycles, scooters, bicycles, and their riders |
| 6 | `pedestrian` | people not riding a vehicle |
| 7 | `traffic_light` | the signal head and its housing |
| 8 | `traffic_sign` | the sign face and its plate |
| 9 | `road_boundary_or_obstacle` | kerbs, guardrails, barriers, bollards, obstacles bounding the road |
| 10 | `mirror` | see below |
| 11 | `instrument_display` | see below |
| 12 | `control_and_ego_vehicle` | see below |
| 13 | `other_environment` | everything else that is genuinely not one of the above |

`unknown` (0) exists in the tool only as a temporary parking place for a region you
cannot resolve during the session. **A delivered frame must contain no `unknown`
pixel**: resolve it, or assign `other_environment` and tick `ambiguity_flag`.

## The three cockpit classes

These are the classes the shared car/motorcycle cockpit model will be trained on,
so their boundaries matter more than anywhere else.

**`mirror` (10)** — the reflecting surface of a mirror, plus its housing and stalk
when they read as one object with it. This covers the car's rear-view and door
mirrors and the motorcycle's bar-end mirrors. Label what is reflected *in* the
mirror as mirror, not as the class of the reflected object: a car seen inside a
mirror is `mirror`.

**`instrument_display` (11)** — the instrument cluster, speedometer, tachometer,
any screen or readable indicator area, and the infotainment display. Include the
bezel only when it is not separable from the display area. On the motorcycle this
is the round gauge cluster above the handlebar.

**`control_and_ego_vehicle` (12)** — the steering wheel, the handlebar, levers,
grips, switchgear, stalks, the non-display dashboard, the fairing, the tank, and
any interior structure of the ego vehicle: pillars, door cards, roof lining,
bonnet. **Visible hands and forearms of the driver or rider also belong here.**

**`other_environment` (13)** — use it for pixels that are genuinely not assignable
to 1-12, not as a dumping ground for "hard". If a region is hard but clearly a
kerb, it is `road_boundary_or_obstacle`.

## Hands

Hands are **not** a separate class. Paint them as `control_and_ego_vehicle` (12)
and describe them through the attributes.

On the motorcycle the rider's hands are frequently **not visible**, and that is
normal, not an error. Causes include: hands below the camera's field of view,
occlusion by the handlebar or fairing, head movement, vibration and motion blur.

Set, for each side, exactly what you see:

- `*_hand_visible` — the hand is clearly identifiable in the frame;
- `*_hand_partially_visible` — part of the hand is visible or it is cut by the
  image border;
- `*_hand_occluded` — you can tell the hand is there but something hides it;
- `*_hand_out_of_frame` — the hand is outside the image;
- `*_hand_motion_blurred` — visible but too smeared to delineate reliably;
- `*_arm_visible` — the forearm is visible (independent of the hand).

Do **not** invent a hand you cannot see, and do **not** extend a mask from a
neighbouring frame into a frame where the hand is absent. If you are unsure whether
something is a hand, tick `ambiguity_flag` and leave the region as whatever it
visually is.

`hand_tracking_available` / `hand_tracking_valid` describe the device's own hand
tracker and are pre-filled. They are context, not instructions: the tracker is
often wrong in both directions.

## Positive / negative / ambiguous examples

`examples/positive/` — clean cases of each cockpit class.
`examples/negative/` — cases that look like a class but are not: a reflection on
the windscreen that is not a mirror, a sticker that is not a display, a bright
highlight on the tank that is not a control.
`examples/ambiguous/` — genuinely debatable cases; annotate them the way the
accompanying note says, and flag them.

## Class boundary rules

1. **Mirror vs control** — the reflecting surface and its immediate housing are
   `mirror`; the stalk where it merges into the fairing or door is
   `control_and_ego_vehicle`.
2. **Display vs control** — only the readable/emissive area and its inseparable
   bezel are `instrument_display`; the surrounding binnacle is
   `control_and_ego_vehicle`.
3. **Through glass** — on the car, objects seen through the windscreen get their
   real class (`vehicle`, `road_surface`, ...), not a glass class. Visible dirt,
   wiper streaks and strong reflections that hide the scene are
   `control_and_ego_vehicle`.
4. **Ego vehicle vs road** — the bonnet, tank and fairing are
   `control_and_ego_vehicle` even when they occupy the bottom of the frame; do not
   let them absorb road pixels and do not let the road absorb them.
5. **Two-wheeler riders** — a rider on another motorcycle is part of
   `two_wheeler`, not `pedestrian`.

## Thin structures

Lane and regulatory markings are thin and the pre-annotation frequently breaks
them into fragments.

- Follow the **painted** extent, not the pre-annotation's fragments; reconnect a
  dashed line only where paint is actually visible.
- Do not thicken a line to make it easier to see; annotate its real width.
- A worn or partially covered marking is still `lane_marking` where paint remains.
- Stop lines, arrows, crossings and painted text are `regulatory_road_marking`
  even when they are as thin as a lane line.
- Where a marking passes under a vehicle or a shadow, do not bridge it.

## Quality flags

Set `image_quality` honestly. A frame flagged `unusable` is excluded from
evaluation rather than annotated badly. `motion_blurred` and `vibration` frames are
deliberately included in the set: annotate what is discernible and flag the rest.

## What is NOT your job

Do not correct or complete the pre-annotation outside the visible evidence, do not
propagate a mask across frames, and do not try to make consecutive frames look
temporally smooth. Each frame is annotated on its own evidence.
