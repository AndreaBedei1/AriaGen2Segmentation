"""Build a CVAT-importable annotation package.

The package deliberately separates two things that must never be confused:

* **images + task configuration**, which the annotator works on;
* **pre-annotations**, which are automatic model output offered as a starting point
  and are marked as such in every artefact (directory name, manifest field, guide).

No automatic mask is ever written as ground truth. The reviewed result of the
annotation session is the first ground truth this project will have.

Pre-annotations use the CVAT "Segmentation mask 1.1" layout, so the class ids in the
PNGs and the palette in `labelmap.txt` are the Article 1 taxonomy itself.
"""
from __future__ import annotations

import datetime as dt
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from ..hashing import sha256_file
from ..taxonomy import Taxonomy
from .hand_audit import HAND_ATTRIBUTES

# Attributes attached to every annotation task.
FRAME_ATTRIBUTES = [
    {"name": "domain", "input_type": "select", "mutable": False,
     "values": ["car", "motorcycle"], "default_value": "car"},
    {"name": "recording_id", "input_type": "text", "mutable": False,
     "values": [], "default_value": ""},
    {"name": "route_segment", "input_type": "text", "mutable": False,
     "values": [], "default_value": ""},
    {"name": "image_quality", "input_type": "select", "mutable": False,
     "values": ["good", "motion_blurred", "vibration", "low_light",
                "overexposed", "unusable"],
     "default_value": "good"},
    {"name": "ambiguity_flag", "input_type": "checkbox", "mutable": True,
     "values": [], "default_value": "false"},
]

_BOOL_HAND_ATTRIBUTES = [a for a in HAND_ATTRIBUTES]


def _hand_attribute_spec() -> List[Dict[str, Any]]:
    spec = []
    for name in _BOOL_HAND_ATTRIBUTES:
        spec.append({"name": name, "input_type": "checkbox", "mutable": True,
                     "values": [], "default_value": "false"})
    return spec


# Which attributes belong on which label.
COCKPIT_LABELS = {"mirror", "instrument_display", "control_and_ego_vehicle"}


def build_label_spec(taxonomy: Taxonomy) -> List[Dict[str, Any]]:
    """CVAT task label specification for the Article 1 macro taxonomy.

    `unknown` is included because a reviewer needs somewhere to park a genuinely
    unresolved region during the session, but the guide states it must not survive
    into the delivered annotation.
    """
    labels = []
    for c in taxonomy.classes:
        attributes = list(FRAME_ATTRIBUTES)
        if c.name in COCKPIT_LABELS:
            attributes = attributes + _hand_attribute_spec()
        labels.append({
            "name": c.name,
            "id": c.id,
            "color": "#{:02x}{:02x}{:02x}".format(*c.color),
            "type": "mask",
            "attributes": attributes,
        })
    return labels


def write_labelmap(path: Path, taxonomy: Taxonomy) -> None:
    """CVAT `Segmentation mask 1.1` labelmap.txt."""
    lines = ["# label:color_rgb:parts:actions"]
    for c in taxonomy.classes:
        lines.append(f"{c.name}:{c.color[0]},{c.color[1]},{c.color[2]}::")
    path.write_text("\n".join(lines) + "\n")


def write_palette(path: Path, taxonomy: Taxonomy) -> None:
    path.write_text(json.dumps(
        {str(c.id): {"name": c.name, "rgb": list(c.color)} for c in taxonomy.classes},
        indent=2) + "\n")


@dataclass
class PackageItem:
    image_name: str
    image_relative_path: str
    preannotation_relative_path: Optional[str]
    domain: str
    recording_id: str
    source_frame_index: int
    timestamp_ns: int
    split_group: str
    selection_reason: str
    expected_classes: List[str]
    failure_mode_candidate: Optional[str]
    hand_visibility_candidate: Dict[str, str]
    route_segment: Optional[str]
    route_progression: Optional[float]
    pairing_id: Optional[str]
    image_sha256: Optional[str] = None
    preannotation_sha256: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        from dataclasses import asdict
        return asdict(self)


def write_manifest(root: Path, items: Sequence[PackageItem],
                   taxonomy: Taxonomy, extra: Optional[Dict[str, Any]] = None
                   ) -> Dict[str, Any]:
    doc = {
        "schema": "article1_annotation_package_v1",
        "generated_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "taxonomy": {"classes": [{"id": c.id, "name": c.name, "group": c.group,
                                  "color": list(c.color)} for c in taxonomy.classes]},
        "preannotations": {
            "present": any(i.preannotation_relative_path for i in items),
            "source": "frozen Article 1 semantic camera (Mask2Former external + "
                      "Grounding DINO/SAM2.1 cockpit proxy + geometric fallback)",
            "status": "PRE_ANNOTATION_NOT_GROUND_TRUTH",
            "instruction": "every pre-annotated pixel must be reviewed; nothing here "
                           "is validated",
        },
        "counts": {
            "items": len(items),
            "per_domain": {d: sum(1 for i in items if i.domain == d)
                           for d in sorted({i.domain for i in items})},
            "per_group": {g: sum(1 for i in items if i.split_group == g)
                          for g in sorted({i.split_group for i in items})},
        },
        "items": [i.to_dict() for i in items],
    }
    if extra:
        doc.update(extra)
    (root / "manifest.json").write_text(json.dumps(doc, indent=2) + "\n")
    return doc


def write_checksums(root: Path, paths: Sequence[Path]) -> Path:
    out = root / "checksums.sha256"
    lines = []
    for p in sorted(paths):
        if p.is_file():
            lines.append(f"{sha256_file(p)}  {p.relative_to(root)}")
    out.write_text("\n".join(lines) + "\n")
    return out


ANNOTATION_GUIDE = """# Article 1 annotation guide — car and motorcycle

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
"""


def write_guide(root: Path) -> Path:
    path = root / "ANNOTATION_GUIDE.md"
    path.write_text(ANNOTATION_GUIDE)
    return path
