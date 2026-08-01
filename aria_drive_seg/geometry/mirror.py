"""Shared mirror-candidate gating and a minimal, evidence-led cockpit prior.

Two problems this addresses, both visible on the motorcycle and both harmless on
the car:

* the frozen mirror prompts describe a car ("interior rear view mirror", "car side
  wing mirror", "exterior door mirror"), so a bar-end mirror has no phrasing that
  can match it;
* the frozen cockpit fallback paints the bottom of the frame as
  `control_and_ego_vehicle` on a geometric prior alone, which is a reasonable
  description of a dashboard and a poor one of open road under a handlebar.

Nothing here reads the vehicle domain, the file name or the frame rate. The gates
are geometry, shape, size, confidence and temporal consistency, all of which the
car satisfies too. The same configuration is meant to run in both domains.

A mirror touching the image border is explicitly **not** rejected: on a motorcycle
that is the normal case, and the instruction is that keeping a slightly distorted
peripheral region is preferable to losing a real mirror.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

# Prompt phrasings offered to the open-vocabulary detector. Both vehicles are
# described in the same list; the detector decides what is present.
SHARED_MIRROR_PROMPTS = (
    "rear view mirror",
    "side mirror",
    "wing mirror",
    "motorcycle mirror",
    "handlebar mirror",
    "bar end mirror",
    "car side mirror",
    "round mirror on a stalk",
)
SHARED_COCKPIT_PROMPTS = (
    "instrument cluster",
    "dashboard display",
    "speedometer",
    "handlebar",
    "steering wheel",
    "vehicle controls",
    "brake lever",
    "windshield",
    "motorcycle fairing",
    "fuel tank",
    "cockpit structure",
)


@dataclass
class MirrorGate:
    """Size, shape and confidence limits for a mirror proposal.

    The bounds are deliberately generous at the small end and strict at the large
    end: missing a mirror is the failure this refinement exists to fix, while a
    proposal covering a quarter of the frame is never a mirror.
    """

    min_area_fraction: float = 0.00004     # ~120 px at 2016x1512
    max_area_fraction: float = 0.030       # 3% of the frame
    max_aspect_ratio: float = 5.0          # allows a mirror plus its stalk
    min_fill_ratio: float = 0.18           # mask area over bounding-box area
    min_score: float = 0.20
    allow_edge_contact: bool = True
    max_centre_distance_fraction: float = 0.95

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class MirrorCandidate:
    frame_index: int
    score: float
    area_fraction: float
    bbox: Tuple[float, float, float, float]
    centroid: Tuple[float, float]
    aspect_ratio: float
    fill_ratio: float
    touches_edge: bool
    lateral_position: float          # 0 at the image centre, 1 at the side edge
    accepted: bool = False
    rejection_reasons: List[str] = field(default_factory=list)
    support_frames: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def describe_proposal(mask: np.ndarray, score: float, frame_index: int,
                      image_size: Optional[Tuple[int, int]] = None
                      ) -> MirrorCandidate:
    """Measure a binary proposal without judging it."""
    h, w = mask.shape[:2] if image_size is None else (image_size[1], image_size[0])
    ys, xs = np.nonzero(mask)
    if xs.size == 0:
        return MirrorCandidate(frame_index, score, 0.0, (0, 0, 0, 0), (0, 0),
                               0.0, 0.0, False, 0.0, False, ["empty proposal"])
    x0, x1 = float(xs.min()), float(xs.max())
    y0, y1 = float(ys.min()), float(ys.max())
    box_w, box_h = max(1.0, x1 - x0 + 1), max(1.0, y1 - y0 + 1)
    area = float(xs.size)
    cx, cy = float(xs.mean()), float(ys.mean())
    return MirrorCandidate(
        frame_index=frame_index, score=float(score),
        area_fraction=area / float(w * h),
        bbox=(x0, y0, x1, y1), centroid=(cx, cy),
        aspect_ratio=float(max(box_w, box_h) / min(box_w, box_h)),
        fill_ratio=float(area / (box_w * box_h)),
        touches_edge=bool(x0 <= 1 or y0 <= 1 or x1 >= w - 2 or y1 >= h - 2),
        lateral_position=float(abs(cx - (w - 1) / 2.0) / ((w - 1) / 2.0)),
    )


def gate_candidate(candidate: MirrorCandidate,
                   gate: Optional[MirrorGate] = None) -> MirrorCandidate:
    """Accept or reject a proposal on shape, size and confidence alone."""
    gate = gate or MirrorGate()
    reasons: List[str] = list(candidate.rejection_reasons)

    if candidate.area_fraction <= 0:
        reasons.append("empty proposal")
    if 0 < candidate.area_fraction < gate.min_area_fraction:
        reasons.append(
            f"area {candidate.area_fraction:.5%} below the {gate.min_area_fraction:.5%} floor")
    if candidate.area_fraction > gate.max_area_fraction:
        reasons.append(
            f"area {candidate.area_fraction:.2%} above the {gate.max_area_fraction:.2%} "
            "ceiling: too large to be a mirror")
    if candidate.aspect_ratio > gate.max_aspect_ratio:
        reasons.append(
            f"aspect ratio {candidate.aspect_ratio:.1f} above {gate.max_aspect_ratio}")
    if candidate.fill_ratio < gate.min_fill_ratio:
        reasons.append(
            f"fill ratio {candidate.fill_ratio:.2f} below {gate.min_fill_ratio}: "
            "the mask does not occupy its own bounding box")
    if candidate.score < gate.min_score:
        reasons.append(f"score {candidate.score:.2f} below {gate.min_score}")
    # Edge contact is explicitly not a rejection reason.
    if candidate.touches_edge and not gate.allow_edge_contact:
        reasons.append("touches the image border")

    candidate.rejection_reasons = reasons
    candidate.accepted = not reasons
    return candidate


def temporal_support(history: Sequence[bool], window: int = 5,
                     minimum: int = 2) -> int:
    """How many of the last `window` frames also proposed a mirror here."""
    if window <= 0:
        return 0
    recent = list(history)[-window:]
    return int(sum(1 for x in recent if x))


def should_persist(frames_since_last_detection: int,
                   max_persistence_frames: int = 3) -> bool:
    """Whether a mirror mask may survive a frame with no detection.

    Short persistence bridges a single missed detection; anything longer would
    keep painting a mirror that has left the frame, which is the failure mode the
    hand audit already showed for hands.
    """
    return 0 <= frames_since_last_detection <= max_persistence_frames


@dataclass
class CockpitPrior:
    """A weak geometric hint, never a substitute for visual evidence.

    The frozen configuration paints the bottom 32% of the frame as
    `control_and_ego_vehicle` at confidence 0.40 wherever the detector found
    nothing. On a motorcycle most of that region is road.
    """

    enabled: bool = True
    bottom_start_fraction: float = 0.86
    confidence: float = 0.15
    is_fallback: bool = True
    may_override_classes: Tuple[str, ...] = ("unknown",)
    excluded_from_scientific_conclusions: bool = True

    def region(self, height: int, width: int) -> np.ndarray:
        mask = np.zeros((height, width), dtype=bool)
        if not self.enabled:
            return mask
        start = int(round(height * self.bottom_start_fraction))
        mask[max(0, min(start, height)):] = True
        return mask

    def can_override(self, class_name: str) -> bool:
        """A geometric guess may only displace an absent decision."""
        return class_name in self.may_override_classes

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["may_override_classes"] = list(self.may_override_classes)
        d["note"] = ("weak positional hint; it cannot outrank a confident external "
                     "class and is marked as fallback provenance")
        return d


def road_is_protected(prior: CockpitPrior) -> bool:
    """Road surface must never be replaced by a positional guess."""
    return not prior.can_override("road_surface")
