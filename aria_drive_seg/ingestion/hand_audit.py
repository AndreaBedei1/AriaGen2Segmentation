"""Lightweight hand-visibility audit for the motorcycle recording.

Scope discipline: this produces **review candidates**, not measurements. Every state
below is a proposal for a human annotator to confirm or overturn, and the module
never reports recall, precision or any accuracy figure, because there is no reviewed
ground truth for hands.

Intermittent hand visibility is normal on a motorcycle. A hand can leave the RGB
field of view, be hidden by the handlebar, be blurred by vibration, or simply be
below the camera. None of those is a model error, and the audit is built so that
"no mask because no hand is visible" is a *correct* outcome rather than a miss. The
failure that does matter is the opposite one: a mask where no hand is visible, or a
mask that keeps living after the hand has gone.

Hands are not a taxonomy class. They stay inside `control_and_ego_vehicle` (12) and
are described by separate attributes.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

# Candidate visibility states proposed for review.
STATES = ("visible", "partially_visible", "occluded", "out_of_frame",
          "motion_blurred", "uncertain", "not_visible")

# States in which a model is allowed to be evaluated on hands once GT exists.
EVALUABLE_STATES = ("visible", "partially_visible")

# States in which a missing mask must NOT be penalised, but a present mask is
# still worth inspecting as a possible false positive.
NON_PENALISED_STATES = ("out_of_frame", "occluded", "not_visible", "uncertain")

# The attribute vocabulary carried into the annotation package.
HAND_ATTRIBUTES = (
    "left_hand_visible", "right_hand_visible",
    "left_hand_partially_visible", "right_hand_partially_visible",
    "left_hand_occluded", "right_hand_occluded",
    "left_hand_out_of_frame", "right_hand_out_of_frame",
    "left_hand_motion_blurred", "right_hand_motion_blurred",
    "left_arm_visible", "right_arm_visible",
    "hand_tracking_available", "hand_tracking_valid",
    "hand_segmentation_proxy_available",
)


@dataclass
class HandSignals:
    """Auxiliary evidence for one hand in one frame. None of it is ground truth."""

    # on-device hand tracking
    tracking_available: bool = False        # the stream exists for this recording
    tracking_valid: bool = False            # this side was tracked in this sample
    tracking_confidence: Optional[float] = None
    tracking_dt_ms: Optional[float] = None  # temporal distance to the RGB frame
    landmarks_total: int = 0
    landmarks_in_frame: int = 0             # projected inside the image bounds
    centroid_xy: Optional[Sequence[float]] = None

    # open-vocabulary segmentation proxy (diagnostic pass, not the frozen pipeline)
    proxy_available: bool = False           # the proxy ran on this frame
    proxy_mask_present: bool = False
    proxy_score: Optional[float] = None
    proxy_area_fraction: float = 0.0
    proxy_iou_with_tracking_region: Optional[float] = None

    # image evidence
    local_blur_variance: Optional[float] = None
    reference_blur_variance: Optional[float] = None
    local_motion: Optional[float] = None

    # temporal context
    frames_since_last_tracking: Optional[int] = None
    proxy_persisted_frames: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class HandCandidate:
    recording_id: str
    domain: str
    source_frame_index: int
    timestamp_ns: int
    side: str                      # left | right
    state_candidate: str
    state_confidence: float
    evaluable: bool
    missing_mask_penalised: bool
    rationale: List[str]
    agreement_case: int
    agreement_label: str
    signals: Dict[str, Any] = field(default_factory=dict)
    review_required: bool = True
    is_ground_truth: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# Agreement cases required by the audit specification.
AGREEMENT_CASES = {
    1: "hand visible and proxy present",
    2: "hand visible and proxy absent",
    3: "hand not visible and proxy absent",
    4: "hand not visible but a false mask is present",
    5: "tracking present but the hand is not visible in the RGB frame",
    6: "hand visible but tracking absent",
    7: "blurred hand",
    8: "partially visible hand",
    9: "mask propagated for too long",
    10: "confusion with the handlebar or a mirror",
}

# A proxy mask that survives this many consecutive frames without any tracking or
# in-frame landmark support is flagged as possible over-propagation.
PROPAGATION_ALERT_FRAMES = 8
# Local sharpness below this fraction of the frame reference is treated as blur.
BLUR_RATIO = 0.45


def _visibility_from_tracking(s: HandSignals) -> Optional[str]:
    """What the tracker alone suggests, before any image evidence."""
    if not s.tracking_valid:
        return None
    if s.landmarks_total <= 0:
        return "uncertain"
    ratio = s.landmarks_in_frame / s.landmarks_total
    if ratio <= 0.0:
        return "out_of_frame"
    if ratio < 0.85:
        return "partially_visible"
    return "visible"


def propose_state(signals: HandSignals) -> HandCandidate:
    """Propose a visibility state candidate from the auxiliary signals.

    Deliberately conservative: when the signals disagree the result is `uncertain`
    rather than a confident label, because a wrong confident label costs a reviewer
    more than an explicit "please look at this one".
    """
    reasons: List[str] = []
    tracked_state = _visibility_from_tracking(signals)

    blurred = (signals.local_blur_variance is not None
               and signals.reference_blur_variance
               and signals.local_blur_variance
               < BLUR_RATIO * signals.reference_blur_variance)

    state: str
    confidence: float

    if tracked_state == "out_of_frame":
        state, confidence = "out_of_frame", 0.75
        reasons.append(
            "hand tracking projects every landmark outside the RGB image: the hand "
            "is tracked by the device but not in the camera's field of view")
        if signals.proxy_mask_present:
            reasons.append(
                "a proxy mask exists while no landmark falls inside the image: "
                "candidate false mask")
    elif tracked_state in ("visible", "partially_visible"):
        state = tracked_state
        confidence = 0.7 if tracked_state == "visible" else 0.6
        reasons.append(
            f"{signals.landmarks_in_frame}/{signals.landmarks_total} tracked "
            "landmarks project inside the image")
        if signals.proxy_mask_present:
            confidence = min(0.9, confidence + 0.2)
            reasons.append("the segmentation proxy also proposes a hand region here")
        else:
            reasons.append(
                "the segmentation proxy proposes no hand region: the hand may be "
                "occluded by the handlebar, or the proxy may have missed it")
            state = "uncertain" if tracked_state == "visible" else state
            confidence = min(confidence, 0.5)
        if blurred:
            state, confidence = "motion_blurred", min(confidence, 0.55)
            reasons.append(
                "local sharpness is well below the frame reference: the hand region "
                "is blurred by motion or vibration")
    elif signals.proxy_mask_present:
        state, confidence = "uncertain", 0.4
        reasons.append(
            "the segmentation proxy proposes a hand region but the tracker reports "
            "nothing: either the tracker failed or the proxy is a false positive")
        if signals.proxy_persisted_frames >= PROPAGATION_ALERT_FRAMES:
            confidence = 0.3
            reasons.append(
                f"the proxy region has persisted for {signals.proxy_persisted_frames} "
                "frames with no tracking support: candidate over-propagation")
    else:
        state, confidence = "not_visible", 0.5
        reasons.append(
            "neither hand tracking nor the segmentation proxy reports a hand; on a "
            "motorcycle this is an expected, non-erroneous situation")
        if not signals.tracking_available:
            state, confidence = "uncertain", 0.25
            reasons.append(
                "no hand-tracking stream is available for this recording, so absence "
                "cannot be distinguished from a missing signal")

    case, label = _agreement_case(signals, state, blurred)
    return HandCandidate(
        recording_id="", domain="", source_frame_index=-1, timestamp_ns=-1, side="",
        state_candidate=state, state_confidence=float(confidence),
        evaluable=state in EVALUABLE_STATES,
        missing_mask_penalised=state in EVALUABLE_STATES,
        rationale=reasons, agreement_case=case, agreement_label=label,
        signals=signals.to_dict(),
    )


def _agreement_case(s: HandSignals, state: str, blurred: bool) -> tuple:
    """Classify the tracking/proxy/image agreement into the audit's case list."""
    visible_like = state in ("visible", "partially_visible")
    if state == "motion_blurred" or blurred:
        return 7, AGREEMENT_CASES[7]
    if state == "partially_visible":
        return 8, AGREEMENT_CASES[8]
    if s.proxy_persisted_frames >= PROPAGATION_ALERT_FRAMES and not s.tracking_valid:
        return 9, AGREEMENT_CASES[9]
    if (s.proxy_mask_present and s.proxy_iou_with_tracking_region is not None
            and s.proxy_iou_with_tracking_region < 0.05 and s.tracking_valid):
        return 10, AGREEMENT_CASES[10]
    if state == "out_of_frame" and s.tracking_valid:
        return 5, AGREEMENT_CASES[5]
    if visible_like and s.proxy_mask_present:
        return 1, AGREEMENT_CASES[1]
    if visible_like and not s.proxy_mask_present:
        return 2, AGREEMENT_CASES[2]
    if visible_like and not s.tracking_valid:
        return 6, AGREEMENT_CASES[6]
    if not visible_like and s.proxy_mask_present:
        return 4, AGREEMENT_CASES[4]
    return 3, AGREEMENT_CASES[3]


def attributes_for(candidate: HandCandidate) -> Dict[str, Any]:
    """Render one candidate into the per-side annotation attribute vocabulary."""
    side = candidate.side or "left"
    s = candidate.signals
    state = candidate.state_candidate
    return {
        f"{side}_hand_visible": state == "visible",
        f"{side}_hand_partially_visible": state == "partially_visible",
        f"{side}_hand_occluded": state == "occluded",
        f"{side}_hand_out_of_frame": state == "out_of_frame",
        f"{side}_hand_motion_blurred": state == "motion_blurred",
        # Arm visibility is not inferable from the available signals; it is left for
        # the reviewer rather than guessed.
        f"{side}_arm_visible": None,
        "hand_tracking_available": bool(s.get("tracking_available", False)),
        "hand_tracking_valid": bool(s.get("tracking_valid", False)),
        "hand_segmentation_proxy_available": bool(s.get("proxy_available", False)),
    }


def summarise(candidates: Sequence[HandCandidate]) -> Dict[str, Any]:
    """Aggregate candidate states and agreement cases.

    All counts are candidate counts. None of them is a performance metric.
    """
    total = len(candidates)
    by_state: Dict[str, int] = {s: 0 for s in STATES}
    by_case: Dict[str, int] = {str(k): 0 for k in AGREEMENT_CASES}
    by_side: Dict[str, Dict[str, int]] = {"left": {s: 0 for s in STATES},
                                          "right": {s: 0 for s in STATES}}
    for c in candidates:
        by_state[c.state_candidate] = by_state.get(c.state_candidate, 0) + 1
        by_case[str(c.agreement_case)] += 1
        if c.side in by_side:
            by_side[c.side][c.state_candidate] = \
                by_side[c.side].get(c.state_candidate, 0) + 1

    evaluable = sum(1 for c in candidates if c.evaluable)
    false_mask = by_case["4"]
    propagation = by_case["9"]
    return {
        "status": "review_candidates_only",
        "is_ground_truth": False,
        "total_candidates": total,
        "per_state": by_state,
        "per_state_by_side": by_side,
        "per_agreement_case": {
            k: {"count": v, "label": AGREEMENT_CASES[int(k)]}
            for k, v in by_case.items()},
        "evaluable_candidates": evaluable,
        "evaluable_fraction": (evaluable / total) if total else 0.0,
        "candidate_false_mask_count": false_mask,
        "candidate_over_propagation_count": propagation,
        "notes": [
            "Intermittent hand visibility is expected on a motorcycle; a frame with "
            "no hand mask is not by itself a failure.",
            "A model may be evaluated on hands only in frames a reviewer confirms as "
            "visible or partially_visible.",
            "Recall, precision and accuracy are deliberately not reported: there is "
            "no reviewed hand ground truth yet.",
            "Hands remain part of control_and_ego_vehicle (12); they are not a new "
            "taxonomy class.",
        ],
    }


def local_patch_metrics(gray: np.ndarray, centre_xy: Sequence[float],
                        radius_px: int = 90) -> Dict[str, float]:
    """Sharpness of an image patch around an expected hand position."""
    import cv2

    h, w = gray.shape[:2]
    cx, cy = int(round(float(centre_xy[0]))), int(round(float(centre_xy[1])))
    x0, x1 = max(0, cx - radius_px), min(w, cx + radius_px)
    y0, y1 = max(0, cy - radius_px), min(h, cy + radius_px)
    if x1 <= x0 or y1 <= y0:
        return {"blur_variance": 0.0, "pixels": 0}
    patch = gray[y0:y1, x0:x1]
    return {"blur_variance": float(cv2.Laplacian(patch, cv2.CV_32F).var()),
            "pixels": int(patch.size)}


def landmarks_in_bounds(points: Sequence[Optional[Sequence[float]]],
                        width: int, height: int) -> int:
    """Count projected landmarks that fall inside the image."""
    n = 0
    for p in points:
        if p is None:
            continue
        if 0 <= p[0] < width and 0 <= p[1] < height:
            n += 1
    return n
