"""Domain resolution never uses the frame rate; hand states stay candidates."""
from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from aria_drive_seg.ingestion.hand_audit import (AGREEMENT_CASES, EVALUABLE_STATES,
                                                 HAND_ATTRIBUTES, HandSignals,
                                                 STATES, attributes_for,
                                                 propose_state, summarise)
from aria_drive_seg.ingestion.inventory import (DomainEvidence, FileRecord,
                                                classify_domain, recording_id_for)


def car_evidence(**kw) -> DomainEvidence:
    base = dict(static_fraction=0.33, top_band_static_fraction=0.75,
                bottom_band_static_fraction=0.38, top_band_luminance=0.15,
                border_static_fraction=0.45, sampled_frames=14)
    base.update(kw)
    return DomainEvidence(**base)


def moto_evidence(**kw) -> DomainEvidence:
    base = dict(static_fraction=0.05, top_band_static_fraction=0.04,
                bottom_band_static_fraction=0.12, top_band_luminance=0.39,
                border_static_fraction=0.07, sampled_frames=14)
    base.update(kw)
    return DomainEvidence(**base)


# --------------------------------------------------------------------------- #
# the classifier structurally cannot see the frame rate
# --------------------------------------------------------------------------- #
def test_domain_evidence_has_no_rate_or_count_field():
    """The 10 vs 15 fps difference must be un-observable to the classifier."""
    fields = {f.name for f in dataclasses.fields(DomainEvidence)}
    for banned in ("fps", "frame_rate", "effective_fps", "rate_hz", "frame_count",
                   "duration_s", "duration", "num_frames", "period"):
        assert banned not in fields


def test_classifier_signature_takes_only_evidence():
    import inspect
    params = list(inspect.signature(classify_domain).parameters)
    assert params == ["evidence"]


def test_car_and_motorcycle_are_separated_by_ego_structure():
    assert classify_domain(car_evidence()).domain == "car"
    assert classify_domain(moto_evidence()).domain == "motorcycle"


def test_sample_count_does_not_flip_the_decision():
    """`sampled_frames` is a sample-size qualifier, not a rate proxy."""
    for n in (2, 6, 14, 30, 200):
        assert classify_domain(car_evidence(sampled_frames=n)).domain == "car"
        assert classify_domain(moto_evidence(sampled_frames=n)).domain == "motorcycle"


def test_insufficient_evidence_yields_unknown_rather_than_a_guess():
    assert classify_domain(car_evidence(sampled_frames=1)).domain == "unknown"
    flat = DomainEvidence(0.5, 0.45, 0.45, 0.28, 0.45, 14)   # inside every dead band
    assert classify_domain(flat).domain == "unknown"


def test_decision_carries_reviewable_rationale_and_evidence():
    d = classify_domain(moto_evidence())
    assert d.rationale and all(isinstance(r, str) for r in d.rationale)
    assert d.evidence["top_band_static_fraction"] == pytest.approx(0.04)
    assert d.confidence > 0.5


def test_recording_id_is_content_derived_not_name_derived():
    a = FileRecord("a.vrs", "/x/a.vrs", "a.vrs", ".vrs", 1, "abcdef1234567890",
                   "t", 0.0, "vrs_recording", estimated_domain="motorcycle")
    b = FileRecord("renamed.vrs", "/y/renamed.vrs", "renamed.vrs", ".vrs", 1,
                   "abcdef1234567890", "t", 0.0, "vrs_recording",
                   estimated_domain="motorcycle")
    assert recording_id_for(a) == recording_id_for(b)


# --------------------------------------------------------------------------- #
# hand visibility: candidates, never measurements
# --------------------------------------------------------------------------- #
def signals(**kw) -> HandSignals:
    return HandSignals(**kw)


def test_tracked_and_proxy_agreeing_gives_visible():
    c = propose_state(signals(tracking_available=True, tracking_valid=True,
                              landmarks_total=21, landmarks_in_frame=21,
                              proxy_available=True, proxy_mask_present=True))
    assert c.state_candidate == "visible"
    assert c.agreement_case == 1
    assert c.evaluable and c.missing_mask_penalised


def test_landmarks_outside_the_image_give_out_of_frame_not_a_failure():
    c = propose_state(signals(tracking_available=True, tracking_valid=True,
                              landmarks_total=21, landmarks_in_frame=0,
                              proxy_available=True, proxy_mask_present=False))
    assert c.state_candidate == "out_of_frame"
    assert not c.evaluable
    assert not c.missing_mask_penalised
    assert c.agreement_case == 5


def test_nothing_reported_is_not_visible_and_is_not_penalised():
    c = propose_state(signals(tracking_available=True, tracking_valid=False,
                              proxy_available=True, proxy_mask_present=False))
    assert c.state_candidate == "not_visible"
    assert not c.missing_mask_penalised
    assert c.agreement_case == 3
    assert any("expected, non-erroneous" in r for r in c.rationale)


def test_a_mask_without_any_hand_evidence_is_a_candidate_false_mask():
    c = propose_state(signals(tracking_available=True, tracking_valid=False,
                              proxy_available=True, proxy_mask_present=True))
    assert c.agreement_case == 4
    assert not c.evaluable


def test_long_lived_mask_without_support_is_flagged_as_over_propagation():
    c = propose_state(signals(tracking_available=True, tracking_valid=False,
                              proxy_available=True, proxy_mask_present=True,
                              proxy_persisted_frames=20))
    assert c.agreement_case == 9
    assert any("over-propagation" in r for r in c.rationale)


def test_partial_landmarks_give_partially_visible():
    c = propose_state(signals(tracking_available=True, tracking_valid=True,
                              landmarks_total=21, landmarks_in_frame=9,
                              proxy_available=True, proxy_mask_present=True))
    assert c.state_candidate == "partially_visible"
    assert c.evaluable


def test_low_local_sharpness_gives_motion_blurred():
    c = propose_state(signals(tracking_available=True, tracking_valid=True,
                              landmarks_total=21, landmarks_in_frame=21,
                              proxy_available=True, proxy_mask_present=True,
                              local_blur_variance=10.0,
                              reference_blur_variance=500.0))
    assert c.state_candidate == "motion_blurred"
    assert c.agreement_case == 7
    assert not c.evaluable


def test_absent_tracking_stream_yields_uncertain_not_not_visible():
    c = propose_state(signals(tracking_available=False, tracking_valid=False))
    assert c.state_candidate == "uncertain"


def test_only_visible_states_are_evaluable():
    assert set(EVALUABLE_STATES) == {"visible", "partially_visible"}
    for state in STATES:
        assert state in STATES
    for state in ("out_of_frame", "occluded", "not_visible", "uncertain"):
        assert state not in EVALUABLE_STATES


def test_every_candidate_requires_review_and_is_not_ground_truth():
    c = propose_state(signals(tracking_available=True, tracking_valid=True,
                              landmarks_total=21, landmarks_in_frame=21))
    assert c.review_required
    assert not c.is_ground_truth


def test_summary_reports_no_accuracy_metric():
    cands = []
    for _ in range(3):
        c = propose_state(signals(tracking_available=True, tracking_valid=True,
                                  landmarks_total=21, landmarks_in_frame=21,
                                  proxy_available=True, proxy_mask_present=True))
        c.side = "left"
        cands.append(c)
    s = summarise(cands)
    assert s["is_ground_truth"] is False
    assert s["status"] == "review_candidates_only"
    for banned in ("recall", "precision", "accuracy", "iou", "f1"):
        assert not any(banned in k.lower() for k in s)


def test_attributes_cover_the_required_vocabulary():
    c = propose_state(signals(tracking_available=True, tracking_valid=True,
                              landmarks_total=21, landmarks_in_frame=21))
    c.side = "left"
    attrs = attributes_for(c)
    for name in ("left_hand_visible", "left_hand_partially_visible",
                 "left_hand_occluded", "left_hand_out_of_frame",
                 "left_hand_motion_blurred", "left_arm_visible",
                 "hand_tracking_available", "hand_tracking_valid",
                 "hand_segmentation_proxy_available"):
        assert name in attrs
        assert name in HAND_ATTRIBUTES


def test_arm_visibility_is_left_to_the_reviewer_not_guessed():
    c = propose_state(signals(tracking_available=True, tracking_valid=True,
                              landmarks_total=21, landmarks_in_frame=21))
    c.side = "right"
    assert attributes_for(c)["right_arm_visible"] is None


def test_all_ten_agreement_cases_are_defined():
    assert sorted(AGREEMENT_CASES) == list(range(1, 11))
