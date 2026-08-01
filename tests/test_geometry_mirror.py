"""Mirror gating keeps edge mirrors, rejects impostors, and stays domain-free."""
from __future__ import annotations

import inspect

import numpy as np
import pytest

from aria_drive_seg.geometry import mirror as mirror_module
from aria_drive_seg.geometry.mirror import (SHARED_COCKPIT_PROMPTS,
                                            SHARED_MIRROR_PROMPTS, CockpitPrior,
                                            MirrorGate, describe_proposal,
                                            gate_candidate, road_is_protected,
                                            should_persist, temporal_support)

W, H = 2016, 1512


def proposal(x0, y0, x1, y1, score=0.5):
    mask = np.zeros((H, W), bool)
    mask[y0:y1, x0:x1] = True
    return describe_proposal(mask, score, frame_index=0)


# --------------------------------------------------------------------------- #
# the failure this exists to fix
# --------------------------------------------------------------------------- #
def test_a_mirror_touching_the_left_border_is_kept():
    """On a motorcycle the bar-end mirrors sit at the very edge."""
    c = gate_candidate(proposal(0, 1020, 150, 1180, score=0.35))
    assert c.touches_edge
    assert c.accepted, c.rejection_reasons


def test_a_mirror_touching_the_right_border_is_kept():
    c = gate_candidate(proposal(W - 150, 1020, W, 1180, score=0.35))
    assert c.touches_edge
    assert c.accepted, c.rejection_reasons


def test_edge_contact_is_never_a_rejection_reason_by_default():
    c = gate_candidate(proposal(0, 1000, 140, 1150, score=0.4))
    assert not any("border" in r for r in c.rejection_reasons)


def test_edge_contact_can_be_rejected_only_when_explicitly_configured():
    gate = MirrorGate(allow_edge_contact=False)
    c = gate_candidate(proposal(0, 1000, 140, 1150, score=0.4), gate)
    assert not c.accepted
    assert any("border" in r for r in c.rejection_reasons)


# --------------------------------------------------------------------------- #
# impostors
# --------------------------------------------------------------------------- #
def test_a_quadrant_sized_proposal_is_rejected():
    """The hand probe produced exactly this: a whole quadrant called an object."""
    c = gate_candidate(proposal(0, 760, 1400, H, score=0.9))
    assert not c.accepted
    assert any("too large to be a mirror" in r for r in c.rejection_reasons)


def test_a_long_thin_streak_is_rejected():
    c = gate_candidate(proposal(100, 1000, 1900, 1030, score=0.6))
    assert not c.accepted
    assert any("aspect ratio" in r for r in c.rejection_reasons)


def test_a_speck_is_rejected():
    c = gate_candidate(proposal(500, 500, 505, 505, score=0.9))
    assert not c.accepted
    assert any("below" in r for r in c.rejection_reasons)


def test_a_low_confidence_proposal_is_rejected():
    c = gate_candidate(proposal(100, 1000, 240, 1150, score=0.05))
    assert not c.accepted
    assert any("score" in r for r in c.rejection_reasons)


def test_a_scattered_proposal_is_rejected():
    """Speckle spread across a large box is not an object."""
    mask = np.zeros((H, W), bool)
    mask[1000:1300:12, 100:400:12] = True
    c = gate_candidate(describe_proposal(mask, 0.6, 0))
    assert not c.accepted
    assert any("fill ratio" in r for r in c.rejection_reasons)


def test_a_mirror_whose_glass_is_missed_is_still_kept():
    """SAM2 sometimes returns the housing ring without the reflecting surface.

    The taxonomy counts housing as mirror, so a partial detection is useful
    evidence rather than an impostor.
    """
    mask = np.zeros((H, W), bool)
    mask[1000:1200, 100:300] = True
    mask[1020:1180, 120:280] = False
    c = gate_candidate(describe_proposal(mask, 0.6, 0))
    assert c.accepted, c.rejection_reasons


def test_an_empty_proposal_is_rejected():
    c = gate_candidate(describe_proposal(np.zeros((H, W), bool), 0.9, 0))
    assert not c.accepted


# --------------------------------------------------------------------------- #
# measurements are descriptive, not judgemental
# --------------------------------------------------------------------------- #
def test_lateral_position_is_zero_at_the_centre_and_one_at_the_edge():
    centre = proposal(W // 2 - 70, H // 2 - 70, W // 2 + 70, H // 2 + 70)
    assert centre.lateral_position < 0.05
    edge = proposal(0, 1000, 140, 1140)
    assert edge.lateral_position > 0.85


def test_a_central_mirror_is_also_accepted():
    """A car's interior rear-view mirror is central; the gate must not be lateral."""
    c = gate_candidate(proposal(W // 2 - 120, 180, W // 2 + 120, 320, score=0.4))
    assert c.accepted, c.rejection_reasons


# --------------------------------------------------------------------------- #
# temporal behaviour
# --------------------------------------------------------------------------- #
def test_temporal_support_counts_recent_detections():
    assert temporal_support([True, False, True, True, True], window=5) == 4
    assert temporal_support([], window=5) == 0


def test_a_mask_may_bridge_a_short_dropout():
    assert should_persist(0)
    assert should_persist(2)


def test_a_mask_must_disappear_after_the_mirror_leaves():
    assert not should_persist(4)
    assert not should_persist(30)


# --------------------------------------------------------------------------- #
# the cockpit prior is weak, marked and cannot take road
# --------------------------------------------------------------------------- #
def test_the_prior_covers_far_less_than_the_frozen_one():
    prior = CockpitPrior()
    frozen_start = 0.68
    assert prior.bottom_start_fraction > frozen_start
    region = prior.region(H, W)
    assert region.mean() < (1.0 - frozen_start)


def test_the_prior_is_low_confidence_and_flagged_as_fallback():
    prior = CockpitPrior()
    assert prior.confidence <= 0.2
    assert prior.is_fallback
    assert prior.excluded_from_scientific_conclusions


def test_the_prior_can_never_take_road_surface():
    prior = CockpitPrior()
    assert road_is_protected(prior)
    assert not prior.can_override("road_surface")


def test_the_prior_cannot_displace_real_external_classes():
    prior = CockpitPrior()
    for name in ("road_surface", "vehicle", "lane_marking", "two_wheeler",
                 "road_boundary_or_obstacle", "other_environment", "pedestrian"):
        assert not prior.can_override(name), name
    assert prior.can_override("unknown")


def test_a_disabled_prior_proposes_nothing():
    assert not CockpitPrior(enabled=False).region(H, W).any()


# --------------------------------------------------------------------------- #
# domain neutrality
# --------------------------------------------------------------------------- #
def test_prompts_name_both_vehicles():
    joined = " ".join(SHARED_MIRROR_PROMPTS).lower()
    assert "motorcycle" in joined and "car" in joined
    cockpit = " ".join(SHARED_COCKPIT_PROMPTS).lower()
    assert "handlebar" in cockpit and "steering wheel" in cockpit


def test_gating_never_branches_on_the_domain():
    source = inspect.getsource(mirror_module)
    for banned in ("if domain", "domain ==", "vehicle_type ==", "== 'motorcycle'",
                   '== "motorcycle"', "== 'car'", '== "car"', "fps"):
        assert banned not in source, banned


def test_gate_signature_takes_no_domain():
    params = list(inspect.signature(gate_candidate).parameters)
    assert params == ["candidate", "gate"]
