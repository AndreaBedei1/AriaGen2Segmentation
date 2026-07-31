"""The shared cockpit stage is leakage-safe, two-domain, and gated on review."""
from __future__ import annotations

import json

import pandas as pd
import pytest

from aria_drive_seg.article1.cockpit_segformer import (COCKPIT_CLASSES,
                                                       require_reviewed_annotations)
from aria_drive_seg.article1.cockpit_training import (FORBIDDEN_SPLIT_UNITS,
                                                      SAFE_SPLIT_UNITS,
                                                      AugmentationPolicy,
                                                      assert_no_temporal_leakage,
                                                      future_fusion_contract,
                                                      grouped_split, planned_metrics,
                                                      swap_side_attributes,
                                                      training_readiness,
                                                      validate_shared_domains,
                                                      validate_split_unit)

NS = 1_000_000_000


def frames(recording: str, domain: str, n: int, start_s: float = 0.0,
           fps: float = 15.0, session: str = "s1"):
    return pd.DataFrame({
        "recording_id": [recording] * n,
        "session_id": [session] * n,
        "domain": [domain] * n,
        "source_frame_index": list(range(n)),
        "timestamp_ns": [int((start_s + i / fps) * NS) for i in range(n)],
    })


@pytest.fixture
def dataset():
    return pd.concat([
        frames("car_a", "car", 20, 0.0, 10.0),
        frames("car_b", "car", 20, 0.0, 10.0, session="s2"),
        frames("moto_a", "motorcycle", 30, 0.0, 15.0, session="s3"),
        frames("moto_b", "motorcycle", 30, 0.0, 15.0, session="s4"),
    ], ignore_index=True)


# --------------------------------------------------------------------------- #
# split units
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("unit", FORBIDDEN_SPLIT_UNITS)
def test_per_frame_split_units_are_rejected(unit):
    with pytest.raises(ValueError, match="single frame"):
        validate_split_unit(unit)


@pytest.mark.parametrize("unit", SAFE_SPLIT_UNITS)
def test_group_split_units_are_accepted(unit):
    validate_split_unit(unit)


def test_frame_index_can_never_become_a_split_unit():
    assert "frame_index" in FORBIDDEN_SPLIT_UNITS
    assert "source_frame_index" in FORBIDDEN_SPLIT_UNITS
    assert not set(SAFE_SPLIT_UNITS) & set(FORBIDDEN_SPLIT_UNITS)


# --------------------------------------------------------------------------- #
# leakage
# --------------------------------------------------------------------------- #
def test_grouped_split_keeps_whole_recordings_on_one_side(dataset):
    train, valid = grouped_split(dataset, "recording_id", ["car_b", "moto_b"])
    assert set(train["recording_id"]) == {"car_a", "moto_a"}
    assert set(valid["recording_id"]) == {"car_b", "moto_b"}
    assert not set(train["recording_id"]) & set(valid["recording_id"])


def test_a_group_on_both_sides_is_rejected(dataset):
    train = dataset[dataset.recording_id.isin(["car_a", "moto_a", "car_b"])]
    valid = dataset[dataset.recording_id.isin(["car_b", "moto_b"])]
    with pytest.raises(ValueError, match="leakage"):
        from aria_drive_seg.article1.splits import assert_no_group_leakage
        assert_no_group_leakage(train, valid, "recording_id")


def test_consecutive_frames_of_one_sequence_cannot_straddle_the_split():
    """The exact failure the grouping exists to prevent."""
    df = frames("moto_a", "motorcycle", 40, 0.0, 15.0)
    train = df.iloc[:20]
    valid = df.iloc[20:]      # frame 19 and frame 20 are 67 ms apart
    with pytest.raises(ValueError, match="temporal leakage"):
        assert_no_temporal_leakage(train, valid, min_separation_s=5.0)


def test_temporal_guarantee_is_in_seconds_not_frames():
    """10 frames apart is 1.0 s at 10 fps but only 0.67 s at 15 fps."""
    car = frames("car_a", "car", 40, 0.0, 10.0)
    moto = frames("moto_a", "motorcycle", 40, 0.0, 15.0)
    # a 10-frame gap satisfies a 0.9 s rule on the car but not on the motorcycle
    assert_no_temporal_leakage(car.iloc[:10], car.iloc[20:], min_separation_s=0.9)
    with pytest.raises(ValueError):
        assert_no_temporal_leakage(moto.iloc[:10], moto.iloc[20:],
                                   min_separation_s=0.9)


def test_well_separated_groups_pass_the_temporal_check(dataset):
    train, valid = grouped_split(dataset, "recording_id", ["car_b", "moto_b"])
    assert_no_temporal_leakage(train, valid, min_separation_s=5.0)


def test_temporal_check_needs_the_columns_it_relies_on():
    df = pd.DataFrame({"domain": ["car"], "x": [1]})
    with pytest.raises(ValueError, match="recording_id and timestamp_ns"):
        assert_no_temporal_leakage(df, df)


# --------------------------------------------------------------------------- #
# the model is shared between the two domains
# --------------------------------------------------------------------------- #
def test_a_single_domain_dataset_is_rejected():
    only_car = frames("car_a", "car", 10)
    with pytest.raises(RuntimeError, match="shared between car and motorcycle"):
        validate_shared_domains(only_car)


def test_both_domains_are_accepted(dataset):
    validate_shared_domains(dataset)


def test_grouped_split_enforces_the_shared_domain_rule():
    only_moto = pd.concat([frames("moto_a", "motorcycle", 10),
                           frames("moto_b", "motorcycle", 10)], ignore_index=True)
    with pytest.raises(RuntimeError):
        grouped_split(only_moto, "recording_id", ["moto_b"])


# --------------------------------------------------------------------------- #
# augmentation
# --------------------------------------------------------------------------- #
def test_horizontal_flip_swaps_the_side_attributes():
    policy = AugmentationPolicy()
    assert policy.horizontal_flip
    assert policy.horizontal_flip_swaps_side_attributes
    swapped = swap_side_attributes({"left_hand_visible": True,
                                    "right_hand_visible": False,
                                    "left_arm_visible": True,
                                    "right_arm_visible": None})
    assert swapped["left_hand_visible"] is False
    assert swapped["right_hand_visible"] is True
    assert swapped["right_arm_visible"] is True


def test_geometry_breaking_augmentations_are_forbidden():
    forbidden = AugmentationPolicy().forbidden
    assert "vertical_flip" in forbidden
    assert any("cutout" in f for f in forbidden)


# --------------------------------------------------------------------------- #
# the training gate
# --------------------------------------------------------------------------- #
def test_training_is_blocked_without_a_review_marker(tmp_path):
    readiness = training_readiness(tmp_path)
    assert readiness["ready_to_train"] is False
    assert readiness["blockers"]
    with pytest.raises(RuntimeError, match="blocked"):
        require_reviewed_annotations(tmp_path)


def test_the_gate_opens_only_with_reviewed_masks(tmp_path):
    (tmp_path / "masks").mkdir()
    (tmp_path / "REVIEWED_ANNOTATIONS.json").write_text(
        json.dumps({"reviewer": "human", "frames": 10}))
    readiness = training_readiness(tmp_path)
    assert readiness["ready_to_train"] is True
    require_reviewed_annotations(tmp_path)


def test_pseudo_labels_are_never_the_reference():
    metrics = planned_metrics()
    joined = json.dumps(metrics).lower()
    assert "no metric is computed against pseudo-labels" in joined
    assert "matched pair" in joined or "recording" in joined


# --------------------------------------------------------------------------- #
# hands are evaluated only where a reviewer confirmed visibility
# --------------------------------------------------------------------------- #
def test_hand_metric_policy_does_not_penalise_absence():
    hands = " ".join(planned_metrics()["hands"]).lower()
    assert "visible or " in hands or "partially_visible" in hands
    assert "not counted as an error" in hands
    assert "false hand masks" in hands


# --------------------------------------------------------------------------- #
# fusion contract
# --------------------------------------------------------------------------- #
def test_fusion_contract_keeps_the_proxy_marked_until_replaced():
    contract = future_fusion_contract()
    assert "internal_is_fallback=true" in contract["fallback_policy"]
    assert contract["output_contract"]["probabilities"].startswith("4xHxW")
    assert set(contract["mapping_to_macro_taxonomy"]) <= set(COCKPIT_CLASSES)
    assert any("recalibrated" in r for r in contract["revalidation_required"])
