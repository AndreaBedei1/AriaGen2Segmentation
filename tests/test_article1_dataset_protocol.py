"""The declared protocol targets 15 fps and forbids rate-derived shortcuts."""
from __future__ import annotations

from pathlib import Path

import pytest
import yaml

PROTOCOL_PATH = Path("configs/article1/dataset_protocol.yaml")


@pytest.fixture(scope="module")
def protocol():
    return yaml.safe_load(PROTOCOL_PATH.read_text())


def test_target_rate_is_declared_as_fifteen(protocol):
    assert protocol["dataset_protocol"]["target_dataset_rgb_fps"] == 15


def test_both_domains_are_in_the_target_protocol(protocol):
    assert set(protocol["dataset_protocol"]["target_domains"]) == {"car", "motorcycle"}


def test_current_car_is_marked_provisional(protocol):
    car = protocol["dataset_protocol"]["status_of_current_data"]["car"]
    assert car["final"] is False
    assert "provisional" in car["role"]


def test_current_motorcycle_is_at_the_target_rate(protocol):
    moto = protocol["dataset_protocol"]["status_of_current_data"]["motorcycle"]
    assert moto["measured_rgb_fps"] == pytest.approx(15.0, abs=0.01)
    assert moto["final"] is True


def test_declaring_the_target_does_not_rewrite_existing_data(protocol):
    """The declared target must not equal the measured value of the car."""
    car = protocol["dataset_protocol"]["status_of_current_data"]["car"]
    assert car["measured_rgb_fps"] == pytest.approx(10.0, abs=0.01)
    assert car["measured_rgb_fps"] != protocol["dataset_protocol"]["target_dataset_rgb_fps"]


def test_rate_is_never_a_feature(protocol):
    policy = protocol["dataset_protocol"]["rate_policy"]
    assert policy["forbid_rate_as_feature"] is True
    assert policy["measure_from_timestamps"] is True
    assert policy["trust_profile_name"] is False
    for banned in ("forbid_upsampling", "forbid_interpolation",
                   "forbid_synthetic_frames", "forbid_silent_duplication"):
        assert policy[banned] is True


def test_only_the_native_mode_may_support_a_claim(protocol):
    modes = protocol["analysis_modes"]
    assert modes["native"]["valid_for_scientific_claims"] is True
    assert modes["native"]["resampling"] == "none"
    assert modes["comparable_timeline"]["valid_for_scientific_claims"] is False
    assert modes["comparable_timeline"]["derived"] is True
    assert modes["comparable_timeline"]["preliminary"] is True


def test_comparable_timeline_forbids_inventing_frames(protocol):
    mode = protocol["analysis_modes"]["comparable_timeline"]
    assert mode["interpolation"] == "forbidden"
    assert mode["frame_creation"] == "forbidden"
    assert mode["upsampling"] == "forbidden"
    assert mode["record_temporal_error"] is True
    assert "silent" not in mode["duplicate_policy"] or \
        "never silent" in mode["duplicate_policy"]


def test_comparable_timeline_may_not_train_the_vehicle_classifier(protocol):
    forbidden = " ".join(protocol["analysis_modes"]["comparable_timeline"]
                         ["forbidden_uses"]).lower()
    assert "classifier" in forbidden
    assert "dataset format" in forbidden


def test_every_temporal_window_is_expressed_in_seconds(protocol):
    windows = protocol["temporal_windows_s"]
    assert windows
    for name, value in windows.items():
        assert isinstance(value, (int, float)), name
        # a window in seconds for driving footage is a small number; a value that
        # looks like a frame count would betray a unit mistake
        assert 0 < value <= 120, f"{name}={value} does not look like seconds"
