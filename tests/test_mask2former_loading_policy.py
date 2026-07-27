import pytest

from aria_drive_seg.segmentation.oneformer import (
    ALLOWED_MASK2FORMER_MISSING,
    ALLOWED_MASK2FORMER_UNEXPECTED_PATTERNS,
    validate_mask2former_loading_info,
)


def test_only_diagnosed_unused_layernorm_is_allowed_missing():
    assert ALLOWED_MASK2FORMER_MISSING == {
        "model.pixel_level_module.encoder.swin.layernorm.weight",
        "model.pixel_level_module.encoder.swin.layernorm.bias",
    }


def loading_info(**updates):
    info = {
        "missing_keys": list(ALLOWED_MASK2FORMER_MISSING),
        "unexpected_keys": [],
        "mismatched_keys": [],
        "error_msgs": [],
    }
    info.update(updates)
    return info


def test_legacy_relative_position_buffer_is_allowed():
    key = ("model.pixel_level_module.encoder.swin.encoder.layers.2.blocks.17"
           ".attention.self.relative_position_index")
    report = validate_mask2former_loading_info(
        loading_info(unexpected_keys=[key]))
    assert report["gate_passed"]
    assert report["unexpected_keys"] == [key]
    assert report["allowed_unexpected_patterns"] == list(
        ALLOWED_MASK2FORMER_UNEXPECTED_PATTERNS)


@pytest.mark.parametrize("updates,match", [
    ({"missing_keys": list(ALLOWED_MASK2FORMER_MISSING) + ["decoder.weight"]},
     "missing keys"),
    ({"unexpected_keys": ["class_predictor.weight"]}, "unexpected keys"),
    ({"mismatched_keys": [("classifier.weight", (2, 2), (3, 2))]},
     "mismatched shapes"),
    ({"error_msgs": ["checkpoint read failed"]}, "loader errors"),
])
def test_loading_gate_fails_closed(updates, match):
    with pytest.raises(RuntimeError, match=match):
        validate_mask2former_loading_info(loading_info(**updates))
