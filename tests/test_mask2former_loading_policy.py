from aria_drive_seg.segmentation.oneformer import ALLOWED_MASK2FORMER_MISSING


def test_only_diagnosed_unused_layernorm_is_allowed_missing():
    assert ALLOWED_MASK2FORMER_MISSING == {
        "model.pixel_level_module.encoder.swin.layernorm.weight",
        "model.pixel_level_module.encoder.swin.layernorm.bias",
    }
