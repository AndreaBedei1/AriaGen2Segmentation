"""Field of view is preserved: no crop, no aspect deformation, exact inverse."""
from __future__ import annotations

import numpy as np
import pytest

from aria_drive_seg.geometry.audit import aspect, detect_centre_crop, edge_losses
from aria_drive_seg.geometry.fov import (LetterboxTransform, build_letterbox,
                                         probe_points, round_trip_error,
                                         round_trip_report, vignette_mask)

SOURCE = (2016, 1512)          # the real Aria RGB geometry


# --------------------------------------------------------------------------- #
# no crop, no deformation
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("target", [(384, 384), (800, 800), (1333, 1000),
                                    (1024, 768), (640, 640), (512, 512)])
def test_letterbox_never_crops(target):
    t = build_letterbox(SOURCE, target)
    # the whole source fits inside the content area
    assert t.content_width <= target[0] and t.content_height <= target[1]
    assert t.content_width == pytest.approx(SOURCE[0] * t.scale, abs=1)
    assert t.content_height == pytest.approx(SOURCE[1] * t.scale, abs=1)
    assert not t.to_dict()["crop"]


@pytest.mark.parametrize("target", [(384, 384), (1333, 1000), (640, 480)])
def test_letterbox_preserves_aspect_ratio(target):
    t = build_letterbox(SOURCE, target)
    source_aspect = aspect(*SOURCE)
    content_aspect = t.content_width / t.content_height
    assert content_aspect == pytest.approx(source_aspect, rel=2e-3)


def test_letterbox_uses_one_scale_for_both_axes():
    """A single scale is what makes deformation impossible by construction."""
    t = build_letterbox(SOURCE, (384, 384))
    assert isinstance(t.scale, float)
    # a hypothetical anisotropic fit would need different factors
    assert t.content_width / SOURCE[0] == pytest.approx(
        t.content_height / SOURCE[1], rel=2e-3)


def test_padding_accounts_for_every_target_pixel():
    for target in [(384, 384), (1333, 1000), (500, 377)]:
        t = build_letterbox(SOURCE, target)
        assert t.pad_left + t.content_width + t.pad_right == target[0]
        assert t.pad_top + t.content_height + t.pad_bottom == target[1]


def test_square_target_pads_vertically_for_a_wide_source():
    t = build_letterbox(SOURCE, (384, 384))
    assert t.pad_top > 0 and t.pad_bottom > 0
    assert t.pad_left == 0 and t.pad_right == 0


def test_invalid_sizes_are_rejected():
    for bad in [(0, 100), (100, 0), (-1, 10)]:
        with pytest.raises(ValueError):
            build_letterbox(bad, (384, 384))
        with pytest.raises(ValueError):
            build_letterbox(SOURCE, bad)


# --------------------------------------------------------------------------- #
# reversibility
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("target", [(384, 384), (1333, 1000), (1024, 768)])
def test_coordinate_round_trip_is_exact(target):
    t = build_letterbox(SOURCE, target)
    report = round_trip_report(t, *SOURCE)
    assert report["median_error_px"] < 0.5
    assert report["p99_error_px"] < 1.0
    assert report["passes"]


def test_round_trip_is_exact_at_the_edges_and_corners():
    t = build_letterbox(SOURCE, (384, 384))
    for name, pts in probe_points(*SOURCE).items():
        assert round_trip_error(t, pts).max() < 1e-6, name


def test_mirror_band_points_survive_the_round_trip():
    """The lateral bands are where the motorcycle mirrors are."""
    t = build_letterbox(SOURCE, (384, 384))
    points = probe_points(*SOURCE)
    for band in ("mirror_band_left", "mirror_band_right"):
        assert round_trip_error(t, points[band]).max() < 1e-6


def test_mask_round_trip_preserves_labels_and_geometry():
    t = build_letterbox(SOURCE, (384, 384))
    mask = np.zeros((SOURCE[1], SOURCE[0]), np.uint16)
    mask[700:900, 60:260] = 10        # a left mirror
    mask[700:900, 1760:1960] = 10     # a right mirror
    mask[1300:, :] = 12               # a cockpit strip

    boxed = t.apply(mask, nearest=True)
    assert boxed.shape == (384, 384)
    back = t.invert_mask(boxed)

    assert back.shape == mask.shape
    # both mirrors survive back to the source geometry
    assert (back[700:900, 60:260] == 10).mean() > 0.8
    assert (back[700:900, 1760:1960] == 10).mean() > 0.8
    # nearest-neighbour resampling must not invent an intermediate label
    assert set(np.unique(back)) <= {0, 10, 12}


def test_rgb_round_trip_keeps_three_channels():
    t = build_letterbox(SOURCE, (384, 384))
    rgb = np.random.default_rng(0).integers(
        0, 255, (SOURCE[1], SOURCE[0], 3), dtype=np.uint8)
    boxed = t.apply(rgb)
    assert boxed.shape == (384, 384, 3)


def test_edge_content_is_not_silently_dropped():
    t = build_letterbox(SOURCE, (384, 384))
    mask = np.zeros((SOURCE[1], SOURCE[0]), np.uint8)
    mask[:, :8] = 1        # the very first columns
    mask[:, -8:] = 1       # the very last columns
    back = t.invert_mask(t.apply(mask, nearest=True))
    assert back[:, :8].any(), "left edge content vanished"
    assert back[:, -8:].any(), "right edge content vanished"


# --------------------------------------------------------------------------- #
# valid region bookkeeping
# --------------------------------------------------------------------------- #
def test_vignette_mask_marks_optically_black_pixels():
    image = np.full((100, 100, 3), 200, np.uint8)
    image[:10, :] = 0
    valid = vignette_mask(image)
    assert not valid[:10].any()
    assert valid[20:].all()


def test_edge_losses_report_zero_when_nothing_is_lost():
    valid = np.zeros((100, 100), bool)
    valid[10:90, 10:90] = True
    losses = edge_losses(valid, valid.copy())
    for side in ("left", "right", "top", "bottom"):
        assert losses[f"pixels_lost_{side}"] == 0


def test_edge_losses_measure_a_real_shrink():
    valid = np.zeros((100, 100), bool)
    valid[10:90, 10:90] = True
    retained = np.zeros_like(valid)
    retained[20:80, 25:75] = True
    losses = edge_losses(valid, retained)
    assert losses["pixels_lost_left"] == 15
    assert losses["pixels_lost_right"] == 15
    assert losses["pixels_lost_top"] == 10
    assert losses["pixels_lost_bottom"] == 10


# --------------------------------------------------------------------------- #
# crop detection
# --------------------------------------------------------------------------- #
def test_centre_crop_to_widescreen_is_detected():
    result = detect_centre_crop((2016, 1512), (2016, 1134),
                                crop_box=[0, 189, 2016, 1323])
    assert result["aspect_changed"]
    assert result["widescreen_conversion"]
    assert result["is_centre_crop"]


def test_a_same_aspect_resize_is_not_a_crop():
    result = detect_centre_crop((2016, 1512), (1008, 756), crop_box=None)
    assert not result["aspect_changed"]
    assert not result["is_centre_crop"]
    assert not result["widescreen_conversion"]


# --------------------------------------------------------------------------- #
# the transform is domain agnostic
# --------------------------------------------------------------------------- #
def test_letterbox_has_no_domain_input():
    import inspect
    signature = inspect.signature(build_letterbox)
    assert list(signature.parameters) == ["source_size", "target_size"]
    source = inspect.getsource(LetterboxTransform)
    for banned in ("motorcycle", "car", "domain", "vehicle_type", "fps"):
        assert banned not in source.lower()
