"""Multi-layer semantics + structured gaze resolution (Phase 3 / §18)."""
import numpy as np

from aria_drive_seg.segmentation.layers import (layer_of_name, resolve_gaze_target)


def test_layer_of_name(taxonomy):
    assert layer_of_name(taxonomy, "windshield") == "transparent"
    assert layer_of_name(taxonomy, "left_window") == "transparent"
    assert layer_of_name(taxonomy, "rear_view_mirror") == "mirror"
    assert layer_of_name(taxonomy, "left_side_mirror") == "mirror"
    assert layer_of_name(taxonomy, "car") == "exterior"
    assert layer_of_name(taxonomy, "pedestrian") == "exterior"
    assert layer_of_name(taxonomy, "dashboard") == "cockpit"
    assert layer_of_name(taxonomy, "speedometer_display") == "cockpit"


def _masks(taxonomy, h=20, w=20, **placements):
    """Build layer masks with a single class id painted at (10,10)."""
    out = {}
    for layer, name in placements.items():
        m = np.zeros((h, w), np.uint16)
        if name:
            m[10, 10] = taxonomy.id_of(name)
        out[layer] = m
    return out


def test_through_glass_prefers_exterior(taxonomy):
    lm = _masks(taxonomy, exterior="car", transparent="windshield")
    r = resolve_gaze_target(lm, 10, 10, taxonomy)
    assert r["primary_target"] == "car"
    assert r["resolution_reason"] == "through_glass"
    assert r["transparent_surface"] == "windshield"


def test_transparent_only_is_unknown_exterior(taxonomy):
    lm = _masks(taxonomy, exterior=None, transparent="windshield")
    r = resolve_gaze_target(lm, 10, 10, taxonomy)
    assert r["primary_target"] == "unknown_exterior"   # NOT "windshield"
    assert r["resolution_reason"] == "transparent_only_no_content"


def test_mirror_is_primary_with_optional_reflection(taxonomy):
    lm = _masks(taxonomy, mirror="rear_view_mirror", exterior="car")
    r = resolve_gaze_target(lm, 10, 10, taxonomy)
    assert r["primary_target"] == "rear_view_mirror"
    assert r["resolution_reason"] == "gaze_on_mirror"
    assert r["secondary_target"] == "car"              # reflected/behind content


def test_cockpit_object(taxonomy):
    lm = _masks(taxonomy, cockpit="steering_wheel")
    r = resolve_gaze_target(lm, 10, 10, taxonomy)
    assert r["primary_target"] == "steering_wheel"
    assert r["resolution_reason"] == "cockpit_object"


def test_nothing_at_gaze(taxonomy):
    lm = _masks(taxonomy, exterior=None, cockpit=None, transparent=None, mirror=None)
    r = resolve_gaze_target(lm, 10, 10, taxonomy)
    assert r["primary_target"] == "unknown"
    assert r["resolution_reason"] == "nothing_at_gaze"


def test_direct_exterior_no_glass(taxonomy):
    lm = _masks(taxonomy, exterior="person")
    r = resolve_gaze_target(lm, 10, 10, taxonomy)
    assert r["primary_target"] == "person"
    assert r["resolution_reason"] == "direct_exterior"
