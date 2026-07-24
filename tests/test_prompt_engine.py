"""Prompt engine: variants really sent, per-class thresholds, robust mapping,
geometric filters (Phase 1 / §18)."""
import pytest

from aria_drive_seg.segmentation.prompt_engine import (PhraseMapper, box_geom_ok,
                                                      build_specs, group_captions,
                                                      normalize, singularize)


def _doc():
    return {
        "defaults": {"box_threshold": 0.35, "candidate_box_threshold": 0.15,
                     "min_map_confidence": 0.3, "min_area": 400, "prompt_mode": "combined"},
        "classes": {
            "traffic_light": {"prompt": "traffic light",
                              "prompt_variants": ["traffic signal", "road traffic light"],
                              "synonyms": ["stop light"], "priority": 75,
                              "box_threshold": 0.25, "max_instances": 12},
            "traffic_sign": {"prompt": "traffic sign", "prompt_variants": ["road sign"],
                             "negative_contexts": ["billboard"], "priority": 72},
            "car": {"prompt": "car", "prompt_variants": ["automobile", "sedan"],
                    "priority": 40, "max_area_frac": 0.3,
                    "min_aspect_ratio": 0.3, "max_aspect_ratio": 4.0,
                    "allowed_regions": [[0.0, 0.0, 1.0, 0.8]]},
        },
    }


@pytest.fixture
def specs():
    return build_specs(_doc(), {"traffic_light", "traffic_sign", "car"})


def test_variants_and_synonyms_really_in_caption(specs):
    caps = group_captions(specs, ["traffic_light"])
    caption = " ".join(c for c, _ in caps)
    for phrase in ["traffic light", "traffic signal", "road traffic light", "stop light"]:
        assert phrase in caption, f"{phrase!r} missing from caption (bug: variants not sent)"


def test_phrase_dedup(specs):
    doc = _doc()
    doc["classes"]["car"]["prompt_variants"] = ["automobile", "automobile", "car"]
    s = build_specs(doc, {"car"})["car"]
    # 'car' duplicates the prompt; 'automobile' duplicated -> deduped
    assert s.all_phrases.count("automobile") == 1
    keys = s.phrase_keys
    assert len(keys) == len(set(keys))


def test_per_class_thresholds_read(specs):
    assert specs["traffic_light"].box_threshold == 0.25
    assert specs["car"].box_threshold == 0.35  # from defaults
    assert specs["traffic_light"].candidate_box_threshold == 0.15
    assert specs["traffic_light"].max_instances == 12


def test_mapping_exact_and_synonym(specs):
    m = PhraseMapper(specs)
    assert m.map("traffic light", ["traffic_light", "traffic_sign"]).name == "traffic_light"
    assert m.map("stop light", ["traffic_light"]).name == "traffic_light"       # synonym
    assert m.map("automobile", ["car"]).name == "car"


def test_mapping_singular_plural(specs):
    m = PhraseMapper(specs)
    assert m.map("cars", ["car"]).name == "car"
    assert m.map("road signs", ["traffic_sign"]).name == "traffic_sign"


def test_mapping_ambiguity_rejected():
    doc = {"defaults": {}, "classes": {
        "traffic_light": {"prompt": "traffic light"},
        "traffic_sign": {"prompt": "traffic light"}}}  # identical -> ambiguous
    specs = build_specs(doc, {"traffic_light", "traffic_sign"})
    m = PhraseMapper(specs)
    r = m.map("traffic light", ["traffic_light", "traffic_sign"])
    # exact map picks the deterministic first; but a partial 'traffic' is ambiguous
    r2 = m.map("traffic", ["traffic_light", "traffic_sign"])
    assert r2.name is None and "ambiguous" in r2.reason


def test_negative_context_rejected(specs):
    m = PhraseMapper(specs, negatives=["billboard"])
    r = m.map("billboard", ["traffic_sign"])
    assert r.name is None and r.reason == "negative_context"


def test_geom_aspect_ratio_and_region():
    specs = build_specs(_doc(), {"car"})
    car = specs["car"]
    # a very tall thin box fails max/min aspect ratio
    ok, reason = box_geom_ok(car, (0, 0, 10, 500), 5000, 1000, 1000)
    assert not ok and "aspect_ratio" in reason
    # a box centred in the bottom 20% is outside the allowed region (top 80%)
    ok, reason = box_geom_ok(car, (100, 900, 300, 980), 16000, 1000, 1000)
    assert not ok and reason == "outside_allowed_region"
    # a normal car box passes
    ok, reason = box_geom_ok(car, (100, 100, 300, 250), 30000, 1000, 1000)
    assert ok and reason == "ok"


def test_geom_min_area_and_box_dims():
    doc = _doc()
    doc["classes"]["car"]["min_box_width"] = 20
    car = build_specs(doc, {"car"})["car"]
    ok, reason = box_geom_ok(car, (0, 0, 5, 100), 300, 1000, 1000)
    assert not ok and reason in ("below_min_area", "below_min_box_width")


def test_normalize_singularize():
    assert normalize("Traffic Sign (Front)!") == "traffic sign front"
    assert singularize("buses") == "buse" or singularize("buses") == "bus"
    assert singularize("cars") == "car"
