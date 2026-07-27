from pathlib import Path

import numpy as np
import pytest
import yaml

from aria_drive_seg.cli import _load_cfg, build_parser
from aria_drive_seg.article1.external import (
    Article1Mapper,
    apply_article1_policy,
    aggregate_native_probabilities,
    filter_thin_markings,
    preserve_thin_markings,
)
from aria_drive_seg.taxonomy import Taxonomy


ROOT = Path(__file__).resolve().parents[1]


def test_standard_article1_commands_default_to_static_v2_configs():
    parser = build_parser()
    common = ["--input", "unused", "--vehicle-type", "car",
              "--session-id", "s", "--participant-id", "p"]
    direct = _load_cfg(parser.parse_args(["article1", "segment-external", *common]))
    replay = _load_cfg(parser.parse_args(["article1", "reprocess-external", *common]))
    assert direct.get("article1.output_subdir") == "article1_external"
    assert direct.get("article1.unknown.enabled") is True
    assert direct.get("article1.thin_markings.accepted_component_policy") == \
        "supported_pixels_only"
    assert replay.get("article1.output_subdir") == "article1_external_checkpoint2"


def test_probabilistic_vehicle_and_two_wheeler_aggregation():
    # car+truck -> vehicle, bicycle+motorcycle -> two_wheeler
    native = np.array([[[.20]], [[.15]], [[.10]], [[.05]], [[.50]]], np.float32)
    lut = np.array([4, 4, 5, 5, 13])
    result = aggregate_native_probabilities(native, lut, 14, confidence_threshold=.1)
    p = result["probabilities"][:, 0, 0]
    assert np.isclose(p.sum(), 1)
    assert np.isclose(p[4], .35)
    assert np.isclose(p[5], .15)
    assert result["mask"][0, 0] == 13


def test_unknown_is_not_other_environment():
    native = np.full((4, 1, 1), .25, np.float32)
    lut = np.array([1, 4, 8, 13])
    result = aggregate_native_probabilities(native, lut, 14, confidence_threshold=.30)
    assert result["mask"][0, 0] == 0
    assert result["probabilities"][0, 0, 0] == 1
    assert result["probabilities"][13, 0, 0] == 0


def test_mapping_covers_mapillary_labels_and_key_macros():
    tax = Taxonomy.load(ROOT / "configs/article1/classes_article1.yaml")
    labels = {
        0: "Car", 1: "Truck", 2: "Bus", 3: "Bicycle", 4: "Motorcycle",
        5: "Bicyclist", 6: "Crosswalk - Plain", 7: "Lane Marking - General",
        8: "Building",
    }
    mapper = Article1Mapper(labels, ROOT / "configs/article1/mapillary_to_article1.yaml", tax)
    assert not mapper.unmapped
    assert mapper.native_to_article[0] == tax.id_of("vehicle")
    assert mapper.native_to_article[3] == tax.id_of("two_wheeler")
    assert mapper.native_to_article[6] == tax.id_of("regulatory_road_marking")
    assert mapper.native_to_article[8] == tax.id_of("other_environment")


def test_mapping_declares_all_65_mapillary_eval_classes():
    doc = yaml.safe_load((ROOT / "configs/article1/mapillary_to_article1.yaml").read_text())
    assert len(doc["mapping"]) == 65


def test_mapillary_ego_is_environment_with_preserved_flag():
    tax = Taxonomy.load(ROOT / "configs/article1/classes_article1.yaml")
    mapper = Article1Mapper(
        {0: "Ego Vehicle", 1: "Car Mount"},
        ROOT / "configs/article1/mapillary_to_article1.yaml", tax)
    assert set(mapper.native_to_article) == {tax.id_of("other_environment")}
    assert mapper.mapillary_ego_ids == {0, 1}
    assert all(mapper.attributes[i]["mapillary_ego_region"] for i in (0, 1))
    assert tax.id_of("unknown") != tax.id_of("other_environment")


def test_thin_markings_override_road_without_aggressive_merge():
    base = np.ones((12, 12), np.uint16)
    probs = np.zeros((14, 12, 12), np.float32)
    probs[1] = 1
    probs[2, 2:10, 3] = .8
    probs[2, 2:10, 8] = .8
    probs[3, 6, 2:10] = .9
    thin = preserve_thin_markings(base, probs, lane_threshold=.5,
                                  regulatory_threshold=.5,
                                  min_component_area=2, max_gap=1)
    assert thin["composite"][3, 3] == 2
    assert thin["composite"][6, 5] == 3
    assert thin["composite"][3, 5] == 1
    assert not thin["lane_mask"][:, 4:8].all()


def test_unknown_reason_low_probability_margin_and_entropy():
    native = np.array([[[.26]], [[.25]], [[.25]], [[.24]]], np.float32)
    lut = np.array([1, 4, 8, 13])
    policy = {"enabled": True, "min_top1_probability": .45,
              "min_top1_top2_margin": .08, "max_normalized_entropy": .50,
              "unsupported_native_to_unknown": True, "combine_rule": "any"}
    out = aggregate_native_probabilities(native, lut, 14, unknown_policy=policy)
    reason = int(out["unknown_reason"][0, 0])
    assert reason & 1  # low probability
    assert reason & 2  # low margin
    assert reason & 4  # high normalized entropy
    assert out["mask"][0, 0] == 0


def test_unsupported_dominant_native_reason():
    native = np.array([[[.8]], [[.2]]], np.float32)
    out = aggregate_native_probabilities(
        native, np.array([0, 4]), 14, unknown_policy={
            "enabled": True, "min_top1_probability": 0,
            "min_top1_top2_margin": 0, "max_normalized_entropy": 1,
            "unsupported_native_to_unknown": True, "combine_rule": "any"})
    assert int(out["unknown_reason"][0, 0]) & 8
    assert out["mask"][0, 0] == 0


def test_thin_filter_requires_road_support_and_preserves_raw():
    base = np.full((20, 20), 13, np.uint16)
    base[10:, :] = 1
    p = np.zeros((14, 20, 20), np.float32)
    p[13] = 1
    p[2, 2:8, 4] = .9       # high probability, outside road
    p[2, 12:19, 10] = .9    # supported by road
    p[1, 10:, :] = .8
    result = filter_thin_markings(base, p, {
        "lane_threshold": .4, "regulatory_threshold": .45,
        "candidate_probability_floor": .1, "road_support_dilation_px": 1,
        "min_component_area_px": 3, "max_component_area_frac": .5,
        "min_top1_top2_margin": 0, "suppress_on_nonroad_classes": False})
    assert result["raw_thin_mask"][3, 4] == 2
    assert result["filtered_thin_mask"][3, 4] == 0
    assert result["reason_map"][3, 4] == 3
    assert result["filtered_thin_mask"][15, 10] == 2


def _thin_cfg(**updates):
    cfg = {
        "lane_threshold": .4, "regulatory_threshold": .4,
        "candidate_probability_floor": .1,
        "road_probability_threshold": .25,
        "use_road_probability_support": True, "use_base_mask_support": False,
        "road_support_dilation_px": 0, "nonroad_conflict_threshold": .55,
        "min_road_overlap": .60,
        "accepted_component_policy": "supported_pixels_only",
        "min_component_area_px": 1, "max_component_area_frac": 1,
        "min_top1_top2_margin": 0, "suppress_on_nonroad_classes": True,
        "morphology_open_kernel": 1, "morphology_close_kernel": 1,
    }
    cfg.update(updates)
    return cfg


@pytest.mark.parametrize("class_id", [2, 3])
@pytest.mark.parametrize("overlap,accepted", [
    (0, False), (.30, False), (.59, False), (.60, True), (.90, True), (1, True),
])
def test_real_component_road_overlap_threshold(class_id, overlap, accepted):
    base = np.full((5, 100), 13, np.uint16)
    p = np.zeros((14, 5, 100), np.float32)
    p[class_id, 2, :] = .9
    supported = int(round(overlap * 100))
    p[1, 2, :supported] = .3
    result = filter_thin_markings(base, p, _thin_cfg())
    kept = int((result["filtered_thin_mask"] == class_id).sum())
    assert kept == (supported if accepted else 0)
    if not accepted:
        assert np.all(result["reason_map"][2, :] == 3)


def test_full_component_policy_keeps_unsupported_tail_once_supported():
    base = np.full((3, 10), 13, np.uint16)
    p = np.zeros((14, 3, 10), np.float32)
    p[2, 1, :] = .9
    p[1, 1, :6] = .3
    out = filter_thin_markings(
        base, p, _thin_cfg(accepted_component_policy="full_component_if_supported",
                           suppress_on_nonroad_classes=False))
    assert np.all(out["filtered_thin_mask"][1] == 2)


def test_large_component_is_rejected_before_road_clipping():
    base = np.ones((20, 20), np.uint16)
    p = np.zeros((14, 20, 20), np.float32)
    p[1] = .4
    p[2, 2:18, 2:18] = .9
    out = filter_thin_markings(
        base, p, _thin_cfg(max_component_area_frac=.10))
    assert not out["filtered_thin_mask"].any()
    assert (out["reason_map"] == 6).sum() == 16 * 16


@pytest.mark.parametrize("blocking_id", [4, 10])
def test_vehicle_or_cockpit_conflict_counts_against_overlap(blocking_id):
    base = np.full((3, 10), 13, np.uint16)
    p = np.zeros((14, 3, 10), np.float32)
    p[2, 1, :] = .9
    p[1, 1, :] = .3
    p[blocking_id, 1, :5] = .8
    out = filter_thin_markings(base, p, _thin_cfg())
    assert not out["filtered_thin_mask"].any()
    assert np.all(out["reason_map"][1] == 3)


def test_component_near_road_edge_keeps_only_supported_pixels():
    base = np.full((3, 10), 13, np.uint16)
    p = np.zeros((14, 3, 10), np.float32)
    p[3, 1, :] = .9
    p[1, 1, :9] = .3
    out = filter_thin_markings(base, p, _thin_cfg())
    assert np.all(out["filtered_thin_mask"][1, :9] == 3)
    assert out["filtered_thin_mask"][1, 9] == 0
    assert out["reason_map"][1, 9] == 3


def test_road_support_combines_probability_base_dilation_and_nonroad_exclusion():
    base = np.full((9, 9), 13, np.uint16)
    base[4, 2] = 1
    p = np.zeros((14, 9, 9), np.float32)
    p[1, 4, 6] = .3
    p[4, 4, 6] = .8
    out = filter_thin_markings(
        base, p, _thin_cfg(road_support_dilation_px=1,
                           use_base_mask_support=True))
    assert out["road_support_raw"][4, 2]       # base-mask support
    assert out["road_support_raw"][4, 6]       # probability support
    assert out["road_support_dilated"][4, 3]   # controlled dilation
    assert not out["road_support_final"][4, 6] # confident vehicle exclusion


def test_direct_and_saved_reprocess_policy_are_equivalent(tmp_path):
    tax = Taxonomy.load(ROOT / "configs/article1/classes_article1.yaml")
    mapper = Article1Mapper(
        {0: "Road", 1: "Lane Marking - General", 2: "Car", 3: "Building"},
        ROOT / "configs/article1/mapillary_to_article1.yaml", tax)
    rng = np.random.default_rng(2026)
    native = rng.random((4, 16, 18), dtype=np.float32)
    cfg = {
        "probability_dtype": "float16",
        "unknown": {"enabled": True, "min_top1_probability": .45,
                    "min_top1_top2_margin": .08,
                    "max_normalized_entropy": .75,
                    "unsupported_native_to_unknown": True, "combine_rule": "any"},
        "thin_markings": _thin_cfg(min_component_area_px=2),
    }
    direct = apply_article1_policy(native, mapper, cfg)
    saved = tmp_path / "native.npz"
    np.savez_compressed(saved, probabilities=native.astype(np.float16))
    reprocessed = apply_article1_policy(
        np.load(saved)["probabilities"].astype(np.float32), mapper, cfg)
    for key in ("mask", "unknown_reason"):
        assert np.array_equal(direct["aggregate"][key],
                              reprocessed["aggregate"][key])
    for key in ("raw_thin_mask", "filtered_thin_mask", "composite", "reason_map"):
        assert np.array_equal(direct["thin"][key], reprocessed["thin"][key])
    assert np.allclose(direct["aggregate"]["probabilities"],
                       reprocessed["aggregate"]["probabilities"],
                       atol=np.finfo(np.float16).eps)
