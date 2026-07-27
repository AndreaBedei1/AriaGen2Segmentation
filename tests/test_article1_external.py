from pathlib import Path

import numpy as np
import yaml

from aria_drive_seg.article1.external import (
    Article1Mapper,
    aggregate_native_probabilities,
    preserve_thin_markings,
)
from aria_drive_seg.taxonomy import Taxonomy


ROOT = Path(__file__).resolve().parents[1]


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
