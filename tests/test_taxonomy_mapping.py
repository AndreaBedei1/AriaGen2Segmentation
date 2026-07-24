"""Canonical taxonomy + Mapillary mapping (§6, §18)."""
import numpy as np
import yaml

from aria_drive_seg.segmentation.oneformer import MapillaryMapper, _norm


def test_taxonomy_has_unknown_zero(taxonomy):
    assert taxonomy.by_id[0].name == "unknown"
    assert 0 not in taxonomy.eval_ids()


def test_palette_shape_and_unknown_black(taxonomy):
    lut = taxonomy.palette()
    assert lut.shape[1] == 3
    assert tuple(lut[0]) == (0, 0, 0)


def test_colorize_roundtrip(taxonomy):
    m = np.array([[0, 1], [7, 24]], dtype=np.uint16)
    rgb = taxonomy.colorize(m)
    assert rgb.shape == (2, 2, 3)
    assert tuple(rgb[0, 1]) == taxonomy.by_id[1].color


def test_cockpit_classes_excluded_from_eval(taxonomy):
    for name in ["steering_wheel", "dashboard", "windshield", "driver_hand"]:
        assert not taxonomy.by_name[name].eval


def test_mapillary_mapping_covers_all_65(project_root, taxonomy):
    doc = yaml.safe_load((project_root / "configs" / "mapillary_to_canonical.yaml").read_text())
    # simulate a 65-class id2label like the model's
    labels = list(doc["mapping"].keys())
    id2label = {i: lbl for i, lbl in enumerate(labels)}
    mapper = MapillaryMapper(id2label, project_root / "configs" / "mapillary_to_canonical.yaml", taxonomy)
    assert mapper.unmapped == [], f"unmapped labels: {mapper.unmapped}"
    # every canonical target exists in taxonomy
    for nid in id2label:
        assert mapper.canonical_of[nid] in taxonomy.by_name


def test_ego_vehicle_maps_to_cockpit(project_root, taxonomy):
    doc = yaml.safe_load((project_root / "configs" / "mapillary_to_canonical.yaml").read_text())
    assert doc["mapping"]["Ego Vehicle"]["canonical"] == "other_cockpit"
    # ego vehicle must NOT be mapped to car
    assert doc["mapping"]["Ego Vehicle"]["canonical"] != "car"


def test_mapper_to_canonical_lut(project_root, taxonomy):
    id2label = {0: "Road", 1: "Car", 2: "Sky", 3: "Ego Vehicle"}
    mapper = MapillaryMapper(id2label, project_root / "configs" / "mapillary_to_canonical.yaml", taxonomy)
    native = np.array([[0, 1], [2, 3]], np.uint16)
    canon = mapper.to_canonical(native)
    assert canon[0, 0] == taxonomy.id_of("road_surface")
    assert canon[0, 1] == taxonomy.id_of("car")
    assert canon[1, 0] == taxonomy.id_of("sky")


def test_norm():
    assert _norm("Lane Marking - General") == "lane marking general"
    assert _norm("Traffic Sign (Front)") == "traffic sign front"
