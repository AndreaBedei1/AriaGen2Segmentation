"""Extended taxonomy (Phase 2): backward compatibility, parent/rollup, prompt<->taxonomy
consistency (§6, §18)."""
import yaml

from aria_drive_seg.taxonomy import Taxonomy, auto_color


# frozen snapshot of the original 0-39 ids -> names (must never change meaning)
_BASELINE = {
    0: "unknown", 1: "road_surface", 2: "lane_marking", 7: "person", 8: "rider",
    9: "car", 14: "traffic_light", 15: "traffic_sign", 24: "sky", 28: "steering_wheel",
    34: "windshield", 36: "driver_hand", 39: "unknown_cockpit",
}


def test_backward_compatible_ids(taxonomy):
    for cid, name in _BASELINE.items():
        assert taxonomy.by_id[cid].name == name, f"id {cid} changed meaning!"


def test_extended_taxonomy_size(taxonomy):
    assert len(taxonomy.classes) >= 90
    # ids are unique and contiguous-ish
    ids = [c.id for c in taxonomy.classes]
    assert len(ids) == len(set(ids))


def test_parents_are_real_classes(taxonomy):
    for c in taxonomy.classes:
        if c.parent is not None:
            assert c.parent in taxonomy.by_name, f"{c.name} has unknown parent {c.parent}"


def test_rollup_to_generic(taxonomy):
    assert taxonomy.rollup_name("pedestrian") == "person"
    assert taxonomy.rollup_name("cyclist") == "rider"
    assert taxonomy.rollup_name("stop_sign") == "traffic_sign"
    assert taxonomy.rollup_name("left_side_mirror") == "side_mirror"
    assert taxonomy.rollup_name("speedometer_display") == "instrument_cluster"
    assert taxonomy.rollup_name("van") == "car"
    # a generic class rolls up to itself
    assert taxonomy.rollup_name("car") == "car"
    # rollup_id agrees
    assert taxonomy.rollup_id(taxonomy.id_of("pedestrian")) == taxonomy.id_of("person")


def test_no_rollup_cycles(taxonomy):
    for c in taxonomy.classes:
        # rollup terminates (no infinite loop) and lands on a parentless class
        top = taxonomy.rollup_name(c.name)
        assert taxonomy.by_name[top].parent is None


def test_subclasses_of(taxonomy):
    subs = set(taxonomy.subclasses_of("traffic_sign"))
    assert {"stop_sign", "yield_sign", "speed_limit_sign"} <= subs


def test_auto_color_deterministic_and_in_range():
    a = auto_color(55)
    assert a == auto_color(55)                 # deterministic
    assert all(0 <= ch <= 255 for ch in a)
    assert len(a) == 3


def test_palette_covers_all_ids(taxonomy):
    lut = taxonomy.palette()
    assert lut.shape[0] == taxonomy.max_id + 1
    # extended classes get a non-black auto colour
    assert tuple(lut[taxonomy.id_of("pedestrian")]) != (0, 0, 0)


def test_every_prompt_class_in_taxonomy(project_root, taxonomy):
    doc = yaml.safe_load((project_root / "configs" / "grounded_prompts.yaml").read_text())
    for name in doc["classes"]:
        assert name in taxonomy.by_name, f"prompt class {name!r} missing from taxonomy"
