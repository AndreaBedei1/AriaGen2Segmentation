"""Hierarchical parent->subclass decision (Phase 3 fix / §3). The user's required cases:
a normal car must not become emergency_vehicle on low confidence; an adult must not
become child on a marginal difference; rejected/ambiguous subclasses keep the parent."""
import yaml

from aria_drive_seg.segmentation.hierarchy import decide_subclass, load_hierarchy


def test_no_candidates():
    name, status, conf = decide_subclass({}, 0.35, 0.08)
    assert name is None and status == "no_candidates"


def test_clear_winner_accepted():
    name, status, _ = decide_subclass({"van": 0.6, "pickup": 0.2}, 0.35, 0.08)
    assert name == "van" and status == "accepted"


def test_normal_car_not_emergency_on_low_confidence():
    # all subtype scores below min_score -> keep the parent (car), not emergency_vehicle
    name, status, _ = decide_subclass({"emergency_vehicle": 0.25, "van": 0.2}, 0.35, 0.08)
    assert name is None and status == "rejected"


def test_adult_not_child_on_marginal_difference():
    # child barely beats pedestrian -> ambiguous -> keep parent (person)
    name, status, _ = decide_subclass({"child": 0.55, "pedestrian": 0.50}, 0.35, 0.08)
    assert name is None and status == "ambiguous"


def test_child_accepted_when_clear():
    name, status, _ = decide_subclass({"child": 0.7, "pedestrian": 0.4}, 0.35, 0.08)
    assert name == "child" and status == "accepted"


def test_single_candidate_above_threshold_accepted():
    name, status, _ = decide_subclass({"cyclist": 0.6}, 0.35, 0.08)
    assert name == "cyclist" and status == "accepted"


def test_load_hierarchy(project_root):
    doc = yaml.safe_load((project_root / "configs" / "grounded_prompts.yaml").read_text())
    h = load_hierarchy(doc)
    assert "person" in h["parents"] and "car" in h["parents"]
    assert "child" in h["subclasses"] and "emergency_vehicle" in h["subclasses"]
    # every hierarchy subclass exists in the taxonomy classes config
    classes = yaml.safe_load((project_root / "configs" / "classes.yaml").read_text())["classes"]
    names = {c["name"] for c in classes}
    for s in h["subclasses"]:
        assert s in names, f"hierarchy subclass {s} not in taxonomy"
    assert 0.0 < h["min_score"] < 1.0 and h["min_margin"] >= 0
