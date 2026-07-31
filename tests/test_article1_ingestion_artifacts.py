"""Guarantees about the committed ingestion artefacts themselves.

These assert on the real outputs of the ingestion run, so a regression that only
shows up in the produced files (a missing checksum, a pre-annotation presented as
ground truth, a mutated source recording) is caught rather than argued about.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

REPORTS = Path("reports/article1_motorcycle_ingestion")
PACKAGE = Path("datasets/article1_annotation_package")
MANIFEST = REPORTS / "acquisition_manifest.json"

requires_manifest = pytest.mark.skipif(
    not MANIFEST.exists(), reason="the ingestion run has not been executed here")
requires_package = pytest.mark.skipif(
    not (PACKAGE / "manifest.json").exists(),
    reason="the annotation package has not been built here")


@pytest.fixture(scope="module")
def manifest():
    return json.loads(MANIFEST.read_text())


@pytest.fixture(scope="module")
def package():
    return json.loads((PACKAGE / "manifest.json").read_text())


# --------------------------------------------------------------------------- #
# manifest carries checksums and provenance
# --------------------------------------------------------------------------- #
@requires_manifest
def test_every_recording_has_a_sha256(manifest):
    recordings = [f for f in manifest["files"] if f["kind"] == "vrs_recording"]
    assert recordings
    for f in recordings:
        assert re.fullmatch(r"[0-9a-f]{64}", f["sha256"] or ""), f["relative_path"]


@requires_manifest
def test_recording_identity_is_derived_from_content(manifest):
    for f in manifest["files"]:
        if f["kind"] == "vrs_recording" and f.get("recording_id"):
            assert f["sha256"][:12] in f["recording_id"]


@requires_manifest
def test_manifest_records_what_domain_resolution_may_not_use(manifest):
    forbidden = manifest["domain_resolution"]["explicitly_not_used"]
    joined = " ".join(forbidden).lower()
    assert "frame rate" in joined
    assert "file name" in joined
    assert "profile name" in joined


@requires_manifest
def test_both_domains_were_resolved(manifest):
    selection = manifest["selection"]
    assert selection["car"]["selected_recording_id"]
    assert selection["motorcycle"]["selected_recording_id"]
    assert (selection["car"]["selected_recording_id"]
            != selection["motorcycle"]["selected_recording_id"])


@requires_manifest
def test_domain_decision_carries_its_evidence(manifest):
    for f in manifest["files"]:
        if f["kind"] != "vrs_recording" or not f.get("recording_id"):
            continue
        assert f["domain_rationale"]
        evidence = f["domain_evidence"]
        for key in ("top_band_static_fraction", "top_band_luminance",
                    "border_static_fraction", "sampled_frames"):
            assert key in evidence
        # the evidence must not smuggle in a rate
        for banned in ("fps", "rate", "frame_count", "duration"):
            assert not any(banned in k for k in evidence)


# --------------------------------------------------------------------------- #
# the source recordings were not modified
# --------------------------------------------------------------------------- #
@requires_manifest
def test_source_recordings_are_unmodified(manifest):
    from aria_drive_seg.hashing import sha256_file

    checked = 0
    for f in manifest["files"]:
        if f["kind"] != "vrs_recording":
            continue
        path = Path(f["absolute_path"])
        if not path.exists():
            continue
        assert sha256_file(path) == f["sha256"], (
            f"{path} no longer matches the hash recorded at ingestion time")
        checked += 1
    if checked == 0:
        pytest.skip("no source recording available on this machine")


# --------------------------------------------------------------------------- #
# stream QA measured the rate rather than trusting the profile
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("tag,expected", [("auto", 10.0), ("moto", 15.0)])
def test_measured_rate_is_recorded_next_to_the_nominal_one(tag, expected):
    path = REPORTS / f"stream_qa_{tag}.json"
    if not path.exists():
        pytest.skip("stream QA not produced here")
    qa = json.loads(path.read_text())
    rgb = qa["streams"]["camera-rgb"]
    assert rgb["nominal_rate_hz"] is not None
    assert rgb["rate"]["effective_fps"] == pytest.approx(expected, rel=0.01)
    # both recordings share a profile name yet run at different rates
    assert qa["metadata"]["recording_profile"]


def test_the_two_recordings_share_a_profile_name_but_not_a_rate():
    auto_path, moto_path = (REPORTS / "stream_qa_auto.json",
                            REPORTS / "stream_qa_moto.json")
    if not (auto_path.exists() and moto_path.exists()):
        pytest.skip("stream QA not produced here")
    auto = json.loads(auto_path.read_text())
    moto = json.loads(moto_path.read_text())
    assert (auto["metadata"]["recording_profile"]
            == moto["metadata"]["recording_profile"])
    assert (auto["streams"]["camera-rgb"]["rate"]["effective_fps"]
            != moto["streams"]["camera-rgb"]["rate"]["effective_fps"])


# --------------------------------------------------------------------------- #
# pseudo-labels are kept separate from ground truth
# --------------------------------------------------------------------------- #
@requires_package
def test_preannotations_are_labelled_as_not_ground_truth(package):
    pre = package["preannotations"]
    assert pre["status"] == "PRE_ANNOTATION_NOT_GROUND_TRUTH"
    assert "review" in pre["instruction"].lower()


@requires_package
def test_preannotation_directory_name_says_what_it_is():
    directory = PACKAGE / "preannotations_not_ground_truth"
    assert directory.exists()
    readme = (directory / "README.md").read_text()
    assert "NOT GROUND TRUTH" in readme
    assert "automatic model output" in readme


@requires_package
def test_package_never_calls_automatic_output_ground_truth(package):
    text = json.dumps(package).lower()
    assert "pre_annotation_not_ground_truth" in text
    # no item may be marked as reviewed by the automatic pass
    for item in package["items"]:
        assert "review_status" not in item or item["review_status"] != "reviewed"


@requires_package
def test_package_checksums_cover_the_tracked_files():
    checksums = (PACKAGE / "checksums.sha256").read_text().splitlines()
    listed = {line.split("  ", 1)[1] for line in checksums if "  " in line}
    for required in ("manifest.json", "labels.json", "labelmap.txt",
                     "palette.json", "ANNOTATION_GUIDE.md", "frame_list.csv"):
        assert required in listed


@requires_package
def test_every_item_records_its_image_checksum(package):
    for item in package["items"]:
        assert re.fullmatch(r"[0-9a-f]{64}", item["image_sha256"] or "")


@requires_package
def test_cockpit_labels_carry_the_hand_attribute_vocabulary():
    from aria_drive_seg.ingestion.hand_audit import HAND_ATTRIBUTES

    labels = json.loads((PACKAGE / "labels.json").read_text())
    for name in ("mirror", "instrument_display", "control_and_ego_vehicle"):
        label = next(l for l in labels if l["name"] == name)
        attributes = {a["name"] for a in label["attributes"]}
        assert set(HAND_ATTRIBUTES) <= attributes, name


@requires_package
def test_hands_are_not_a_taxonomy_class():
    labels = json.loads((PACKAGE / "labels.json").read_text())
    names = {l["name"] for l in labels}
    for banned in ("hand", "left_hand", "right_hand", "driver_hand", "rider_hand"):
        assert banned not in names
    assert "control_and_ego_vehicle" in names


@requires_package
def test_the_delivered_taxonomy_is_the_article_one():
    labels = json.loads((PACKAGE / "labels.json").read_text())
    assert sorted(l["id"] for l in labels) == list(range(0, 14))


def _plain(text: str) -> str:
    """Strip markdown emphasis and collapse whitespace, so these assertions test
    what the guide says rather than how it is formatted."""
    return re.sub(r"\s+", " ", re.sub(r"[*`_]", "", text)).lower()


@requires_package
def test_guide_forbids_unknown_in_the_delivered_annotation():
    guide = _plain((PACKAGE / "ANNOTATION_GUIDE.md").read_text())
    assert "must contain no unknown pixel" in guide
    assert "never as a reference" in guide


@requires_package
def test_guide_states_that_an_absent_hand_is_normal():
    guide = _plain((PACKAGE / "ANNOTATION_GUIDE.md").read_text())
    assert "normal, not an error" in guide
    assert "do not invent a hand you cannot see" in guide
    assert "do not extend a mask from a neighbouring frame" in guide


# --------------------------------------------------------------------------- #
# heavy artefacts stay out of git
# --------------------------------------------------------------------------- #
def test_gitignore_excludes_the_heavy_package_content():
    ignore = Path(".gitignore").read_text()
    assert "datasets/article1_annotation_package/images/" in ignore
    assert ("datasets/article1_annotation_package/"
            "preannotations_not_ground_truth/SegmentationClass/") in ignore


def test_gitignore_excludes_recordings_and_run_outputs():
    ignore = Path(".gitignore").read_text()
    assert "*.[vV][rR][sS]" in ignore
    assert "output/" in ignore
