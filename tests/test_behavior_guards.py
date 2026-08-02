"""Repository-level guards for the Article 1 behaviour analysis.

These tests do not check that the analysis is right. They check that the rules it
was built under are still being followed — that nobody has hard-coded a frame
rate, resurrected the retired branch, committed a coordinate, or quietly turned a
candidate into a verdict. They are cheap and they fail loudly, which is the point.
"""
from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
BEHAVIOR = ROOT / "aria_drive_seg" / "behavior"
SCRIPTS = ROOT / "scripts"
REPORTS = ROOT / "reports" / "article1_behavior_analysis"

BASELINE_COMMIT = "a9ee61bc77cdf61f364eb0eec98c2975fd933829"
RETIRED_BRANCH = "feature/article1-motorcycle-fov-mirror-refinement"

BEHAVIOR_SOURCES = sorted(BEHAVIOR.glob("*.py"))
BEHAVIOR_SCRIPTS = sorted(
    p for p in SCRIPTS.glob("*article1_behavior*.py")) + sorted(
    p for p in SCRIPTS.glob("*article1_*.py")
    if any(k in p.name for k in ("shared_route", "solid_line", "paired_route",
                                 "road_events", "semantic_gaze", "dynamics",
                                 "ppg", "routes")))


def _git(*args: str) -> str:
    return subprocess.run(["git", "-C", str(ROOT), *args],
                          capture_output=True, text=True, check=False).stdout


# --------------------------------------------------------------------------- #
# Provenance of the branch
# --------------------------------------------------------------------------- #
def test_branch_descends_from_the_baseline_commit():
    """The behaviour work must build on the frozen baseline, not on something else."""
    merge_base = _git("merge-base", "HEAD", BASELINE_COMMIT).strip()
    assert merge_base == BASELINE_COMMIT, (
        f"HEAD does not descend from the baseline commit {BASELINE_COMMIT}; "
        f"merge-base is {merge_base!r}")


def test_retired_full_fov_branch_is_not_present_locally():
    branches = _git("branch", "--list", RETIRED_BRANCH).strip()
    assert branches == "", (
        f"the retired branch {RETIRED_BRANCH} is present again: {branches!r}")


def test_no_full_fov_artefacts_are_tracked():
    """None of the retired branch's files may come back into the tree."""
    tracked = _git("ls-files").splitlines()
    banned = [p for p in tracked
              if "fov_mirror_refinement" in p
              or p.startswith("aria_drive_seg/geometry/")
              or p.endswith("semantic_camera_full_fov.yaml")]
    assert banned == [], f"full-FOV artefacts are tracked again: {banned}"


def test_behaviour_code_never_imports_the_retired_geometry_package():
    offenders = [p.name for p in BEHAVIOR_SOURCES + BEHAVIOR_SCRIPTS
                 if re.search(r"\bfrom\s+\S*geometry\s+import|\bimport\s+\S*\.geometry",
                              p.read_text())]
    assert offenders == [], f"{offenders} import the retired geometry package"


# --------------------------------------------------------------------------- #
# Frame rate must never be baked in or used as a feature
# --------------------------------------------------------------------------- #
def _code_only(path: Path) -> str:
    """Source with comments and string literals removed.

    Docstrings are where these modules *document* the 10/15 fps problem, so a
    naive text scan flags the very prose that explains the rule. Only executable
    code can hard-code anything, so only executable code is searched.
    """
    import io
    import tokenize

    out = []
    with path.open("rb") as fh:
        for tok in tokenize.tokenize(fh.readline):
            if tok.type in (tokenize.COMMENT, tokenize.STRING,
                            tokenize.NL, tokenize.NEWLINE):
                continue
            out.append((tok.start[0], tok.string))
    return "\n".join(f"{line}:{text}" for line, text in out)


@pytest.mark.parametrize("path", BEHAVIOR_SOURCES, ids=lambda p: p.name)
def test_no_hard_coded_frame_rate(path: Path):
    """No 10 fps or 15 fps constant anywhere in the behaviour package.

    The car currently runs at 10 fps and the motorcycle at 15. Both are temporary
    acquisition artefacts, and a constant of either would silently break the day
    the protocol is unified at 15 fps.
    """
    code = _code_only(path)
    bad = []
    for pattern in (r"fps\S*\n:\n=\n(10|15)(\.0+)?\b",
                    r"(frame_rate|fps)\S*:?\n?=\n?(10|15)(\.0+)?\b"):
        bad += re.findall(pattern, code, re.I)
    # Also catch a bare 10/15 assigned to anything rate-shaped, on one line.
    src = path.read_text()
    for i, line in enumerate(src.splitlines(), 1):
        if re.search(r"\b(fps|frame_rate|framerate|rate_hz)\s*[:=]\s*(10|15)(\.0+)?\b",
                     line, re.I) and not line.strip().startswith("#"):
            bad.append((i, line.strip()))
    assert bad == [], f"{path.name} hard-codes a frame rate: {bad}"


@pytest.mark.parametrize("path", BEHAVIOR_SOURCES, ids=lambda p: p.name)
def test_rates_are_measured_not_assumed(path: Path):
    """Anything that needs a rate must measure it from timestamps."""
    code = _code_only(path)
    if "sampling_rate" not in code and "effective_fps" not in code:
        pytest.skip("module does not deal with sampling rates")
    src = path.read_text()
    assert ("np.median(np.diff" in src or "measure_rate" in src
            or "np.diff(ts" in src), (
        f"{path.name} handles a sampling rate without measuring it from "
        "timestamps")


@pytest.mark.parametrize("path", BEHAVIOR_SOURCES + BEHAVIOR_SCRIPTS,
                         ids=lambda p: p.name)
def test_frame_rate_and_recording_id_are_not_used_as_features(path: Path):
    """Neither may enter a model or a comparison as an input variable.

    A classifier given the frame rate learns which file it is reading, not which
    vehicle it is looking at; the same is true of the recording id.
    """
    text = path.read_text()
    for pattern, what in (
            (r"features?\s*=.*\b(fps|frame_rate|effective_fps)\b", "frame rate"),
            (r"features?\s*=.*\brecording_id\b", "recording id"),
            (r"\bX\s*=.*\b(fps|frame_rate|recording_id)\b", "rate or id")):
        assert not re.search(pattern, text), (
            f"{path.name} appears to use {what} as a feature")


def test_metrics_are_normalised_per_unit_time():
    """Counts crossing recordings must go through a per-second/per-minute helper."""
    dynamics = (BEHAVIOR / "dynamics.py").read_text()
    assert "_per_minute" in dynamics
    gaze = (BEHAVIOR / "gaze_semantics.py").read_text()
    assert "per_minute" in gaze and "per_s" in gaze


# --------------------------------------------------------------------------- #
# Timestamps, no synthetic data
# --------------------------------------------------------------------------- #
def test_multimodal_timeline_declares_no_interpolation():
    from aria_drive_seg.behavior.multimodal import (build_multimodal_timeline,
                                                    derive_view)
    import numpy as np
    ref = np.arange(0, 10) * 100_000_000
    tl = build_multimodal_timeline("r", "car", ref, {}, {"gps-app": ref[::3]})
    s = tl.summary()
    assert s["interpolated"] is False
    assert s["resampled"] is False
    assert s["synthetic_samples"] == 0
    view = derive_view(tl, 1.0, "test")
    assert view.interpolated is False
    assert view.synthetic_samples == 0


def test_derived_views_drop_rather_than_fill():
    """A grid point with no nearby real frame must be dropped, never invented."""
    from aria_drive_seg.behavior.multimodal import (build_multimodal_timeline,
                                                    derive_view)
    import numpy as np
    # A one-second hole in an otherwise 10 Hz stream.
    ref = np.concatenate([np.arange(0, 5), np.arange(25, 30)]) * 100_000_000
    tl = build_multimodal_timeline("r", "car", ref, {})
    view = derive_view(tl, 10.0, "test", max_temporal_error_s=0.05)
    assert view.dropped_grid_points > 0
    assert view.row_indices.size <= ref.size


def test_association_tolerance_depends_on_measured_cadence():
    """The same configuration must adapt to a 10 fps and a 15 fps recording."""
    from aria_drive_seg.behavior.multimodal import association_tolerance_s
    at10 = association_tolerance_s(1 / 10.0, 1 / 30.0)
    at15 = association_tolerance_s(1 / 15.0, 1 / 30.0)
    assert at10 > at15, "tolerance must shrink as the reference rate rises"


# --------------------------------------------------------------------------- #
# Gaze must not feed segmentation
# --------------------------------------------------------------------------- #
def test_gaze_is_never_used_to_produce_segmentation():
    """The behaviour package reads the frozen masks; it never writes one."""
    text = (BEHAVIOR / "gaze_semantics.py").read_text()
    for banned in ("write_mask", "cv2.imwrite", "segmenter", "model(", ".predict("):
        assert banned not in text, (
            f"gaze_semantics.py contains {banned!r}: gaze must not influence "
            "segmentation")


def test_semantic_gaze_report_declares_the_separation():
    path = REPORTS / "semantic_gaze_summary.json"
    if not path.exists():
        pytest.skip("semantic gaze summary not generated in this checkout")
    method = json.loads(path.read_text())["method"]
    assert method["gaze_used_for_segmentation"] is False
    assert "frozen" in method["segmentation_source"].lower()


# --------------------------------------------------------------------------- #
# Privacy
# --------------------------------------------------------------------------- #
COORD_COLUMNS = ("latitude", "longitude", "matched_lat", "matched_lon")


def test_committed_csvs_carry_no_absolute_coordinates():
    """Nothing under reports/ may contain a coordinate column."""
    offenders = []
    for csv in REPORTS.rglob("*.csv"):
        header = csv.read_text(errors="ignore").split("\n", 1)[0].lower()
        cols = {c.strip().strip('"') for c in header.split(",")}
        hit = sorted(cols & set(COORD_COLUMNS))
        if hit:
            offenders.append((str(csv.relative_to(ROOT)), hit))
    assert offenders == [], f"committed CSVs carry coordinates: {offenders}"


def test_osm_cache_is_not_tracked():
    tracked = _git("ls-files").splitlines()
    cached = [p for p in tracked if "osm_cache" in p or p.endswith("overpass.json")]
    assert cached == [], f"the OSM cache is tracked: {cached}"


def test_privacy_guard_rejects_a_coordinate_table():
    import pandas as pd
    from aria_drive_seg.behavior.privacy import assert_no_absolute_coordinates
    with pytest.raises(ValueError, match="absolute coordinates"):
        assert_no_absolute_coordinates(
            pd.DataFrame({"latitude": [1.0], "longitude": [2.0]}), "test table")


def test_redaction_trims_both_ends_of_a_track():
    import numpy as np
    from aria_drive_seg.behavior.privacy import redact_track
    x = np.arange(0, 2000, 10.0)
    y = np.zeros_like(x)
    red = redact_track(x, y, endpoint_redaction_m=250.0)
    assert red.trimmed_start > 0 and red.trimmed_end > 0
    assert red.keep.sum() < x.size
    # The published origin is the first surviving point, so it is not the home end.
    assert abs(red.x_m[red.keep][0]) < 1e-6


def test_a_track_shorter_than_its_redaction_publishes_nothing():
    import numpy as np
    from aria_drive_seg.behavior.privacy import redact_track
    x = np.arange(0, 100, 10.0)
    red = redact_track(x, np.zeros_like(x), endpoint_redaction_m=250.0)
    assert red.keep.sum() == 0


# --------------------------------------------------------------------------- #
# Result status
# --------------------------------------------------------------------------- #
def test_every_generated_summary_is_marked_exploratory():
    found = list(REPORTS.rglob("*.json"))
    if not found:
        pytest.skip("no generated summaries in this checkout")
    unmarked = []
    for path in found:
        try:
            payload = json.loads(path.read_text())
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict) and "schema" in payload:
            if payload.get("result_status") != "exploratory_pilot":
                unmarked.append(str(path.relative_to(ROOT)))
    assert unmarked == [], f"summaries not marked exploratory_pilot: {unmarked}"


# --------------------------------------------------------------------------- #
# Source data integrity
# --------------------------------------------------------------------------- #
def test_vrs_recordings_are_not_tracked_and_not_written():
    tracked = [p for p in _git("ls-files").splitlines() if p.lower().endswith(".vrs")]
    assert tracked == [], f"VRS recordings are tracked: {tracked}"
    for path in BEHAVIOR_SOURCES + BEHAVIOR_SCRIPTS:
        text = path.read_text()
        assert not re.search(r"open\([^)]*\.vrs[^)]*['\"][waxr]\+?b?['\"]", text), (
            f"{path.name} appears to open a VRS for writing")


def test_frozen_baseline_outputs_are_only_read():
    """The behaviour package must not write into a frozen baseline run."""
    for path in BEHAVIOR_SOURCES:
        text = path.read_text()
        if "semantic_camera_final_pass" not in text:
            continue
        assert "imwrite" not in text and "write_mask" not in text, (
            f"{path.name} writes into the frozen semantic-camera output")


# --------------------------------------------------------------------------- #
# Visual output completeness
# --------------------------------------------------------------------------- #
def _visualiser():
    """The visualisation script, imported without running it."""
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "_vis_behavior", SCRIPTS / "visualise_article1_behavior.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_the_analysis_plan_s_twenty_figures_are_all_declared():
    """The plan asks for twenty figures; the script must claim all twenty."""
    module = _visualiser()
    assert len(module.REQUIRED_FIGURES) == 20, (
        f"expected 20 required figures, found {len(module.REQUIRED_FIGURES)}")
    numbers = sorted(int(stem.split("_")[0]) for stem in module.REQUIRED_FIGURES)
    assert numbers == list(range(1, 21)), (
        f"required figures are not numbered 1..20: {numbers}")


def test_a_run_that_drops_a_required_figure_is_rejected():
    """The completeness check must actually fail on a short gallery."""
    module = _visualiser()
    complete = [f"/figures/{stem}_x.png" for stem in module.REQUIRED_FIGURES]
    assert module.missing_required(complete) == []
    assert module.missing_required(complete[:-1]) == [module.REQUIRED_FIGURES[-1]]


def test_the_generated_gallery_contains_every_required_figure():
    manifest = (ROOT / "output" / "article1" / "behavior_analysis" / "figures"
                / "figure_manifest.json")
    if not manifest.exists():
        pytest.skip("figures have not been generated in this checkout")
    module = _visualiser()
    payload = json.loads(manifest.read_text())
    absent = module.missing_required(payload["figures"])
    assert absent == [], f"generated gallery is missing required figures: {absent}"
