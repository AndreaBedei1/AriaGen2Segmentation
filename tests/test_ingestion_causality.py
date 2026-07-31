"""The ingestion code never leaks the future into a causal result, and never
lets gaze influence segmentation.

The offline diagnostics deliberately look both ways; the point of these tests is
that they are the *only* things that do, that they say so, and that what they see
never flows back into a mask.
"""
from __future__ import annotations

import ast
import inspect
from pathlib import Path

import numpy as np
import pytest

from aria_drive_seg.ingestion import (compare, extract, failure_modes, hand_audit,
                                      inventory, rgb_scan, route_align,
                                      segment_select, stream_qa, streams, timeline)
from aria_drive_seg.ingestion.streams import associate_by_timestamp
from aria_drive_seg.ingestion.timeline import NS_PER_S

ALL_MODULES = (compare, extract, failure_modes, hand_audit, inventory, rgb_scan,
               route_align, segment_select, stream_qa, streams, timeline)

# Modules allowed to compare a frame with its neighbours, because they are offline
# QA over an already-decoded stream or an already-produced run. None of them writes
# a mask; they only describe what was produced.
#
#   failure_modes  - describes flicker and trailing, which are inherently two-sided
#   compare        - measures an existing run against another existing run
#   segment_select - ranks windows of a recording that has already been decoded
#   rgb_scan       - duplicate detection compares each frame with its successor
BIDIRECTIONAL_DIAGNOSTICS = {failure_modes, compare, segment_select, rgb_scan}


def module_path(module) -> Path:
    return Path(module.__file__)


# --------------------------------------------------------------------------- #
# gaze never enters segmentation
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("module", ALL_MODULES, ids=lambda m: m.__name__)
def test_no_module_feeds_gaze_into_segmentation(module):
    """No ingestion module may pass a gaze coordinate to a segmenter."""
    source = module_path(module).read_text()
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        for kw in node.keywords:
            if kw.arg and "gaze" in kw.arg.lower():
                # the only allowed use is reading/associating gaze, never
                # conditioning a segmentation call on it
                func = getattr(node.func, "attr", getattr(node.func, "id", ""))
                assert "segment" not in func.lower() and "infer" not in func.lower(), (
                    f"{module.__name__} passes {kw.arg} into {func}")


def test_the_extractor_writes_gaze_only_as_a_sidecar():
    """Gaze is associated after the fact; it is not an input to anything."""
    source = inspect.getsource(extract.run_timestamped_extract)
    # the sidecars are written after the frame loop has finished
    assert source.index("_write_gaze_sidecar") > source.index("frame_df")
    assert "gaze" not in inspect.getsource(extract.select_indices).lower()


def test_gaze_sidecar_records_distance_and_validity_without_deciding():
    source = inspect.getsource(extract._write_gaze_sidecar)
    for field in ("nearest_gaze_dt_ms", "nearest_gaze_combined_valid",
                  "gaze_valid_samples_in_window"):
        assert field in source
    # it must not collapse a gaze sample into a class decision
    assert "class" not in source.lower()


# --------------------------------------------------------------------------- #
# no future access in the causal parts
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "module", [m for m in ALL_MODULES if m not in BIDIRECTIONAL_DIAGNOSTICS],
    ids=lambda m: m.__name__)
def test_causal_modules_do_not_index_a_later_frame(module):
    """Reject `frames[i + k]` style forward indexing outside the diagnostics."""
    tree = ast.parse(module_path(module).read_text())
    for node in ast.walk(tree):
        if not isinstance(node, ast.Subscript):
            continue
        index = node.slice
        if isinstance(index, ast.BinOp) and isinstance(index.op, ast.Add):
            right = index.right
            if isinstance(right, ast.Constant) and isinstance(right.value, int) \
                    and right.value > 0:
                target = getattr(node.value, "id", "") or \
                    getattr(node.value, "attr", "")
                assert "frame" not in target.lower(), (
                    f"{module.__name__} indexes {target}[... + {right.value}]")


def test_the_diagnostics_declare_that_they_look_both_ways():
    doc = failure_modes.__doc__ or ""
    assert "offline diagnostic" in doc
    assert "both directions" in doc
    assert "never part of the scientific segmentation path" in doc


@pytest.mark.parametrize("module", sorted(BIDIRECTIONAL_DIAGNOSTICS,
                                          key=lambda m: m.__name__),
                         ids=lambda m: m.__name__)
def test_every_exempt_module_declares_itself_as_qa_or_diagnostic(module):
    """A module is only exempt from the causality rule if it says what it is."""
    doc = (module.__doc__ or "").lower()
    assert any(word in doc for word in
               ("diagnostic", "qa", "proxy statistic", "ranking")), module.__name__


def test_the_diagnostic_result_carries_the_causality_note():
    diagnostics = [
        failure_modes.FrameDiagnostics(
            frame_index=i, timestamp_ns=int(i / 15 * NS_PER_S),
            class_fraction={"road_surface": 0.5, "unknown": 0.0},
            component_count={"road_surface": 1, "unknown": 0},
            mean_confidence=0.9, mean_entropy=0.1, fallback_fraction=0.0,
            conflict_fraction=0.0, external_fraction=1.0,
            internal_model_fraction=0.0, geometric_fraction=0.0,
            dense_fill_fraction=0.0)
        for i in range(5)]

    class Tax:
        def names(self):
            return ["unknown", "road_surface"]

    result = failure_modes.detect(diagnostics, Tax())
    assert result["status"] == "candidate_failure_modes_pre_ground_truth"
    assert result["is_confirmed_error_list"] is False


# --------------------------------------------------------------------------- #
# association windows are symmetric by construction, and that is deliberate
# --------------------------------------------------------------------------- #
def test_association_window_is_symmetric_and_is_not_a_segmentation_input():
    """A +/- window around a frame does look forward in time.

    That is correct for *sensor association*: a gaze sample 20 ms after the shutter
    describes the same instant of the drive. It is not a causality violation
    because the associated sample never influences a mask; it is only ever read
    after segmentation.
    """
    frames = np.array([1_000_000_000], dtype=np.int64)
    samples = np.array([1_000_000_000 - 30_000_000,
                        1_000_000_000 + 30_000_000], dtype=np.int64)
    assoc = associate_by_timestamp(frames, samples, window_s=0.05)[0]
    assert len(assoc.window_indices) == 2
    assert min(assoc.window_dt_ms) < 0 < max(assoc.window_dt_ms)


def test_segmentation_stages_are_not_reachable_from_the_ingestion_package():
    """The ingestion package must not import a segmenter at all."""
    for module in ALL_MODULES:
        source = module_path(module).read_text()
        for banned in ("GroundedSAM2Segmenter", "OneFormerMapillarySegmenter",
                       "CockpitSegFormer", "run_semantic_camera"):
            assert banned not in source, f"{module.__name__} imports {banned}"
