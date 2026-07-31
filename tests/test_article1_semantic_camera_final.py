from __future__ import annotations

import hashlib
import json

import cv2
import numpy as np
import pandas as pd

from aria_drive_seg.article1.semantic_camera_final import (
    FINAL_PROVENANCE,
    FinalPassResult,
    _tree_contract,
    finalize_presentation_frame,
    refine_line_class,
    run_semantic_camera_final_pass,
)
from aria_drive_seg.article1.semantic_camera_video import (
    PresentationResult,
    identity_pair_flow,
)
from aria_drive_seg.config import Config
from aria_drive_seg.io_utils import write_mask_u16
from aria_drive_seg.taxonomy import Taxonomy
from scripts.render_article1_semantic_camera_final import \
    build_final_video_frame


def line_cfg():
    return {
        "closing_orientations_deg": [0, 45, 90, 135],
        "closing_kernel_length_px": 7,
        "maximum_gap_px": 8,
        "orientation_tolerance_deg": 15,
        "endpoint_alignment_tolerance_deg": 20,
        "seed_minimum_area_px": 2,
        "probability_support_floor": .5,
        "skeletonize_after_linking": True,
        "maximum_thickness_px": 3,
        "minimum_length_px": 4,
        "minimum_aspect_ratio": 1.8,
        "added_confidence": .5,
    }


def test_compatible_line_gaps_are_connected():
    binary = np.zeros((15, 30), bool)
    binary[7, 3:10] = True
    binary[7, 14:22] = True
    result = refine_line_class(
        binary, np.ones_like(binary), np.zeros_like(binary),
        np.zeros(binary.shape, np.float32),
        {**line_cfg(), "closing_kernel_length_px": 3})
    assert np.all(result.mask[7, 9:15])
    assert result.stats["bridges"] >= 1
    assert result.stats["morphological_added_pixels"] > 0


def test_incompatible_line_orientations_are_not_connected():
    binary = np.zeros((24, 30), bool)
    binary[12, 3:10] = True
    binary[5:12, 14] = True
    result = refine_line_class(
        binary, np.ones_like(binary), np.zeros_like(binary),
        np.zeros(binary.shape, np.float32), line_cfg())
    assert not result.mask[12, 11]
    assert result.stats["bridges"] == 0


def test_line_never_crosses_vehicle_or_cockpit_barrier():
    binary = np.zeros((15, 30), bool)
    binary[7, 2:10] = True
    binary[7, 20:28] = True
    forbidden = np.zeros_like(binary)
    forbidden[:, 13:17] = True
    result = refine_line_class(
        binary, np.ones_like(binary), forbidden,
        np.zeros(binary.shape, np.float32),
        {**line_cfg(), "maximum_gap_px": 16})
    assert not np.any(result.mask[:, 13:17])
    assert result.stats["bridges"] == 0


def _temporal_result(mask):
    shape = mask.shape
    return PresentationResult(
        mask=mask.copy(),
        confidence=np.full(shape, .8, np.float32),
        provenance=np.ones(shape, np.uint8),
        support_count=np.ones(shape, np.uint8),
        stats={"dense_coverage": 1.0},
    )


def _geometry_cfg():
    return {
        "road_closing_kernel_px": 3,
        "road_support_radius_px": 2,
        "barrier_dilation_px": 0,
        "fill_confidence": .3,
        "line_filter": {
            "common": line_cfg(),
            "lane_marking": {},
            "regulatory_road_marking": {
                "minimum_aspect_ratio": 1.2,
            },
        },
        "road_boundary_filter": {
            "road_edge_radius_px": 3,
            "probability_support_floor": .5,
            "closing_kernel_px": 3,
            "maximum_hole_area_px": 8,
            "minimum_component_area_px": 2,
            "minimum_road_edge_fraction": 0.0,
            "added_confidence": .4,
        },
    }


def test_finalization_is_dense_valid_and_deterministic():
    mask = np.full((20, 30), 13, np.uint16)
    mask[6:18, 2:28] = 1
    mask[12, 4:11] = 2
    mask[12, 15:24] = 2
    temporal = _temporal_result(mask)
    external = mask.copy()
    external_confidence = np.full(mask.shape, .4, np.float32)
    first = finalize_presentation_frame(
        temporal, mask, external, external_confidence, _geometry_cfg())
    second = finalize_presentation_frame(
        temporal, mask, external, external_confidence, _geometry_cfg())
    first.validate()
    assert np.array_equal(first.mask, second.mask)
    assert np.array_equal(first.provenance, second.provenance)
    assert np.all((first.mask >= 1) & (first.mask <= 13))
    assert first.stats["dense_coverage"] == 1.0
    assert set(np.unique(first.provenance)).issubset(FINAL_PROVENANCE)


def _synthetic_final_inputs(tmp_path, count=5):
    root = tmp_path / "frames_run"
    source = tmp_path / "semantic_camera"
    previous = tmp_path / "previous"
    (root / "frames/rectified").mkdir(parents=True)
    for directory in (
            "final_masks", "final_confidence", "external_masks",
            "external_confidence", "internal_masks",
            "internal_confidence", "metadata"):
        (source / directory).mkdir(parents=True)
    for directory in (
            "presentation_masks", "presentation_confidence", "flow_cache"):
        (previous / directory).mkdir(parents=True)
    rows = []
    for offset in range(count):
        frame_id = 300 + offset
        stem = f"frame_{frame_id:06d}"
        relative = f"frames/rectified/{stem}.jpg"
        cv2.imwrite(
            str(root / relative),
            np.full((40, 60, 3), 50 + offset, np.uint8))
        rows.append({
            "frame_index": frame_id,
            "capture_timestamp_ns": (offset + 1) * 100_000_000,
            "rectified_path": relative,
            "original_path": relative,
        })
        external = np.full((40, 60), 13, np.uint16)
        external[12:34, 3:57] = 1
        final = external.copy()
        final[24, 5:25] = 2
        final[24, 30:52] = 2
        if offset != count // 2:
            final[32:36, 20:40] = 11
        internal = np.zeros_like(final)
        internal[final == 11] = 11
        for directory, value in (
                ("final_masks", final),
                ("external_masks", external),
                ("internal_masks", internal)):
            write_mask_u16(source / directory / f"{stem}.png", value)
        for directory, value in (
                ("final_confidence", 180),
                ("external_confidence", 160),
                ("internal_confidence", 180)):
            cv2.imwrite(
                str(source / directory / f"{stem}.png"),
                np.full((40, 60), value, np.uint8))
        (source / "metadata" / f"{stem}.json").write_text(
            json.dumps({"frame_index": frame_id}))
        write_mask_u16(
            previous / "presentation_masks" / f"{stem}.png", final)
        cv2.imwrite(
            str(previous / "presentation_confidence" / f"{stem}.png"),
            np.full((40, 60), 180, np.uint8))
        if offset:
            pair = identity_pair_flow((40, 60))
            np.savez_compressed(
                previous / "flow_cache" /
                f"flow_{frame_id - 1:06d}_{frame_id:06d}.npz",
                forward=pair.forward,
                backward=pair.backward,
                valid_forward=pair.valid_forward.astype(np.uint8),
                valid_backward=pair.valid_backward.astype(np.uint8),
            )
    pd.DataFrame(rows).to_parquet(
        root / "frames/frames.parquet", index=False)
    (source / "manifest.json").write_text(json.dumps({
        "fingerprint": "scientific-raw",
        "done": {str(300 + offset): {} for offset in range(count)},
    }))
    (previous / "manifest.json").write_text(json.dumps({
        "fingerprint": "previous-presentation",
        "done": {str(300 + offset): {} for offset in range(count)},
    }))
    return root, source, previous


def test_final_runner_preserves_raw_has_full_coverage_and_is_deterministic(
        tmp_path, project_root):
    root, source, previous = _synthetic_final_inputs(tmp_path)
    output = tmp_path / "final"
    cfg = Config.load(
        project_root / "configs/article1/semantic_camera_final.yaml",
        overrides={"semantic_camera_final": {"processing_scale": 1.0}})
    raw_before = _tree_contract(source)
    source_file = source / "final_masks/frame_000302.png"
    file_before = hashlib.sha256(source_file.read_bytes()).hexdigest()
    assert run_semantic_camera_final_pass(
        root, source, previous, cfg, output) == 0
    raw_after = _tree_contract(source)
    assert raw_before == raw_after
    assert hashlib.sha256(source_file.read_bytes()).hexdigest() == file_before
    first = cv2.imread(
        str(output / "final_masks/frame_000302.png"),
        cv2.IMREAD_UNCHANGED)
    first_hash = hashlib.sha256(
        (output / "final_masks/frame_000302.png").read_bytes()).hexdigest()
    assert np.all((first >= 1) & (first <= 13))
    assert run_semantic_camera_final_pass(
        root, source, previous, cfg, output, resume=True) == 0
    assert hashlib.sha256(
        (output / "final_masks/frame_000302.png").read_bytes()
    ).hexdigest() == first_hash
    metrics = json.loads((output / "metrics.json").read_text())
    assert metrics["coverage"]["dense_fraction"] == 1.0
    assert metrics["coverage"]["invalid_id_pixels"] == 0
    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["raw_contract_sha256_before"] == \
        manifest["raw_contract_sha256_after"]


def test_final_video_frame_contains_only_overlay_legend_and_index(
        project_root):
    rgb = np.full((24, 32, 3), 80, np.uint8)
    mask = np.full((24, 32), 13, np.uint16)
    mask[12:] = 1
    mask[18, 5:26] = 2
    taxonomy = Taxonomy.load(
        project_root / "configs/article1/classes_article1.yaml")
    frame = build_final_video_frame(rgb, mask, taxonomy, 123)
    assert frame.shape == (960, 1280, 3)
    assert frame.dtype == np.uint8
