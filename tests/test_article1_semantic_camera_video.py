from __future__ import annotations

import hashlib
import json

import cv2
import numpy as np
import pandas as pd
import pytest

from aria_drive_seg.article1.semantic_camera_video import (
    PairFlow,
    compute_flow_aligned_metrics,
    compute_presentation_metrics,
    identity_pair_flow,
    improve_fusion_frame,
    stabilize_presentation_frame,
    warp_source_to_target,
)
from aria_drive_seg.cli import _load_cfg, build_parser
from aria_drive_seg.config import Config
from aria_drive_seg.io_utils import write_mask_u16
from aria_drive_seg.taxonomy import Taxonomy
from scripts.render_article1_semantic_camera_video import build_video_frames


def fusion_cfg():
    return {
        "entropy_penalty": 0.0,
        "conflict_margin": 0.0,
        "cockpit_bottom_start_fraction": .5,
        "cockpit_side_fraction": .2,
        "cockpit_position_bonus": .2,
        "internal_min_confidence": {
            "mirror": .2,
            "instrument_display": .2,
            "control_and_ego_vehicle": .2,
        },
        "internal_priorities": {
            "mirror": 150,
            "instrument_display": 140,
            "control_and_ego_vehicle": 130,
        },
        "strong_external_thresholds": {
            "road_surface": .7,
            "lane_marking": .7,
            "vehicle": .8,
            "traffic_sign": .7,
        },
    }


def presentation_cfg():
    classes = {
        name: {"ttl_frames": 0, "vote_weight": .5,
               "minimum_confidence": 0.0}
        for name in (
            "unknown", "road_surface", "lane_marking",
            "regulatory_road_marking", "vehicle", "two_wheeler",
            "pedestrian", "traffic_light", "traffic_sign",
            "road_boundary_or_obstacle", "mirror", "instrument_display",
            "control_and_ego_vehicle", "other_environment")
    }
    classes["mirror"] = {
        "ttl_frames": 2, "vote_weight": 2.0,
        "minimum_confidence": .1, "temporal_decay": 1.0}
    classes["instrument_display"] = dict(classes["mirror"])
    classes["control_and_ego_vehicle"] = dict(classes["mirror"])
    classes["lane_marking"] = {
        "ttl_frames": 2, "vote_weight": 1.5,
        "minimum_confidence": .1, "temporal_decay": 1.0}
    classes["vehicle"] = {
        "ttl_frames": 1, "vote_weight": .6,
        "minimum_confidence": .4, "temporal_decay": 1.0}
    return {
        "window_radius_frames": 2,
        "current_confidence_floor": .2,
        "switch_margin": 0.0,
        "thin_switch_margin": -.1,
        "internal_activation_relaxation": 1.0,
        "internal_current_keep_confidence": .1,
        "classes": classes,
        "strong_external_thresholds": {
            "road_surface": .8, "lane_marking": .8,
            "vehicle": .8, "traffic_sign": .8},
    }


def test_improved_fusion_adds_cockpit_but_protects_external():
    shape = (6, 8)
    current = np.full(shape, 13, np.uint16)
    current_conf = np.full(shape, .7, np.float32)
    external = current.copy()
    external_conf = np.full(shape, .5, np.float32)
    internal = np.zeros(shape, np.uint16)
    internal_conf = np.zeros(shape, np.float32)
    internal[4:, :] = 12
    internal_conf[4:, :] = .4
    external[4, 2] = 2
    external_conf[4, 2] = .95
    current[4, 2] = 2
    result = improve_fusion_frame(
        current, current_conf, external, external_conf,
        internal, internal_conf, fusion_cfg())
    assert np.all(result.mask[5] == 12)
    assert result.mask[4, 2] == 2
    assert result.provenance[4, 2] == 3
    assert result.stats["dense_coverage"] == 1.0


def test_improved_fusion_never_removes_existing_internal():
    shape = (4, 5)
    current = np.full(shape, 13, np.uint16)
    current[1, 1] = 10
    external = np.full(shape, 4, np.uint16)
    internal = np.zeros(shape, np.uint16)
    internal[1, 1] = 10
    result = improve_fusion_frame(
        current, np.full(shape, .9), external, np.full(shape, .99),
        internal, np.full(shape, .01), fusion_cfg())
    assert result.mask[1, 1] == 10
    assert np.all(result.mask > 0)


def test_bidirectional_warp_known_translation():
    shape = (5, 7)
    forward = np.zeros((*shape, 2), np.float32)
    backward = np.zeros((*shape, 2), np.float32)
    forward[..., 0] = 1
    backward[..., 0] = -1
    valid_forward = np.ones(shape, bool)
    valid_backward = np.ones(shape, bool)
    valid_forward[:, -1] = False
    valid_backward[:, 0] = False
    pair = PairFlow(
        forward, backward, valid_forward, valid_backward)
    mask = np.zeros(shape, np.uint16)
    mask[2, 2] = 10
    warped, _, valid = warp_source_to_target(
        mask, np.ones(shape, np.float32), 0, 1, [pair])
    assert warped[2, 3] == 10
    restored, _, _ = warp_source_to_target(
        warped, np.ones(shape, np.float32), 1, 0, [pair])
    assert restored[2, 2] == 10
    assert valid[2, 3]


def test_future_and_past_internal_evidence_fill_dropout():
    shape = (4, 6)
    masks = [np.full(shape, 13, np.uint16) for _ in range(3)]
    confidences = [np.full(shape, .8, np.float32) for _ in range(3)]
    masks[0][2, 3] = 11
    masks[2][2, 3] = 11
    confidences[0][2, 3] = confidences[2][2, 3] = .7
    external = [np.full(shape, 13, np.uint16) for _ in masks]
    external_conf = [np.full(shape, .5, np.float32) for _ in masks]
    result = stabilize_presentation_frame(
        1, masks, confidences, external, external_conf,
        [identity_pair_flow(shape), identity_pair_flow(shape)],
        presentation_cfg())
    assert result.mask[2, 3] == 11
    assert result.provenance[2, 3] == 4


def test_future_only_evidence_is_backward_propagation():
    shape = (3, 4)
    masks = [np.full(shape, 13, np.uint16) for _ in range(2)]
    masks[1][1, 1] = 10
    confidence = [np.full(shape, .4, np.float32) for _ in masks]
    external = [np.full(shape, 13, np.uint16) for _ in masks]
    result = stabilize_presentation_frame(
        0, masks, confidence, external, confidence,
        [identity_pair_flow(shape)], presentation_cfg())
    assert result.mask[1, 1] == 10
    assert result.provenance[1, 1] == 3


def test_internal_ttl_exceeds_vehicle_ttl():
    shape = (3, 5)
    masks = [np.full(shape, 13, np.uint16) for _ in range(3)]
    masks[0][1, 1] = 10
    masks[0][1, 3] = 4
    confidence = [np.full(shape, .5, np.float32) for _ in masks]
    external = [np.full(shape, 13, np.uint16) for _ in masks]
    result = stabilize_presentation_frame(
        2, masks, confidence, external, confidence,
        [identity_pair_flow(shape), identity_pair_flow(shape)],
        presentation_cfg())
    assert result.mask[1, 1] == 10
    assert result.mask[1, 3] == 13


def test_strong_external_class_is_protected_in_presentation():
    shape = (3, 4)
    masks = [np.full(shape, 13, np.uint16) for _ in range(3)]
    masks[0][1, 1] = masks[2][1, 1] = 12
    masks[1][1, 1] = 2
    confidence = [np.full(shape, .7, np.float32) for _ in masks]
    external = [np.full(shape, 13, np.uint16) for _ in masks]
    external[1][1, 1] = 2
    external_conf = [np.full(shape, .5, np.float32) for _ in masks]
    external_conf[1][1, 1] = .95
    result = stabilize_presentation_frame(
        1, masks, confidence, external, external_conf,
        [identity_pair_flow(shape), identity_pair_flow(shape)],
        presentation_cfg())
    assert result.mask[1, 1] == 2
    assert result.provenance[1, 1] == 6


def test_metrics_report_flicker_and_persistence_improvement():
    shape = (3, 4)
    raw = [np.full(shape, 13, np.uint16) for _ in range(3)]
    raw[0][1, 1] = raw[2][1, 1] = 11
    stable = [mask.copy() for mask in raw]
    stable[1][1, 1] = 11
    raw_metrics = compute_presentation_metrics(raw)
    stable_metrics = compute_presentation_metrics(stable)
    assert stable_metrics["class_switch_rate"] < \
        raw_metrics["class_switch_rate"]
    assert stable_metrics["isolated_one_frame_flicker_rate"] < \
        raw_metrics["isolated_one_frame_flicker_rate"]
    assert stable_metrics["internal_class_persistence"] > \
        raw_metrics["internal_class_persistence"]


def test_flow_aligned_metrics_do_not_count_known_motion_as_flicker():
    shape = (5, 7)
    first = np.full(shape, 13, np.uint16)
    second = first.copy()
    first[2, 2] = 12
    second[2, 3] = 12
    second[2, 2] = 13
    forward = np.zeros((*shape, 2), np.float32)
    backward = np.zeros((*shape, 2), np.float32)
    forward[..., 0] = 1
    backward[..., 0] = -1
    valid_forward = np.ones(shape, bool)
    valid_backward = np.ones(shape, bool)
    valid_forward[:, -1] = False
    valid_backward[:, 0] = False
    pair = PairFlow(
        forward, backward, valid_forward, valid_backward)
    raw = compute_presentation_metrics([first, second])
    aligned = compute_flow_aligned_metrics([first, second], [pair])
    assert raw["class_switch_rate"] > 0
    assert aligned["class_switch_rate"] == 0
    assert aligned["internal_class_persistence"] == 1


def _synthetic_semantic_camera(tmp_path):
    root = tmp_path / "frames_run"
    source = tmp_path / "semantic_camera"
    (root / "frames/rectified").mkdir(parents=True)
    for directory in (
            "final_masks", "final_confidence", "external_masks",
            "external_confidence", "internal_masks",
            "internal_confidence", "metadata"):
        (source / directory).mkdir(parents=True)
    rows = []
    for offset in range(3):
        frame_id = 200 + offset
        stem = f"frame_{frame_id:06d}"
        relative = f"frames/rectified/{stem}.jpg"
        cv2.imwrite(
            str(root / relative),
            np.full((24, 32, 3), 50 + offset, np.uint8))
        rows.append({
            "frame_index": frame_id,
            "capture_timestamp_ns": (offset + 1) * 100_000_000,
            "rectified_path": relative,
            "original_path": relative,
        })
        external = np.full((24, 32), 13, np.uint16)
        final = external.copy()
        internal = np.zeros((24, 32), np.uint16)
        if offset != 1:
            internal[18:22, 10:20] = 11
            final[18:22, 10:20] = 11
        for directory, mask in (
                ("final_masks", final),
                ("external_masks", external),
                ("internal_masks", internal)):
            write_mask_u16(source / directory / f"{stem}.png", mask)
        for directory, value in (
                ("final_confidence", 180),
                ("external_confidence", 160),
                ("internal_confidence", 180)):
            cv2.imwrite(
                str(source / directory / f"{stem}.png"),
                np.full((24, 32), value, np.uint8))
        (source / "metadata" / f"{stem}.json").write_text(json.dumps({
            "frame_index": frame_id,
            "capture_timestamp_ns": (offset + 1) * 100_000_000,
            "internal_source": "synthetic_proxy",
            "internal_is_fallback": True,
        }))
    pd.DataFrame(rows).to_parquet(
        root / "frames/frames.parquet", index=False)
    (source / "manifest.json").write_text(json.dumps({
        "fingerprint": "synthetic-semantic-camera",
        "done": {str(200 + index): {} for index in range(3)},
    }))
    return root, source


def _sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_video_runner_is_resumable_dense_and_preserves_source(
        tmp_path, project_root):
    from aria_drive_seg.article1.semantic_camera_video import \
        run_semantic_camera_video

    root, source = _synthetic_semantic_camera(tmp_path)
    output = tmp_path / "video"
    cfg = Config.load(
        project_root / "configs/article1/semantic_camera_video.yaml",
        overrides={"semantic_camera_video": {"processing_scale": 1.0}})
    source_mask = source / "final_masks/frame_000201.png"
    source_hash = _sha(source_mask)

    def identity_provider(previous, current, _cfg):
        return identity_pair_flow(previous.shape[:2])

    assert run_semantic_camera_video(
        root, source, cfg, output,
        pair_flow_provider=identity_provider) == 0
    presentation = cv2.imread(
        str(output / "presentation_masks/frame_000201.png"),
        cv2.IMREAD_UNCHANGED)
    assert presentation.shape == (24, 32)
    assert np.all(presentation > 0)
    assert np.all(presentation[18:22, 10:20] == 11)
    assert _sha(source_mask) == source_hash
    metrics = json.loads((output / "metrics.json").read_text())
    assert metrics["stabilized_presentation"]["class_switch_rate"] < \
        metrics["current_scientific_fusion"]["class_switch_rate"]

    summary_path = output / "summary.json"
    summary = json.loads(summary_path.read_text())
    initial_timing = summary["mean_presentation_ms_per_frame"]
    summary["peak_ram_mb"] = 9999
    summary_path.write_text(json.dumps(summary))
    target = output / "presentation_masks/frame_000201.png"
    before = target.stat().st_mtime_ns
    run_semantic_camera_video(
        root, source, cfg, output, resume=True,
        pair_flow_provider=identity_provider)
    assert target.stat().st_mtime_ns == before
    resumed = json.loads(summary_path.read_text())
    assert resumed["peak_ram_mb"] >= 9999
    assert resumed["mean_presentation_ms_per_frame"] == initial_timing
    assert resumed["initial_run_mean_presentation_ms_per_frame"] == \
        initial_timing


def test_video_runner_fails_closed_on_incompatible_resume(
        tmp_path, project_root):
    from aria_drive_seg.article1.semantic_camera_video import \
        run_semantic_camera_video

    root, source = _synthetic_semantic_camera(tmp_path)
    output = tmp_path / "video"
    cfg = Config.load(
        project_root / "configs/article1/semantic_camera_video.yaml",
        overrides={"semantic_camera_video": {"processing_scale": 1.0}})

    def identity_provider(previous, current, _cfg):
        return identity_pair_flow(previous.shape[:2])

    run_semantic_camera_video(
        root, source, cfg, output,
        pair_flow_provider=identity_provider)
    changed = Config.load(
        project_root / "configs/article1/semantic_camera_video.yaml",
        overrides={"semantic_camera_video": {
            "processing_scale": 1.0,
            "presentation": {"window_radius_frames": 1}}})
    with pytest.raises(RuntimeError, match="fingerprint"):
        run_semantic_camera_video(
            root, source, changed, output, resume=True,
            pair_flow_provider=identity_provider)


def test_video_cli_defaults_to_presentation_config():
    args = build_parser().parse_args([
        "article1", "stabilize-semantic-camera-video",
        "--input", "frames", "--semantic-camera", "semantic",
        "--vehicle-type", "car", "--session-id", "s",
        "--participant-id", "p",
    ])
    cfg = _load_cfg(args)
    assert cfg.get("semantic_camera_video.presentation.non_causal") is True
    assert cfg.get(
        "semantic_camera_video.presentation.window_radius_frames") == 5


def test_video_renderer_products_have_codec_aligned_geometry(project_root):
    shape = (24, 32)
    rgb = np.zeros((*shape, 3), np.uint8)
    current = np.full(shape, 13, np.uint16)
    improved = current.copy()
    improved[-6:] = 12
    presentation = improved.copy()
    presentation[8:10, 4:20] = 2
    taxonomy = Taxonomy.load(
        project_root / "configs/article1/classes_article1.yaml")
    metadata = {
        "frame_index": 1,
        "capture_timestamp_ns": 100,
        "presentation": {
            "full_resolution_internal_fraction": .2,
        },
    }
    metrics = {
        "comparison": {
            "presentation_vs_current": {
                "flicker_reduction": .5,
            },
        },
    }
    products = build_video_frames(
        rgb, current, improved, presentation,
        metadata, taxonomy, metrics)
    assert [product.shape for product in products] == [
        (960, 1280, 3),
        (960, 1280, 3),
        (560, 1920, 3),
        (960, 1920, 3),
        (960, 1920, 3),
    ]
    assert all(
        dimension % 8 == 0
        for product in products for dimension in product.shape[:2])
