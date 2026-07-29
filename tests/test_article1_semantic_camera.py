import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import pytest

from aria_drive_seg.article1.semantic_camera import (
    ExternalEvidence, ExternalEvidenceReader, InternalEvidence,
    PROVENANCE, fuse_semantic_camera, grounded_output_to_internal,
    run_semantic_camera,
)
from aria_drive_seg.cli import _load_cfg, build_parser
from aria_drive_seg.config import Config
from aria_drive_seg.io_utils import write_mask_u16
from aria_drive_seg.segmentation.base import FrameOutput
from aria_drive_seg.taxonomy import Taxonomy
from scripts.render_article1_semantic_camera import build_video_frames


def _entropy(probabilities):
    return (
        -(probabilities * np.log(np.maximum(probabilities, 1e-8))).sum(0)
        / np.log(probabilities.shape[0])
    ).astype(np.float32)


def external(mask, confidence=.7):
    mask = np.asarray(mask, np.uint16)
    return ExternalEvidence(
        mask, np.full(mask.shape, confidence, np.float32),
        np.full(mask.shape, .2, np.float32), None, "test_external",
        "test_entropy")


def internal(mask, confidence=.8, kind=1, source="grounded_sam2_cockpit_proxy"):
    mask = np.asarray(mask, np.uint16)
    conf = np.zeros(mask.shape, np.float32)
    conf[mask > 0] = confidence
    probabilities = np.zeros((4, *mask.shape), np.float32)
    probabilities[0] = 1
    active = mask > 0
    rows, cols = np.where(active)
    probabilities[0, rows, cols] = 1 - conf[active]
    probabilities[mask[active], rows, cols] = conf[active]
    evidence = np.zeros(mask.shape, np.uint8)
    evidence[active] = kind
    return InternalEvidence(
        probabilities, mask, conf, _entropy(probabilities), evidence,
        source, "test", source != "segformer_cockpit_reviewed")


@pytest.fixture
def fusion_cfg(project_root):
    cfg = Config.load(project_root / "configs/article1/semantic_camera.yaml")
    return cfg.get("semantic_camera.fusion")


def test_dense_fill_has_no_unknown_and_complete_provenance(fusion_cfg):
    shape = (8, 11)
    result = fuse_semantic_camera(
        external(np.zeros(shape, np.uint16), .1),
        internal(np.zeros(shape, np.uint16)), fusion_cfg)
    assert np.all(result.mask == 13)
    assert np.all(result.provenance == 5)
    assert result.stats["dense_coverage"] == 1
    assert not (result.mask == 0).any()
    assert set(np.unique(result.provenance)).issubset(PROVENANCE)


def test_external_road_scene_dominates_geometric_proxy(fusion_cfg):
    ext = external(np.ones((6, 9), np.uint16), .2)
    proxy = internal(np.full((6, 9), 3, np.uint16), .9, kind=2)
    result = fuse_semantic_camera(ext, proxy, fusion_cfg)
    assert np.all(result.mask == 1)
    assert np.all(np.isin(result.provenance, [1, 6]))


def test_internal_mirror_detection_can_win_ambiguous_vehicle(fusion_cfg):
    ext = external(np.full((6, 9), 4, np.uint16), .35)
    proxy = internal(np.full((6, 9), 1, np.uint16), .80, kind=1)
    result = fuse_semantic_camera(ext, proxy, fusion_cfg)
    assert np.all(result.mask == 10)
    assert np.all(result.conflict == 2)
    assert np.all(result.provenance == 7)


def test_confident_external_safety_class_beats_weak_control(fusion_cfg):
    ext = external(np.full((6, 9), 8, np.uint16), .95)
    proxy = internal(np.full((6, 9), 3, np.uint16), .30, kind=1)
    result = fuse_semantic_camera(ext, proxy, fusion_cfg)
    assert np.all(result.mask == 8)
    assert np.all(result.conflict == 1)
    assert np.all(result.provenance == 6)


def test_geometric_proxy_only_overrides_authorized_ambiguous_classes(fusion_cfg):
    mask = np.asarray([[4, 9, 13, 1, 2, 6]], np.uint16)
    ext = external(mask, .20)
    proxy = internal(np.full(mask.shape, 3, np.uint16), .40, kind=2)
    result = fuse_semantic_camera(ext, proxy, fusion_cfg)
    assert result.mask.tolist()[0][:3] == [12, 12, 12]
    assert result.mask.tolist()[0][3:] == [1, 2, 6]


def test_fusion_does_not_mutate_input_evidence(fusion_cfg):
    ext = external(np.full((5, 7), 4, np.uint16), .4)
    proxy = internal(np.full((5, 7), 1, np.uint16), .8)
    ext_before = ext.mask.copy()
    internal_before = proxy.mask.copy()
    fuse_semantic_camera(ext, proxy, fusion_cfg)
    assert np.array_equal(ext.mask, ext_before)
    assert np.array_equal(proxy.mask, internal_before)


def test_grounded_proxy_rolls_cockpit_classes_and_optional_geometry(project_root):
    tax = Taxonomy.load(project_root / "configs/classes.yaml")
    shape = (10, 12)
    cockpit = np.zeros(shape, np.uint16)
    mirror = np.zeros(shape, np.uint16)
    cockpit[1:3, 1:3] = tax.id_of("instrument_cluster")
    cockpit[5:8, 2:7] = tax.id_of("dashboard")
    mirror[2:5, 8:11] = tax.id_of("rear_view_mirror")
    confidence = np.zeros(shape, np.float32)
    confidence[(cockpit > 0) | (mirror > 0)] = .8
    output = FrameOutput(
        1, 100, "grounded_sam2", shape, np.zeros(shape, np.uint16),
        confidence=confidence, layers={"cockpit": cockpit, "mirror": mirror})
    result = grounded_output_to_internal(
        output, tax, {"enabled": False})
    assert np.all(result.mask[1:3, 1:3] == 2)
    assert np.all(result.mask[5:8, 2:7] == 3)
    assert np.all(result.mask[2:5, 8:11] == 1)
    assert not (result.evidence_kind == 2).any()
    result.validate()


def _synthetic_run(tmp_path, frames=3):
    root = tmp_path / "frames_run"
    ext = tmp_path / "external"
    (root / "frames/rectified").mkdir(parents=True)
    (ext / "static_masks").mkdir(parents=True)
    (ext / "static_confidence").mkdir(parents=True)
    rows = []
    for offset in range(frames):
        frame_id = 100 + offset
        relative = f"frames/rectified/frame_{frame_id:06d}.jpg"
        image = np.full((24, 32, 3), 40 + offset, np.uint8)
        cv2.imwrite(str(root / relative), image)
        rows.append({
            "frame_index": frame_id,
            "capture_timestamp_ns": (offset + 1) * 100_000_000,
            "rectified_path": relative,
            "original_path": relative,
        })
        write_mask_u16(
            ext / "static_masks" / f"frame_{frame_id:06d}.png",
            np.ones((24, 32), np.uint16))
        cv2.imwrite(
            str(ext / "static_confidence" / f"frame_{frame_id:06d}.png"),
            np.full((24, 32), 200, np.uint8))
    pd.DataFrame(rows).to_parquet(
        root / "frames/frames.parquet", index=False)
    (ext / "manifest.json").write_text(json.dumps({
        "static_policy_fingerprint": "test-static"}))
    return root, ext


class FakeInternalProvider:
    source_kind = "segformer_cockpit_reviewed"
    fallback = False

    def descriptor(self):
        return {"kind": self.source_kind, "checkpoint": "synthetic"}

    def infer(self, image_rgb, frame_index, timestamp_ns, vehicle_type):
        mask = np.zeros(image_rgb.shape[:2], np.uint16)
        mask[-4:, :5] = 3
        return internal(
            mask, .95, kind=1, source="segformer_cockpit_reviewed")


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def test_streaming_runner_is_dense_resumable_and_preserves_external(
        tmp_path, project_root):
    root, ext = _synthetic_run(tmp_path)
    output = tmp_path / "semantic"
    cfg = Config.load(project_root / "configs/article1/semantic_camera.yaml")
    source = ext / "static_masks/frame_000100.png"
    source_hash = _sha(source)
    assert run_semantic_camera(
        root, ext, cfg, "car", output, internal_provider=FakeInternalProvider()
    ) == 0
    for frame_id in (100, 101, 102):
        mask = cv2.imread(
            str(output / f"final_masks/frame_{frame_id:06d}.png"),
            cv2.IMREAD_UNCHANGED)
        assert mask.shape == (24, 32)
        assert not (mask == 0).any()
    assert _sha(source) == source_hash
    manifest = json.loads((output / "manifest.json").read_text())
    assert len(manifest["done"]) == 3
    assert manifest["internal_is_fallback"] is False
    summary_path = output / "summary.json"
    summary = json.loads(summary_path.read_text())
    summary["peak_ram_mb"] = 999.0
    summary["peak_vram_mb"] = 888.0
    summary_path.write_text(json.dumps(summary))
    before = (
        output / "final_masks/frame_000102.png").stat().st_mtime_ns
    run_semantic_camera(
        root, ext, cfg, "car", output, resume=True,
        internal_provider=FakeInternalProvider())
    assert (
        output / "final_masks/frame_000102.png").stat().st_mtime_ns == before
    resumed_summary = json.loads(summary_path.read_text())
    assert resumed_summary["peak_ram_mb"] >= 999.0
    assert resumed_summary["peak_vram_mb"] >= 888.0


def test_incompatible_resume_and_missing_external_fail_closed(
        tmp_path, project_root):
    root, ext = _synthetic_run(tmp_path, frames=1)
    output = tmp_path / "semantic"
    cfg = Config.load(project_root / "configs/article1/semantic_camera.yaml")
    run_semantic_camera(
        root, ext, cfg, "car", output,
        internal_provider=FakeInternalProvider())
    changed = Config.load(
        project_root / "configs/article1/semantic_camera.yaml",
        overrides={"semantic_camera": {
            "fusion": {"conflict_margin": .99}}})
    with pytest.raises(RuntimeError, match="fingerprint"):
        run_semantic_camera(
            root, ext, changed, "car", output, resume=True,
            internal_provider=FakeInternalProvider())
    (ext / "static_confidence/frame_000100.png").unlink()
    with pytest.raises(RuntimeError, match="missing external confidence"):
        ExternalEvidenceReader(ext).read(100)


def test_semantic_camera_cli_defaults_to_dedicated_config(project_root):
    args = build_parser().parse_args([
        "article1", "semantic-camera", "--input", "frames",
        "--external", "external", "--vehicle-type", "car",
        "--session-id", "s", "--participant-id", "p",
    ])
    cfg = _load_cfg(args)
    assert cfg.get("semantic_camera.output_subdir") == "semantic_camera"
    assert cfg.get("grounded_sam2.gaze_conditioned") is False


def test_semantic_camera_video_frames_have_codec_aligned_geometry(project_root):
    shape = (24, 32)
    rgb = np.zeros((*shape, 3), np.uint8)
    ext = np.ones(shape, np.uint16)
    inside = np.zeros(shape, np.uint16)
    inside[-5:] = 12
    final = ext.copy()
    final[-5:] = 12
    metadata = {
        "frame_index": 1,
        "capture_timestamp_ns": 100,
        "external_selected_fraction": .8,
        "internal_model_selected_fraction": .1,
        "geometric_proxy_selected_fraction": .1,
        "dense_fill_fraction": 0,
        "conflict_fraction": .1,
        "internal_source": "synthetic",
        "internal_is_fallback": True,
    }
    taxonomy = Taxonomy.load(
        project_root / "configs/article1/classes_article1.yaml")
    dense, comparison = build_video_frames(
        rgb, ext, inside, final, metadata, taxonomy)
    assert dense.shape == (960, 1280, 3)
    assert comparison.shape == (560, 1920, 3)
    assert all(
        value % 8 == 0
        for frame in (dense, comparison) for value in frame.shape[:2])
