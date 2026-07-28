from pathlib import Path
import inspect

import cv2
import json
import numpy as np
import pandas as pd
import pytest

from aria_drive_seg.article1.optical_flow import (
    FlowResult, compute_optical_flow, validate_flow, warp_with_backward)
from aria_drive_seg.article1.temporal import (
    PROVENANCE, reset_reason_for, stabilize_frame)
from aria_drive_seg.article1.temporal_state import TemporalState
from aria_drive_seg.article1.temporal_pipeline import run_temporal
from aria_drive_seg.config import Config


NAMES = [
    "unknown", "road_surface", "lane_marking", "regulatory_road_marking",
    "vehicle", "two_wheeler", "pedestrian", "traffic_light", "traffic_sign",
    "road_boundary_or_obstacle", "mirror", "instrument_display",
    "control_and_ego_vehicle", "other_environment",
]


def cfg():
    return {
        "reset": {"max_frame_gap_s": .25, "max_invalid_flow_fraction": .60,
                  "max_photometric_error": .15, "max_median_flow_px": 80},
        "fusion": {"current_weight": 1, "previous_weight": .65,
                   "minimum_previous_confidence": .55,
                   "uncertain_current_threshold": .55,
                   "uncertain_current_margin": .08, "temporal_decay": .85},
        "hysteresis": {"enabled": True, "switch_margin": .10,
                       "minimum_confirmation_frames": 2,
                       "immediate_switch_confidence": .80,
                       "previous_drop_threshold": .25},
        "classes": {
            "unknown": {"ttl_frames": 0, "previous_weight": 0},
            "road_surface": {"ttl_frames": 3, "previous_weight": .55},
            "lane_marking": {"ttl_frames": 3, "previous_weight": .75},
            "regulatory_road_marking": {"ttl_frames": 2, "previous_weight": .65},
            "vehicle": {"ttl_frames": 1, "previous_weight": .35,
                        "require_current_support": True},
            "pedestrian": {"ttl_frames": 1, "previous_weight": .2,
                           "require_current_support": True},
        },
        "thin_markings": {"enabled": True, "lane_ttl_frames": 3,
                          "regulatory_ttl_frames": 2,
                          "minimum_previous_confidence": .55},
    }


def one_hot(cid, shape=(8, 9), confidence=1.0):
    p = np.zeros((14, *shape), np.float32)
    p[cid] = confidence
    if confidence < 1:
        p[0] = 1 - confidence
    return p


def valid_flow(shape=(8, 9), dx=0, dy=0):
    h, w = shape
    backward = np.zeros((h, w, 2), np.float32)
    backward[..., 0], backward[..., 1] = -dx, -dy
    return FlowResult(
        -backward, backward, np.ones(shape, bool), np.zeros(shape, bool),
        np.zeros(shape, np.float32), 1.0, float(np.hypot(dx, dy)), 0.0)


def initial(cid=1, thin_id=0, shape=(8, 9)):
    p = one_hot(cid, shape)
    thin = np.full(shape, thin_id, np.uint16)
    rgb = np.zeros((*shape, 3), np.uint8)
    return stabilize_frame(
        p, thin, np.ones(shape, bool), rgb, 1, 100_000_000,
        NAMES, cfg(), "c", "s", None, None)


def next_frame(state, p, frame=2, timestamp=200_000_000, flow=None,
               thin=None, mode="T4", road=None):
    shape = p.shape[1:]
    return stabilize_frame(
        p, np.zeros(shape, np.uint16) if thin is None else thin,
        np.ones(shape, bool) if road is None else road,
        np.zeros((*shape, 3), np.uint8), frame, timestamp, NAMES, cfg(),
        "c", "s", state, flow or valid_flow(shape), mode)


def test_identity_flow_and_probability_normalization_are_deterministic():
    first = initial(1)
    current = one_hot(0)
    a = next_frame(first.state, current)
    b = next_frame(first.state, current)
    assert np.array_equal(a.mask, b.mask)
    assert np.array_equal(a.provenance, b.provenance)
    assert np.allclose(a.probabilities.sum(0), 1)


@pytest.mark.parametrize("dx,dy", [(2, 0), (0, 2)])
def test_known_translation_warp(dx, dy):
    previous = np.zeros((7, 8), np.uint8)
    previous[2, 2] = 9
    backward = np.zeros((7, 8, 2), np.float32)
    backward[..., 0], backward[..., 1] = -dx, -dy
    warped = warp_with_backward(previous, backward, cv2.INTER_NEAREST)
    assert warped[2 + dy, 2 + dx] == 9


def test_forward_backward_inconsistency_and_occlusion():
    image = np.zeros((10, 10, 3), np.uint8)
    forward = np.zeros((10, 10, 2), np.float32)
    backward = np.zeros_like(forward)
    backward[..., 0] = 3
    result = validate_flow(image, image, forward, backward, {
        "use_forward_backward_check": True,
        "forward_backward_threshold_px": 1,
        "photometric_error_threshold": 1,
    })
    assert not result.valid.any()
    assert result.occlusion.all()


def test_dis_identity_flow():
    rng = np.random.default_rng(4)
    image = rng.integers(0, 255, (40, 50, 3), np.uint8)
    result = compute_optical_flow(image, image, {
        "backend": "opencv_dis", "processing_scale": .5,
        "grayscale": True, "use_forward_backward_check": True,
        "forward_backward_threshold_px": 1.5,
        "photometric_error_threshold": .15,
    })
    assert result.valid_fraction > .9
    assert result.median_flow_px < .1


@pytest.mark.parametrize("change,expected", [
    ({"frame_index": 3}, 2),
    ({"timestamp_ns": 400_000_000}, 5),
])
def test_missing_frame_and_timestamp_gap_reset(change, expected):
    state = initial().state
    assert reset_reason_for(
        state, change.get("frame_index", 2),
        change.get("timestamp_ns", 200_000_000), state.mask.shape,
        valid_flow(state.mask.shape), cfg(), "c", "s") == expected


def test_resolution_change_scene_cut_and_invalid_flow_reset():
    state = initial().state
    assert reset_reason_for(
        state, 2, 200_000_000, (9, 9), None, cfg(), "c", "s") == 3
    bad = valid_flow(state.mask.shape)
    bad.valid[:] = False
    bad.valid_fraction = 0
    assert reset_reason_for(
        state, 2, 200_000_000, state.mask.shape, bad, cfg(), "c", "s") == 7
    bad.valid[:] = True
    bad.valid_fraction = 1
    bad.mean_photometric_error = .8
    assert reset_reason_for(
        state, 2, 200_000_000, state.mask.shape, bad, cfg(), "c", "s") == 8


def test_previous_fills_unknown_but_confident_current_wins():
    first = initial(1)
    recovered = next_frame(first.state, one_hot(0))
    assert np.all(recovered.mask == 1)
    assert recovered.unknown_recovered.all()
    confident = next_frame(first.state, one_hot(4))
    assert np.all(confident.mask == 4)
    assert not confident.unknown_recovered.any()


def test_ttl_expiration_and_propagation_age():
    first = initial(1)
    a = next_frame(first.state, one_hot(0), frame=2)
    b = next_frame(a.state, one_hot(0), frame=3, timestamp=300_000_000)
    c = next_frame(b.state, one_hot(0), frame=4, timestamp=400_000_000)
    assert a.propagation_age.max() == 1
    assert b.propagation_age.max() == 2
    assert np.all(c.mask == 0)


def test_hysteresis_hold_then_confirmation_and_immediate_switch():
    first = initial(1)
    weak = one_hot(4, confidence=.55)
    a = next_frame(first.state, weak)
    assert np.all(a.mask == 1)
    b = next_frame(a.state, weak, frame=3, timestamp=300_000_000)
    assert np.all(b.mask == 4)
    immediate = next_frame(first.state, one_hot(4, confidence=.9))
    assert np.all(immediate.mask == 4)


def test_lane_one_frame_recovery_and_ttl_expiration():
    first = initial(1, thin_id=2)
    a = next_frame(first.state, one_hot(1))
    assert np.all(a.thin_mask == 2)
    assert np.all(a.provenance == 5)
    b = next_frame(a.state, one_hot(1), frame=3, timestamp=300_000_000)
    c = next_frame(b.state, one_hot(1), frame=4, timestamp=400_000_000)
    d = next_frame(c.state, one_hot(1), frame=5, timestamp=500_000_000)
    assert np.all(b.thin_mask == 2)
    assert np.all(c.thin_mask == 2)
    assert not d.thin_mask.any()


def test_regulatory_ttl_expires_before_lane():
    first = initial(1, thin_id=3)
    a = next_frame(first.state, one_hot(1))
    b = next_frame(a.state, one_hot(1), frame=3, timestamp=300_000_000)
    c = next_frame(b.state, one_hot(1), frame=4, timestamp=400_000_000)
    assert np.all(a.thin_mask == 3)
    assert np.all(b.thin_mask == 3)
    assert not c.thin_mask.any()


@pytest.mark.parametrize("cid", [4, 6])
def test_dynamic_classes_require_current_support(cid):
    first = initial(cid)
    current = one_hot(1)
    result = next_frame(first.state, current)
    assert np.all(result.mask == 1)


def test_raw_input_is_never_mutated_and_provenance_is_explicit():
    first = initial(1)
    current = one_hot(0)
    original = current.copy()
    result = next_frame(first.state, current)
    assert np.array_equal(current, original)
    assert set(np.unique(result.provenance)).issubset(PROVENANCE)


def test_resume_state_and_incompatible_failure(tmp_path):
    state = initial(2).state
    path = tmp_path / "state.npz"
    state.save(path)
    loaded = TemporalState.load(path, "c", "s")
    assert np.array_equal(loaded.mask, state.mask)
    with pytest.raises(RuntimeError, match="config fingerprint"):
        TemporalState.load(path, "different", "s")


def test_primary_segmentation_has_no_future_or_gaze_input():
    import aria_drive_seg.article1.temporal as module
    source = inspect.getsource(module)
    forbidden = ("gaze", "frame_t_plus_1", "future_frame", "bidirectional")
    assert not any(token in source.lower() for token in forbidden)
    signature = inspect.signature(stabilize_frame)
    assert "next_frame" not in signature.parameters


def _synthetic_replay(tmp_path, frames=3):
    root, static = tmp_path / "run", tmp_path / "static"
    (root / "frames" / "rectified").mkdir(parents=True)
    rows = []
    for index in range(frames):
        frame_id = 100 + index
        relative = f"frames/rectified/frame_{frame_id:06d}.jpg"
        image = np.zeros((24, 32, 3), np.uint8)
        image[:, 2 + index:6 + index] = 180
        cv2.imwrite(str(root / relative), image)
        rows.append({
            "frame_index": frame_id,
            "capture_timestamp_ns": (index + 1) * 100_000_000,
            "rectified_path": relative, "original_path": relative,
        })
        stem = f"frame_{frame_id:06d}"
        for directory in ("probabilities", "base_masks",
                          "filtered_thin_masks", "masks"):
            (static / directory).mkdir(parents=True, exist_ok=True)
        probability = one_hot(1, (24, 32))
        with open(static / "probabilities" / f"{stem}.npz", "wb") as handle:
            np.savez_compressed(handle, probabilities=probability.astype(np.float16))
        cv2.imwrite(str(static / "base_masks" / f"{stem}.png"),
                    np.ones((24, 32), np.uint16))
        cv2.imwrite(str(static / "filtered_thin_masks" / f"{stem}.png"),
                    np.zeros((24, 32), np.uint16))
        cv2.imwrite(str(static / "masks" / f"{stem}.png"),
                    np.ones((24, 32), np.uint16))
    pd.DataFrame(rows).to_parquet(root / "frames" / "frames.parquet", index=False)
    config = Config.load(
        "configs/article1/temporal_segmentation.yaml",
        overrides={"temporal": {
            "processing_scale": .5, "state_checkpoint_interval": 2,
            "experiment_modes": ["T0", "T4"], "diagnostic_outputs": False,
        }})
    return root, static, config


def test_streaming_manifest_raw_preservation_and_resume(tmp_path):
    root, static, config = _synthetic_replay(tmp_path)
    assert run_temporal(root, config, static_output=str(static)) == 0
    out = root / "article1_temporal"
    manifest = json.loads((out / "manifest.json").read_text())
    assert manifest["causal"] is True
    assert manifest["gaze_assisted"] is False
    assert manifest["frame_count"] == 3
    assert manifest["processed_frame_indices"] == [100, 101, 102]
    raw = cv2.imread(str(out / "static_masks/frame_000100.png"),
                     cv2.IMREAD_UNCHANGED)
    assert np.all(raw == 1)
    before = (out / "temporal_masks/frame_000102.png").stat().st_mtime_ns
    assert run_temporal(root, config, static_output=str(static), resume=True) == 0
    assert (out / "temporal_masks/frame_000102.png").stat().st_mtime_ns == before


def test_replay_missing_static_probability_fails_closed(tmp_path):
    root, static, config = _synthetic_replay(tmp_path, frames=1)
    (static / "probabilities/frame_000100.npz").unlink()
    with pytest.raises(RuntimeError, match="missing static probability"):
        run_temporal(root, config, static_output=str(static))


def test_resume_rejects_missing_intermediate_frame(tmp_path):
    root, static, config = _synthetic_replay(tmp_path)
    run_temporal(root, config, static_output=str(static))
    manifest_path = root / "article1_temporal/manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["processed_frame_indices"] = [100, 102]
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(RuntimeError, match="missing or non-contiguous"):
        run_temporal(root, config, static_output=str(static), resume=True)


def test_streaming_runner_never_indexes_a_future_frame_or_reads_gaze_coordinates():
    import aria_drive_seg.article1.temporal_pipeline as pipeline
    source = inspect.getsource(pipeline).lower()
    assert "frames[position + 1]" not in source
    assert "rect_u" not in source and "rect_v" not in source
