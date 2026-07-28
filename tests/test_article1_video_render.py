import numpy as np

from aria_drive_seg.article1.render import _hud
from aria_drive_seg.taxonomy import Taxonomy
from scripts.render_article1_temporal import build_frames


def test_external_video_panel_geometry_is_codec_aligned():
    image = np.zeros((24, 32, 3), np.uint8)
    mask = np.zeros((24, 32), np.uint16)
    panel = _hud(image, "test", 1, 0.0, {0: "unknown"}, mask, ["ok"])
    assert panel.shape == (24, 448, 3)
    assert panel.shape[1] % 8 == 0


def test_temporal_video_frames_have_stable_codec_geometry():
    image = np.zeros((24, 32, 3), np.uint8)
    mask = np.ones((24, 32), np.uint16)
    zero = np.zeros((24, 32), np.uint8)
    metadata = {
        "frame_index": 1, "capture_timestamp_ns": 10,
        "flow_valid_fraction": 1.0, "propagated_pixels": 0,
        "mean_propagation_age": 0.0, "reset_reason": "initial_frame",
    }
    taxonomy = Taxonomy.load("configs/article1/classes_article1.yaml")
    frames = build_frames(
        image, mask, mask, mask, mask, np.full_like(zero, 255), zero,
        zero, zero, zero, np.full_like(zero, 255), np.full_like(zero, 255),
        zero, metadata, None, taxonomy)
    assert len(frames) == 8
    assert frames[0].shape == frames[1].shape == (756, 1288, 3)
    assert all(frame.shape[0] % 4 == 0 and frame.shape[1] % 8 == 0
               for frame in frames)
