import numpy as np

from aria_drive_seg.article1.render import _hud


def test_external_video_panel_geometry_is_codec_aligned():
    image = np.zeros((24, 32, 3), np.uint8)
    mask = np.zeros((24, 32), np.uint16)
    panel = _hud(image, "test", 1, 0.0, {0: "unknown"}, mask, ["ok"])
    assert panel.shape == (24, 448, 3)
    assert panel.shape[1] % 8 == 0
