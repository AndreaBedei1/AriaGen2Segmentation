"""Hierarchical ROI + multi-scale helpers (Phase 4 & 5 / §18)."""
import numpy as np

from aria_drive_seg.segmentation import roi as R


def test_derive_rois_from_detections():
    dets = [
        {"name": "windshield", "box": (400, 200, 1600, 900)},
        {"name": "rear_view_mirror", "box": (1500, 300, 1700, 450)},
        {"name": "dashboard", "box": (300, 900, 1700, 1200)},
        {"name": "car", "box": (800, 500, 950, 650)},  # not structural
    ]
    rois = R.derive_rois(dets, h=1512, w=2016, pad_frac=0.0)
    assert "windshield_view" in rois and "rear_view_mirror_zone" in rois
    assert "dashboard_zone" in rois
    assert rois["windshield_view"] == (400, 200, 1600, 900)


def test_roi_class_subset():
    avail = {"traffic_light", "pedestrian", "car", "dashboard", "infotainment_screen"}
    ws = R.roi_class_subset("windshield_view", avail)
    assert "traffic_light" in ws and "pedestrian" in ws and "dashboard" not in ws
    mir = R.roi_class_subset("rear_view_mirror_zone", avail)
    assert "car" in mir
    console = R.roi_class_subset("center_console_zone", avail)
    assert "infotainment_screen" in console


def test_tiles_cover_box_with_overlap():
    box = (0, 0, 300, 100)
    tiles = R.tiles_of(box, 3, 0.2)
    assert len(tiles) == 3
    # tiles collectively span the full width
    assert tiles[0][0] == 0 and tiles[-1][2] == 300
    # adjacent tiles overlap
    assert tiles[1][0] < tiles[0][2]


def test_remap_box_offset_scale():
    # crop at offset (100,50) upscaled 2x: a box at (20,10,40,30) in crop -> full
    fb = R.remap_box((20, 10, 40, 30), (100, 50), 2.0)
    assert fb == (110.0, 55.0, 120.0, 65.0)


def test_paste_mask_places_and_downscales():
    crop = np.zeros((20, 20), bool); crop[:, :] = True  # 20x20 all true (2x upscaled 10x10)
    full = R.paste_mask(crop, (5, 5), full_h=40, full_w=40, scale=2.0)
    assert full.shape == (40, 40)
    # 20x20 crop at scale 2 -> 10x10 region at (5,5)
    assert full[5:15, 5:15].all()
    assert not full[0, 0]
    assert not full[20, 20]


def test_roistate_snaps_on_large_move_and_smooths_small():
    st = R.ROIState(alpha=0.5, move_thresh_frac=0.1)
    r0 = st.update({"windshield_view": (100, 100, 200, 200)}, 1000, 1000)
    assert r0["windshield_view"] == (100, 100, 200, 200)
    # small move -> smoothed (EMA toward new)
    r1 = st.update({"windshield_view": (110, 100, 210, 200)}, 1000, 1000)
    assert 100 < r1["windshield_view"][0] < 110
    # large move (>10% of width) -> snap
    r2 = st.update({"windshield_view": (500, 100, 600, 200)}, 1000, 1000)
    assert r2["windshield_view"][0] >= 500 - 1
