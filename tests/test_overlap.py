"""Overlap resolution + priority rules (§7, §18)."""
import numpy as np

from aria_drive_seg.segmentation.overlap import (Instance, class_aware_nms,
                                                mask_iou, resolve_overlaps)


def _full(h, w):
    return np.ones((h, w), bool)


def test_unassigned_stays_unknown():
    idm, sc = resolve_overlaps(4, 4, [])
    assert (idm == 0).all()


def test_lane_marking_beats_road_regardless_of_order():
    h = w = 10
    road = Instance(_full(h, w), canonical_id=1, score=0.9, priority=10)     # road_surface
    lane = Instance(_full(h, w), canonical_id=2, score=0.3, priority=70)     # lane_marking
    for order in ([road, lane], [lane, road]):
        idm, _ = resolve_overlaps(h, w, order)
        assert (idm == 2).all(), "high-priority lane_marking must win over road even with lower score"


def test_big_background_does_not_cover_small_object():
    h = w = 20
    sky = Instance(_full(h, w), canonical_id=24, score=0.9, priority=5)
    small = np.zeros((h, w), bool)
    small[8:12, 8:12] = True
    person = Instance(small, canonical_id=7, score=0.5, priority=80)
    idm, _ = resolve_overlaps(h, w, [sky, person])
    assert (idm[8:12, 8:12] == 7).all()
    assert idm[0, 0] == 24


def test_equal_priority_higher_score_wins():
    h = w = 6
    a = Instance(_full(h, w), canonical_id=9, score=0.4, priority=40)
    b = Instance(_full(h, w), canonical_id=10, score=0.8, priority=40)
    idm, sc = resolve_overlaps(h, w, [a, b])
    assert (idm == 10).all()
    assert np.allclose(sc, 0.8)


def test_low_conf_does_not_overwrite_high_conf_same_priority():
    h = w = 5
    hi = Instance(_full(h, w), canonical_id=9, score=0.9, priority=40)
    lo = Instance(_full(h, w), canonical_id=10, score=0.2, priority=40)
    idm, _ = resolve_overlaps(h, w, [hi, lo])
    assert (idm == 9).all()


def test_class_aware_nms_suppresses_same_class_overlap():
    boxes = np.array([[0, 0, 10, 10], [1, 1, 11, 11], [50, 50, 60, 60]], float)
    scores = np.array([0.9, 0.8, 0.7])
    cls = np.array([1, 1, 1])
    keep = class_aware_nms(boxes, scores, cls, iou_thr=0.5)
    assert 0 in keep and 2 in keep and 1 not in keep


def test_mask_iou():
    a = np.zeros((4, 4), bool); a[:2] = True
    b = np.zeros((4, 4), bool); b[1:3] = True
    assert abs(mask_iou(a, b) - (4 / 12)) < 1e-6
