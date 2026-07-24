"""RGB<->gaze association, interpolation, dt gating, out-of-image (§9, §18).

Uses a fake projector + fake EyeGaze so the temporal/logic paths are tested without
projectaria_tools or a GPU."""
from dataclasses import dataclass

import numpy as np

from aria_drive_seg.gaze.align import _align_one, _lerp
from aria_drive_seg.gaze.project import GazePixel


@dataclass
class FakeEG:
    yaw: float = 0.0
    pitch: float = 0.0
    depth: float = 5.0
    combined_gaze_valid: bool = True


class FakeProjector:
    fallback_depth_m = 8.0

    def __init__(self, in_image=True):
        self.in_image = in_image

    def depth_of(self, eg):
        return (eg.depth, "device") if eg.depth > 0 else (self.fallback_depth_m, "fallback")

    def reproject_official(self, eg, depth, which):
        return GazePixel(1000.0, 700.0, self.in_image)

    def reproject_yawpitch(self, yaw, pitch, depth, which):
        return GazePixel(1000.0, 700.0, self.in_image)


def _mk(eg_ts, gazes):
    ts = np.array(eg_ts, dtype=np.int64)
    return ts, (lambda i: gazes[i])


def test_nearest_valid_within_dt():
    ts, get = _mk([0, 10_000_000, 20_000_000], [FakeEG(), FakeEG(), FakeEG()])
    rec = _align_one(0, 9_000_000, ts, get, FakeProjector(), max_dt_ns=20e6, do_interp=False)
    assert rec["valid"] and rec["validity_reason"] == "ok"
    assert abs(rec["dt_ms"] - 1.0) < 1e-6  # nearest is 10ms sample, 1ms away


def test_far_dt_marked_invalid():
    ts, get = _mk([0, 100_000_000], [FakeEG(), FakeEG()])
    rec = _align_one(0, 50_000_000, ts, get, FakeProjector(), max_dt_ns=20e6, do_interp=False)
    assert not rec["valid"]
    assert "far_dt" in rec["validity_reason"]


def test_invalid_flag_propagates():
    ts, get = _mk([0, 10_000_000], [FakeEG(combined_gaze_valid=False),
                                    FakeEG(combined_gaze_valid=False)])
    rec = _align_one(0, 1_000_000, ts, get, FakeProjector(), max_dt_ns=20e6, do_interp=False)
    assert not rec["valid"]
    assert "invalid_flag" in rec["validity_reason"]


def test_interpolation_used_when_bracketing_valid():
    ts, get = _mk([0, 10_000_000], [FakeEG(yaw=0.0), FakeEG(yaw=0.2)])
    rec = _align_one(0, 5_000_000, ts, get, FakeProjector(), max_dt_ns=20e6, do_interp=True)
    assert rec["interpolated"] is True
    assert abs(rec["yaw"] - 0.1) < 1e-6  # midpoint


def test_out_of_image_not_valid():
    ts, get = _mk([0, 10_000_000], [FakeEG(), FakeEG()])
    rec = _align_one(0, 1_000_000, ts, get, FakeProjector(in_image=False),
                     max_dt_ns=20e6, do_interp=False)
    assert not rec["valid"]
    assert rec["validity_reason"] == "out_of_image"


def test_fallback_depth_when_invalid_depth():
    ts, get = _mk([0, 10_000_000], [FakeEG(depth=0.0), FakeEG(depth=0.0)])
    rec = _align_one(0, 1_000_000, ts, get, FakeProjector(), max_dt_ns=20e6, do_interp=False)
    assert rec["depth_source"] == "fallback"
    assert rec["depth"] == 8.0


def test_lerp():
    assert _lerp(0, 10, 0.3) == 3.0
