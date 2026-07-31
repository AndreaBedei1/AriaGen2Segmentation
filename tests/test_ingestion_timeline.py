"""Frame-rate independence, gap detection and derived-timeline guarantees."""
from __future__ import annotations

import numpy as np
import pytest

from aria_drive_seg.ingestion.timeline import (NS_PER_S, build_comparable_timeline,
                                               continuous_segments, find_gaps,
                                               frames_for_seconds, measure_rate,
                                               nearest_indices, per_second,
                                               synchronisation_report)


def regular(fps: float, seconds: float, start: int = 1_000_000_000) -> np.ndarray:
    step = int(round(NS_PER_S / fps))
    return np.arange(start, start + int(seconds * fps) * step, step, dtype=np.int64)


# --------------------------------------------------------------------------- #
# effective rate is measured, never assumed
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("fps", [10.0, 15.0, 20.0, 30.0, 7.5])
def test_effective_fps_is_measured_from_timestamps(fps):
    rate = measure_rate(regular(fps, 20))
    assert rate.effective_fps == pytest.approx(fps, rel=1e-6)
    assert rate.effective_fps_span == pytest.approx(fps, rel=1e-2)
    assert rate.non_monotonic_count == 0


def test_effective_fps_ignores_a_pause():
    """A pause must be reported as a gap, not smeared into the rate."""
    a = regular(15.0, 10)
    b = a[-1] + 3 * NS_PER_S + np.arange(150, dtype=np.int64) * int(NS_PER_S / 15)
    rate = measure_rate(np.concatenate([a, b]))
    assert rate.effective_fps == pytest.approx(15.0, rel=1e-6)
    # the span rate is dragged down by the pause: that is why it is not the primary
    assert rate.effective_fps_span < 14.0


def test_measure_rate_handles_degenerate_streams():
    assert measure_rate(np.array([], dtype=np.int64)).count == 0
    assert measure_rate(np.array([5], dtype=np.int64)).effective_fps is None


def test_duplicate_and_non_monotonic_timestamps_are_counted():
    ts = np.array([0, 100, 100, 300, 200, 400], dtype=np.int64)
    rate = measure_rate(ts)
    assert rate.duplicate_timestamp_count == 1
    assert rate.non_monotonic_count == 2  # the repeat and the backwards step


# --------------------------------------------------------------------------- #
# temporal windows are declared in seconds
# --------------------------------------------------------------------------- #
def test_window_in_seconds_yields_different_frame_counts_per_rate():
    assert frames_for_seconds(1.0, 10.0) == 10
    assert frames_for_seconds(1.0, 15.0) == 15
    assert frames_for_seconds(2.0, 15.0) == 30


def test_window_never_collapses_to_zero_frames():
    assert frames_for_seconds(0.01, 10.0) == 1


def test_window_rejects_an_unusable_rate():
    for bad in (0.0, -5.0, float("nan"), float("inf")):
        with pytest.raises(ValueError):
            frames_for_seconds(1.0, bad)


def test_per_second_normalisation_makes_rates_comparable():
    """30 events over 20 frames at 10 fps and 30 frames at 15 fps are the same rate."""
    assert per_second(30, 2.0) == per_second(30, 2.0)
    car = per_second(20, 2.0)     # 20 events in 2 s at 10 fps
    moto = per_second(30, 3.0)    # 30 events in 3 s at 15 fps
    assert car == pytest.approx(moto)


def test_per_second_rejects_a_zero_duration():
    assert per_second(5, 0.0) is None


# --------------------------------------------------------------------------- #
# gaps and segments
# --------------------------------------------------------------------------- #
def test_gap_thresholds_are_relative_to_the_streams_own_period():
    ts = regular(15.0, 5)
    # a single dropped frame -> one interval of two periods
    ts = np.delete(ts, 30)
    gaps = find_gaps(ts)
    assert len(gaps) == 1
    assert gaps[0].over_1p5_periods
    assert gaps[0].periods == pytest.approx(2.0)
    assert gaps[0].missing_estimate == 1
    assert not gaps[0].over_250ms  # 133 ms at 15 fps is not an absolute pause


def test_a_regular_slow_stream_reports_no_gap():
    """A 1 Hz GPS samples every 1000 ms by design; that is not 4000 pauses."""
    ts = (np.arange(60, dtype=np.int64) * NS_PER_S)
    assert find_gaps(ts) == []
    assert len(continuous_segments(ts)) == 1


def test_a_pause_in_a_slow_stream_is_still_flagged_by_its_own_cadence():
    ts = np.array([0, NS_PER_S, 2 * NS_PER_S, 4 * NS_PER_S], dtype=np.int64)
    gaps = find_gaps(ts)
    assert len(gaps) == 1
    assert gaps[0].periods == pytest.approx(2.0)
    # the absolute 250 ms rule does not apply to a stream whose period is 1000 ms
    assert not gaps[0].over_250ms


def test_absolute_pause_is_flagged_for_a_fast_stream():
    ts = np.concatenate([regular(15.0, 2),
                         regular(15.0, 2)[-1] + NS_PER_S
                         + np.arange(30, dtype=np.int64) * int(NS_PER_S / 15)])
    assert any(g.over_250ms for g in find_gaps(ts))


def test_continuous_segments_split_on_a_pause_and_keep_indices():
    a = regular(15.0, 4)
    b = a[-1] + 2 * NS_PER_S + np.arange(60, dtype=np.int64) * int(NS_PER_S / 15)
    segs = continuous_segments(np.concatenate([a, b]))
    assert len(segs) == 2
    assert segs[0].start_index == 0
    assert segs[1].start_index == a.size
    assert segs[1].count == 60


def test_a_clean_stream_is_one_segment_with_no_gaps():
    ts = regular(15.0, 30)
    assert len(continuous_segments(ts)) == 1
    assert find_gaps(ts) == []


# --------------------------------------------------------------------------- #
# cross-stream synchronisation
# --------------------------------------------------------------------------- #
def test_nearest_indices_pick_the_closest_sample_on_either_side():
    target = np.array([0, 100, 200, 300], dtype=np.int64)
    assert list(nearest_indices(np.array([149, 151]), target)) == [1, 2]


def test_synchronisation_report_measures_distance_to_rgb():
    rgb = regular(15.0, 5)
    gaze = regular(30.0, 5)
    rep = synchronisation_report(rgb, gaze)
    assert rep["available"]
    assert rep["abs_dt_to_reference_ms"]["max"] < 34.0


# --------------------------------------------------------------------------- #
# derived comparable timeline: nearest real sample, never invented data
# --------------------------------------------------------------------------- #
def test_comparable_timeline_only_selects_real_samples():
    ts = regular(15.0, 10)
    tl = build_comparable_timeline(ts, grid_hz=10.0)
    assert tl.derived and tl.preliminary
    assert not tl.interpolated
    assert tl.synthetic_frames == 0
    assert set(tl.selected_ns).issubset(set(int(x) for x in ts))
    assert len(tl.selected_indices) == len(tl.grid_ns) == len(tl.temporal_error_ms)


def test_comparable_timeline_reports_duplicates_instead_of_hiding_them():
    """Snapping a 10 fps stream onto a 15 Hz grid must reuse frames, and say so."""
    ts = regular(10.0, 6)
    tl = build_comparable_timeline(ts, grid_hz=15.0)
    assert tl.duplicate_selection_count > 0
    assert any("never silently duplicated" in n for n in tl.notes)


def test_comparable_timeline_bounds_the_temporal_error():
    ts = regular(15.0, 10)
    tl = build_comparable_timeline(ts, grid_hz=10.0)
    # nearest neighbour on a 15 fps stream is never farther than half a period
    assert tl.max_temporal_error_ms <= (1000.0 / 15.0) / 2 + 1e-6


def test_comparable_timeline_drops_rather_than_fills_a_hole():
    ts = np.concatenate([regular(15.0, 3),
                         regular(15.0, 3)[-1] + 5 * NS_PER_S
                         + np.arange(45, dtype=np.int64) * int(NS_PER_S / 15)])
    tl = build_comparable_timeline(ts, grid_hz=10.0, max_temporal_error_ms=60.0)
    assert any("not filled by design" in n for n in tl.notes)
    assert tl.max_temporal_error_ms <= 60.0


def test_comparable_timeline_rejects_an_invalid_grid():
    with pytest.raises(ValueError):
        build_comparable_timeline(regular(15.0, 2), grid_hz=0.0)
    with pytest.raises(ValueError):
        build_comparable_timeline(np.array([], dtype=np.int64), grid_hz=10.0)
