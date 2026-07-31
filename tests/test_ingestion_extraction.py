"""Extraction is timestamp-driven, rate agnostic and loses nothing silently."""
from __future__ import annotations

import inspect
from pathlib import Path

import numpy as np
import pytest

from aria_drive_seg.ingestion import extract as extract_module
from aria_drive_seg.ingestion.extract import ExtractionRequest, select_indices
from aria_drive_seg.ingestion.streams import associate_by_timestamp
from aria_drive_seg.ingestion.timeline import NS_PER_S

SOURCE_FILES = [
    Path(extract_module.__file__),
    Path(extract_module.__file__).parent / "timeline.py",
    Path(extract_module.__file__).parent / "segment_select.py",
    Path(extract_module.__file__).parent / "stream_qa.py",
    Path(extract_module.__file__).parent / "rgb_scan.py",
    Path(extract_module.__file__).parent / "compare.py",
    Path(extract_module.__file__).parent / "failure_modes.py",
    Path(extract_module.__file__).parent / "annotation_select.py",
    Path(extract_module.__file__).parent / "hand_audit.py",
    Path(extract_module.__file__).parent / "route_align.py",
    Path(extract_module.__file__).parent / "inventory.py",
]


def regular(fps: float, seconds: float, start: int = 5_000_000_000) -> np.ndarray:
    step = int(round(NS_PER_S / fps))
    return np.arange(start, start + int(seconds * fps) * step, step, dtype=np.int64)


def request(**kw) -> ExtractionRequest:
    base = dict(recording_id="rec", domain="motorcycle", source_file="x.vrs",
                source_file_sha256="deadbeef")
    base.update(kw)
    return ExtractionRequest(**base)


# --------------------------------------------------------------------------- #
# the same request selects the same wall-clock window at any rate
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("fps", [10.0, 15.0, 24.0])
def test_time_window_selects_the_same_duration_at_any_rate(fps):
    ts = regular(fps, 60)
    idx = select_indices(ts, request(start_time_s=10.0, end_time_s=40.0))
    span = (ts[idx[-1]] - ts[idx[0]]) / NS_PER_S
    assert span == pytest.approx(30.0, abs=1.0 / fps)
    # the frame count scales with the rate; the duration does not
    assert len(idx) == pytest.approx(30 * fps, rel=0.05)


def test_absolute_timestamps_reproduce_a_segment_exactly():
    ts = regular(15.0, 60)
    idx = select_indices(ts, request(start_timestamp_ns=int(ts[100]),
                                     end_timestamp_ns=int(ts[549])))
    assert idx[0] == 100 and idx[-1] == 549
    assert len(idx) == 450


def test_absolute_timestamps_take_precedence_over_relative_offsets():
    ts = regular(15.0, 60)
    idx = select_indices(ts, request(start_time_s=30.0,
                                     start_timestamp_ns=int(ts[10]),
                                     end_timestamp_ns=int(ts[20])))
    assert idx[0] == 10


def test_selection_preserves_every_frame_in_the_window():
    ts = regular(15.0, 20)
    idx = select_indices(ts, request(start_time_s=5.0, end_time_s=10.0))
    assert idx == list(range(idx[0], idx[-1] + 1))  # contiguous, nothing dropped


def test_no_decimation_unless_explicitly_requested():
    ts = regular(15.0, 10)
    assert len(select_indices(ts, request())) == ts.size


def test_declared_decimation_never_repeats_a_source_frame():
    ts = regular(15.0, 60)
    idx = select_indices(ts, request(sample_interval_s=1.0))
    assert len(idx) == len(set(idx))
    assert len(idx) == pytest.approx(60, abs=2)


def test_decimation_modes_are_mutually_exclusive():
    with pytest.raises(ValueError):
        select_indices(regular(15.0, 5),
                       request(frame_step=2, sample_interval_s=1.0))


def test_invalid_requests_are_rejected():
    for kw in (dict(frame_step=0), dict(sample_interval_s=0.0),
               dict(start_time_s=10.0, end_time_s=5.0),
               dict(start_timestamp_ns=10, end_timestamp_ns=5)):
        with pytest.raises(ValueError):
            select_indices(regular(15.0, 5), request(**kw))


# --------------------------------------------------------------------------- #
# no hard-coded frame rate anywhere in the ingestion package
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("path", SOURCE_FILES, ids=lambda p: p.name)
def test_no_hard_coded_frame_rate_constant(path):
    """Guard against `fps = 10` / `fps = 15` style constants creeping back in."""
    banned = ("fps = 10", "fps=10", "fps = 15", "fps=15",
              "FPS = 10", "FPS = 15", "/ 10.0)  # fps", "* 10  # frames")
    text = path.read_text()
    for token in banned:
        assert token not in text, f"{path.name} contains a hard-coded rate: {token!r}"


@pytest.mark.parametrize("path", SOURCE_FILES, ids=lambda p: p.name)
def test_no_silent_interpolation_or_synthetic_frames(path):
    text = path.read_text().lower()
    for token in ("cv2.inter_cubic", "np.interp(", "scipy.interpolate"):
        assert token not in text, f"{path.name} interpolates: {token!r}"


def test_extraction_summary_declares_it_did_not_resample():
    src = inspect.getsource(extract_module.run_timestamped_extract)
    for field in ('"resampled": False', '"interpolated": False',
                  '"synthetic_frames": 0', '"declared_decimation"'):
        assert field in src


def test_every_record_carries_full_provenance():
    src = inspect.getsource(extract_module.run_timestamped_extract)
    for field in ("recording_id", "domain", "source_file_sha256", "source_stream_id",
                  "source_frame_index", "timestamp_ns", "timestamp_s",
                  "effective_fps", "width", "height", "valid", "extraction_status"):
        assert f'"{field}"' in src, f"record is missing {field}"


def test_frame_index_is_the_source_index_not_a_local_counter():
    src = inspect.getsource(extract_module.run_timestamped_extract)
    assert '"frame_index": int(i)' in src
    assert '"source_frame_index": int(i)' in src
    assert 'f"frame_{i:06d}.jpg"' in src


# --------------------------------------------------------------------------- #
# timestamp-based association of gaze and hand tracking
# --------------------------------------------------------------------------- #
def test_association_is_by_timestamp_and_keeps_the_whole_window():
    frames = regular(15.0, 2)
    gaze = regular(30.0, 2)
    assoc = associate_by_timestamp(frames, gaze, window_s=0.05)
    assert len(assoc) == frames.size
    # a 30 Hz stream has ~3 samples inside a +/-50 ms window: none are discarded
    assert max(len(a.window_indices) for a in assoc) >= 3
    assert all(a.nearest_dt_ms is not None for a in assoc)


def test_association_window_adapts_to_the_stream_rate():
    frames = regular(15.0, 2)
    narrow = associate_by_timestamp(frames, regular(30.0, 2), window_s=0.02)
    wide = associate_by_timestamp(frames, regular(30.0, 2), window_s=0.20)
    assert sum(len(a.window_indices) for a in wide) > \
        sum(len(a.window_indices) for a in narrow)


def test_association_keeps_the_signed_temporal_distance():
    frames = np.array([1_000_000_000], dtype=np.int64)
    later = np.array([1_000_000_000 + 7_000_000], dtype=np.int64)
    earlier = np.array([1_000_000_000 - 7_000_000], dtype=np.int64)
    assert associate_by_timestamp(frames, later, 0.05)[0].nearest_dt_ms == pytest.approx(7.0)
    assert associate_by_timestamp(frames, earlier, 0.05)[0].nearest_dt_ms == pytest.approx(-7.0)


def test_association_never_assumes_one_to_one():
    frames = regular(15.0, 1)
    sparse = regular(1.0, 1)          # far fewer samples than frames
    assoc = associate_by_timestamp(frames, sparse, window_s=0.05)
    assert len(assoc) == frames.size
    assert any(not a.window_indices for a in assoc)   # some frames have no sample
    assert all(a.nearest_index is not None for a in assoc)  # nearest still recorded


def test_association_handles_an_empty_stream_without_inventing_data():
    assoc = associate_by_timestamp(regular(15.0, 1), np.array([], dtype=np.int64), 0.05)
    assert all(a.nearest_index is None and a.window_indices == [] for a in assoc)


def test_association_rejects_a_negative_window():
    with pytest.raises(ValueError):
        associate_by_timestamp(regular(15.0, 1), regular(30.0, 1), -0.01)
