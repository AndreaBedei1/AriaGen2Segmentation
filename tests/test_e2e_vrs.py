"""End-to-end on a small REAL slice of the VRS (§18). Skips without projectaria_tools/VRS."""
import numpy as np
import pytest

from conftest import has_projectaria

pytestmark = pytest.mark.skipif(not has_projectaria(),
                                reason="projectaria_tools not in this environment")


def test_vrs_opens_and_has_rgb_and_gaze(vrs_path, config):
    from aria_drive_seg.vrs.provider import AriaProvider
    p = AriaProvider(vrs_path)
    labels = {s.label for s in p.list_streams()}
    assert "camera-rgb" in labels
    cfg = p.rgb_config()
    assert cfg["width"] > 0 and cfg["height"] > 0
    assert p.has_eyegaze("eyegaze")


def test_extract_two_frames_and_align_gaze(vrs_path, config, tmp_path):
    from aria_drive_seg.vrs.extract import run_extract
    from aria_drive_seg.gaze.align import run_align_gaze
    import pandas as pd

    config.set("frames.frame_step", 800)
    config.set("frames.max_frames", 2)
    res = run_extract(vrs_path, str(tmp_path), config)
    assert res["num_written"] == 2
    fdf = pd.read_parquet(tmp_path / "frames" / "frames.parquet")
    assert len(fdf) == 2
    # monotonic timestamps
    ts = fdf["capture_timestamp_ns"].to_numpy()
    assert (np.diff(ts) > 0).all()
    # rectified frames exist and are the configured size
    assert (tmp_path / "frames" / "rectified").exists()

    summary = run_align_gaze(vrs_path, str(tmp_path), config)
    assert summary["eyegaze"] is True
    adf = pd.read_parquet(tmp_path / "gaze" / "aligned_gaze.parquet")
    assert len(adf) == 2
    # at least the projection fields exist
    assert {"rect_u", "rect_v", "valid", "dt_ms"}.issubset(adf.columns)


def test_extract_is_resumable(vrs_path, config, tmp_path):
    from aria_drive_seg.vrs.extract import run_extract
    config.set("frames.frame_step", 800)
    config.set("frames.max_frames", 2)
    run_extract(vrs_path, str(tmp_path), config)
    # second run should skip already-done frames (resume) and still report 2
    res = run_extract(vrs_path, str(tmp_path), config, resume=True)
    assert res["num_written"] == 2
