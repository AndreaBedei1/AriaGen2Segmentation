import pandas as pd
import pytest

from aria_drive_seg.article1.splits import (
    assert_no_group_leakage,
    assert_no_near_frame_leakage,
    grouped_split,
)


def test_grouped_split_has_no_session_leakage():
    df = pd.DataFrame({"session_id": ["a", "a", "b", "b"],
                       "frame_index": [1, 2, 1, 2],
                       "vehicle_type": ["car", "car", "motorcycle", "motorcycle"]})
    train, valid = grouped_split(df, "session_id", ["b"])
    assert set(train.session_id) == {"a"}
    assert set(valid.session_id) == {"b"}


def test_leakage_detection_fails_closed():
    train = pd.DataFrame({"session_id": ["a"], "frame_index": [10]})
    valid = pd.DataFrame({"session_id": ["a"], "frame_index": [12]})
    with pytest.raises(ValueError, match="leakage"):
        assert_no_group_leakage(train, valid, "session_id")
    with pytest.raises(ValueError, match="near-frame"):
        assert_no_near_frame_leakage(train, valid, min_frame_distance=5)
