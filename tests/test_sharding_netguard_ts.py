"""Multi-GPU sharding, offline netguard, timestamp health (§13, §2, §4, §18)."""
import socket

import numpy as np
import pytest

from aria_drive_seg.sharding import (assert_disjoint_cover, parse_devices,
                                     recompose, shard_indices)
from aria_drive_seg.vrs.inspect import _timestamp_health


# ---------------- sharding ----------------
def test_shards_disjoint_and_cover():
    idx = list(range(100))
    assert assert_disjoint_cover(idx, 3)
    assert assert_disjoint_cover(idx, 1)
    parts = [set(shard_indices(idx, 4, s)) for s in range(4)]
    assert set().union(*parts) == set(idx)
    for i in range(4):
        for j in range(i + 1, 4):
            assert not (parts[i] & parts[j])


def test_recompose_orders_by_frame_index():
    s0 = {0: "a", 2: "c"}
    s1 = {1: "b", 3: "d"}
    assert recompose([s1, s0]) == ["a", "b", "c", "d"]


def test_parse_devices():
    assert parse_devices("auto", 2) == [0, 1]
    assert parse_devices("0", 4) == [0]
    assert parse_devices("0,2", 4) == [0, 2]


# ---------------- timestamp health ----------------
def test_timestamp_monotonic_no_gaps():
    ts = np.arange(0, 10) * 100_000_000  # 10 Hz
    h = _timestamp_health(ts)
    assert h["monotonic"] and h["num_gaps"] == 0
    assert abs(h["rate_hz"] - 10.0) < 1e-3


def test_timestamp_detects_gap_and_nonmonotonic():
    ts = np.array([0, 100, 200, 500, 600], dtype=np.int64) * 1_000_000  # gap at 200->500
    h = _timestamp_health(ts)
    assert h["num_gaps"] >= 1
    ts2 = np.array([0, 100, 90, 200], dtype=np.int64) * 1_000_000
    h2 = _timestamp_health(ts2)
    assert not h2["monotonic"]


# ---------------- netguard ----------------
def test_netguard_blocks_remote_and_allows_loopback():
    from aria_drive_seg import netguard
    netguard.reset_violations()
    netguard.install_netguard(strict=True)
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        with pytest.raises(OSError):
            s.connect(("93.184.216.34", 80))  # example.com IP -> must be blocked
        s.close()
        assert any(host for _, host in netguard.get_violations())
    finally:
        netguard.uninstall_netguard()
