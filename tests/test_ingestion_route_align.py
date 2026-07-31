"""Route pairing is spatial, direction-aware and never forced."""
from __future__ import annotations

import inspect

import numpy as np
import pytest

from aria_drive_seg.ingestion import route_align
from aria_drive_seg.ingestion.route_align import (angular_difference_deg, bearing_deg,
                                                  build_track, detect_route_events,
                                                  haversine_m, pair_tracks,
                                                  progression_of_timestamp)

NS = 1_000_000_000


def straight_track(recording_id: str, domain: str, n: int = 60,
                   lat0: float = 44.1400, lon0: float = 12.2200,
                   dlat: float = 0.0002, start_ns: int = 0, dt_ns: int = NS,
                   speed: float = 12.0):
    lat = [lat0 + i * dlat for i in range(n)]
    lon = [lon0] * n
    ts = [start_ns + i * dt_ns for i in range(n)]
    return build_track(recording_id, domain, ts, lat, lon,
                       [3.0] * n, [speed] * n)


def test_haversine_and_bearing_are_sane():
    # one degree of latitude is about 111 km
    assert haversine_m(0.0, 0.0, 1.0, 0.0) == pytest.approx(111_195, rel=0.01)
    assert bearing_deg(44.0, 12.0, 45.0, 12.0) == pytest.approx(0.0, abs=0.1)
    assert bearing_deg(44.0, 12.0, 44.0, 13.0) == pytest.approx(90.0, abs=0.5)


def test_angular_difference_wraps():
    assert angular_difference_deg(350.0, 10.0) == pytest.approx(20.0)
    assert angular_difference_deg(10.0, 350.0) == pytest.approx(20.0)
    assert angular_difference_deg(0.0, 180.0) == pytest.approx(180.0)


def test_progression_is_normalised_and_monotonic():
    t = straight_track("a", "car")
    assert t.progression[0] == pytest.approx(0.0)
    assert t.progression[-1] == pytest.approx(1.0)
    assert np.all(np.diff(t.progression) >= 0)


# --------------------------------------------------------------------------- #
# pairing uses space and direction, never time or frame index
# --------------------------------------------------------------------------- #
def test_pairing_never_reads_time_of_day_or_frame_index():
    src = inspect.getsource(route_align.pair_tracks)
    for banned in ("frame_index", "utc", "wall_clock", "epoch"):
        assert banned not in src.lower()


def test_two_drives_of_the_same_road_pair_regardless_of_absolute_time():
    """Same road, two months apart, different rates: pairing must still work."""
    car = straight_track("car", "car", start_ns=0, dt_ns=NS)
    moto = straight_track("moto", "motorcycle",
                          start_ns=5_000_000 * NS,     # ~two months later
                          dt_ns=NS // 2)               # sampled twice as densely
    result = pair_tracks(car, moto)
    assert result["summary"]["paired"]
    assert result["summary"]["accepted_fraction"] > 0.9


def test_a_distant_route_is_not_paired():
    car = straight_track("car", "car", lon0=12.2200)
    far = straight_track("moto", "motorcycle", lon0=12.5000)   # tens of km away
    result = pair_tracks(car, far)
    assert not result["summary"]["paired"]
    assert result["summary"]["accepted_count"] == 0
    assert "do not share a route segment" in result["summary"]["reason"]
    # rejected candidates are still emitted rather than hidden
    assert result["summary"]["candidate_count"] > 0


def test_opposite_direction_on_the_same_road_is_rejected():
    car = straight_track("car", "car")
    reversed_lat = list(reversed([44.1400 + i * 0.0002 for i in range(60)]))
    moto = build_track("moto", "motorcycle", [i * NS for i in range(60)],
                       reversed_lat, [12.2200] * 60, [3.0] * 60, [12.0] * 60)
    result = pair_tracks(car, moto)
    accepted = [p for p in result["pairs"] if p["accepted"]]
    assert not accepted
    assert any("comparable direction" in w
               for p in result["pairs"] for w in p["warnings"])


def test_every_pair_carries_its_quality_evidence():
    car = straight_track("car", "car")
    moto = straight_track("moto", "motorcycle", dt_ns=NS // 2)
    for p in pair_tracks(car, moto)["pairs"]:
        for field in ("pairing_id", "recording_id_car", "recording_id_motorcycle",
                      "timestamp_ns_car", "timestamp_ns_motorcycle",
                      "progression_car", "progression_motorcycle", "distance_m",
                      "heading_difference_deg", "pairing_quality", "sensors_used",
                      "warnings", "accepted"):
            assert field in p


def test_result_is_labelled_exploratory():
    car = straight_track("car", "car")
    moto = straight_track("moto", "motorcycle")
    assert pair_tracks(car, moto)["summary"]["status"] == "exploratory_preliminary"


def test_empty_trajectory_does_not_force_a_pairing():
    empty = build_track("x", "car", [], [], [])
    moto = straight_track("moto", "motorcycle")
    assert not pair_tracks(empty, moto)["summary"]["paired"]


# --------------------------------------------------------------------------- #
# route events
# --------------------------------------------------------------------------- #
def test_a_stop_is_detected_with_its_duration():
    lat = [44.14 + i * 0.0002 for i in range(20)] + [44.14 + 19 * 0.0002] * 15
    lon = [12.22] * len(lat)
    ts = [i * NS for i in range(len(lat))]
    track = build_track("s", "car", ts, lat, lon, [3.0] * len(lat), None)
    stops = detect_route_events(track)["stops"]
    assert stops and stops[0]["duration_s"] >= 5.0


def test_gps_dropout_is_reported():
    ts = [i * NS for i in range(10)] + [(10 + 30) * NS + i * NS for i in range(10)]
    lat = [44.14 + i * 0.0002 for i in range(20)]
    track = build_track("d", "car", ts, lat, [12.22] * 20, [3.0] * 20, [12.0] * 20)
    assert detect_route_events(track)["gps_dropouts"]


def test_degraded_accuracy_is_warned_about():
    n = 30
    track = build_track("g", "car", [i * NS for i in range(n)],
                        [44.14 + i * 0.0002 for i in range(n)], [12.22] * n,
                        [45.0] * n, [12.0] * n)
    assert any("degraded" in w for w in track.warnings)


def test_progression_lookup_reports_its_temporal_distance():
    track = straight_track("p", "car")
    progression, dt = progression_of_timestamp(track, int(10.4 * NS))
    assert 0.0 <= progression <= 1.0
    assert dt == pytest.approx(0.4, abs=0.01)
