"""Behavioural guarantees of the Article 1 multimodal analysis.

Each test pins one of the methodological rules the analysis was built under, in
the code rather than in a document: the two motion families stay apart, gravity
gets removed, PPG passes a gate before it becomes a heart rate, HRV is refused on
a window too short to support it, a line-crossing candidate needs several kinds of
evidence, and nothing can turn a candidate into a verdict without a person.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
REPORTS = ROOT / "reports" / "article1_behavior_analysis"


# --------------------------------------------------------------------------- #
# Vehicle dynamics vs head dynamics
# --------------------------------------------------------------------------- #
def _synthetic_imu(seconds=20.0, fs=800.0, seed=0):
    """A head IMU: gravity, a slow head turn, and vehicle vibration on top."""
    rng = np.random.default_rng(seed)
    n = int(seconds * fs)
    t = np.arange(n) / fs
    ts = (t * 1e9).astype(np.int64)
    gravity = np.tile(np.array([0.0, 0.0, 9.80665]), (n, 1))
    vibration = 0.4 * np.sin(2 * np.pi * 12.0 * t)[:, None] * np.array([1, 1, 1])
    accel = gravity + vibration + rng.normal(0, 0.02, (n, 3))
    gyro = np.zeros((n, 3))
    gyro[:, 2] = 0.6 * np.sin(2 * np.pi * 0.5 * t)      # head yaw about vertical
    return ts, accel, gyro


def test_head_dynamics_removes_gravity_and_recovers_it():
    from aria_drive_seg.behavior.dynamics import G_MSEC2, compute_head_dynamics
    ts, accel, gyro = _synthetic_imu()
    head = compute_head_dynamics(ts, accel, gyro,
                                 camera_forward_device=[0, 0, 1],
                                 camera_right_device=[1, 0, 0])
    # The estimated gravity magnitude must land on standard gravity ...
    assert abs(head.meta["gravity_magnitude_median"] - G_MSEC2) < 0.15
    # ... and what is left must be small, i.e. gravity really was subtracted.
    interior = slice(int(0.2 * ts.size), int(0.8 * ts.size))
    assert np.nanmedian(head.linear_accel_magnitude[interior]) < 1.0
    assert head.meta["gravity_removed"] is True
    assert head.meta["is_vehicle_acceleration"] is False


def test_head_dynamics_is_labelled_as_head_motion_everywhere():
    from aria_drive_seg.behavior.dynamics import (compute_head_dynamics,
                                                  head_dynamics_summary)
    ts, accel, gyro = _synthetic_imu(seconds=10.0)
    head = compute_head_dynamics(ts, accel, gyro)
    frame = head.to_frame()
    assert (frame["measures"] == "head_motion").all()
    assert (~frame["is_vehicle_acceleration"]).all()
    summary = head_dynamics_summary(head)
    assert summary["is_vehicle_acceleration"] is False
    assert "not vehicle acceleration" in summary["caveat"].lower()


def test_vehicle_dynamics_never_reads_the_imu():
    """Vehicle speed comes from position/speed sources only."""
    import inspect
    from aria_drive_seg.behavior import dynamics
    src = inspect.getsource(dynamics.compute_vehicle_dynamics)
    for banned in ("accel_msec2", "gyro_rad", "imu"):
        assert banned not in src, (
            f"compute_vehicle_dynamics references {banned!r}: the head IMU must "
            "never become vehicle acceleration")


def test_vehicle_dynamics_records_its_speed_source():
    from aria_drive_seg.behavior.dynamics import compute_vehicle_dynamics
    ts = (np.arange(30) * 1e9).astype(np.int64)
    x = np.arange(30) * 12.0
    y = np.zeros(30)
    dyn = compute_vehicle_dynamics(ts, x, y, gps_speed_mps=np.full(30, 12.0))
    assert dyn.speed_source == "gps_speed_field"
    assert (dyn.to_frame()["derived_from_head_imu"] == False).all()  # noqa: E712
    assert any("no VIO/SLAM pose stream" in n for n in dyn.notes)


def test_reported_dynamics_keep_the_two_families_separate():
    path = REPORTS / "dynamics_summary.json"
    if not path.exists():
        pytest.skip("dynamics summary not generated")
    report = json.loads(path.read_text())
    assert "different sensors" in report["separation"]
    for domain, r in report["domains"].items():
        assert r["vehicle"]["derived_from_head_imu"] is False
        assert r["head"]["is_vehicle_acceleration"] is False


def test_braking_detectability_is_reported_when_the_threshold_is_unreachable():
    """A zero count must be accompanied by whether it could have been non-zero."""
    path = REPORTS / "dynamics_summary.json"
    if not path.exists():
        pytest.skip("dynamics summary not generated")
    for domain, r in json.loads(path.read_text())["domains"].items():
        d = r["vehicle"]["detectability"]
        assert "most_negative_observed_acceleration_mps2" in d
        if r["vehicle"]["hard_braking_episodes"] == 0:
            # However the zero arose - threshold never reached, or reached but
            # never for long enough to be an episode - it must be explained.
            assert d["caveat"], (
                f"{domain} reports zero hard braking with no detectability note")
            assert "detection limit" in d["caveat"] or "limit of the sampling" \
                in d["caveat"]


# --------------------------------------------------------------------------- #
# PPG
# --------------------------------------------------------------------------- #
def _synthetic_ppg(seconds=90.0, fs=256.0, bpm=72.0, noise=0.02, seed=0):
    rng = np.random.default_rng(seed)
    n = int(seconds * fs)
    t = np.arange(n) / fs
    ts = (t * 1e9).astype(np.int64)
    pulse = np.sin(2 * np.pi * (bpm / 60.0) * t)
    pulse += 0.3 * np.sin(2 * np.pi * 2 * (bpm / 60.0) * t)   # dicrotic shoulder
    return ts, 100_000 + 5_000 * pulse + rng.normal(0, 5_000 * noise, n)


def test_ppg_detects_a_plausible_rate_on_a_clean_signal():
    from aria_drive_seg.behavior.ppg import detect_beats
    ts, v = _synthetic_ppg(bpm=72.0)
    beats, filtered, fs = detect_beats(ts, v)
    assert abs(fs - 256.0) < 1.0
    hr = beats.heart_rate_bpm[beats.valid]
    assert 65 <= float(np.median(hr)) <= 79


def test_hrv_is_refused_on_a_window_shorter_than_its_minimum():
    from aria_drive_seg.behavior.ppg import detect_beats, hrv_window
    ts, v = _synthetic_ppg(seconds=90.0)
    beats, _, _ = detect_beats(ts, v)

    short = hrv_window(beats, 0, int(20e9), min_window_s=30.0)
    assert short["rmssd_ms"] is None
    assert "below the 30 s minimum" in short["refusals"]["rmssd"]

    medium = hrv_window(beats, 0, int(40e9), min_window_s=30.0,
                        sdnn_min_window_s=60.0)
    assert medium["rmssd_ms"] is not None
    assert medium["sdnn_ms"] is None, "SDNN must be refused below 60 s"
    assert "60 s" in medium["refusals"]["sdnn"]

    long = hrv_window(beats, 0, int(80e9), min_window_s=30.0,
                      sdnn_min_window_s=60.0)
    assert long["sdnn_ms"] is not None


def test_pnn50_is_refused_without_enough_intervals():
    from aria_drive_seg.behavior.ppg import detect_beats, hrv_window
    ts, v = _synthetic_ppg(seconds=90.0)
    beats, _, _ = detect_beats(ts, v)
    w = hrv_window(beats, 0, int(70e9), pnn50_min_beats=10_000)
    assert w["pnn50"] is None
    assert "pnn50" in w["refusals"]


def test_ppg_quality_gate_rejects_a_motion_contaminated_window():
    from aria_drive_seg.behavior.ppg import assess_quality, detect_beats
    ts, v = _synthetic_ppg(seconds=60.0)
    beats, filtered, fs = detect_beats(ts, v)
    imu_ts = np.arange(0, int(60e9), int(1e9 / 800), dtype=np.int64)
    calm = np.full(imu_ts.size, 9.8)
    q_calm = assess_quality(ts, v, filtered, fs, beats, window_s=10.0,
                            imu_timestamp_ns=imu_ts, imu_accel_magnitude=calm,
                            motion_accel_std_threshold=2.0)
    shaken = calm + np.random.default_rng(1).normal(0, 6.0, imu_ts.size)
    q_shaken = assess_quality(ts, v, filtered, fs, beats, window_s=10.0,
                              imu_timestamp_ns=imu_ts, imu_accel_magnitude=shaken,
                              motion_accel_std_threshold=2.0)
    assert q_calm.motion_artefact.sum() == 0
    assert q_shaken.motion_artefact.all()
    assert q_shaken.usable.sum() < q_calm.usable.sum()
    assert any("motion artefact" in r for r in q_shaken.reason)


def test_unusable_windows_invalidate_their_beats():
    from aria_drive_seg.behavior.ppg import (assess_quality, detect_beats,
                                             tag_beats_with_quality)
    ts, v = _synthetic_ppg(seconds=40.0)
    beats, filtered, fs = detect_beats(ts, v)
    before = int(beats.valid.sum())
    imu_ts = np.arange(0, int(40e9), int(1e9 / 800), dtype=np.int64)
    shaken = 9.8 + np.random.default_rng(2).normal(0, 8.0, imu_ts.size)
    q = assess_quality(ts, v, filtered, fs, beats, window_s=10.0,
                       imu_timestamp_ns=imu_ts, imu_accel_magnitude=shaken)
    beats = tag_beats_with_quality(beats, q)
    assert int(beats.valid.sum()) < before
    assert any("window rejected" in r for r in beats.reason)


def test_double_detection_check_fires_on_alternating_intervals():
    """The notch artefact must be detectable, not merely mentioned."""
    from aria_drive_seg.behavior.ppg import Beats, beat_detection_diagnostics
    n = 60
    ibi = np.where(np.arange(n) % 2 == 0, 350.0, 500.0)   # alternating: artefact
    beats = Beats(timestamp_ns=np.cumsum(ibi * 1e6).astype(np.int64),
                  amplitude=np.ones(n), ibi_ms=ibi,
                  heart_rate_bpm=60_000 / ibi, valid=np.ones(n, bool),
                  reason=["ok"] * n, sqi_at_beat=np.ones(n))
    diag = beat_detection_diagnostics(beats)
    assert diag["ibi_lag1_autocorrelation"] < -0.3
    assert diag["alternation_suggests_double_detection"] is True


def test_ppg_report_never_claims_a_medical_meaning():
    path = REPORTS / "ppg_summary.json"
    if not path.exists():
        pytest.skip("ppg summary not generated")
    report = json.loads(path.read_text())
    assert report["interpretation"] == "physiological_proxy_not_medical"
    text = json.dumps(report).lower()
    # Clinical *claims*, not clinical vocabulary: "beat_detection_diagnostics" is
    # a signal-processing check and must not trip this.
    for banned in ("diagnosis", "diagnosed", "arrhythmia", "pathological",
                   "stress level", "clinically", "medical condition"):
        assert banned not in text, f"PPG report contains a clinical claim: {banned}"
    for domain, r in report["domains"].items():
        assert r["caveat"], f"{domain} PPG summary carries no proxy caveat"
        assert r["interpretation"] == "physiological_proxy_not_medical"


# --------------------------------------------------------------------------- #
# Road events
# --------------------------------------------------------------------------- #
def test_event_windows_are_clipped_and_say_so():
    from aria_drive_seg.behavior.events import RoadEvent, build_windows
    ev = RoadEvent("e1", "roundabout_traverse", "car", "rec", int(5e9), int(9e9),
                   int(7e9), 100.0, 1, "secondary")
    wins = build_windows(ev, {"phys_baseline": (-30.0, -10.0),
                              "phys_immediate": (0.0, 10.0)},
                         recording_start_ns=0, recording_end_ns=int(60e9),
                         other_events=[], avoid_overlap=False)
    assert wins["phys_baseline"].clipped is True
    assert "start of the recording" in wins["phys_baseline"].clip_reason
    assert wins["phys_immediate"].clipped is False


def test_sub_events_of_one_manoeuvre_do_not_trim_each_other():
    """A roundabout's entry must not destroy its own traverse baseline."""
    from aria_drive_seg.behavior.events import RoadEvent, build_windows, same_family
    assert same_family("roundabout_entry", "roundabout_traverse")
    assert not same_family("roundabout_traverse", "pedestrian_crossing")
    traverse = RoadEvent("t", "roundabout_traverse", "car", "r", int(60e9),
                         int(70e9), int(65e9), 0.0, 1, None)
    entry = RoadEvent("e", "roundabout_entry", "car", "r", int(60e9), int(61e9),
                      int(60e9), 0.0, 1, None)
    wins = build_windows(traverse, {"phys_baseline": (-30.0, -10.0)},
                         0, int(200e9), [traverse, entry], avoid_overlap=True)
    assert wins["phys_baseline"].duration_s > 15.0


def test_background_classifications_do_not_contaminate_baselines():
    """curve/straight partition the drive; they cannot be contaminants."""
    from aria_drive_seg.behavior.events import CONTAMINATING_KINDS
    for kind in ("curve", "straight", "road_class_change"):
        assert kind not in CONTAMINATING_KINDS
    for kind in ("roundabout_traverse", "junction_crossing", "stop"):
        assert kind in CONTAMINATING_KINDS


def test_events_come_from_the_map_not_from_the_signals():
    path = REPORTS / "events" / "road_events_summary.json"
    if not path.exists():
        pytest.skip("road events summary not generated")
    report = json.loads(path.read_text())
    assert "OpenStreetMap" in report["event_source"]
    assert "not the analysed signals" in report["event_source"]


# --------------------------------------------------------------------------- #
# Solid-line candidates
# --------------------------------------------------------------------------- #
def test_only_the_four_candidate_states_exist():
    from aria_drive_seg.behavior.solid_line import CANDIDATE_STATES
    assert set(CANDIDATE_STATES) == {"candidate_compliant", "candidate_noncompliant",
                                     "uncertain", "not_evaluable"}
    for banned in ("illegal", "violation", "offence", "noncompliant_confirmed"):
        assert banned not in CANDIDATE_STATES


def test_a_candidate_needs_several_evidence_sources():
    from aria_drive_seg.behavior.solid_line import _decide_state
    evidence = {"min_distance_to_junction_m": 200.0,
                "min_distance_to_roundabout_m": 200.0}
    marking = {"continuity": "solid"}
    # One source is never enough, however suggestive the marking is.
    assert _decide_state(evidence, marking, "candidate_solid_line_crossing",
                         ["signed_lateral_offset"], 3, True, 25.0) == "not_evaluable"
    # With enough sources, and only then, it can become a candidate.
    assert _decide_state(evidence, marking, "candidate_solid_line_crossing",
                         ["a", "b", "c"], 3, True, 25.0) == "candidate_noncompliant"


def test_unresolvable_geometry_forces_not_evaluable():
    from aria_drive_seg.behavior.solid_line import _decide_state
    state = _decide_state({"min_distance_to_junction_m": 200.0,
                           "min_distance_to_roundabout_m": 200.0},
                          {"continuity": "solid"}, "candidate_solid_line_crossing",
                          ["a", "b", "c", "d"], 3, False, 25.0)
    assert state == "not_evaluable"


def test_a_junction_manoeuvre_is_not_treated_as_a_violation():
    from aria_drive_seg.behavior.solid_line import _decide_state
    state = _decide_state({"min_distance_to_junction_m": 5.0,
                           "min_distance_to_roundabout_m": 200.0},
                          {"continuity": "solid"}, "junction_manoeuvre",
                          ["a", "b", "c"], 3, True, 25.0)
    assert state == "candidate_compliant"


def test_line_continuity_is_unknown_on_too_short_an_observation():
    from aria_drive_seg.behavior.solid_line import (MarkingObservation,
                                                    classify_continuity)
    obs = [MarkingObservation(i, i, True, True, .2, .2, 100, .9) for i in range(3)]
    out = classify_continuity(obs, "left", frame_interval_s=0.066,
                              min_persistence_s=0.5)
    assert out["continuity"] == "unknown"
    assert "solid line from a dashed one" in out["reason"]


def test_candidates_carry_a_pending_review_status():
    path = REPORTS / "solid_line" / "solid_line_candidates.csv"
    if not path.exists():
        pytest.skip("candidates not generated")
    df = pd.read_csv(path)
    if df.empty:
        return
    assert (df["review_status"] == "pending_human_review").all()
    assert df["automatic_label_is_not_final"].all()
    from aria_drive_seg.behavior.solid_line import CANDIDATE_STATES
    assert set(df["state"]) <= set(CANDIDATE_STATES)


def test_no_classifier_is_trained_without_reviewed_labels():
    path = REPORTS / "solid_line" / "solid_line_summary.json"
    if not path.exists():
        pytest.skip("solid line summary not generated")
    report = json.loads(path.read_text())
    assert report["definitive_label_requires_human_review"] is True
    training = report["classifier_training"]
    assert training["attempted"] is False
    assert "20 confirmed" in training["reason"]


def test_review_package_permits_only_the_four_human_labels():
    manifest = ROOT / "datasets" / "article1_action_review" / "manifest.json"
    if not manifest.exists():
        pytest.skip("review package not generated")
    payload = json.loads(manifest.read_text())
    assert set(payload["permitted_human_labels"]) == {
        "compliant", "noncompliant", "uncertain", "not_evaluable"}


# --------------------------------------------------------------------------- #
# Statistics
# --------------------------------------------------------------------------- #
def test_block_bootstrap_is_wider_than_an_iid_bootstrap_on_correlated_data():
    """The whole reason for the block bootstrap, pinned as a test."""
    from aria_drive_seg.behavior.stats import block_bootstrap_ci
    rng = np.random.default_rng(3)
    # A strongly autocorrelated series, like any signal along a drive.
    noise = rng.normal(0, 1, 400)
    series = np.convolve(noise, np.ones(20) / 20, mode="same")
    lo_iid, hi_iid = block_bootstrap_ci(series, np.median, block_size=1,
                                        iterations=800, rng=rng)
    lo_blk, hi_blk = block_bootstrap_ci(series, np.median, block_size=20,
                                        iterations=800, rng=rng)
    assert (hi_blk - lo_blk) > (hi_iid - lo_iid), (
        "a block bootstrap must not be narrower than the i.i.d. one on "
        "autocorrelated data")


def test_permutation_p_value_is_never_zero():
    from aria_drive_seg.behavior.stats import block_permutation_test
    rng = np.random.default_rng(4)
    a = rng.normal(10, 1, 60)
    b = rng.normal(-10, 1, 60)
    out = block_permutation_test(a, b, lambda x, y: float(np.median(x) - np.median(y)),
                                 block_size=4, iterations=500, rng=rng)
    assert out["p_value"] > 0.0
    assert out["exchangeable_unit"] == "contiguous block"


def test_fdr_correction_raises_p_values():
    from aria_drive_seg.behavior.stats import benjamini_hochberg
    raw = [0.001, 0.01, 0.02, 0.2, 0.5]
    out = benjamini_hochberg(raw, alpha=0.05)
    assert all(adj >= p for adj, p in zip(out["adjusted"], raw))
    assert out["tested"] == 5


def test_small_sample_intervals_carry_a_caveat():
    from aria_drive_seg.behavior.stats import paired_block_bootstrap
    rng = np.random.default_rng(5)
    eff = paired_block_bootstrap(rng.normal(0, 1, 5), rng.normal(0, 1, 5), rng=rng)
    assert eff.caveat is not None and "paired units" in eff.caveat


def test_paired_comparison_reports_effective_sample_size_not_row_count():
    path = REPORTS / "paired" / "paired_comparison_summary.json"
    if not path.exists():
        pytest.skip("paired comparison not generated")
    report = json.loads(path.read_text())
    assert report["frames_are_not_independent_samples"] is True
    assert report["unit_of_analysis"].endswith("route bin")
    assert report["effective_independent_blocks"] <= report["paired_bins_with_metrics"]
    assert "effective sample size" in report["effective_sample_size_note"]
    assert "one session per vehicle" in report["pilot_caveat"]


def test_paired_comparison_uses_only_validated_pairs():
    path = REPORTS / "route" / "paired_route_segments.csv"
    if not path.exists():
        pytest.skip("route pairing not generated")
    pairs = pd.read_csv(path)
    paired = pairs[pairs["paired"].astype(bool)]
    if paired.empty:
        return
    assert (paired["direction_a"] == paired["direction_b"]).all()
    assert paired["rejection_reason"].isna().all()
    rejected = pairs[~pairs["paired"].astype(bool)]
    assert rejected["rejection_reason"].notna().all(), (
        "every rejected pair must record why")


def test_route_bins_are_map_defined_and_spatial_only():
    path = REPORTS / "route" / "route_alignment_summary.json"
    if not path.exists():
        pytest.skip("route alignment not generated")
    report = json.loads(path.read_text())
    assert report["alignment"] == "spatial_only_never_temporal"
    assert "osm_way_id" in report["bin_definition"]
    assert report["pairing_criteria"]["require_same_direction"] is True


def test_undersampled_bin_sizes_are_flagged_rather_than_reported_as_findings():
    path = REPORTS / "route" / "route_alignment_summary.json"
    if not path.exists():
        pytest.skip("route alignment not generated")
    report = json.loads(path.read_text())
    for name, s in report["bin_sizes"].items():
        if s["gps_cadence_undersamples_this_bin"]:
            assert s["resolution_caveat"], (
                f"{name} is undersampled but carries no caveat")
            assert "sampling limit" in s["resolution_caveat"]
