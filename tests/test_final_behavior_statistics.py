"""Tests for the final behaviour statistics: blink, pupil, gaze, stats, reports.

Two kinds of test live here.

**Behavioural tests** exercise the new modules on synthetic data whose right
answer is known by construction, so a regression in the blink grouping or the
light correction fails on arithmetic rather than on a golden file.

**Guards** check that the rules this work was done under are still being
followed: no interpolated blink, no frame treated as an independent sample, no
medical interpretation, no head check relabelled as a mirror check, the 5 Hz and
native runs untouched, and no report still carrying the retired 8% / 3% gaze
coverage claim.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pytest

from aria_drive_seg.behavior.blink import (BLINK_CATEGORIES, EVENT_CATEGORIES,
                                           blink_metrics, build_blink_events,
                                           closed_series,
                                           determine_blink_polarity,
                                           windowed_blink_rate)
from aria_drive_seg.behavior.eye_state import (REQUIRED_COLUMNS, LayoutField,
                                               decode_record, parse_data_layout)
from aria_drive_seg.behavior.head_eye import (find_excursions, gaze_yaw_rate,
                                              head_eye_metrics)
from aria_drive_seg.behavior.lane_position import (crossing_candidates,
                                                   measure_frame,
                                                   select_measurement_band,
                                                   summarise)
from aria_drive_seg.behavior.pupil import (apply_log_lux_model, attach_nearest,
                                           build_pupil_table, fit_log_lux_model,
                                           local_baseline)
from aria_drive_seg.behavior.semantic_attention import (CLASS_GROUPS,
                                                        attention_metrics,
                                                        flatten_unit)
from aria_drive_seg.behavior.stats import (effective_sample_size,
                                           two_sample_block_bootstrap)

pd = pytest.importorskip("pandas")

ROOT = Path(__file__).resolve().parent.parent
ANALYSIS = ROOT / "output" / "article1" / "final_behavior_statistics"
FINAL_REPORTS = ROOT / "reports" / "article1_final_behavior_statistics"
BEHAVIOR_REPORTS = ROOT / "reports" / "article1_behavior_analysis"
BEHAVIOR = ROOT / "aria_drive_seg" / "behavior"
SCRIPTS = ROOT / "scripts"

NS = 1_000_000_000

#: Modules and runners this branch added or changed.
NEW_SOURCES = [BEHAVIOR / name for name in
               ("eye_state.py", "blink.py", "pupil.py", "semantic_attention.py",
                "head_eye.py", "lane_position.py")]
NEW_SCRIPTS = [SCRIPTS / name for name in
               ("extract_article1_eye_state.py",
                "analyze_article1_final_behavior.py",
                "measure_article1_lane_position.py",
                "compare_article1_final_paired.py",
                "visualise_article1_final_behavior.py")]


def _needs(path: Path):
    if not path.exists():
        pytest.skip(f"{path.relative_to(ROOT)} not present; run the pipeline first")
    return path


# --------------------------------------------------------------------------- #
# Eye-state extraction: layout, validity flags, no interpolation
# --------------------------------------------------------------------------- #
def test_data_layout_is_parsed_from_its_declaration():
    """Offsets and widths come from the descriptor, never from a constant."""
    descriptor = json.dumps({"data_layout": [
        {"name": "tracker_timestamp_ns", "type": "DataPieceValue<int64_t>",
         "offset": 0},
        {"name": "left_eye/pupil_diameter_valid", "type": "DataPieceValue<Bool>",
         "offset": 8},
        {"name": "left_eye/pupil_diameter_meter", "type": "DataPieceValue<float>",
         "offset": 9},
        {"name": "left_eye/entrance_pupil_position_in_device_meter_xyz",
         "type": "DataPieceValue<Point3Df>", "offset": 13},
        {"name": "algorithm_name", "type": "DataPieceString", "index": 0},
    ]})
    fields, size = parse_data_layout(descriptor)
    assert size == 25, "fixed-buffer size must be the maximum field end"
    assert [f.name for f in fields][-1].endswith("xyz")
    assert all(f.piece_type != "DataPieceString" for f in fields), \
        "a variable-size piece is not part of the fixed buffer"


def test_data_layout_refuses_a_type_it_cannot_decode():
    """A partly-understood layout is an error, not a silently skipped field."""
    descriptor = json.dumps({"data_layout": [
        {"name": "weird", "type": "DataPieceValue<Quaternionf>", "offset": 0}]})
    with pytest.raises(ValueError, match="unsupported fixed-size type"):
        parse_data_layout(descriptor)


def test_record_decode_reads_the_declared_offsets():
    import struct
    fields = [LayoutField("valid", "DataPieceValue<Bool>", 0, "<?", 1),
              LayoutField("diameter", "DataPieceValue<float>", 1, "<f", 4),
              LayoutField("blink", "DataPieceValue<Bool>", 5, "<?", 1)]
    payload = struct.pack("<?f?", True, 0.00231, False)
    decoded = decode_record(payload, fields)
    assert decoded["valid"] is True
    assert decoded["blink"] is False
    assert decoded["diameter"] == pytest.approx(0.00231, abs=1e-9)


def test_required_columns_are_the_ones_the_brief_asks_for():
    for name in ("left_blink", "right_blink", "left_blink_valid",
                 "right_blink_valid", "left_pupil_diameter_meter",
                 "right_pupil_diameter_meter", "left_pupil_diameter_valid",
                 "right_pupil_diameter_valid", "timestamp_ns"):
        assert name in REQUIRED_COLUMNS


def test_extracted_samples_carry_every_required_column_and_a_real_timestamp():
    for recording in _recording_dirs():
        frame = pd.read_parquet(recording / "eye_state_samples.parquet")
        for name in REQUIRED_COLUMNS:
            assert name in frame.columns, f"{recording.name} is missing {name}"
        ts = frame["timestamp_ns"].to_numpy(np.int64)
        assert np.all(np.diff(ts) > 0), "timestamps must be strictly increasing"
        # The device writes at 30 Hz; every gap must be one real sampling
        # interval, which is what "nothing was interpolated or dropped" looks like.
        gaps_ms = np.diff(ts) / 1e6
        assert gaps_ms.max() < 40.0 and gaps_ms.min() > 25.0


def test_invalid_pupil_readings_carry_no_number():
    for recording in _recording_dirs():
        frame = pd.read_parquet(recording / "eye_state_samples.parquet")
        for side in ("left", "right"):
            invalid = ~frame[f"{side}_pupil_diameter_valid"].to_numpy(bool)
            values = frame[f"{side}_pupil_diameter_meter"].to_numpy(float)
            assert np.all(np.isnan(values[invalid])), (
                f"{recording.name}: a device-invalid {side} pupil reading still "
                "carries a usable-looking number")


def test_extraction_declares_that_nothing_was_interpolated():
    for recording in _recording_dirs():
        summary = json.loads((recording / "eye_state_summary.json").read_text())
        extraction = summary["extraction"]
        assert extraction["interpolated"] is False
        assert extraction["resampled"] is False
        assert extraction["gap_filled"] is False
        assert extraction["blink_interpolated"] is False
        assert summary["decode_verification"]["verified"] is True
        assert summary["decode_verification"]["flag_mismatches"] == {}
        assert summary["decode_verification"]["max_timestamp_error_ns"] == 0


def _recording_dirs():
    _needs(ANALYSIS)
    dirs = sorted(p for p in ANALYSIS.iterdir()
                  if p.is_dir() and (p / "eye_state_samples.parquet").exists())
    if not dirs:
        pytest.skip("no extracted eye-state tables; run the pipeline first")
    return dirs


# --------------------------------------------------------------------------- #
# Blink polarity, grouping and windows
# --------------------------------------------------------------------------- #
def _synthetic_blink_stream(n: int = 900, period: int = 30, closed_len: int = 4,
                            closed_value: bool = False):
    """30 Hz stream with one closure every `period` samples."""
    ts = np.arange(n, dtype=np.int64) * (NS // 30)
    closed = np.zeros(n, bool)
    for start in range(period, n - closed_len, period):
        closed[start:start + closed_len] = True
    flag = np.where(closed, closed_value, not closed_value)
    return ts, flag.astype(bool), np.ones(n, bool)


def test_polarity_is_decided_from_run_lengths_not_from_the_field_name():
    for closed_value in (False, True):
        ts, flag, valid = _synthetic_blink_stream(closed_value=closed_value)
        polarity = determine_blink_polarity(flag, valid, ts)
        assert polarity["decided"] is True
        assert polarity["closed_value"] is closed_value, (
            "the closed state is whichever value is rarer and shorter-running, "
            "whatever the field is called")


def test_polarity_is_refused_when_the_two_states_are_not_separated():
    ts = np.arange(600, dtype=np.int64) * (NS // 30)
    flag = np.tile([True] * 10 + [False] * 10, 30).astype(bool)
    polarity = determine_blink_polarity(flag, np.ones(600, bool), ts)
    assert polarity["decided"] is False
    assert polarity["closed_value"] is None
    assert "cannot be established" in polarity["reason"]


def test_blink_events_group_consecutive_samples_only():
    ts, flag, valid = _synthetic_blink_stream(n=300, period=30, closed_len=4)
    frame = pd.DataFrame({
        "timestamp_ns": ts, "left_blink": flag, "right_blink": flag,
        "left_blink_valid": valid, "right_blink_valid": valid,
        "combined_gaze_direction_valid": ~np.isin(flag, [False])})
    polarity = determine_blink_polarity(flag, valid, ts)
    events = build_blink_events(frame, polarity)
    assert len(events) == 9, "one event per synthetic closure, not one per sample"
    assert {e.category for e in events} == {"bilateral_blink"}
    assert all(e.samples == 4 for e in events)


def test_a_sampling_gap_splits_an_event_rather_than_being_bridged():
    """A blink is never interpolated across a hole in the stream."""
    step = NS // 30
    ts = np.array([0, step, 10 * step, 11 * step], dtype=np.int64)
    closed = np.array([False, False, False, False])       # raw flag: closed=False
    frame = pd.DataFrame({
        "timestamp_ns": ts, "left_blink": closed, "right_blink": closed,
        "left_blink_valid": np.ones(4, bool), "right_blink_valid": np.ones(4, bool),
        "combined_gaze_direction_valid": np.zeros(4, bool)})
    polarity = {"decided": True, "closed_value": False}
    events = build_blink_events(frame, polarity, max_gap_s=0.075)
    assert len(events) == 2, ("the 300 ms hole must end the run; bridging it "
                              "would invent eye state that was never recorded")
    assert [e.samples for e in events] == [2, 2]


def test_event_categories_are_exactly_the_five_permitted_ones():
    assert set(EVENT_CATEGORIES) == {
        "bilateral_blink", "left_only", "right_only", "uncertain",
        "invalid_tracking_gap"}
    assert set(BLINK_CATEGORIES) <= set(EVENT_CATEGORIES)


def test_monocular_and_invalid_events_are_distinguished():
    step = NS // 30
    ts = np.arange(12, dtype=np.int64) * step
    left = np.ones(12, bool)
    right = np.ones(12, bool)
    left_valid = np.ones(12, bool)
    right_valid = np.ones(12, bool)
    left[2:4] = False                       # left-only closure
    right[6:8] = False                      # right-only closure
    left_valid[10:12] = right_valid[10:12] = False   # tracking gap
    frame = pd.DataFrame({
        "timestamp_ns": ts, "left_blink": left, "right_blink": right,
        "left_blink_valid": left_valid, "right_blink_valid": right_valid,
        "combined_gaze_direction_valid": np.ones(12, bool)})
    events = build_blink_events(frame, {"decided": True, "closed_value": False})
    categories = [e.category for e in events]
    assert categories == ["left_only", "right_only", "invalid_tracking_gap"]
    assert events[-1].reason and "invalid" in events[-1].reason


def test_a_prolonged_closure_is_not_counted_as_a_blink():
    step = NS // 30
    n = 90
    ts = np.arange(n, dtype=np.int64) * step
    flag = np.ones(n, bool)
    flag[10:70] = False                     # 2 s closure
    frame = pd.DataFrame({
        "timestamp_ns": ts, "left_blink": flag, "right_blink": flag,
        "left_blink_valid": np.ones(n, bool), "right_blink_valid": np.ones(n, bool),
        "combined_gaze_direction_valid": np.ones(n, bool)})
    events = build_blink_events(frame, {"decided": True, "closed_value": False},
                                max_blink_duration_s=1.0)
    assert [e.category for e in events] == ["uncertain"]
    assert "longer than" in events[0].reason


def test_blink_rate_windows_are_defined_in_seconds():
    step = NS // 30
    ts = np.arange(30 * 200, dtype=np.int64) * step        # 200 s
    flag = np.ones(ts.size, bool)
    for start in range(30, ts.size - 4, 30):               # one blink per second
        flag[start:start + 2] = False
    frame = pd.DataFrame({
        "timestamp_ns": ts, "left_blink": flag, "right_blink": flag,
        "left_blink_valid": np.ones(ts.size, bool),
        "right_blink_valid": np.ones(ts.size, bool),
        "combined_gaze_direction_valid": np.ones(ts.size, bool)})
    polarity = determine_blink_polarity(flag, np.ones(ts.size, bool), ts)
    events = build_blink_events(frame, polarity)
    windows = windowed_blink_rate(events, int(ts[0]), int(ts[-1]),
                                  window_s=60.0, step_s=10.0)
    assert all(w["window_s"] == 60.0 and w["step_s"] == 10.0 for w in windows)
    starts = [w["window_start_ns"] for w in windows]
    assert np.allclose(np.diff(starts), 10.0 * NS), "step must be 10 real seconds"
    assert all(w["window_end_ns"] <= ts[-1] for w in windows), \
        "a partial window would report a rate over less time than it claims"
    assert all(55 <= w["blinks_per_minute"] <= 61 for w in windows)


def test_windowed_rate_rejects_a_non_positive_window():
    with pytest.raises(ValueError):
        windowed_blink_rate([], 0, NS, window_s=0.0, step_s=1.0)


def test_eye_closure_fraction_is_not_labelled_a_drowsiness_measure():
    ts, flag, valid = _synthetic_blink_stream()
    frame = pd.DataFrame({
        "timestamp_ns": ts, "left_blink": flag, "right_blink": flag,
        "left_blink_valid": valid, "right_blink_valid": valid,
        "combined_gaze_direction_valid": np.ones(ts.size, bool)})
    polarity = determine_blink_polarity(flag, valid, ts)
    metrics = blink_metrics(frame, build_blink_events(frame, polarity), polarity)
    closure = metrics["eye_closure"]
    assert closure["is_drowsiness_measure"] is False
    assert "not a drowsiness score" in closure["interpretation"]
    assert metrics["blink_interpolated"] is False


# --------------------------------------------------------------------------- #
# Pupil and the light correction
# --------------------------------------------------------------------------- #
def test_light_correction_recovers_a_known_slope():
    rng = np.random.default_rng(0)
    lux = np.exp(rng.uniform(np.log(50), np.log(20000), 4000))
    diameter = 0.005 - 0.0009 * np.log10(lux + 1) + rng.normal(0, 5e-6, lux.size)
    model = fit_log_lux_model(diameter, lux)
    assert model["fitted"] is True
    assert model["slope_m_per_log10lux"] == pytest.approx(-0.0009, rel=0.02)
    assert model["r_squared"] > 0.99
    assert model["status"] == "exploratory"
    residual = apply_log_lux_model(diameter, lux, model)
    assert abs(float(np.nanmean(residual))) < 1e-6
    assert float(np.nanstd(residual)) < float(np.std(diameter))


def test_light_correction_is_refused_when_it_cannot_be_identified():
    diameter = np.full(200, 0.0022)
    assert fit_log_lux_model(diameter, np.full(200, 1000.0))["fitted"] is False
    assert fit_log_lux_model(diameter[:10], np.arange(10.0))["fitted"] is False


def test_light_is_attached_from_real_samples_and_never_interpolated():
    target = np.array([0, NS, 2 * NS], dtype=np.int64)
    source = np.array([0, 2 * NS], dtype=np.int64)
    values, dt_s = attach_nearest(target, source, [100.0, 300.0], max_dt_s=0.25)
    assert values[0] == 100.0 and values[2] == 300.0
    assert np.isnan(values[1]), ("a target 1 s from the nearest real sample gets "
                                 "no value, not an interpolated one")
    assert np.allclose(np.abs(dt_s), [0.0, 1.0, 0.0])


def test_local_baseline_is_a_window_in_seconds():
    ts = np.arange(600, dtype=np.int64) * (NS // 30)
    values = np.concatenate([np.full(300, 1.0), np.full(300, 2.0)])
    baseline = local_baseline(ts, values, window_s=2.0)
    assert baseline[0] == pytest.approx(1.0)
    assert baseline[-1] == pytest.approx(2.0)
    assert np.all(np.isfinite(baseline))


def test_pupil_table_excludes_closed_eyes_but_keeps_the_flags():
    n = 60
    ts = np.arange(n, dtype=np.int64) * (NS // 30)
    closed = np.zeros(n, bool)
    closed[10:14] = True
    eye_state = pd.DataFrame({
        "timestamp_ns": ts,
        "left_pupil_diameter_meter": np.full(n, 0.0022),
        "right_pupil_diameter_meter": np.full(n, 0.0023),
        "left_pupil_diameter_valid": np.ones(n, bool),
        "right_pupil_diameter_valid": np.ones(n, bool)})
    table = build_pupil_table(eye_state, ts, np.full(n, 1000.0), closed=closed)
    assert bool(table["left_pupil_diameter_valid"].all()), \
        "the device flag stays as the device wrote it"
    assert np.all(np.isnan(table.loc[closed, "left_pupil_usable_m"])), \
        "a diameter measured through an eyelid is not a pupil measurement"
    assert np.all(np.isfinite(table.loc[~closed, "left_pupil_usable_m"]))


def test_pupil_summary_states_it_is_not_clinical():
    for recording in _recording_dirs():
        summary = json.loads((recording / "eye_state_summary.json").read_text())
        assert "Not a clinical measurement" in summary["pupil"][
            "clinical_interpretation"]
        assert summary["pupil"]["light_model"]["status"] == "exploratory"
        assert summary["pupil"]["light_model"]["r_squared"] is not None, (
            "the fit quality must be reported so a reader can weigh the "
            "correction")


# --------------------------------------------------------------------------- #
# Semantic attention
# --------------------------------------------------------------------------- #
def _synthetic_gaze(classes, n_per: int = 10):
    names = sorted({c for c in classes})
    rows = []
    for i, name in enumerate(classes):
        row = {"timestamp_ns": int(i * NS // 5), "top1_class": name,
               "foveal_entropy": 0.2, "is_fixation": True, "fixation_id": i // 3,
               "rect_u": 500.0 + i, "rect_v": 380.0}
        for other in names:
            row[f"p_{other}"] = 1.0 if other == name else 0.0
        rows.append(row)
    return pd.DataFrame(rows), names


def test_attention_metrics_count_revisits_and_transitions():
    classes = ["road_surface", "road_surface", "sky", "road_surface", "sky"]
    frame, names = _synthetic_gaze(classes)
    metrics = attention_metrics(frame, names, sample_interval_s=0.2)
    road = metrics["per_class"]["road_surface"]
    assert road["top1_samples"] == 3
    assert road["visit_runs"] == 2 and road["revisits"] == 1
    assert road["dwell_time_s"] == pytest.approx(0.6)
    assert road["time_to_first_fixation_s"] == pytest.approx(0.0)
    assert metrics["semantic_transitions"] == 3
    assert metrics["gaze_entropy"]["foveal_mass_distribution_entropy"] > 0


def test_foveal_mass_is_the_declared_primary_metric():
    frame, names = _synthetic_gaze(["road_surface", "sky"])
    metrics = attention_metrics(frame, names, sample_interval_s=0.2)
    assert metrics["primary_metric"] == "foveal_mass_percent"
    assert "vanishing point" in metrics["primary_metric_note"]


def test_class_groups_cover_the_required_shares():
    for name in ("road_relevant", "interior_cockpit", "vulnerable_road_user",
                 "sign_and_signal"):
        assert name in CLASS_GROUPS and CLASS_GROUPS[name]


def test_an_empty_unit_reports_a_reason_rather_than_a_zero():
    frame, names = _synthetic_gaze(["road_surface"])
    metrics = attention_metrics(frame.iloc[0:0], names, sample_interval_s=0.2)
    assert metrics["usable"] is False and metrics["reason"]
    assert flatten_unit(metrics)["road_relevant_mass_percent"] is None


# --------------------------------------------------------------------------- #
# Head-eye coordination
# --------------------------------------------------------------------------- #
def test_excursion_direction_comes_from_the_run_mean():
    ts = np.arange(100, dtype=np.int64) * (NS // 100)
    rate = np.zeros(100)
    rate[10:40] = 0.6                      # sustained left
    rate[60:90] = -0.6                     # sustained right
    events = find_excursions(ts, rate, threshold=0.35, min_duration_s=0.15)
    assert [e.direction for e in events] == ["left", "right"]


def test_a_head_check_is_never_called_a_mirror_check():
    ts = np.arange(600, dtype=np.int64) * (NS // 30)
    head_ts = np.arange(6000, dtype=np.int64) * (NS // 300)
    head_rate = np.zeros(6000)
    head_rate[1000:1200] = 0.6
    gaze_rate = np.zeros(600)
    gaze_rate[100:120] = 2.0
    metrics = head_eye_metrics(ts, gaze_rate, np.zeros(600), head_ts, head_rate)
    checks = metrics["lateral_head_checks"]
    assert checks["is_mirror_check_count"] is False
    assert "not attributed to a mirror" in checks["attribution"]
    assert metrics["head_motion_is_not_vehicle_motion"] is True


def test_gaze_rate_is_blanked_across_an_invalid_sample():
    ts = np.arange(5, dtype=np.int64) * (NS // 30)
    yaw = np.array([0.0, 0.1, 0.2, 0.3, 0.4])
    valid = np.array([True, True, False, True, True])
    rate = gaze_yaw_rate(ts, yaw, valid)
    assert np.isnan(rate[1]) and np.isnan(rate[2]), \
        "a rate spanning an unmeasured sample is a rate across a gap"
    assert np.isfinite(rate[0]) and np.isfinite(rate[3])


# --------------------------------------------------------------------------- #
# Visual lane position
# --------------------------------------------------------------------------- #
def _lane_mask(left_px: int, right_px: int, width: int = 200, height: int = 100,
               lane_id: int = 2, road_id: int = 1) -> np.ndarray:
    mask = np.zeros((height, width), np.uint16)
    mask[70:, left_px:right_px] = road_id
    mask[70:, left_px:left_px + 3] = lane_id
    mask[70:, right_px - 3:right_px] = lane_id
    return mask


def test_lane_offset_is_zero_when_the_camera_is_centred():
    mask = _lane_mask(60, 140)             # centred on a 200 px image
    record = measure_frame(mask, lane_marking_id=2, road_surface_id=1,
                           band_top_fraction=0.7, band_bottom_fraction=1.0)
    assert record["measured"] is True
    assert record["normalized_offset"] == pytest.approx(0.0, abs=0.05)


def test_lane_offset_can_exceed_one_so_a_crossing_is_detectable():
    """Both markings on one side of the axis must not be clipped to +-1."""
    mask = _lane_mask(20, 90)              # lane centre well left of the axis
    record = measure_frame(mask, lane_marking_id=2, road_surface_id=1,
                           band_top_fraction=0.7, band_bottom_fraction=1.0)
    assert record["measured"] is True
    assert record["normalized_offset"] > 1.0, (
        "a rule that takes one marking either side of the camera axis bounds its "
        "own output to +-1 and can never report a crossing")


def test_a_pair_with_no_road_between_it_is_rejected():
    mask = np.zeros((100, 200), np.uint16)
    mask[70:, 60:63] = 2
    mask[70:, 137:140] = 2                 # markings, but nothing between them
    record = measure_frame(mask, lane_marking_id=2, road_surface_id=1,
                           band_top_fraction=0.7, band_bottom_fraction=1.0)
    assert record["measured"] is False
    assert "road surface" in record["reason"]


def test_measurement_band_is_selected_from_the_row_profile():
    """The band is measured per recording; a fixed one measures a car dashboard."""
    profile = np.zeros(100)
    profile[60:73] = 0.35                  # car-like: road, then dashboard below
    band = select_measurement_band(profile, min_road_row_fraction=0.15)
    assert band["selected"] is True
    assert band["bottom_fraction"] == pytest.approx(0.73)
    assert band["top_fraction"] > 0.6

    profile = np.zeros(100)
    profile[55:100] = 0.8                  # motorcycle-like: road to the bottom
    band = select_measurement_band(profile, min_road_row_fraction=0.15)
    assert band["bottom_fraction"] == pytest.approx(1.0)


def test_crossing_candidates_need_a_minimum_duration_and_are_never_verdicts():
    ts = np.arange(20, dtype=np.int64) * (NS // 5)
    offset = np.zeros(20)
    offset[2:4] = 1.4                      # 0.2 s, too short
    offset[8:16] = 1.5                     # 1.4 s
    candidates = crossing_candidates(ts, offset, min_duration_s=0.5)
    assert len(candidates) == 1
    assert candidates[0]["state"] == "candidate_requires_review"
    assert candidates[0]["is_traffic_violation_claim"] is False


def test_lane_summary_is_labelled_a_proxy():
    records = [{"measured": True, "normalized_offset": 0.1, "timestamp_ns": i * NS,
                "reason": None} for i in range(20)]
    summary = summarise(records, sample_interval_s=0.2)
    assert summary["metric"] == "visual_lane_position_proxy"
    assert summary["is_vehicle_metric_position"] is False


# --------------------------------------------------------------------------- #
# Statistics: units, never the frame
# --------------------------------------------------------------------------- #
def test_effective_sample_size_is_blocks_not_rows():
    assert effective_sample_size(42, 8) == 6
    assert effective_sample_size(0, 8) == 0


def test_two_sample_block_bootstrap_is_wider_than_an_iid_one():
    rng = np.random.default_rng(1)
    walk = np.cumsum(rng.normal(0, 1, 200))          # strongly autocorrelated
    other = np.cumsum(rng.normal(0, 1, 200))
    blocked = two_sample_block_bootstrap(walk, other, block_size=20, rng=rng)
    iid = two_sample_block_bootstrap(walk, other, block_size=1, rng=rng)
    assert (blocked.ci_high - blocked.ci_low) > (iid.ci_high - iid.ci_low), (
        "resampling single values assumes independence these samples do not have")


def test_paired_statistics_never_use_the_frame_as_a_unit():
    path = _needs(FINAL_REPORTS / "paired_statistics.csv")
    table = pd.read_csv(path)
    assert set(table["statistical_unit"]) <= {
        "50 m route bin", "road event", "time block"}
    assert not table["frames_are_independent_samples"].any()
    assert (table["effective_sample_size"] <= table["n_units"]).all()


def test_shared_bins_carry_gaze_for_both_vehicles():
    path = _needs(ANALYSIS / "semantic_attention_bins.parquet")
    bins = pd.read_parquet(path)
    with_gaze = bins.dropna(subset=["road_relevant_mass_percent"])
    per_bin = with_gaze.groupby("bin_key")["domain"].nunique()
    assert len(per_bin) == 42, f"expected 42 shared bins, found {len(per_bin)}"
    assert (per_bin >= 2).all(), (
        "every shared bin must carry a semantic-gaze reading for both vehicles")


def test_paired_summary_records_that_gaze_is_available():
    path = _needs(FINAL_REPORTS / "paired_comparison_summary.json")
    summary = json.loads(path.read_text())
    assert summary["gaze_in_paired_comparison"]["available"] is True
    assert summary["gaze_in_paired_comparison"]["bins_with_gaze_on_both_sides"] == 42
    assert summary["unit_of_analysis_is_never_the_frame"] is True

    legacy = BEHAVIOR_REPORTS / "paired" / "paired_comparison_summary.json"
    if legacy.exists():
        entry = json.loads(legacy.read_text())["semantic_gaze_in_paired_comparison"]
        assert entry["available"] is True
        assert entry["bins_with_gaze_on_both_sides"] == 42


# --------------------------------------------------------------------------- #
# Guards
# --------------------------------------------------------------------------- #
#: The retired claim, in every phrasing that appeared in the old reports.
OBSOLETE_COVERAGE = re.compile(
    r"(coverage[^.\n]{0,40}\b8\s?%|\b8%\s?(and|/)\s?3%|"
    r"covers 8% and 3%|no semantic coverage on the shared route|"
    r"no bin carries gaze for \*?\*?both)", re.IGNORECASE)


#: A quotation of the retired claim is allowed only where the surrounding
#: sentence marks it as history. Markdown wraps sentences across lines, so the
#: marker is looked for in a small window rather than on the same line.
SUPERSEDED_MARKER = re.compile(
    r"supersed|no longer|earlier revision|previously|retired|"
    r"is not true|are no longer true|removed", re.IGNORECASE)
CONTEXT_LINES = 3


def _unmarked_claims(text: str, pattern: re.Pattern) -> list[str]:
    lines = text.splitlines()
    out = []
    for i, line in enumerate(lines):
        if not pattern.search(line):
            continue
        window = "\n".join(lines[max(0, i - CONTEXT_LINES):
                                 i + CONTEXT_LINES + 1])
        if not SUPERSEDED_MARKER.search(window):
            out.append(line.strip())
    return out


@pytest.mark.parametrize("report", sorted(
    list((ROOT / "reports" / "article1_behavior_analysis").glob("*.md")) +
    list((ROOT / "reports" / "article1_final_behavior_statistics").glob("*.md"))),
    ids=lambda p: p.name)
def test_no_report_still_claims_the_retired_gaze_coverage(report: Path):
    unmarked = _unmarked_claims(report.read_text(), OBSOLETE_COVERAGE)
    assert not unmarked, (
        f"{report.name} still asserts the retired gaze-coverage claim: {unmarked}")


@pytest.mark.parametrize("report", ["REPORT_SEMANTIC_GAZE.md",
                                    "REPORT_PAIRED_AUTO_MOTO.md",
                                    "REPORT_FINAL_SUMMARY.md"])
def test_updated_reports_state_the_real_coverage(report: str):
    text = (BEHAVIOR_REPORTS / report).read_text()
    assert "93.2%" in text and "89.6%" in text
    assert "42" in text


def test_new_reports_exist_and_carry_the_required_disclaimers():
    for name in ("REPORT_BLINK_PUPIL.md", "REPORT_MULTIMODAL_COMPARISON.md",
                 "REPORT_FINAL_SUMMARY.md"):
        text = (FINAL_REPORTS / name).read_text()
        assert "exploratory_pilot" in text
    blink = (FINAL_REPORTS / "REPORT_BLINK_PUPIL.md").read_text()
    assert "not a drowsiness" in blink.lower()
    assert "no anisocoria" in blink.lower()
    assert "not a clinical" in blink.lower() or "not clinical" in blink.lower()


#: Words that would turn an ocular or physiological measurement into a diagnosis.
#: Bare "diagnostic" is deliberately absent — it has an ordinary engineering sense
#: ("diagnostic output") that has nothing to do with medicine.
MEDICAL_CLAIMS = (
    r"\bdrowsiness (?:score|index|level|measure)\b", r"\bmicrosleep\b",
    r"\bdiagnos(?:is|es|ed)\b", r"\bdiagnostic (?:sign|criteri|marker|test)\b",
    r"\bfitness to drive\b", r"\banisocoria\b", r"\bpatholog", r"\bimpairment\b",
    r"\bclinical(?:ly)? (?:significant|relevant|validated)\b",
)

#: A negation anywhere in the sentence discharges the claim. Sentences wrap, so
#: the window is a few lines rather than one.
NEGATION = re.compile(r"\bnot\b|\bnever\b|\bno\b|\bnothing\b|\bfalse\b|\bcannot\b",
                      re.IGNORECASE)


@pytest.mark.parametrize("path", NEW_SOURCES + NEW_SCRIPTS,
                         ids=lambda p: p.name)
def test_new_code_makes_no_medical_claim(path: Path):
    text = path.read_text()
    lines = text.splitlines()
    for pattern in MEDICAL_CLAIMS:
        for match in re.finditer(pattern, text, re.IGNORECASE):
            index = text[:match.start()].count("\n")
            window = "\n".join(lines[max(0, index - 2):index + 3])
            assert NEGATION.search(window), (
                f"{path.name}:{index + 1} uses {match.group()!r} without a "
                f"negation nearby: {lines[index].strip()!r}")


@pytest.mark.parametrize("path", NEW_SOURCES + NEW_SCRIPTS,
                         ids=lambda p: p.name)
def test_new_code_never_writes_into_the_frozen_runs(path: Path):
    """The 5 Hz and native runs are read-only inputs to this analysis."""
    text = path.read_text()
    for match in re.finditer(r"fast_semantic_gaze(_native)?", text):
        line_no = text[:match.start()].count("\n")
        line = text.splitlines()[line_no]
        assert not re.search(r"to_parquet|atomic_write|\.write_text|open\([^)]*['\"]w",
                             line), (
            f"{path.name}:{line_no + 1} writes into a frozen run: {line.strip()!r}")


def test_the_frozen_runs_still_match_their_manifest():
    from aria_drive_seg.hashing import sha256_file
    manifest_path = _needs(FINAL_REPORTS / "frozen_run_manifest.json")
    manifest = json.loads(manifest_path.read_text())
    changed = []
    for run, entry in manifest["runs"].items():
        root = ROOT / run
        if not root.exists():
            pytest.skip(f"{run} is not present locally")
        for name, digest in entry["files"].items():
            path = root / name
            if not path.exists():
                changed.append(f"{run}/{name}: missing")
            elif sha256_file(path) != digest:
                changed.append(f"{run}/{name}: content changed")
    assert not changed, (
        "the 5 Hz and native semantic-gaze runs must not be modified by this "
        "analysis: " + "; ".join(changed))


def test_figure_manifest_locks_the_shared_scales_and_the_figure_count():
    manifest_path = _needs(FINAL_REPORTS / "figures" / "figure_manifest.json")
    manifest = json.loads(manifest_path.read_text())
    assert manifest["comparisons_use_common_scales"] is True
    assert manifest["one_question_per_figure"] is True
    assert manifest["eye_closure_is_not_a_drowsiness_measure"] is True
    assert manifest["pupil_has_no_clinical_interpretation"] is True
    stems = [entry["stem"] for entry in manifest["figures"]]
    assert len(stems) == 12, (
        "nine publication figures plus at most three additions; "
        f"found {len(stems)}: {stems}")
    assert stems[:9] == ["01_route_map_shared", "02_semantic_gaze_distribution",
                         "03_dwell_fixation_by_class",
                         "04_semantic_gaze_shared_route", "05_event_related_gaze",
                         "06_speed_head_motion", "07_ppg_hr_event_related",
                         "08_solid_line_candidates",
                         "09_paired_summary_difference"], \
        "the nine existing figures keep their identity and their order"
    assert manifest["additions"] == ["10_blink_pupil_comparison",
                                     "11_event_related_eye_response",
                                     "12_paired_gaze_physiology_summary"]
    assert all(entry.get("same_scale_car_motorcycle") for entry in
               manifest["figures"][:9])
    for stem in stems:
        for suffix in (".png", ".pdf"):
            assert (FINAL_REPORTS / "figures" / f"{stem}{suffix}").exists()


def test_figures_use_one_colour_per_vehicle_everywhere():
    from aria_drive_seg.article1.fast_plots import DOMAIN_COLORS
    from aria_drive_seg.article1.final_plots import plot_blink_pupil  # noqa: F401
    manifest = json.loads(
        _needs(FINAL_REPORTS / "figures" / "figure_manifest.json").read_text())
    assert manifest["domain_colors"] == DOMAIN_COLORS
    assert set(DOMAIN_COLORS) == {"car", "motorcycle"}


def test_the_final_summary_declares_the_units_and_refuses_the_frame():
    summary = json.loads(_needs(ANALYSIS / "final_behavior_summary.json").read_text())
    assert summary["frames_are_independent_samples"] is False
    assert summary["paired_bins_with_gaze_on_both_sides"] == 42
    assert summary["semantic_gaze_covers_whole_recording"] is True
    assert set(summary["unit_of_analysis"]) == {"route bin", "event", "time block"}


def test_event_windows_are_durations_in_seconds():
    events = pd.read_parquet(_needs(ANALYSIS / "event_response.parquet"))
    assert not events["frames_are_independent_samples"].any()
    assert (events["unit_of_analysis"] == "event").all()
    # Every window's declared duration must equal its real bounds in seconds.
    declared = events["window_duration_s"].to_numpy(float)
    actual = (events["window_end_ns"] - events["window_start_ns"]).to_numpy(float) / NS
    assert np.allclose(declared, actual)
    assert (declared >= 0).all()


def test_a_window_trimmed_to_nothing_says_so_and_reports_no_rate():
    """Zero duration is a legitimate outcome, and must never look like data."""
    events = pd.read_parquet(_needs(ANALYSIS / "event_response.parquet"))
    empty = events[events["window_duration_s"] <= 0]
    if empty.empty:
        pytest.skip("no window was trimmed to nothing in this run")
    assert empty["window_clipped"].all()
    assert empty["window_clip_reason"].notna().all()
    assert (empty["gaze_samples"] == 0).all()
    for column in ("heart_rate_bpm", "road_relevant_mass_percent", "speed_mps",
                   "blink_rate_per_minute", "eye_closure_fraction"):
        assert empty[column].isna().all(), (
            f"{column} has a value in a window of zero duration")


def test_event_windows_are_anchored_the_way_the_road_event_analysis_anchors_them():
    """Anticipation ends at the event start; the immediate window begins at its end."""
    events = pd.read_parquet(_needs(ANALYSIS / "event_response.parquet"))
    for window in ("approach", "phys_anticipation", "pre_event", "phys_baseline"):
        rows = events[events["window"] == window]
        assert (rows["window_end_ns"] <= rows["event_start_ns"]).all(), (
            f"{window} must not extend past the event's start")
    for window in ("post_event", "phys_immediate", "phys_delayed"):
        rows = events[events["window"] == window]
        assert (rows["window_start_ns"] >= rows["event_end_ns"]).all(), (
            f"{window} must not begin before the event's end")
