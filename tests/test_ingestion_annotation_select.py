"""Annotation selection is balanced, deterministic, diverse and non-duplicating."""
from __future__ import annotations

import pytest

from aria_drive_seg.ingestion.annotation_select import (GROUPS, FrameCandidate,
                                                        build_selection,
                                                        build_stratifier,
                                                        select_group)

NS = 1_000_000_000


def candidate(domain: str, i: int, fps: float, **kw) -> FrameCandidate:
    base = dict(
        domain=domain, recording_id=f"{domain}_rec",
        source_frame_index=i, timestamp_ns=int(i / fps * NS),
        timestamp_s=i / fps, image_path=f"{domain}/{i}.jpg",
        mean_luminance=0.2 + 0.3 * ((i % 7) / 6),
        blur_variance=100.0 + 900.0 * ((i % 5) / 4),
        frame_difference=0.01 + 0.05 * ((i % 3) / 2),
        # spread the hashes so consecutive frames are not near duplicates
        dhash=(i * 0x9E3779B97F4A7C15) & ((1 << 64) - 1),
        class_fraction={"road_surface": 0.4, "vehicle": 0.02 * (i % 3),
                        "mapillary_ego_region": 0.05 + 0.01 * (i % 4),
                        "lane_marking": 0.005 * (i % 2)},
    )
    base.update(kw)
    return FrameCandidate(**base)


def pool(domain: str, n: int, fps: float) -> list:
    return [candidate(domain, i, fps) for i in range(n)]


# --------------------------------------------------------------------------- #
# balancing is by duration and need, not by frame availability
# --------------------------------------------------------------------------- #
def test_denser_sampling_does_not_buy_a_bigger_share():
    """Same duration, 10 fps vs 15 fps: both domains must get the same quota."""
    pools = {"car": pool("car", 300, 10.0),        # 30 s at 10 fps
             "motorcycle": pool("motorcycle", 450, 15.0)}   # 30 s at 15 fps
    quota = {g: 10 for g in GROUPS}
    result = build_selection(pools, {"car": quota, "motorcycle": quota},
                             min_separation_s=0.5, min_hamming=4)
    counts = result["counts"]["per_domain"]
    assert counts["car"] == counts["motorcycle"]


def test_balancing_rule_is_documented_in_the_output():
    pools = {"car": pool("car", 60, 10.0), "motorcycle": pool("motorcycle", 90, 15.0)}
    result = build_selection(pools, {d: {"external_validation": 5} for d in pools},
                             min_separation_s=0.5, min_hamming=4)
    assert "higher rate" in result["balancing_rule"]
    assert result["is_ground_truth"] is False


# --------------------------------------------------------------------------- #
# no near duplicates
# --------------------------------------------------------------------------- #
def test_selected_frames_respect_the_minimum_temporal_separation():
    p = pool("car", 300, 10.0)
    picks = select_group(p, 20, build_stratifier(p), min_separation_s=2.0,
                         min_hamming=4)
    chosen = sorted(c.timestamp_ns for c, _, _ in picks)
    assert all(b - a >= 2.0 * NS for a, b in zip(chosen, chosen[1:]))


def test_identical_looking_neighbours_are_rejected():
    """Frames close in time with the same hash must not both be selected."""
    p = [candidate("car", i, 10.0, dhash=12345) for i in range(60)]
    picks = select_group(p, 20, build_stratifier(p), min_separation_s=0.2,
                         min_hamming=8)
    assert len(picks) == 1  # everything else is a perceptual duplicate nearby


def test_a_revisit_far_apart_in_time_is_not_a_duplicate():
    """Same view an hour later is a distinct sample, not a duplicate."""
    p = [candidate("car", 0, 10.0, dhash=12345),
         candidate("car", 1, 10.0, dhash=12345)]
    p[1].timestamp_ns = int(600 * NS)   # ten minutes later, identical appearance
    p[1].timestamp_s = 600.0
    picks = select_group(p, 5, build_stratifier(p), min_separation_s=2.0,
                         min_hamming=8)
    assert len(picks) == 2


# --------------------------------------------------------------------------- #
# not random, and justified
# --------------------------------------------------------------------------- #
def test_selection_is_deterministic():
    p = pool("car", 200, 10.0)
    a = [c.source_frame_index for c, _, _ in
         select_group(p, 15, build_stratifier(p), 1.0, 4)]
    b = [c.source_frame_index for c, _, _ in
         select_group(p, 15, build_stratifier(p), 1.0, 4)]
    assert a == b


def test_every_selected_frame_records_why():
    pools = {"car": pool("car", 120, 10.0), "motorcycle": pool("motorcycle", 180, 15.0)}
    result = build_selection(pools, {d: {"external_validation": 8} for d in pools},
                             min_separation_s=1.0, min_hamming=4)
    for s in result["selected"]:
        assert s["selection_reason"]
        assert s["strata_covered"]
        assert s["split_group"] in GROUPS
        for field in ("domain", "recording_id", "source_frame_index", "timestamp_ns",
                      "image_path", "expected_classes", "hand_visibility_candidate",
                      "confidence", "entropy", "provenance", "route_segment",
                      "pairing_id", "failure_mode_candidate"):
            assert field in s


def test_rare_conditions_are_preferred_over_repetition():
    """A single dark frame in a bright pool must be picked early."""
    p = pool("car", 100, 10.0)
    for c in p:
        c.mean_luminance = 0.6
    p[57].mean_luminance = 0.01
    picks = select_group(p, 6, build_stratifier(p), 1.0, 4)
    assert 57 in [c.source_frame_index for c, _, _ in picks]


def test_shortfall_is_reported_rather_than_hidden():
    pools = {"car": pool("car", 30, 10.0), "motorcycle": pool("motorcycle", 30, 15.0)}
    result = build_selection(pools, {d: {"external_validation": 100} for d in pools},
                             min_separation_s=2.0, min_hamming=4)
    stats = result["per_domain_group_stats"]["car"]["external_validation"]
    assert stats["selected"] < stats["requested"]
    assert stats["shortfall_reason"]


# --------------------------------------------------------------------------- #
# groups
# --------------------------------------------------------------------------- #
def test_cockpit_group_only_draws_frames_with_ego_structure():
    p = pool("motorcycle", 120, 15.0)
    for c in p:
        c.class_fraction = {"road_surface": 0.9}      # no ego structure at all
    result = build_selection({"motorcycle": p},
                             {"motorcycle": {"cockpit_training": 10}},
                             min_separation_s=1.0, min_hamming=4)
    assert result["counts"]["per_group"]["cockpit_training"] == 0


def test_failure_group_is_empty_without_diagnostics():
    p = pool("motorcycle", 120, 15.0)
    result = build_selection({"motorcycle": p},
                             {"motorcycle": {"failure_mode_review": 10}},
                             min_separation_s=1.0, min_hamming=4)
    assert result["counts"]["per_group"]["failure_mode_review"] == 0


def test_stratum_coverage_is_reported():
    pools = {"car": pool("car", 150, 10.0), "motorcycle": pool("motorcycle", 220, 15.0)}
    result = build_selection(pools, {d: {"external_validation": 10} for d in pools},
                             min_separation_s=1.0, min_hamming=4)
    coverage = result["stratum_coverage"]
    assert coverage
    assert "domain:car" in coverage and "domain:motorcycle" in coverage
