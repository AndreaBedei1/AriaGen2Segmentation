"""Balanced, non-random selection of frames for human annotation.

Design constraints:

* **Balanced by duration, not by frame availability.** The motorcycle is currently
  sampled 1.5x more densely than the car, so a "take every Nth frame" policy would
  hand the motorcycle 1.5x more annotation budget for no scientific reason. Quotas
  are therefore per domain, and diversity constraints are expressed in seconds.
* **Never random.** Selection is a greedy maximisation of stratum coverage: each
  pick is the candidate that adds the most uncovered strata, so rare conditions
  (bad light, blur, high entropy, visible hands, failure modes) survive instead of
  being drowned by ordinary driving.
* **No near duplicates.** A minimum temporal separation in seconds plus a
  perceptual-hash distance floor.
* Every selected item records *why* it was chosen.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence

import numpy as np

from .rgb_scan import hamming
from .timeline import NS_PER_S

# The three required annotation groups.
GROUPS = ("external_validation", "cockpit_training", "failure_mode_review")


@dataclass
class FrameCandidate:
    """One annotatable frame with everything needed to stratify and justify it."""

    domain: str
    recording_id: str
    source_frame_index: int
    timestamp_ns: int
    timestamp_s: float
    image_path: str
    # image statistics
    mean_luminance: float = 0.0
    blur_variance: float = 0.0
    frame_difference: float = 0.0
    dhash: int = 0
    # semantics (proxy or semantic-camera derived)
    class_fraction: Dict[str, float] = field(default_factory=dict)
    confidence: Optional[float] = None
    entropy: Optional[float] = None
    provenance: Dict[str, float] = field(default_factory=dict)
    fallback: bool = False
    conflict_fraction: Optional[float] = None
    # audit outcomes
    failure_mode_candidate: Optional[str] = None
    hand_visibility_candidate: Dict[str, str] = field(default_factory=dict)
    # route
    route_progression: Optional[float] = None
    route_segment: Optional[str] = None
    pairing_id: Optional[str] = None
    # bookkeeping
    has_semantic_camera_output: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class SelectedFrame:
    domain: str
    recording_id: str
    source_frame_index: int
    timestamp_ns: int
    timestamp_s: float
    image_path: str
    selection_reason: str
    strata_covered: List[str]
    expected_classes: List[str]
    failure_mode_candidate: Optional[str]
    hand_visibility_candidate: Dict[str, str]
    confidence: Optional[float]
    entropy: Optional[float]
    provenance: Dict[str, float]
    split_group: str
    route_segment: Optional[str]
    route_progression: Optional[float]
    pairing_id: Optional[str]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# --------------------------------------------------------------------------- #
# Strata
# --------------------------------------------------------------------------- #
def _q(values: Sequence[float], frac: float) -> float:
    arr = np.asarray([v for v in values if v is not None], dtype=np.float64)
    return float(np.quantile(arr, frac)) if arr.size else 0.0


def build_stratifier(pool: Sequence[FrameCandidate]) -> Callable[[FrameCandidate], List[str]]:
    """Build a stratum-labelling function calibrated on the pool's own quantiles.

    Absolute thresholds would not transfer between a car interior and an open
    motorcycle view, so "dark", "blurred" and "high entropy" are defined relative to
    the material actually available.
    """
    lum = [c.mean_luminance for c in pool]
    blur = [c.blur_variance for c in pool]
    motion = [c.frame_difference for c in pool]
    entropy = [c.entropy for c in pool if c.entropy is not None]
    conf = [c.confidence for c in pool if c.confidence is not None]

    lum_lo, lum_hi = _q(lum, 0.15), _q(lum, 0.85)
    blur_lo = _q(blur, 0.20)
    motion_lo, motion_hi = _q(motion, 0.25), _q(motion, 0.80)
    ent_hi = _q(entropy, 0.80) if entropy else None
    conf_lo = _q(conf, 0.20) if conf else None

    def strata(c: FrameCandidate) -> List[str]:
        out = [f"domain:{c.domain}"]
        out.append("light:dark" if c.mean_luminance <= lum_lo else
                   "light:bright" if c.mean_luminance >= lum_hi else "light:normal")
        if c.blur_variance <= blur_lo:
            out.append("quality:low_sharpness")
        out.append("motion:low" if c.frame_difference <= motion_lo else
                   "motion:high" if c.frame_difference >= motion_hi else
                   "motion:medium")
        # A near-stationary frame with high scene change is a turn; low change with
        # high motion is forward translation on a straight.
        out.append("geometry:turning" if c.frame_difference >= motion_hi
                   else "geometry:straight")

        cf = c.class_fraction or {}
        if cf.get("vehicle", 0) + cf.get("two_wheeler", 0) > 0.01:
            out.append("scene:traffic")
        else:
            out.append("scene:clear")
        if cf.get("pedestrian", 0) > 0.0005:
            out.append("scene:pedestrian")
        if cf.get("two_wheeler", 0) > 0.001:
            out.append("scene:two_wheeler")
        if cf.get("traffic_sign", 0) + cf.get("traffic_light", 0) > 0.0008:
            out.append("scene:signage")
        if cf.get("lane_marking", 0) + cf.get("regulatory_road_marking", 0) > 0.004:
            out.append("scene:markings")
        if cf.get("road_boundary_or_obstacle", 0) > 0.004:
            out.append("scene:boundary")

        cockpit = (cf.get("mirror", 0) + cf.get("instrument_display", 0)
                   + cf.get("control_and_ego_vehicle", 0)
                   + cf.get("mapillary_ego_region", 0))
        out.append("cockpit:prominent" if cockpit > 0.12 else
                   "cockpit:partial" if cockpit > 0.02 else "cockpit:minimal")
        if cf.get("mirror", 0) > 0.001:
            out.append("cockpit:mirror")
        if cf.get("instrument_display", 0) > 0.001:
            out.append("cockpit:instrument")

        if ent_hi is not None and c.entropy is not None and c.entropy >= ent_hi:
            out.append("uncertainty:high_entropy")
        if conf_lo is not None and c.confidence is not None and c.confidence <= conf_lo:
            out.append("uncertainty:low_confidence")
        if c.conflict_fraction is not None and c.conflict_fraction > 0.3:
            out.append("uncertainty:model_conflict")
        if c.fallback:
            out.append("provenance:fallback")
        for label, frac in (c.provenance or {}).items():
            if frac > 0.05:
                out.append(f"provenance:{label}")

        if c.failure_mode_candidate:
            out.append(f"failure:{c.failure_mode_candidate}")
        for side, state in (c.hand_visibility_candidate or {}).items():
            out.append(f"hand:{side}:{state}")
        if c.route_segment:
            out.append(f"route:{c.route_segment}")
        return out

    return strata


# --------------------------------------------------------------------------- #
# Greedy diverse selection
# --------------------------------------------------------------------------- #
def select_group(pool: Sequence[FrameCandidate], quota: int,
                 strata_fn: Callable[[FrameCandidate], List[str]],
                 min_separation_s: float = 4.0,
                 min_hamming: int = 8,
                 already_selected: Optional[List[FrameCandidate]] = None,
                 priority: Optional[Callable[[FrameCandidate], float]] = None
                 ) -> List[tuple]:
    """Greedily pick frames that maximise newly covered strata.

    Returns (candidate, covered_strata, reason) tuples.
    """
    chosen: List[FrameCandidate] = list(already_selected or [])
    covered: set = set()
    for c in chosen:
        covered.update(strata_fn(c))

    result: List[tuple] = []
    remaining = list(pool)
    while remaining and len(result) < quota:
        best = None
        best_gain: tuple = (-1.0, -1.0)
        for c in remaining:
            if not _separated(c, chosen, min_separation_s, min_hamming):
                continue
            s = set(strata_fn(c))
            gain = len(s - covered)
            score = (float(gain), float(priority(c)) if priority else 0.0)
            if score > best_gain:
                best_gain, best = score, c
        if best is None:
            break
        s = strata_fn(best)
        new = sorted(set(s) - covered)
        covered.update(s)
        chosen.append(best)
        remaining.remove(best)
        reason = (f"adds {len(new)} previously uncovered strata: "
                  f"{', '.join(new[:6])}" if new else
                  "extends coverage of already represented conditions")
        result.append((best, s, reason))
    return result


def _separated(c: FrameCandidate, chosen: Sequence[FrameCandidate],
               min_separation_s: float, min_hamming: int) -> bool:
    """Reject near duplicates in time and in appearance."""
    for other in chosen:
        if other.recording_id != c.recording_id:
            continue
        if abs(c.timestamp_ns - other.timestamp_ns) < min_separation_s * NS_PER_S:
            return False
        if c.dhash and other.dhash:
            d = int(hamming(np.array([c.dhash], dtype=np.uint64),
                            np.array([other.dhash], dtype=np.uint64))[0])
            if d < min_hamming:
                return False
    return True


def _expected_classes(c: FrameCandidate, threshold: float = 0.002) -> List[str]:
    return sorted(k for k, v in (c.class_fraction or {}).items()
                  if v >= threshold and k != "mapillary_ego_region")


def build_selection(pools: Dict[str, List[FrameCandidate]],
                    per_domain_quota: Dict[str, Dict[str, int]],
                    min_separation_s: float = 4.0,
                    min_hamming: int = 8) -> Dict[str, Any]:
    """Select all three annotation groups for every domain.

    `per_domain_quota` maps domain -> group -> number of frames. Quotas are set by
    the caller from recording *duration* and scientific need, never from how many
    frames each recording happens to contain.
    """
    everything = [c for pool in pools.values() for c in pool]
    strata_fn = build_stratifier(everything)

    selected: List[SelectedFrame] = []
    per_group_pick: Dict[str, List[FrameCandidate]] = {g: [] for g in GROUPS}
    stats: Dict[str, Any] = {}

    for domain, pool in pools.items():
        quotas = per_domain_quota.get(domain, {})
        picked_in_domain: List[FrameCandidate] = []
        for group in GROUPS:
            quota = int(quotas.get(group, 0))
            if quota <= 0:
                continue
            sub = _group_pool(pool, group)
            picks = select_group(
                sub, quota, strata_fn, min_separation_s, min_hamming,
                already_selected=picked_in_domain,
                priority=_group_priority(group))
            for cand, strata, reason in picks:
                picked_in_domain.append(cand)
                per_group_pick[group].append(cand)
                selected.append(SelectedFrame(
                    domain=cand.domain, recording_id=cand.recording_id,
                    source_frame_index=cand.source_frame_index,
                    timestamp_ns=cand.timestamp_ns, timestamp_s=cand.timestamp_s,
                    image_path=cand.image_path,
                    selection_reason=f"[{group}] {reason}",
                    strata_covered=sorted(strata),
                    expected_classes=_expected_classes(cand),
                    failure_mode_candidate=cand.failure_mode_candidate,
                    hand_visibility_candidate=cand.hand_visibility_candidate,
                    confidence=cand.confidence, entropy=cand.entropy,
                    provenance=cand.provenance, split_group=group,
                    route_segment=cand.route_segment,
                    route_progression=cand.route_progression,
                    pairing_id=cand.pairing_id,
                ))
            stats.setdefault(domain, {})[group] = {
                "requested": quota, "selected": len(picks),
                "pool_size": len(sub),
                "shortfall_reason": (
                    None if len(picks) >= quota else
                    "the pool ran out of frames separated enough in time and "
                    "appearance to remain non-duplicated"),
            }

    coverage: Dict[str, int] = {}
    for s in selected:
        for stratum in s.strata_covered:
            coverage[stratum] = coverage.get(stratum, 0) + 1

    return {
        "status": "pre_annotation_selection",
        "is_ground_truth": False,
        "balancing_rule": (
            "quotas are assigned per domain and per group by the caller from "
            "recording duration and scientific need; the motorcycle does not receive "
            "more frames merely because it is sampled at a higher rate"),
        "deduplication": {
            "min_temporal_separation_s": min_separation_s,
            "min_perceptual_hash_distance": min_hamming,
        },
        "selected": [s.to_dict() for s in selected],
        "counts": {
            "total": len(selected),
            "per_domain": {d: sum(1 for s in selected if s.domain == d)
                           for d in pools},
            "per_group": {g: sum(1 for s in selected if s.split_group == g)
                          for g in GROUPS},
        },
        "per_domain_group_stats": stats,
        "stratum_coverage": dict(sorted(coverage.items())),
        "uncovered_note": (
            "a stratum with a low count is a genuine scarcity in the source "
            "recordings, not a sampling bug"),
    }


def _group_pool(pool: Sequence[FrameCandidate], group: str) -> List[FrameCandidate]:
    """Restrict the pool to frames that are useful for a given group."""
    if group == "failure_mode_review":
        return [c for c in pool if c.failure_mode_candidate
                or (c.entropy is not None and c.conflict_fraction is not None)]
    if group == "cockpit_training":
        # frames where ego structure is actually present; a frame with no cockpit
        # teaches the cockpit model nothing except background
        return [c for c in pool
                if (c.class_fraction or {}).get("mapillary_ego_region", 0.0)
                + (c.class_fraction or {}).get("mirror", 0.0)
                + (c.class_fraction or {}).get("instrument_display", 0.0)
                + (c.class_fraction or {}).get("control_and_ego_vehicle", 0.0) > 0.005
                or c.hand_visibility_candidate]
    return list(pool)


def _group_priority(group: str) -> Callable[[FrameCandidate], float]:
    if group == "failure_mode_review":
        return lambda c: float(c.entropy or 0.0) + float(c.conflict_fraction or 0.0)
    if group == "cockpit_training":
        return lambda c: float(
            (c.class_fraction or {}).get("mapillary_ego_region", 0.0)
            + (c.class_fraction or {}).get("mirror", 0.0)
            + (c.class_fraction or {}).get("instrument_display", 0.0))
    return lambda c: float(len(_expected_classes(c)))
