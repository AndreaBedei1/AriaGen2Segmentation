"""Automatic, verifiable selection of a continuous analysis segment.

The selection is a ranking, not a single opaque pick: every candidate window keeps
the measurements that produced its score, so a reviewer can disagree with the
weights and re-rank without re-running anything.

The scoring deliberately does **not** reward "easy" footage. Windows that are
static, empty or visually monotonous score low; windows with real driving, traffic,
visual change and a visible cockpit score high. Quality gates (gaps, exposure,
sharpness) are hard constraints applied before scoring, so a beautiful but broken
window is rejected rather than down-weighted.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from .rgb_scan import RgbScan, hamming
from .timeline import NS_PER_S, frames_for_seconds, measure_rate

# Article 1 classes that indicate a semantically rich external scene.
TRAFFIC_CLASSES = ("vehicle", "two_wheeler", "pedestrian", "traffic_light",
                   "traffic_sign")
# The scouting pass runs the external model only, which has no cockpit classes: it
# marks the ego structure through the Mapillary Ego Vehicle / Car Mount labels.
# That region is external provenance, never a scientific `control_and_ego_vehicle`.
COCKPIT_CLASSES = ("mirror", "instrument_display", "control_and_ego_vehicle",
                   "mapillary_ego_region")


@dataclass
class SegmentCandidate:
    rank: int
    recording_id: str
    domain: str
    start_frame_index: int
    end_frame_index: int
    start_timestamp_ns: int
    end_timestamp_ns: int
    duration_s: float
    frame_count: int
    expected_frame_count: int
    missing_frame_estimate: int
    largest_gap_ms: float
    gaps_over_1p5_periods: int
    mean_blur_variance: float
    low_sharpness_fraction: float
    mean_luminance: float
    dark_fraction: float
    clipped_fraction: float
    mean_frame_difference: float
    visual_diversity: float
    static_structure_fraction: float
    semantic_diversity: Optional[float] = None
    traffic_class_fraction: Optional[float] = None
    lane_marking_fraction: Optional[float] = None
    cockpit_proxy_fraction: Optional[float] = None
    mirror_proxy_fraction: Optional[float] = None
    instrument_proxy_fraction: Optional[float] = None
    hand_tracked_fraction: Optional[float] = None
    hand_proxy_frames: Optional[int] = None
    passes_quality_gate: bool = True
    rejection_reasons: List[str] = field(default_factory=list)
    score: float = 0.0
    score_components: Dict[str, float] = field(default_factory=dict)
    rationale: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def static_structure_mask(thumbnails: np.ndarray, std_threshold: float = 8.0
                          ) -> np.ndarray:
    """Pixels that barely change across a set of thumbnails (ego structure proxy)."""
    if thumbnails.shape[0] < 2:
        return np.zeros(thumbnails.shape[1:], dtype=bool)
    return thumbnails.astype(np.float32).std(axis=0) < std_threshold


def _visual_diversity(dhashes: np.ndarray, samples: int = 24) -> float:
    """Mean pairwise perceptual distance of frames sampled across the window."""
    if dhashes.size < 2:
        return 0.0
    idx = np.unique(np.linspace(0, dhashes.size - 1, min(samples, dhashes.size)).astype(int))
    sel = dhashes[idx]
    a, b = np.meshgrid(sel, sel, indexing="ij")
    iu = np.triu_indices(sel.size, k=1)
    if iu[0].size == 0:
        return 0.0
    return float(np.mean(hamming(a[iu], b[iu])) / 64.0)


def _entropy(fractions: np.ndarray) -> float:
    p = np.asarray(fractions, dtype=np.float64)
    p = p[p > 0]
    if p.size == 0:
        return 0.0
    p = p / p.sum()
    return float(-(p * np.log(p)).sum() / np.log(len(p)) if len(p) > 1 else 0.0)


def build_candidates(scan: RgbScan, domain: str, duration_s: float = 30.0,
                     stride_s: float = 5.0,
                     scout: Optional[Dict[str, Any]] = None,
                     hand_tracked_by_frame: Optional[Dict[int, bool]] = None,
                     max_missing_fraction: float = 0.02,
                     max_gap_ms: float = 250.0) -> List[SegmentCandidate]:
    """Slide a fixed-duration window over the recording and measure every position.

    Both the window and the stride are given in seconds; the corresponding frame
    counts are derived from the recording's own measured rate.
    """
    ts = scan.timestamp_ns
    rate = measure_rate(ts)
    fps = rate.effective_fps
    if fps is None:
        raise RuntimeError(f"{scan.recording_id}: cannot measure the RGB rate")
    median_dt_ns = (rate.median_dt_ms or 0.0) * 1e6

    window_frames = frames_for_seconds(duration_s, fps)
    stride_frames = frames_for_seconds(stride_s, fps)
    expected = window_frames

    global_static = static_structure_mask(scan.thumbnails[::max(
        1, scan.thumbnails.shape[0] // 200)])
    blur_median = float(np.median(scan.blur_variance))

    scout_ts = scout_frac = scout_names = None
    if scout:
        scout_ts = np.asarray(scout["timestamp_ns"], dtype=np.int64)
        scout_frac = np.asarray(scout["class_fraction"], dtype=np.float32)
        scout_names = list(scout["class_names"])

    candidates: List[SegmentCandidate] = []
    n = int(ts.size)
    for start in range(0, max(1, n - window_frames + 1), max(1, stride_frames)):
        end = start + window_frames - 1
        if end >= n:
            break
        sl = slice(start, end + 1)
        w_ts = ts[sl]
        actual_duration = float((w_ts[-1] - w_ts[0]) / NS_PER_S)

        dt = np.diff(w_ts)
        largest_gap = float(np.max(dt) / 1e6) if dt.size else 0.0
        gaps_15 = int(np.sum(dt > 1.5 * median_dt_ns)) if median_dt_ns > 0 else 0
        missing = int(np.sum(np.maximum(0, np.round(dt / median_dt_ns) - 1))) \
            if median_dt_ns > 0 else 0

        blur = scan.blur_variance[sl]
        lum = scan.mean_luminance[sl]
        diff = scan.frame_difference[sl]
        thumbs = scan.thumbnails[sl]

        local_static = static_structure_mask(thumbs[::max(1, thumbs.shape[0] // 40)])
        cand = SegmentCandidate(
            rank=0, recording_id=scan.recording_id, domain=domain,
            start_frame_index=int(scan.frame_index[start]),
            end_frame_index=int(scan.frame_index[end]),
            start_timestamp_ns=int(w_ts[0]), end_timestamp_ns=int(w_ts[-1]),
            duration_s=actual_duration,
            frame_count=int(w_ts.size), expected_frame_count=expected,
            missing_frame_estimate=missing, largest_gap_ms=largest_gap,
            gaps_over_1p5_periods=gaps_15,
            mean_blur_variance=float(np.mean(blur)),
            low_sharpness_fraction=float(np.mean(blur < 0.4 * blur_median)),
            mean_luminance=float(np.mean(lum)),
            dark_fraction=float(np.mean(scan.dark_fraction[sl])),
            clipped_fraction=float(np.mean(scan.bright_fraction[sl])),
            mean_frame_difference=float(np.mean(diff)),
            visual_diversity=_visual_diversity(scan.dhash[sl]),
            static_structure_fraction=float(local_static.mean()),
        )

        if scout_ts is not None:
            inside = (scout_ts >= w_ts[0]) & (scout_ts <= w_ts[-1])
            if np.any(inside):
                frac = scout_frac[inside]
                mean_frac = frac.mean(axis=0)
                cand.semantic_diversity = _entropy(mean_frac)
                cand.traffic_class_fraction = float(sum(
                    mean_frac[scout_names.index(c)] for c in TRAFFIC_CLASSES
                    if c in scout_names))
                if "lane_marking" in scout_names:
                    cand.lane_marking_fraction = float(mean_frac[scout_names.index("lane_marking")])
                cand.cockpit_proxy_fraction = float(sum(
                    mean_frac[scout_names.index(c)] for c in COCKPIT_CLASSES
                    if c in scout_names))
                if "mirror" in scout_names:
                    cand.mirror_proxy_fraction = float(mean_frac[scout_names.index("mirror")])
                if "instrument_display" in scout_names:
                    cand.instrument_proxy_fraction = float(
                        mean_frac[scout_names.index("instrument_display")])

        if hand_tracked_by_frame is not None:
            flags = [hand_tracked_by_frame.get(int(scan.frame_index[i]), False)
                     for i in range(start, end + 1)]
            cand.hand_proxy_frames = int(sum(flags))
            cand.hand_tracked_fraction = float(np.mean(flags)) if flags else 0.0

        # --- hard quality gates ---------------------------------------------
        if missing > max_missing_fraction * expected:
            cand.passes_quality_gate = False
            cand.rejection_reasons.append(
                f"{missing} estimated missing frames exceeds "
                f"{max_missing_fraction:.1%} of the window")
        if largest_gap > max_gap_ms:
            cand.passes_quality_gate = False
            cand.rejection_reasons.append(
                f"largest interval {largest_gap:.1f} ms exceeds {max_gap_ms} ms")
        if cand.mean_luminance < 0.10 or cand.mean_luminance > 0.90:
            cand.passes_quality_gate = False
            cand.rejection_reasons.append(
                f"mean luminance {cand.mean_luminance:.2f} outside the usable range")
        if cand.low_sharpness_fraction > 0.5:
            cand.passes_quality_gate = False
            cand.rejection_reasons.append(
                f"{cand.low_sharpness_fraction:.0%} of frames below 40% of the "
                "median sharpness")

        candidates.append(cand)

    _score(candidates, global_static)
    candidates.sort(key=lambda c: (c.passes_quality_gate, c.score), reverse=True)
    for i, c in enumerate(candidates):
        c.rank = i + 1
    return candidates


# Score weights. Motion and visual/semantic variety dominate on purpose so the
# ranking cannot be won by a stationary, visually trivial window.
WEIGHTS = {
    "motion": 0.20,
    "visual_diversity": 0.20,
    "semantic_diversity": 0.15,
    "traffic": 0.15,
    "cockpit": 0.10,
    "sharpness": 0.10,
    "hands": 0.05,
    "continuity": 0.05,
}


def _norm01(values: Sequence[Optional[float]]) -> np.ndarray:
    arr = np.array([np.nan if v is None else float(v) for v in values], dtype=np.float64)
    if np.all(np.isnan(arr)):
        return np.zeros(arr.shape)
    lo = np.nanmin(arr)
    hi = np.nanmax(arr)
    out = np.zeros(arr.shape) if hi <= lo else (arr - lo) / (hi - lo)
    return np.nan_to_num(out, nan=0.0)


def _score(candidates: List[SegmentCandidate], global_static: np.ndarray) -> None:
    if not candidates:
        return
    motion = _norm01([c.mean_frame_difference for c in candidates])
    visual = _norm01([c.visual_diversity for c in candidates])
    semantic = _norm01([c.semantic_diversity for c in candidates])
    traffic = _norm01([c.traffic_class_fraction for c in candidates])
    cockpit = _norm01([c.cockpit_proxy_fraction if c.cockpit_proxy_fraction is not None
                       else c.static_structure_fraction for c in candidates])
    sharp = _norm01([c.mean_blur_variance for c in candidates])
    hands = _norm01([c.hand_tracked_fraction for c in candidates])
    continuity = np.array([1.0 - min(1.0, c.missing_frame_estimate /
                                     max(1, c.expected_frame_count))
                           for c in candidates])

    for i, c in enumerate(candidates):
        comps = {
            "motion": WEIGHTS["motion"] * float(motion[i]),
            "visual_diversity": WEIGHTS["visual_diversity"] * float(visual[i]),
            "semantic_diversity": WEIGHTS["semantic_diversity"] * float(semantic[i]),
            "traffic": WEIGHTS["traffic"] * float(traffic[i]),
            "cockpit": WEIGHTS["cockpit"] * float(cockpit[i]),
            "sharpness": WEIGHTS["sharpness"] * float(sharp[i]),
            "hands": WEIGHTS["hands"] * float(hands[i]),
            "continuity": WEIGHTS["continuity"] * float(continuity[i]),
        }
        c.score_components = comps
        c.score = float(sum(comps.values()))
        c.rationale = _rationale(c, comps)


def _rationale(c: SegmentCandidate, comps: Dict[str, float]) -> List[str]:
    out = []
    top = sorted(comps.items(), key=lambda kv: kv[1], reverse=True)[:3]
    out.append("strongest contributions: " +
               ", ".join(f"{k} {v:.3f}" for k, v in top))
    out.append(f"{c.frame_count} frames over {c.duration_s:.2f} s "
               f"({c.missing_frame_estimate} estimated missing, "
               f"largest interval {c.largest_gap_ms:.1f} ms)")
    if c.traffic_class_fraction is not None:
        out.append(f"external traffic proxy covers {c.traffic_class_fraction:.2%} "
                   "of the sampled pixels")
    if c.cockpit_proxy_fraction is not None:
        out.append(f"cockpit proxy covers {c.cockpit_proxy_fraction:.2%} of the "
                   "sampled pixels")
    if c.hand_tracked_fraction is not None:
        out.append(f"hand tracking reports a hand in {c.hand_tracked_fraction:.1%} "
                   "of the window's frames (proxy signal, not visibility ground truth)")
    if not c.passes_quality_gate:
        out.append("REJECTED: " + "; ".join(c.rejection_reasons))
    return out


def select_segment(candidates: List[SegmentCandidate],
                   min_separation_s: float = 20.0) -> Dict[str, Any]:
    """Pick the best passing candidate and report a non-overlapping shortlist."""
    passing = [c for c in candidates if c.passes_quality_gate]
    if not passing:
        return {"selected": None,
                "reason": "no candidate window passed the quality gates",
                "shortlist": [c.to_dict() for c in candidates[:5]]}

    shortlist: List[SegmentCandidate] = []
    for c in passing:
        if all(abs(c.start_timestamp_ns - s.start_timestamp_ns) >=
               min_separation_s * NS_PER_S for s in shortlist):
            shortlist.append(c)
        if len(shortlist) >= 5:
            break

    best = shortlist[0]
    return {
        "selected": best.to_dict(),
        "selection_rule": (
            "highest weighted score among windows passing every quality gate; "
            "weights favour motion, visual and semantic variety over easy footage"),
        "weights": WEIGHTS,
        "shortlist": [c.to_dict() for c in shortlist],
        "evaluated_windows": len(candidates),
        "passing_windows": len(passing),
    }
