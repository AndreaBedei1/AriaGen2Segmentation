"""Failure-mode detection over a semantic-camera run.

These are **candidate** failure modes derived from the model's own outputs and from
image statistics. Without reviewed ground truth none of them is a confirmed error,
so the vocabulary stays deliberately pre-GT: coverage, stability, fragmentation,
persistence, agreement, fallback share, confidence and entropy.

Causality note: this module is an *offline diagnostic* over an already-produced run.
It deliberately looks at neighbouring frames in both directions in order to describe
flicker and trailing. It is never part of the scientific segmentation path and its
output never feeds back into a mask.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from ..io_utils import read_mask_u16
from ..taxonomy import Taxonomy
from .timeline import frames_for_seconds, measure_rate

# Every failure mode the motorcycle QA is required to cover.
FAILURE_MODES = (
    "missing_external_segmentation",
    "wrong_class_candidate",
    "fragmentation",
    "flicker",
    "temporal_trail",
    "internal_external_fusion_error",
    "false_cockpit_mask",
    "cockpit_absorbed_by_external",
    "mirror_not_detected",
    "instrument_display_not_detected",
    "handlebar_classified_as_other",
    "visible_hand_not_proposed",
    "false_hand",
    "hand_mask_propagated_after_disappearance",
    "excessive_fallback",
    "high_entropy",
    "anomalous_confidence",
    "blur_related_failure",
    "vibration_related_failure",
)

COCKPIT_IDS_BY_NAME = ("mirror", "instrument_display", "control_and_ego_vehicle")


@dataclass
class FrameDiagnostics:
    frame_index: int
    timestamp_ns: int
    class_fraction: Dict[str, float]
    component_count: Dict[str, int]
    mean_confidence: float
    mean_entropy: float
    fallback_fraction: float
    conflict_fraction: float
    external_fraction: float
    internal_model_fraction: float
    geometric_fraction: float
    dense_fill_fraction: float
    blur_variance: Optional[float] = None
    frame_difference: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class FailureEvent:
    mode: str
    frame_index: int
    timestamp_ns: int
    severity: float          # 0-1, relative to the run's own distribution
    evidence: Dict[str, Any]
    description: str
    is_confirmed_error: bool = False
    requires_human_review: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def load_diagnostics(semantic_dir: str | Path, taxonomy: Taxonomy,
                     mask_subdir: str = "final_masks",
                     scan_lookup: Optional[Dict[int, Dict[str, float]]] = None
                     ) -> List[FrameDiagnostics]:
    """Read a semantic-camera run into per-frame diagnostics."""
    import cv2

    root = Path(semantic_dir)
    metas = sorted((root / "metadata").glob("*.json"))
    names = taxonomy.names()
    out: List[FrameDiagnostics] = []
    for path in metas:
        meta = json.loads(path.read_text())
        mask = read_mask_u16(root / mask_subdir / f"{path.stem}.png")
        total = mask.size
        fraction, components = {}, {}
        for cid, name in enumerate(names):
            binary = (mask == cid).astype(np.uint8)
            hits = int(binary.sum())
            fraction[name] = hits / total
            if hits:
                n, _, _, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
                components[name] = int(max(0, n - 1))
            else:
                components[name] = 0
        extra = (scan_lookup or {}).get(int(meta["frame_index"]), {})
        out.append(FrameDiagnostics(
            frame_index=int(meta["frame_index"]),
            timestamp_ns=int(meta["capture_timestamp_ns"]),
            class_fraction=fraction, component_count=components,
            mean_confidence=float(meta.get("mean_final_confidence", float("nan"))),
            mean_entropy=float(meta.get("mean_final_entropy", float("nan"))),
            fallback_fraction=float(meta.get("geometric_proxy_selected_fraction", 0.0)),
            conflict_fraction=float(meta.get("conflict_fraction", 0.0)),
            external_fraction=float(meta.get("external_selected_fraction", 0.0)),
            internal_model_fraction=float(
                meta.get("internal_model_selected_fraction", 0.0)),
            geometric_fraction=float(
                meta.get("geometric_proxy_selected_fraction", 0.0)),
            dense_fill_fraction=float(meta.get("dense_fill_fraction", 0.0)),
            blur_variance=extra.get("blur_variance"),
            frame_difference=extra.get("frame_difference"),
        ))
    return out


def _upper(values: Sequence[float], q: float = 0.9) -> float:
    arr = np.asarray([v for v in values if v is not None and np.isfinite(v)])
    return float(np.quantile(arr, q)) if arr.size else float("inf")


def _lower(values: Sequence[float], q: float = 0.1) -> float:
    arr = np.asarray([v for v in values if v is not None and np.isfinite(v)])
    return float(np.quantile(arr, q)) if arr.size else float("-inf")


def detect(diagnostics: Sequence[FrameDiagnostics], taxonomy: Taxonomy,
           hand_states: Optional[Dict[int, Dict[str, str]]] = None,
           hand_masks_present: Optional[Dict[int, bool]] = None,
           trail_window_s: float = 0.7) -> Dict[str, Any]:
    """Detect candidate failure modes across a run."""
    if not diagnostics:
        return {"events": [], "summary": {}}

    diagnostics = sorted(diagnostics, key=lambda d: d.frame_index)
    ts = np.array([d.timestamp_ns for d in diagnostics], dtype=np.int64)
    rate = measure_rate(ts)
    fps = rate.effective_fps or 0.0
    # the trailing window is declared in seconds and converted with THIS run's rate
    trail_frames = frames_for_seconds(trail_window_s, fps) if fps > 0 else 5

    entropy_hi = _upper([d.mean_entropy for d in diagnostics], 0.9)
    conf_lo = _lower([d.mean_confidence for d in diagnostics], 0.1)
    fallback_hi = _upper([d.fallback_fraction for d in diagnostics], 0.9)
    blur_lo = _lower([d.blur_variance for d in diagnostics], 0.15)
    motion_hi = _upper([d.frame_difference for d in diagnostics], 0.9)

    events: List[FailureEvent] = []

    def add(mode: str, d: FrameDiagnostics, severity: float,
            evidence: Dict[str, Any], description: str) -> None:
        events.append(FailureEvent(mode, d.frame_index, d.timestamp_ns,
                                   float(np.clip(severity, 0.0, 1.0)),
                                   evidence, description))

    frag_thresholds = {
        name: _upper([d.component_count[name] for d in diagnostics], 0.92)
        for name in taxonomy.names()}

    for i, d in enumerate(diagnostics):
        road = d.class_fraction.get("road_surface", 0.0)
        other = d.class_fraction.get("other_environment", 0.0)
        cockpit = sum(d.class_fraction.get(c, 0.0) for c in COCKPIT_IDS_BY_NAME)

        if road < 0.02 and d.dense_fill_fraction > 0.15:
            add("missing_external_segmentation", d,
                d.dense_fill_fraction,
                {"road_surface_fraction": road,
                 "dense_fill_fraction": d.dense_fill_fraction},
                "almost no road surface while a large share of the frame is dense "
                "fill: the external model did not describe the scene here")

        if d.mean_entropy > entropy_hi:
            add("high_entropy", d, (d.mean_entropy - entropy_hi) / max(entropy_hi, 1e-6),
                {"mean_entropy": d.mean_entropy, "run_p90": entropy_hi},
                "final entropy in the top decile of this run")

        if d.mean_confidence < conf_lo:
            add("anomalous_confidence", d, (conf_lo - d.mean_confidence) / max(conf_lo, 1e-6),
                {"mean_confidence": d.mean_confidence, "run_p10": conf_lo},
                "final confidence in the bottom decile of this run")

        if d.fallback_fraction > fallback_hi and d.fallback_fraction > 0.05:
            add("excessive_fallback", d, d.fallback_fraction,
                {"geometric_fallback_fraction": d.fallback_fraction},
                "the geometric cockpit prior, not a model, decided a large share of "
                "the frame")

        if d.conflict_fraction > 0.5:
            add("internal_external_fusion_error", d, d.conflict_fraction,
                {"conflict_fraction": d.conflict_fraction},
                "external and internal streams disagree over most of the frame")

        for name in taxonomy.names():
            n = d.component_count[name]
            if n >= 8 and n > frag_thresholds[name]:
                add("fragmentation", d, min(1.0, n / 40.0),
                    {"class": name, "components": n,
                     "run_p92": frag_thresholds[name]},
                    f"{name} is split into {n} connected components")

        if cockpit < 0.005 and d.geometric_fraction < 0.01:
            add("cockpit_absorbed_by_external", d, 1.0 - cockpit * 100,
                {"cockpit_fraction": cockpit,
                 "external_fraction": d.external_fraction},
                "no cockpit class survives fusion: the ego structure was absorbed "
                "into external classes")

        if d.class_fraction.get("mirror", 0.0) <= 0.0:
            add("mirror_not_detected", d, 0.5, {"mirror_fraction": 0.0},
                "no mirror pixel in this frame")
        if d.class_fraction.get("instrument_display", 0.0) <= 0.0:
            add("instrument_display_not_detected", d, 0.5,
                {"instrument_display_fraction": 0.0},
                "no instrument-display pixel in this frame")

        if (d.blur_variance is not None and d.blur_variance < blur_lo
                and (d.mean_entropy > entropy_hi or d.mean_confidence < conf_lo)):
            add("blur_related_failure", d, 0.6,
                {"blur_variance": d.blur_variance, "run_p15": blur_lo},
                "low sharpness coincides with degraded confidence or entropy")

        if (d.frame_difference is not None and d.frame_difference > motion_hi
                and d.mean_entropy > entropy_hi):
            add("vibration_related_failure", d, 0.6,
                {"frame_difference": d.frame_difference, "run_p90": motion_hi},
                "large inter-frame change coincides with elevated entropy: "
                "consistent with vibration or a fast orientation change")

        # flicker: a class present before and after but absent now
        if 0 < i < len(diagnostics) - 1:
            prev, nxt = diagnostics[i - 1], diagnostics[i + 1]
            for name in taxonomy.names():
                if (prev.class_fraction[name] > 0.002
                        and nxt.class_fraction[name] > 0.002
                        and d.class_fraction[name] <= 0.0):
                    add("flicker", d, 0.5, {"class": name},
                        f"{name} is present in the neighbouring frames but absent "
                        "here")

        # temporal trail: a class persists long after it stopped growing
        if i >= trail_frames:
            window = diagnostics[i - trail_frames:i + 1]
            for name in ("vehicle", "two_wheeler", "pedestrian"):
                series = [w.class_fraction[name] for w in window]
                if series[0] > 0.01 and all(
                        0 < s < 0.4 * series[0] for s in series[1:]):
                    add("temporal_trail", d, 0.5,
                        {"class": name, "start_fraction": series[0],
                         "current_fraction": series[-1],
                         "window_frames": trail_frames,
                         "window_s": trail_window_s},
                        f"{name} shrinks steadily but never disappears over "
                        f"{trail_window_s:.1f} s: candidate temporal trail")

    events.extend(_hand_events(diagnostics, hand_states, hand_masks_present,
                               trail_frames, trail_window_s))

    by_mode: Dict[str, int] = {m: 0 for m in FAILURE_MODES}
    for e in events:
        by_mode[e.mode] = by_mode.get(e.mode, 0) + 1

    duration = rate.duration_s
    return {
        "status": "candidate_failure_modes_pre_ground_truth",
        "is_confirmed_error_list": False,
        "frames": len(diagnostics),
        "duration_s": duration,
        "effective_fps": fps,
        "thresholds": {
            "entropy_p90": entropy_hi, "confidence_p10": conf_lo,
            "fallback_p90": fallback_hi, "blur_p15": blur_lo,
            "motion_p90": motion_hi,
            "trail_window_s": trail_window_s, "trail_window_frames": trail_frames,
        },
        "events": [e.to_dict() for e in events],
        "counts_per_mode": by_mode,
        "rate_per_second_per_mode": {
            k: (v / duration if duration > 0 else None) for k, v in by_mode.items()},
        "note": ("counts are normalised per second so runs sampled at different "
                 "rates remain comparable"),
    }


def _hand_events(diagnostics: Sequence[FrameDiagnostics],
                 hand_states: Optional[Dict[int, Dict[str, str]]],
                 hand_masks_present: Optional[Dict[int, bool]],
                 trail_frames: int, trail_window_s: float) -> List[FailureEvent]:
    """Hand-specific candidates, built so that absence is not treated as failure."""
    if not hand_states:
        return []
    out: List[FailureEvent] = []
    consecutive_mask_without_hand = 0
    for d in diagnostics:
        states = hand_states.get(d.frame_index, {})
        mask_present = bool((hand_masks_present or {}).get(d.frame_index, False))
        visible = any(s in ("visible", "partially_visible") for s in states.values())

        if visible and not mask_present:
            out.append(FailureEvent(
                "visible_hand_not_proposed", d.frame_index, d.timestamp_ns, 0.6,
                {"hand_states": states},
                "a hand is a visibility candidate here but the cockpit proxy "
                "proposes no hand region"))
        if mask_present and not visible:
            out.append(FailureEvent(
                "false_hand", d.frame_index, d.timestamp_ns, 0.7,
                {"hand_states": states},
                "a hand region is proposed while no hand is a visibility "
                "candidate: candidate false mask"))
            consecutive_mask_without_hand += 1
            if consecutive_mask_without_hand >= trail_frames:
                out.append(FailureEvent(
                    "hand_mask_propagated_after_disappearance", d.frame_index,
                    d.timestamp_ns, 0.8,
                    {"consecutive_frames": consecutive_mask_without_hand,
                     "window_s": trail_window_s},
                    "a hand region has persisted with no visibility support for "
                    f"more than {trail_window_s:.1f} s"))
        else:
            consecutive_mask_without_hand = 0
    return out


def pick_sequences(result: Dict[str, Any], modes: Sequence[str],
                   length: int = 5, per_mode: int = 1) -> List[Dict[str, Any]]:
    """Choose consecutive-frame windows illustrating the main failure modes."""
    events = result.get("events", [])
    chosen: List[Dict[str, Any]] = []
    for mode in modes:
        candidates = sorted((e for e in events if e["mode"] == mode),
                            key=lambda e: -e["severity"])
        used: List[int] = []
        for e in candidates:
            centre = e["frame_index"]
            if any(abs(centre - u) < length for u in used):
                continue
            half = length // 2
            chosen.append({
                "mode": mode,
                "centre_frame_index": centre,
                "frame_indices": list(range(centre - half, centre - half + length)),
                "severity": e["severity"],
                "description": e["description"],
                "evidence": e["evidence"],
            })
            used.append(centre)
            if len(used) >= per_mode:
                break
    return chosen
