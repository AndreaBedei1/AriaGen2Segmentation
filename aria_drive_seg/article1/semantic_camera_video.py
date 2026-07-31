"""Offline semantic-camera fusion and bidirectional presentation stabilization.

The scientific semantic-camera output is a read-only input.  This module writes a
separate non-causal presentation product that may use past and future frames.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import resource
import time
from typing import Any, Sequence

import cv2
import numpy as np

from ..config import Config
from ..hashing import stable_hash
from ..io_utils import (
    atomic_write, atomic_write_bytes, atomic_write_json, read_mask_u16,
    write_mask_u16,
)
from ..segmentation.base import iter_frames
from .optical_flow import (
    compute_optical_flow, validate_flow, warp_with_backward,
)
from .semantic_camera import FINAL_CLASS_NAMES


INTERNAL_CLASS_IDS = np.asarray([10, 11, 12], dtype=np.uint16)
THIN_CLASS_IDS = np.asarray([2, 3], dtype=np.uint16)

IMPROVED_PROVENANCE = {
    0: "invalid_unassigned",
    1: "current_scientific_fusion",
    2: "improved_internal_win",
    3: "protected_external_win",
    4: "dense_other_environment_fill",
}

PRESENTATION_PROVENANCE = {
    0: "invalid_unassigned",
    1: "improved_current",
    2: "forward_from_past",
    3: "backward_from_future",
    4: "bidirectional_vote",
    5: "hysteresis_hold",
    6: "protected_current_external",
}


@dataclass
class ImprovedFusionResult:
    mask: np.ndarray
    confidence: np.ndarray
    provenance: np.ndarray
    internal_added: np.ndarray
    stats: dict[str, Any]

    def validate(self) -> None:
        shape = self.mask.shape
        if any(value.shape != shape for value in (
                self.confidence, self.provenance, self.internal_added)):
            raise ValueError("improved fusion geometry mismatch")
        if np.any(self.mask == 0) or np.any(
                self.mask >= len(FINAL_CLASS_NAMES)):
            raise ValueError("improved fusion is not a dense Article 1 mask")
        if np.any(self.provenance == 0):
            raise ValueError("improved fusion provenance is incomplete")


@dataclass
class PairFlow:
    """Adjacent-frame flow and validity in both target coordinate systems."""

    forward: np.ndarray
    backward: np.ndarray
    valid_forward: np.ndarray
    valid_backward: np.ndarray

    def validate(self) -> None:
        shape = self.forward.shape[:2]
        if self.forward.shape != (*shape, 2) or \
                self.backward.shape != (*shape, 2):
            raise ValueError("pair flow must have shape HxWx2")
        if self.valid_forward.shape != shape or \
                self.valid_backward.shape != shape:
            raise ValueError("pair-flow validity geometry mismatch")


@dataclass
class PresentationResult:
    mask: np.ndarray
    confidence: np.ndarray
    provenance: np.ndarray
    support_count: np.ndarray
    stats: dict[str, Any]

    def validate(self) -> None:
        shape = self.mask.shape
        if any(value.shape != shape for value in (
                self.confidence, self.provenance, self.support_count)):
            raise ValueError("presentation result geometry mismatch")
        if np.any(self.mask == 0) or np.any(
                self.mask >= len(FINAL_CLASS_NAMES)):
            raise ValueError("presentation result is not dense")
        if np.any(self.provenance == 0):
            raise ValueError("presentation provenance is incomplete")


def _entropy_proxy(confidence: np.ndarray, classes: int) -> np.ndarray:
    top = np.clip(np.asarray(confidence, np.float32), 1e-8, 1.0)
    residual = np.maximum(1.0 - top, 1e-8)
    tail = residual / max(1, classes - 1)
    entropy = -(
        top * np.log(top) + (classes - 1) * tail * np.log(tail))
    return np.clip(entropy / np.log(classes), 0, 1).astype(np.float32)


def _class_lookup(values: dict[str, Any], default: float = 0.0,
                  dtype=np.float32) -> np.ndarray:
    result = np.full(len(FINAL_CLASS_NAMES), default, dtype=dtype)
    for class_id, name in enumerate(FINAL_CLASS_NAMES):
        if name in values:
            result[class_id] = values[name]
    return result


def _validate_dense_inputs(*arrays: np.ndarray) -> tuple[int, int]:
    if not arrays or arrays[0].ndim != 2:
        raise ValueError("semantic-camera masks must be HxW")
    shape = arrays[0].shape
    if any(value.shape != shape for value in arrays[1:]):
        raise ValueError("semantic-camera evidence geometry mismatch")
    return shape


def improve_fusion_frame(
        current_mask: np.ndarray,
        current_confidence: np.ndarray,
        external_mask: np.ndarray,
        external_confidence: np.ndarray,
        internal_mask: np.ndarray,
        internal_confidence: np.ndarray,
        cfg: dict[str, Any],
) -> ImprovedFusionResult:
    """Increase cockpit recall without replacing strong external evidence."""
    h, w = _validate_dense_inputs(
        current_mask, current_confidence, external_mask,
        external_confidence, internal_mask, internal_confidence)
    current = np.asarray(current_mask, np.uint16)
    external = np.asarray(external_mask, np.uint16)
    internal = np.asarray(internal_mask, np.uint16)
    if np.any(current == 0) or np.any(current >= len(FINAL_CLASS_NAMES)):
        raise ValueError("scientific semantic-camera input must already be dense")
    if np.any(internal >= len(FINAL_CLASS_NAMES)):
        raise ValueError("internal evidence contains an invalid class")
    if np.any((internal > 0) & ~np.isin(internal, INTERNAL_CLASS_IDS)):
        raise ValueError("internal evidence must use mirror/display/control classes")

    current_conf = np.clip(
        np.asarray(current_confidence, np.float32), 0, 1)
    external_conf = np.clip(
        np.asarray(external_confidence, np.float32), 0, 1)
    internal_conf = np.clip(
        np.asarray(internal_confidence, np.float32), 0, 1)
    entropy_penalty = float(cfg.get("entropy_penalty", .12))
    external_score = (
        external_conf
        - entropy_penalty * _entropy_proxy(
            external_conf, len(FINAL_CLASS_NAMES))
        + _class_lookup(cfg.get("external_priorities", {}))[external] / 1000)
    internal_score = (
        internal_conf
        - entropy_penalty * _entropy_proxy(internal_conf, 4)
        + _class_lookup(cfg.get("internal_priorities", {}))[internal] / 1000)

    y, x = np.indices((h, w))
    bottom = y >= int(round(
        h * float(cfg.get("cockpit_bottom_start_fraction", .58))))
    side_fraction = float(cfg.get("cockpit_side_fraction", .20))
    sides = (x < round(w * side_fraction)) | \
        (x >= round(w * (1.0 - side_fraction)))
    spatial = bottom | sides
    internal_score += spatial * float(cfg.get(
        "cockpit_position_bonus", .16))

    minimum = _class_lookup(
        cfg.get("internal_min_confidence", {}), default=1.1)
    candidate = (internal > 0) & (internal_conf >= minimum[internal])
    existing_internal = np.isin(current, INTERNAL_CLASS_IDS)
    score_wins = internal_score >= (
        external_score + float(cfg.get("conflict_margin", -.04)))

    protection_threshold = _class_lookup(
        cfg.get("strong_external_thresholds", {}), default=1.1)
    strong_external = (
        (external > 0)
        & (external_conf >= protection_threshold[external]))
    high_internal_override = (
        spatial
        & np.isin(internal, [10, 11])
        & (internal_conf >= float(cfg.get(
            "strong_external_internal_override_confidence", .78))))
    internal_wins = (
        existing_internal
        | (candidate & score_wins & (~strong_external | high_internal_override))
    )

    result_mask = current.copy()
    result_confidence = current_conf.copy()
    provenance = np.full(current.shape, 1, np.uint8)
    added = internal_wins & (current != internal)
    result_mask[internal_wins] = internal[internal_wins]
    result_confidence[internal_wins] = np.maximum(
        current_conf[internal_wins], internal_conf[internal_wins])
    provenance[internal_wins] = 2
    protected = candidate & strong_external & \
        ~high_internal_override & ~existing_internal
    provenance[protected] = 3

    dense_fill = result_mask == 0
    result_mask[dense_fill] = FINAL_CLASS_NAMES.index("other_environment")
    result_confidence[dense_fill] = 0.0
    provenance[dense_fill] = 4
    stats = {
        "dense_coverage": float((result_mask > 0).mean()),
        "changed_fraction": float((result_mask != current).mean()),
        "internal_fraction_before": float(existing_internal.mean()),
        "internal_fraction_after": float(
            np.isin(result_mask, INTERNAL_CLASS_IDS).mean()),
        "internal_added_fraction": float(added.mean()),
        "protected_external_fraction": float(protected.mean()),
    }
    result = ImprovedFusionResult(
        result_mask, np.clip(result_confidence, 0, 1),
        provenance, added, stats)
    result.validate()
    return result


def identity_pair_flow(shape: tuple[int, int]) -> PairFlow:
    flow = np.zeros((*shape, 2), np.float32)
    valid = np.ones(shape, bool)
    return PairFlow(flow.copy(), flow.copy(), valid.copy(), valid.copy())


def warp_source_to_target(
        mask: np.ndarray,
        confidence: np.ndarray,
        source_index: int,
        target_index: int,
        pair_flows: Sequence[PairFlow],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Warp a source frame to a target through adjacent forward/backward flow."""
    current_mask = np.asarray(mask, np.uint16)
    current_confidence = np.asarray(confidence, np.float32)
    valid = np.ones(current_mask.shape, np.uint8)
    if source_index < target_index:
        for pair_index in range(source_index, target_index):
            pair = pair_flows[pair_index]
            pair.validate()
            current_mask = warp_with_backward(
                current_mask, pair.backward,
                cv2.INTER_NEAREST).astype(np.uint16)
            current_confidence = warp_with_backward(
                current_confidence, pair.backward)
            valid = (
                warp_with_backward(
                    valid, pair.backward, cv2.INTER_NEAREST) > 0
            ) & pair.valid_backward
            valid = valid.astype(np.uint8)
    elif source_index > target_index:
        for pair_index in range(source_index - 1, target_index - 1, -1):
            pair = pair_flows[pair_index]
            pair.validate()
            current_mask = warp_with_backward(
                current_mask, pair.forward,
                cv2.INTER_NEAREST).astype(np.uint16)
            current_confidence = warp_with_backward(
                current_confidence, pair.forward)
            valid = (
                warp_with_backward(
                    valid, pair.forward, cv2.INTER_NEAREST) > 0
            ) & pair.valid_forward
            valid = valid.astype(np.uint8)
    return current_mask, current_confidence, valid.astype(bool)


def _add_votes(scores: np.ndarray, counts: np.ndarray,
               mask: np.ndarray, values: np.ndarray,
               eligible: np.ndarray) -> None:
    flat_mask = mask.ravel().astype(np.int64)
    pixels = np.arange(flat_mask.size, dtype=np.int64)
    active = eligible.ravel() & (flat_mask > 0) & \
        (flat_mask < scores.shape[0])
    offsets = flat_mask[active] * flat_mask.size + pixels[active]
    scores.reshape(-1)[offsets] += values.ravel()[active]
    counts.reshape(-1)[offsets] += 1


def stabilize_presentation_frame(
        target_index: int,
        masks: Sequence[np.ndarray],
        confidences: Sequence[np.ndarray],
        external_masks: Sequence[np.ndarray],
        external_confidences: Sequence[np.ndarray],
        pair_flows: Sequence[PairFlow],
        cfg: dict[str, Any],
) -> PresentationResult:
    """Non-causal local vote with class TTL and activation/deactivation hysteresis."""
    count = len(masks)
    if count == 0 or len(confidences) != count or \
            len(external_masks) != count or \
            len(external_confidences) != count:
        raise ValueError("presentation evidence lists must have equal length")
    if not 0 <= target_index < count:
        raise IndexError(target_index)
    if len(pair_flows) != max(0, count - 1):
        raise ValueError("one pair flow is required per adjacent frame pair")
    shape = _validate_dense_inputs(
        masks[target_index], confidences[target_index],
        external_masks[target_index], external_confidences[target_index])
    current = np.asarray(masks[target_index], np.uint16)
    current_conf = np.clip(
        np.asarray(confidences[target_index], np.float32), 0, 1)
    if np.any(current == 0):
        raise ValueError("presentation input must be dense")

    class_cfg = cfg.get("classes", {})
    ttl = np.zeros(len(FINAL_CLASS_NAMES), np.int16)
    vote_weight = np.ones(len(FINAL_CLASS_NAMES), np.float32)
    minimum_confidence = np.zeros(len(FINAL_CLASS_NAMES), np.float32)
    decay = np.ones(len(FINAL_CLASS_NAMES), np.float32)
    for class_id, name in enumerate(FINAL_CLASS_NAMES):
        values = class_cfg.get(name, {})
        ttl[class_id] = int(values.get("ttl_frames", 0))
        vote_weight[class_id] = float(values.get("vote_weight", 1.0))
        minimum_confidence[class_id] = float(
            values.get("minimum_confidence", 0.0))
        decay[class_id] = float(values.get(
            "temporal_decay", cfg.get("temporal_decay", .86)))
    radius = min(
        int(cfg.get("window_radius_frames", int(ttl.max(initial=0)))),
        int(ttl.max(initial=0)))
    classes = len(FINAL_CLASS_NAMES)
    scores = np.zeros((classes, *shape), np.float32)
    past_scores = np.zeros_like(scores)
    future_scores = np.zeros_like(scores)
    support_counts = np.zeros((classes, *shape), np.uint8)
    current_floor = float(cfg.get("current_confidence_floor", .30))
    current_values = (
        np.maximum(current_conf, current_floor)
        * vote_weight[current]
        * float(cfg.get("current_weight", 1.0))
    )
    _add_votes(
        scores, support_counts, current, current_values,
        np.ones(shape, bool))

    for source_index in range(
            max(0, target_index - radius),
            min(count, target_index + radius + 1)):
        if source_index == target_index:
            continue
        distance = abs(source_index - target_index)
        warped_mask, warped_conf, valid = warp_source_to_target(
            masks[source_index], confidences[source_index],
            source_index, target_index, pair_flows)
        eligible = (
            valid
            & (distance <= ttl[warped_mask])
            & (warped_conf >= minimum_confidence[warped_mask])
        )
        values = (
            warped_conf
            * vote_weight[warped_mask]
            * np.power(decay[warped_mask], distance)
        )
        direction_scores = (
            past_scores if source_index < target_index else future_scores)
        _add_votes(
            direction_scores, support_counts, warped_mask, values, eligible)
        _add_votes(
            scores, np.zeros_like(support_counts),
            warped_mask, values, eligible)

    candidate = scores.argmax(0).astype(np.uint16)
    best_score = np.take_along_axis(
        scores, candidate[None], 0)[0]
    current_score = np.take_along_axis(
        scores, current[None], 0)[0]
    total_score = scores.sum(0)
    switch = candidate != current
    margin = best_score - current_score
    allowed = switch & (
        margin >= float(cfg.get("switch_margin", .06)))

    winner_support = np.take_along_axis(
        support_counts, candidate[None], 0)[0]
    internal_candidate = np.isin(candidate, INTERNAL_CLASS_IDS)
    internal_activation = (
        internal_candidate
        & (winner_support >= int(cfg.get(
            "internal_minimum_support_frames", 1)))
        & (best_score >= current_score - float(cfg.get(
            "internal_activation_relaxation", .18)))
    )
    allowed |= switch & internal_activation
    thin_candidate = np.isin(candidate, THIN_CLASS_IDS)
    allowed |= (
        switch & thin_candidate
        & (margin >= float(cfg.get("thin_switch_margin", -.02)))
        & (winner_support >= int(cfg.get("thin_minimum_support_frames", 1)))
    )

    current_internal = np.isin(current, INTERNAL_CLASS_IDS)
    keep_current_internal = (
        current_internal & switch
        & (
            (current_conf >= float(cfg.get(
                "internal_current_keep_confidence", .20)))
            | (current_score >= best_score * float(cfg.get(
                "internal_deactivation_ratio", .42)))
        )
    )
    allowed[keep_current_internal] = False

    external = np.asarray(external_masks[target_index], np.uint16)
    external_conf = np.clip(
        np.asarray(external_confidences[target_index], np.float32), 0, 1)
    thresholds = _class_lookup(
        cfg.get("strong_external_thresholds", {}), default=1.1)
    protected = (
        switch & (external > 0)
        & (external_conf >= thresholds[external])
        & (candidate != external)
    )
    allowed[protected] = False

    output = current.copy()
    output[allowed] = candidate[allowed]
    confidence = np.divide(
        np.take_along_axis(scores, output[None], 0)[0],
        np.maximum(total_score, 1e-8),
        dtype=np.float32)
    provenance = np.full(shape, 1, np.uint8)
    changed = output != current
    rows, cols = np.indices(shape)
    winner_past = past_scores[output, rows, cols] > 0
    winner_future = future_scores[output, rows, cols] > 0
    provenance[changed & winner_past & ~winner_future] = 2
    provenance[changed & ~winner_past & winner_future] = 3
    provenance[changed & winner_past & winner_future] = 4
    provenance[switch & ~allowed & ~protected] = 5
    provenance[protected] = 6
    output_support = support_counts[output, rows, cols]
    stats = {
        "dense_coverage": float((output > 0).mean()),
        "changed_fraction": float(changed.mean()),
        "forward_fraction": float((provenance == 2).mean()),
        "backward_fraction": float((provenance == 3).mean()),
        "bidirectional_fraction": float((provenance == 4).mean()),
        "hysteresis_hold_fraction": float((provenance == 5).mean()),
        "protected_external_fraction": float((provenance == 6).mean()),
        "internal_fraction": float(
            np.isin(output, INTERNAL_CLASS_IDS).mean()),
        "thin_fraction": float(np.isin(output, THIN_CLASS_IDS).mean()),
    }
    result = PresentationResult(
        output, np.clip(confidence, 0, 1), provenance,
        output_support.astype(np.uint8), stats)
    result.validate()
    return result


def compute_presentation_metrics(
        masks: Sequence[np.ndarray],
) -> dict[str, Any]:
    """Image-coordinate, pre-GT temporal diagnostics for one mask sequence."""
    if not masks:
        raise ValueError("metrics require at least one mask")
    shape = masks[0].shape
    if any(mask.shape != shape for mask in masks):
        raise ValueError("metric masks must share geometry")
    switch_sum = 0.0
    flicker_sum = 0
    internal_intersection = internal_previous = 0
    class_iou: dict[str, list[float]] = {
        name: [] for name in FINAL_CLASS_NAMES[1:]}
    for index in range(1, len(masks)):
        previous = masks[index - 1]
        current = masks[index]
        switch_sum += float((previous != current).mean())
        previous_internal = np.isin(previous, INTERNAL_CLASS_IDS)
        current_internal = np.isin(current, INTERNAL_CLASS_IDS)
        internal_intersection += int(
            (previous_internal & current_internal).sum())
        internal_previous += int(previous_internal.sum())
        for class_id, name in enumerate(FINAL_CLASS_NAMES[1:], start=1):
            union = (previous == class_id) | (current == class_id)
            if union.any():
                class_iou[name].append(float(
                    (((previous == class_id) & (current == class_id)).sum())
                     / union.sum()))
    for index in range(1, len(masks) - 1):
        flicker_sum += int(
            ((masks[index - 1] == masks[index + 1])
             & (masks[index] != masks[index - 1])).sum())
    pairs = max(1, len(masks) - 1)
    flicker_denominator = max(1, (len(masks) - 2) * shape[0] * shape[1])
    switch_rate = switch_sum / pairs
    iou_means = {
        name: float(np.mean(values)) if values else 1.0
        for name, values in class_iou.items()}
    internal_iou = float(np.mean([
        iou_means[FINAL_CLASS_NAMES[class_id]]
        for class_id in INTERNAL_CLASS_IDS]))
    lane_continuity = float(np.mean([
        iou_means["lane_marking"],
        iou_means["regulatory_road_marking"],
    ]))
    return {
        "frame_count": len(masks),
        "evaluation_geometry": list(shape),
        "class_switch_rate": switch_rate,
        "temporal_consistency": 1.0 - switch_rate,
        "isolated_one_frame_flicker_rate": (
            flicker_sum / flicker_denominator),
        "isolated_one_frame_flicker_pixels": flicker_sum,
        "internal_class_persistence": (
            internal_intersection / max(1, internal_previous)),
        "internal_mean_temporal_iou": internal_iou,
        "lane_and_marking_continuity": lane_continuity,
        "class_temporal_iou": iou_means,
        "disclaimer": (
            "Pre-GT image-coordinate continuity diagnostics only; lower "
            "flicker can preserve false positives and is not accuracy."),
    }


def compute_flow_aligned_metrics(
        masks: Sequence[np.ndarray],
        pair_flows: Sequence[PairFlow],
) -> dict[str, Any]:
    """Pre-GT continuity after warping each previous mask into current geometry."""
    if not masks or len(pair_flows) != max(0, len(masks) - 1):
        raise ValueError("flow-aligned metrics require one flow per frame pair")
    shape = masks[0].shape
    if any(mask.shape != shape for mask in masks):
        raise ValueError("flow-aligned metric masks must share geometry")
    switch_sum = 0.0
    internal_intersection = internal_previous = 0
    class_iou: dict[str, list[float]] = {
        name: [] for name in FINAL_CLASS_NAMES[1:]}
    for index, pair in enumerate(pair_flows):
        pair.validate()
        previous = warp_with_backward(
            masks[index], pair.backward,
            cv2.INTER_NEAREST).astype(np.uint16)
        current = masks[index + 1]
        valid = pair.valid_backward
        switch_sum += float(
            ((previous != current) & valid).sum() / max(1, valid.sum()))
        previous_internal = np.isin(previous, INTERNAL_CLASS_IDS) & valid
        current_internal = np.isin(current, INTERNAL_CLASS_IDS) & valid
        internal_intersection += int(
            (previous_internal & current_internal).sum())
        internal_previous += int(previous_internal.sum())
        for class_id, name in enumerate(FINAL_CLASS_NAMES[1:], start=1):
            first = (previous == class_id) & valid
            second = (current == class_id) & valid
            union = first | second
            if union.any():
                class_iou[name].append(float(
                    (first & second).sum() / union.sum()))
    pairs = max(1, len(pair_flows))
    switch_rate = switch_sum / pairs
    iou_means = {
        name: float(np.mean(values)) if values else 1.0
        for name, values in class_iou.items()}
    return {
        "class_switch_rate": switch_rate,
        "temporal_consistency": 1.0 - switch_rate,
        "internal_class_persistence": (
            internal_intersection / max(1, internal_previous)),
        "internal_mean_temporal_iou": float(np.mean([
            iou_means[FINAL_CLASS_NAMES[class_id]]
            for class_id in INTERNAL_CLASS_IDS])),
        "lane_and_marking_continuity": float(np.mean([
            iou_means["lane_marking"],
            iou_means["regulatory_road_marking"],
        ])),
        "class_temporal_iou": iou_means,
        "validity": "backward-flow valid pixels only",
        "disclaimer": (
            "Pre-GT flow-aligned continuity diagnostics only; lower switch "
            "rate can preserve false positives and is not accuracy."),
    }


VIDEO_OUTPUT_DIRS = (
    "improved_masks", "improved_confidence", "improved_provenance",
    "presentation_masks", "presentation_confidence",
    "presentation_provenance", "presentation_support",
    "metadata", "flow_cache", "videos",
)


def _read_u8(path: Path) -> np.ndarray:
    value = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if value is None:
        raise FileNotFoundError(path)
    return np.asarray(value, np.float32) / 255.0


def _write_u8(path: Path, value: np.ndarray) -> None:
    encoded_value = np.asarray(value, np.uint8)
    ok, encoded = cv2.imencode(".png", encoded_value)
    if not ok:
        raise RuntimeError(f"could not encode {path}")
    atomic_write_bytes(path, encoded.tobytes())


def _flow_cache_path(out: Path, first: int, second: int) -> Path:
    return out / "flow_cache" / f"flow_{first:06d}_{second:06d}.npz"


def _save_pair_flow(path: Path, pair: PairFlow) -> None:
    pair.validate()
    with atomic_write(path, "wb") as handle:
        np.savez_compressed(
            handle,
            forward=pair.forward.astype(np.float16),
            backward=pair.backward.astype(np.float16),
            valid_forward=pair.valid_forward.astype(np.uint8),
            valid_backward=pair.valid_backward.astype(np.uint8),
        )


def _load_pair_flow(path: Path) -> PairFlow:
    with np.load(path) as data:
        result = PairFlow(
            np.asarray(data["forward"], np.float32),
            np.asarray(data["backward"], np.float32),
            np.asarray(data["valid_forward"], bool),
            np.asarray(data["valid_backward"], bool),
        )
    result.validate()
    return result


def compute_bidirectional_pair_flow(
        previous_rgb: np.ndarray,
        current_rgb: np.ndarray,
        cfg: dict[str, Any],
) -> PairFlow:
    """Compute DIS flow once and validate both target-coordinate directions."""
    flow_cfg = dict(cfg)
    flow_cfg["processing_scale"] = 1.0
    forward_result = compute_optical_flow(
        previous_rgb, current_rgb, flow_cfg)
    reverse_result = validate_flow(
        current_rgb, previous_rgb,
        forward_result.backward, forward_result.forward, flow_cfg)
    result = PairFlow(
        forward_result.forward,
        forward_result.backward,
        reverse_result.valid,
        forward_result.valid,
    )
    result.validate()
    return result


def _source_frame_complete(source: Path, stem: str) -> bool:
    requirements = (
        ("final_masks", ".png"), ("final_confidence", ".png"),
        ("external_masks", ".png"), ("external_confidence", ".png"),
        ("internal_masks", ".png"), ("internal_confidence", ".png"),
        ("metadata", ".json"),
    )
    return all(
        (source / directory / f"{stem}{suffix}").exists()
        for directory, suffix in requirements)


def _improved_complete(out: Path, stem: str) -> bool:
    return all((out / directory / f"{stem}.png").exists() for directory in (
        "improved_masks", "improved_confidence", "improved_provenance"))


def _presentation_complete(out: Path, stem: str) -> bool:
    return all((out / directory / f"{stem}{suffix}").exists()
               for directory, suffix in (
                   ("presentation_masks", ".png"),
                   ("presentation_confidence", ".png"),
                   ("presentation_provenance", ".png"),
                   ("presentation_support", ".png"),
                   ("metadata", ".json")))


def _resize_mask(value: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    return cv2.resize(
        value, size, interpolation=cv2.INTER_NEAREST).astype(np.uint16)


def _resize_confidence(
        value: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    return cv2.resize(
        np.asarray(value, np.float32), size,
        interpolation=cv2.INTER_LINEAR).astype(np.float32)


def _full_presentation_result(
        improved_mask: np.ndarray,
        improved_confidence: np.ndarray,
        external_mask: np.ndarray,
        external_confidence: np.ndarray,
        working_current: np.ndarray,
        working_result: PresentationResult,
        cfg: dict[str, Any],
) -> PresentationResult:
    h, w = improved_mask.shape
    size = (w, h)
    current_up = _resize_mask(working_current, size)
    candidate_up = _resize_mask(working_result.mask, size)
    confidence_up = _resize_confidence(working_result.confidence, size)
    provenance_up = cv2.resize(
        working_result.provenance, size,
        interpolation=cv2.INTER_NEAREST).astype(np.uint8)
    support_up = cv2.resize(
        working_result.support_count, size,
        interpolation=cv2.INTER_NEAREST).astype(np.uint8)
    change = candidate_up != current_up

    class_cfg = cfg.get("classes", {})
    ttl = np.zeros(len(FINAL_CLASS_NAMES), np.int16)
    for class_id, name in enumerate(FINAL_CLASS_NAMES):
        ttl[class_id] = int(class_cfg.get(name, {}).get("ttl_frames", 0))
    change &= ttl[candidate_up] > 0
    thresholds = _class_lookup(
        cfg.get("strong_external_thresholds", {}), default=1.1)
    protected = (
        change & (external_mask > 0)
        & (external_confidence >= thresholds[external_mask])
        & (candidate_up != external_mask)
    )
    change &= ~protected

    mask = improved_mask.copy()
    confidence = improved_confidence.copy()
    provenance = np.full(mask.shape, 1, np.uint8)
    support = np.ones(mask.shape, np.uint8)
    mask[change] = candidate_up[change]
    confidence[change] = confidence_up[change]
    provenance[change] = provenance_up[change]
    provenance[protected] = 6
    support[change] = support_up[change]
    stats = {
        **working_result.stats,
        "full_resolution_changed_fraction": float(change.mean()),
        "full_resolution_protected_external_fraction": float(
            protected.mean()),
        "full_resolution_internal_fraction": float(
            np.isin(mask, INTERNAL_CLASS_IDS).mean()),
        "full_resolution_thin_fraction": float(
            np.isin(mask, THIN_CLASS_IDS).mean()),
    }
    result = PresentationResult(
        mask, np.clip(confidence, 0, 1), provenance, support, stats)
    result.validate()
    return result


def _comparison_metrics(
        current: dict[str, Any],
        improved: dict[str, Any],
        presentation: dict[str, Any],
) -> dict[str, Any]:
    def reduction(key: str, source: dict, target: dict) -> float:
        initial = float(source[key])
        return (initial - float(target[key])) / max(initial, 1e-12)

    def aligned(source: dict, target: dict) -> dict[str, float]:
        source_flow = source["flow_aligned"]
        target_flow = target["flow_aligned"]
        return {
            "flow_aligned_switch_rate_reduction": reduction(
                "class_switch_rate", source_flow, target_flow),
            "flow_aligned_internal_persistence_gain": (
                target_flow["internal_class_persistence"]
                - source_flow["internal_class_persistence"]),
            "flow_aligned_lane_continuity_gain": (
                target_flow["lane_and_marking_continuity"]
                - source_flow["lane_and_marking_continuity"]),
        }

    return {
        "presentation_vs_current": {
            "class_switch_rate_reduction": reduction(
                "class_switch_rate", current, presentation),
            "flicker_reduction": reduction(
                "isolated_one_frame_flicker_rate",
                current, presentation),
            "internal_persistence_gain": (
                presentation["internal_class_persistence"]
                - current["internal_class_persistence"]),
            "lane_continuity_gain": (
                presentation["lane_and_marking_continuity"]
                - current["lane_and_marking_continuity"]),
            **aligned(current, presentation),
        },
        "presentation_vs_improved": {
            "class_switch_rate_reduction": reduction(
                "class_switch_rate", improved, presentation),
            "flicker_reduction": reduction(
                "isolated_one_frame_flicker_rate",
                improved, presentation),
            "internal_persistence_gain": (
                presentation["internal_class_persistence"]
                - improved["internal_class_persistence"]),
            "lane_continuity_gain": (
                presentation["lane_and_marking_continuity"]
                - improved["lane_and_marking_continuity"]),
            **aligned(improved, presentation),
        },
    }


def run_semantic_camera_video(
        input_dir: str | Path,
        semantic_camera_dir: str | Path,
        cfg: Config,
        output_dir: str | Path | None = None,
        resume: bool = True,
        force: bool = False,
        pair_flow_provider=compute_bidirectional_pair_flow,
) -> int:
    """Create improved raw fusion and a separate non-causal presentation output."""
    root = Path(input_dir)
    source = Path(semantic_camera_dir)
    video_cfg = cfg.get("semantic_camera_video", {})
    out = Path(output_dir) if output_dir is not None else (
        root / video_cfg.get(
            "output_subdir", "semantic_camera_video_stabilized"))
    for directory in VIDEO_OUTPUT_DIRS:
        (out / directory).mkdir(parents=True, exist_ok=True)

    source_manifest_path = source / "manifest.json"
    if not source_manifest_path.exists():
        raise RuntimeError("semantic-camera source manifest is missing")
    source_manifest = json.loads(source_manifest_path.read_text())
    source_fingerprint = source_manifest.get("fingerprint")
    if not source_fingerprint:
        raise RuntimeError("semantic-camera source fingerprint is missing")
    frames = iter_frames(root)
    max_frames = cfg.get("frames.max_frames")
    if max_frames is not None:
        frames = frames[:int(max_frames)]
    if not frames:
        raise RuntimeError("video stabilization input contains no frames")
    for reference in frames:
        stem = f"frame_{reference.frame_index:06d}"
        if not _source_frame_complete(source, stem):
            raise RuntimeError(
                f"semantic-camera source incomplete at {stem}")

    frame_contract = [
        (frame.frame_index, frame.capture_timestamp_ns)
        for frame in frames]
    fingerprint = stable_hash([
        video_cfg, source_fingerprint, frame_contract,
        list(FINAL_CLASS_NAMES)])
    manifest_path = out / "manifest.json"
    previous = (
        json.loads(manifest_path.read_text())
        if manifest_path.exists() else None)
    if previous and previous.get("fingerprint") != fingerprint:
        if resume and not force:
            raise RuntimeError(
                "semantic-camera presentation resume fingerprint is incompatible")
        previous = None
    manifest = previous or {
        "stage": "article1_semantic_camera_video_presentation_v1",
        "fingerprint": fingerprint,
        "source_semantic_camera": str(source.resolve()),
        "source_fingerprint": source_fingerprint,
        "scientific_source_is_read_only": True,
        "presentation_is_non_causal": True,
        "uses_future_frames": True,
        "frame_independent_raw_preserved": True,
        "frame_contract": frame_contract,
        "improved_done": {},
        "flow_done": {},
        "presentation_done": {},
        "available_outputs": list(VIDEO_OUTPUT_DIRS),
    }
    if force:
        manifest["improved_done"] = {}
        manifest["flow_done"] = {}
        manifest["presentation_done"] = {}

    improved_cfg = video_cfg.get("improved_fusion", {})
    for reference in frames:
        stem = f"frame_{reference.frame_index:06d}"
        key = str(reference.frame_index)
        if resume and not force and key in manifest["improved_done"] and \
                _improved_complete(out, stem):
            continue
        current = read_mask_u16(source / "final_masks" / f"{stem}.png")
        current_conf = _read_u8(
            source / "final_confidence" / f"{stem}.png")
        external = read_mask_u16(
            source / "external_masks" / f"{stem}.png")
        external_conf = _read_u8(
            source / "external_confidence" / f"{stem}.png")
        internal = read_mask_u16(
            source / "internal_masks" / f"{stem}.png")
        internal_conf = _read_u8(
            source / "internal_confidence" / f"{stem}.png")
        result = improve_fusion_frame(
            current, current_conf, external, external_conf,
            internal, internal_conf, improved_cfg)
        write_mask_u16(
            out / "improved_masks" / f"{stem}.png", result.mask)
        _write_u8(
            out / "improved_confidence" / f"{stem}.png",
            np.clip(result.confidence * 255, 0, 255))
        _write_u8(
            out / "improved_provenance" / f"{stem}.png",
            result.provenance)
        manifest["improved_done"][key] = result.stats
        atomic_write_json(manifest_path, manifest)

    scale = float(video_cfg.get("processing_scale", .5))
    first_bgr = cv2.imread(
        str(frames[0].rectified_path), cv2.IMREAD_COLOR)
    if first_bgr is None:
        raise FileNotFoundError(frames[0].rectified_path)
    full_h, full_w = first_bgr.shape[:2]
    work_size = (
        max(2, round(full_w * scale)),
        max(2, round(full_h * scale)))
    raw_masks: list[np.ndarray] = []
    improved_masks: list[np.ndarray] = []
    improved_confidences: list[np.ndarray] = []
    external_masks: list[np.ndarray] = []
    external_confidences: list[np.ndarray] = []
    rgb_frames: list[np.ndarray] = []
    for reference in frames:
        stem = f"frame_{reference.frame_index:06d}"
        bgr = cv2.imread(str(reference.rectified_path), cv2.IMREAD_COLOR)
        if bgr is None:
            raise FileNotFoundError(reference.rectified_path)
        if bgr.shape[:2] != (full_h, full_w):
            raise RuntimeError("video stabilization RGB geometry changed")
        rgb = cv2.cvtColor(
            cv2.resize(bgr, work_size, interpolation=cv2.INTER_AREA),
            cv2.COLOR_BGR2RGB)
        raw = read_mask_u16(source / "final_masks" / f"{stem}.png")
        improved = read_mask_u16(
            out / "improved_masks" / f"{stem}.png")
        external = read_mask_u16(
            source / "external_masks" / f"{stem}.png")
        raw_masks.append(_resize_mask(raw, work_size))
        improved_masks.append(_resize_mask(improved, work_size))
        improved_confidences.append(_resize_confidence(
            _read_u8(out / "improved_confidence" / f"{stem}.png"),
            work_size))
        external_masks.append(_resize_mask(external, work_size))
        external_confidences.append(_resize_confidence(
            _read_u8(
                source / "external_confidence" / f"{stem}.png"),
            work_size))
        rgb_frames.append(rgb)

    flow_cfg = video_cfg.get("optical_flow", {})
    pair_flows: list[PairFlow] = []
    flow_rows = []
    for index in range(len(frames) - 1):
        first = frames[index].frame_index
        second = frames[index + 1].frame_index
        key = f"{first}:{second}"
        path = _flow_cache_path(out, first, second)
        started = time.perf_counter()
        cached = (
            resume and not force and key in manifest["flow_done"]
            and path.exists())
        if cached:
            pair = _load_pair_flow(path)
        else:
            pair = pair_flow_provider(
                rgb_frames[index], rgb_frames[index + 1], flow_cfg)
            _save_pair_flow(path, pair)
        pair_flows.append(pair)
        row = {
            "first_frame_index": first,
            "second_frame_index": second,
            "valid_forward_fraction": float(pair.valid_forward.mean()),
            "valid_backward_fraction": float(pair.valid_backward.mean()),
            "cache_hit": cached,
            "elapsed_ms": (time.perf_counter() - started) * 1000,
        }
        flow_rows.append(row)
        manifest["flow_done"][key] = row
        if not cached:
            atomic_write_json(manifest_path, manifest)

    presentation_cfg = video_cfg.get("presentation", {})
    presentation_masks: list[np.ndarray] = []
    timing_rows = []
    for index, reference in enumerate(frames):
        stem = f"frame_{reference.frame_index:06d}"
        key = str(reference.frame_index)
        started = time.perf_counter()
        if resume and not force and \
                key in manifest["presentation_done"] and \
                _presentation_complete(out, stem):
            full_mask = read_mask_u16(
                out / "presentation_masks" / f"{stem}.png")
            presentation_masks.append(_resize_mask(full_mask, work_size))
            timing_rows.append({
                "frame_index": reference.frame_index,
                "cache_hit": True,
                "elapsed_ms": (time.perf_counter() - started) * 1000,
            })
            continue
        working_result = stabilize_presentation_frame(
            index, improved_masks, improved_confidences,
            external_masks, external_confidences,
            pair_flows, presentation_cfg)
        improved_full = read_mask_u16(
            out / "improved_masks" / f"{stem}.png")
        improved_conf_full = _read_u8(
            out / "improved_confidence" / f"{stem}.png")
        external_full = read_mask_u16(
            source / "external_masks" / f"{stem}.png")
        external_conf_full = _read_u8(
            source / "external_confidence" / f"{stem}.png")
        full_result = _full_presentation_result(
            improved_full, improved_conf_full,
            external_full, external_conf_full,
            improved_masks[index], working_result, presentation_cfg)
        write_mask_u16(
            out / "presentation_masks" / f"{stem}.png",
            full_result.mask)
        _write_u8(
            out / "presentation_confidence" / f"{stem}.png",
            np.clip(full_result.confidence * 255, 0, 255))
        _write_u8(
            out / "presentation_provenance" / f"{stem}.png",
            full_result.provenance)
        _write_u8(
            out / "presentation_support" / f"{stem}.png",
            full_result.support_count)
        source_metadata = json.loads(
            (source / "metadata" / f"{stem}.json").read_text())
        metadata = {
            "frame_index": reference.frame_index,
            "capture_timestamp_ns": reference.capture_timestamp_ns,
            "scientific_source_fingerprint": source_fingerprint,
            "raw_scientific_output_preserved": True,
            "presentation_is_non_causal": True,
            "uses_future_frames": True,
            "window_radius_frames": presentation_cfg.get(
                "window_radius_frames", 0),
            "improved_fusion": manifest["improved_done"][key],
            "presentation": full_result.stats,
            "internal_source": source_metadata.get("internal_source"),
            "internal_is_fallback": source_metadata.get(
                "internal_is_fallback"),
            "presentation_provenance_codes": {
                str(code): name for code, name
                in PRESENTATION_PROVENANCE.items()},
        }
        atomic_write_json(
            out / "metadata" / f"{stem}.json", metadata)
        manifest["presentation_done"][key] = full_result.stats
        atomic_write_json(manifest_path, manifest)
        presentation_masks.append(_resize_mask(
            full_result.mask, work_size))
        timing_rows.append({
            "frame_index": reference.frame_index,
            "cache_hit": False,
            "elapsed_ms": (time.perf_counter() - started) * 1000,
        })

    metrics = {
        "current_scientific_fusion": compute_presentation_metrics(
            raw_masks),
        "improved_raw_fusion": compute_presentation_metrics(
            improved_masks),
        "stabilized_presentation": compute_presentation_metrics(
            presentation_masks),
    }
    metrics["current_scientific_fusion"]["flow_aligned"] = \
        compute_flow_aligned_metrics(raw_masks, pair_flows)
    metrics["improved_raw_fusion"]["flow_aligned"] = \
        compute_flow_aligned_metrics(improved_masks, pair_flows)
    metrics["stabilized_presentation"]["flow_aligned"] = \
        compute_flow_aligned_metrics(presentation_masks, pair_flows)
    metrics["comparison"] = _comparison_metrics(
        metrics["current_scientific_fusion"],
        metrics["improved_raw_fusion"],
        metrics["stabilized_presentation"])
    atomic_write_json(out / "metrics.json", metrics)
    summary_path = out / "summary.json"
    previous_summary = (
        json.loads(summary_path.read_text())
        if summary_path.exists() else {})
    storage = sum(
        path.stat().st_size for path in out.rglob("*") if path.is_file())
    current_peak_ram = (
        resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024)
    prior_peak_ram = previous_summary.get("peak_ram_mb")
    current_mean_presentation_ms = float(np.mean([
        row["elapsed_ms"] for row in timing_rows]))
    uncached_timings = [
        row["elapsed_ms"] for row in timing_rows
        if not row["cache_hit"]]
    initial_mean_presentation_ms = previous_summary.get(
        "initial_run_mean_presentation_ms_per_frame")
    if initial_mean_presentation_ms is None:
        initial_mean_presentation_ms = (
            float(np.mean(uncached_timings))
            if uncached_timings
            else previous_summary.get(
                "mean_presentation_ms_per_frame",
                current_mean_presentation_ms))
    summary = {
        "stage": manifest["stage"],
        "fingerprint": fingerprint,
        "source_fingerprint": source_fingerprint,
        "frame_count": len(frames),
        "frame_range": [
            frames[0].frame_index, frames[-1].frame_index],
        "processing_geometry": [work_size[1], work_size[0]],
        "full_geometry": [full_h, full_w],
        "presentation_is_non_causal": True,
        "uses_future_frames": True,
        "flow": {
            "pairs": len(flow_rows),
            "mean_valid_forward_fraction": float(np.mean([
                row["valid_forward_fraction"] for row in flow_rows]))
                if flow_rows else 1.0,
            "mean_valid_backward_fraction": float(np.mean([
                row["valid_backward_fraction"] for row in flow_rows]))
                if flow_rows else 1.0,
            "cache_hits": sum(bool(row["cache_hit"]) for row in flow_rows),
        },
        "mean_presentation_ms_per_frame": (
            initial_mean_presentation_ms),
        "initial_run_mean_presentation_ms_per_frame": (
            initial_mean_presentation_ms),
        "last_invocation_mean_presentation_ms_per_frame": (
            current_mean_presentation_ms),
        "metrics": metrics,
        "peak_ram_mb": max(
            [float(value) for value in (
                current_peak_ram, prior_peak_ram) if value is not None]),
        "storage_bytes": storage,
    }
    atomic_write_json(summary_path, summary)
    return 0
