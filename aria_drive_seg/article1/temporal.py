"""Causal, per-pixel probability stabilization for Article 1."""
from __future__ import annotations

from dataclasses import dataclass, field
import time

import cv2
import numpy as np

from .optical_flow import FlowResult, warp_with_backward
from .temporal_state import TemporalState


PROVENANCE = {
    0: "unknown", 1: "current_static", 2: "warped_previous",
    3: "fused_current_previous", 4: "thin_current",
    5: "thin_propagated", 6: "temporal_hysteresis",
    7: "reset_current_only",
}
RESET_REASON = {
    0: "none", 1: "initial_frame", 2: "missing_frame",
    3: "resolution_change", 4: "non_monotonic_timestamp",
    5: "timestamp_gap", 6: "excessive_flow",
    7: "insufficient_valid_flow", 8: "photometric_or_scene_cut",
    9: "config_change", 10: "incompatible_cache",
}
SWITCH_REASON = {
    0: "unchanged", 1: "new_class_margin", 2: "confirmed_candidate",
    3: "immediate_high_confidence", 4: "previous_confidence_drop",
    5: "ttl_expired", 6: "flow_invalid", 7: "hysteresis_hold",
    8: "unknown_recovery",
}


@dataclass
class TemporalResult:
    probabilities: np.ndarray
    mask: np.ndarray
    confidence: np.ndarray
    provenance: np.ndarray
    propagation_age: np.ndarray
    class_age: np.ndarray
    candidate_class: np.ndarray
    switch_reason: np.ndarray
    thin_mask: np.ndarray
    unknown_recovered: np.ndarray
    flow_validity: np.ndarray
    occlusion: np.ndarray
    reset_reason: int
    state: TemporalState
    timings_ms: dict = field(default_factory=dict)


def _class_params(class_names: list[str], cfg: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    class_cfg = cfg.get("classes", {})
    ttl = np.zeros(len(class_names), np.uint16)
    weight = np.zeros(len(class_names), np.float32)
    require = np.zeros(len(class_names), bool)
    for cid, name in enumerate(class_names):
        values = class_cfg.get(name, {})
        ttl[cid] = int(values.get("ttl_frames", 1))
        weight[cid] = float(values.get(
            "previous_weight", cfg.get("fusion", {}).get("previous_weight", .65)))
        require[cid] = bool(values.get("require_current_support", False))
    return ttl, weight, require


def reset_reason_for(state: TemporalState | None, frame_index: int,
                     timestamp_ns: int, geometry: tuple[int, int],
                     flow: FlowResult | None, cfg: dict,
                     config_fingerprint: str,
                     static_policy_fingerprint: str) -> int:
    if state is None:
        return 1
    if state.config_fingerprint != config_fingerprint or \
            state.static_policy_fingerprint != static_policy_fingerprint:
        return 9
    if state.previous_rgb.shape[:2] != geometry:
        return 3
    if timestamp_ns <= state.timestamp_ns:
        return 4
    if (timestamp_ns - state.timestamp_ns) / 1e9 > float(
            cfg.get("reset", {}).get("max_frame_gap_s", .25)):
        return 5
    if frame_index != state.frame_index + 1:
        return 2
    if flow is not None:
        reset = cfg.get("reset", {})
        if flow.median_flow_px > float(reset.get("max_median_flow_px", 80)):
            return 6
        if flow.valid_fraction < 1 - float(
                reset.get("max_invalid_flow_fraction", .60)):
            return 7
        if flow.mean_photometric_error > float(
                reset.get("max_photometric_error", .15)):
            return 8
    return 0


def _initial_result(current_prob: np.ndarray, current_thin: np.ndarray,
                    current_rgb: np.ndarray, frame_index: int, timestamp_ns: int,
                    config_fingerprint: str, static_policy_fingerprint: str,
                    reset_count: int, reset_reason: int) -> TemporalResult:
    mask = current_prob.argmax(0).astype(np.uint16)
    mask[current_thin > 0] = current_thin[current_thin > 0]
    confidence = current_prob.max(0).astype(np.float32)
    zeros = np.zeros(mask.shape, np.uint16)
    provenance = np.full(mask.shape, 7, np.uint8)
    state = TemporalState(
        current_prob.astype(np.float32), mask, confidence,
        np.ones_like(mask, np.uint16), zeros.copy(), mask.copy(), zeros.copy(),
        current_thin.astype(np.uint16), confidence.copy(), current_rgb.copy(),
        frame_index, timestamp_ns, reset_count, config_fingerprint,
        static_policy_fingerprint)
    return TemporalResult(
        current_prob, mask, confidence, provenance, zeros.copy(),
        state.class_age, state.candidate_class, np.full(mask.shape, 6, np.uint8),
        current_thin, np.zeros(mask.shape, bool), np.ones(mask.shape, bool),
        np.zeros(mask.shape, bool), reset_reason, state,
        {"warping_ms": 0.0, "fusion_ms": 0.0, "hysteresis_ms": 0.0,
         "thin_temporal_ms": 0.0})


def stabilize_frame(current_probabilities: np.ndarray,
                    current_thin: np.ndarray,
                    current_road_support: np.ndarray,
                    current_rgb: np.ndarray,
                    frame_index: int,
                    timestamp_ns: int,
                    class_names: list[str],
                    cfg: dict,
                    config_fingerprint: str,
                    static_policy_fingerprint: str,
                    state: TemporalState | None,
                    flow: FlowResult | None,
                    mode: str = "T4") -> TemporalResult:
    """Stabilize frame *t* from current evidence and state from *t-1* only."""
    p = np.asarray(current_probabilities, np.float32)
    p /= np.maximum(p.sum(0, keepdims=True), 1e-8)
    reset = reset_reason_for(
        state, frame_index, timestamp_ns, p.shape[1:], flow, cfg,
        config_fingerprint, static_policy_fingerprint)
    if mode == "T0" or reset:
        count = (state.reset_count if state else 0) + int(reset > 0)
        return _initial_result(
            p, current_thin, current_rgb, frame_index, timestamp_ns,
            config_fingerprint, static_policy_fingerprint, count, reset)
    assert state is not None
    h, w = p.shape[1:]
    warp_start = time.perf_counter()
    if mode == "T1":
        backward = np.zeros((h, w, 2), np.float32)
        valid = np.ones((h, w), bool)
        occlusion = np.zeros((h, w), bool)
    else:
        assert flow is not None
        backward, valid, occlusion = flow.backward, flow.valid, flow.occlusion
    previous_p = warp_with_backward(state.probabilities, backward)
    previous_p /= np.maximum(previous_p.sum(0, keepdims=True), 1e-8)
    previous_mask = warp_with_backward(
        state.mask, backward, cv2.INTER_NEAREST).astype(np.uint16)
    previous_conf = warp_with_backward(state.confidence, backward)
    previous_class_age = warp_with_backward(
        state.class_age, backward, cv2.INTER_NEAREST).astype(np.uint16)
    previous_propagation_age = warp_with_backward(
        state.propagation_age, backward, cv2.INTER_NEAREST).astype(np.uint16)
    previous_candidate = warp_with_backward(
        state.candidate_class, backward, cv2.INTER_NEAREST).astype(np.uint16)
    previous_candidate_age = warp_with_backward(
        state.candidate_age, backward, cv2.INTER_NEAREST).astype(np.uint16)
    warping_ms = (time.perf_counter() - warp_start) * 1000

    fusion_start = time.perf_counter()
    ttl, class_weight, require_support = _class_params(class_names, cfg)
    current_mask = p.argmax(0).astype(np.uint16)
    ordered = np.argsort(p, axis=0)
    current_top2 = ordered[-2:]
    current_conf = p.max(0)
    current_margin = np.take_along_axis(p, ordered[-1:], 0)[0] - \
        np.take_along_axis(p, ordered[-2:-1], 0)[0]
    previous_ttl = ttl[previous_mask]
    eligible = (valid & (previous_conf >= float(
        cfg.get("fusion", {}).get("minimum_previous_confidence", .55))) &
        (previous_propagation_age < previous_ttl) & (previous_mask != 0))
    support = ((current_top2 == previous_mask[None]).any(0) |
               (current_mask == 0) |
               (np.take_along_axis(p, previous_mask[None], 0)[0] >= .10))
    eligible &= ~require_support[previous_mask] | support
    previous_weight = class_weight[previous_mask] * eligible
    previous_weight *= np.power(
        float(cfg.get("fusion", {}).get("temporal_decay", .85)),
        previous_propagation_age)
    high_current = current_conf >= float(
        cfg.get("hysteresis", {}).get("immediate_switch_confidence", .80))
    previous_weight[high_current & (current_mask != 0)] = 0
    current_weight = np.full((h, w), float(
        cfg.get("fusion", {}).get("current_weight", 1.0)), np.float32)
    uncertain = ((current_conf < float(
        cfg.get("fusion", {}).get("uncertain_current_threshold", .55))) |
        (current_margin < float(
            cfg.get("fusion", {}).get("uncertain_current_margin", .08))) |
        (current_mask == 0))
    current_weight[uncertain] *= .35
    fused = current_weight[None] * p + previous_weight[None] * previous_p
    fused /= np.maximum(fused.sum(0, keepdims=True), 1e-8)
    proposed = fused.argmax(0).astype(np.uint16)
    fusion_ms = (time.perf_counter() - fusion_start) * 1000

    hysteresis_start = time.perf_counter()
    switch = proposed != previous_mask
    hcfg = cfg.get("hysteresis", {})
    candidate_same = proposed == previous_candidate
    candidate_age = np.where(
        switch,
        np.where(candidate_same, previous_candidate_age + 1, 1),
        0).astype(np.uint16)
    switch_reason = np.zeros((h, w), np.uint8)
    immediate = switch & (current_mask != 0) & (current_conf >= float(
        hcfg.get("immediate_switch_confidence", .80)))
    margin_switch = switch & (current_mask != 0) & (current_margin >= float(
        hcfg.get("switch_margin", .10)))
    confirmed = switch & (candidate_age >= int(
        hcfg.get("minimum_confirmation_frames", 2)))
    dropped = switch & (previous_conf < float(
        hcfg.get("previous_drop_threshold", .25)))
    expired = switch & (previous_propagation_age >= previous_ttl)
    allowed_switch = immediate | margin_switch | confirmed | dropped | expired | ~valid
    if mode in {"T2"} or not hcfg.get("enabled", True):
        allowed_switch[:] = True
    temporal_mask = proposed.copy()
    held = switch & ~allowed_switch
    temporal_mask[held] = previous_mask[held]
    if held.any():
        rows, cols = np.where(held)
        held_class = previous_mask[held]
        retained = previous_conf[held] * float(
            cfg.get("fusion", {}).get("temporal_decay", .85))
        fused[held_class, rows, cols] = np.maximum(
            fused[held_class, rows, cols], retained)
        fused[:, rows, cols] /= np.maximum(
            fused[:, rows, cols].sum(0, keepdims=True), 1e-8)
    switch_reason[margin_switch & allowed_switch] = 1
    switch_reason[confirmed & allowed_switch] = 2
    switch_reason[immediate] = 3
    switch_reason[dropped & allowed_switch] = 4
    switch_reason[expired & allowed_switch] = 5
    switch_reason[~valid & switch] = 6
    switch_reason[held] = 7
    unknown_recovered = (current_mask == 0) & (temporal_mask != 0) & eligible
    switch_reason[unknown_recovered] = 8

    used_previous = eligible & (previous_weight > 0)
    propagation_age = np.where(
        used_previous & (current_mask != temporal_mask),
        previous_propagation_age + 1, 0).astype(np.uint16)
    same_class = temporal_mask == previous_mask
    class_age = np.where(
        same_class & valid, previous_class_age + 1, 1).astype(np.uint16)
    confidence = np.take_along_axis(fused, temporal_mask[None], 0)[0]
    provenance = np.ones((h, w), np.uint8)
    provenance[used_previous] = 3
    provenance[used_previous & (current_mask != temporal_mask)] = 2
    provenance[held] = 6
    hysteresis_ms = (time.perf_counter() - hysteresis_start) * 1000

    thin_start = time.perf_counter()
    thin = current_thin.astype(np.uint16).copy()
    thin_propagated = np.zeros((h, w), bool)
    previous_thin_conf = np.zeros((h, w), np.float32)
    if mode == "T4" and cfg.get("thin_markings", {}).get("enabled", True):
        previous_thin = warp_with_backward(
            state.thin_mask, backward, cv2.INTER_NEAREST).astype(np.uint16)
        previous_thin_conf = warp_with_backward(state.thin_confidence, backward)
        thin_cfg = cfg.get("thin_markings", {})
        propagated = (thin == 0) & (previous_thin > 0) & valid & current_road_support
        lane_ttl = int(thin_cfg.get("lane_ttl_frames", 3))
        regulatory_ttl = int(thin_cfg.get("regulatory_ttl_frames", 2))
        thin_ttl = np.where(previous_thin == 2, lane_ttl, regulatory_ttl)
        propagated &= previous_propagation_age < thin_ttl
        propagated &= previous_thin_conf >= float(
            thin_cfg.get("minimum_previous_confidence", .55))
        thin[propagated] = previous_thin[propagated]
        temporal_mask[propagated] = previous_thin[propagated]
        provenance[current_thin > 0] = 4
        provenance[propagated] = 5
        propagation_age[propagated] = previous_propagation_age[propagated] + 1
        thin_propagated = propagated
    thin_confidence = np.zeros((h, w), np.float32)
    current_thin_pixels = current_thin > 0
    thin_confidence[current_thin_pixels] = np.maximum(
        current_conf[current_thin_pixels],
        np.take_along_axis(
            p, np.clip(current_thin, 0, len(class_names) - 1)[None], 0
        )[0][current_thin_pixels])
    thin_confidence[thin_propagated] = (
        previous_thin_conf[thin_propagated] *
        float(cfg.get("thin_markings", {}).get("temporal_decay", .75)))
    thin_temporal_ms = (time.perf_counter() - thin_start) * 1000
    state_out = TemporalState(
        fused, temporal_mask, confidence, class_age, propagation_age,
        proposed, candidate_age, thin, thin_confidence, current_rgb.copy(),
        frame_index, timestamp_ns, state.reset_count, config_fingerprint,
        static_policy_fingerprint)
    state_out.validate()
    return TemporalResult(
        fused, temporal_mask, confidence, provenance, propagation_age,
        class_age, proposed, switch_reason, thin, unknown_recovered,
        valid, occlusion, 0, state_out,
        {"warping_ms": warping_ms, "fusion_ms": fusion_ms,
         "hysteresis_ms": hysteresis_ms,
         "thin_temporal_ms": thin_temporal_ms})
