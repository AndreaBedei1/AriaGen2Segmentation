"""Final offline/presentation pass for the Article 1 semantic camera.

This stage consumes, but never mutates, the scientific semantic-camera output and
the preceding stabilized presentation output.  It adds a wider flow-aligned
temporal window plus road-geometry-aware line and boundary refinement.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import resource
import time
from typing import Any, Sequence

import cv2
import numpy as np

from ..config import Config
from ..hashing import stable_hash
from ..io_utils import atomic_write_bytes, atomic_write_json, read_mask_u16, \
    write_mask_u16
from ..segmentation.base import iter_frames
from .semantic_camera import FINAL_CLASS_NAMES
from .semantic_camera_video import (
    INTERNAL_CLASS_IDS,
    PairFlow,
    PresentationResult,
    _load_pair_flow,
    compute_bidirectional_pair_flow,
    compute_flow_aligned_metrics,
    compute_presentation_metrics,
    stabilize_presentation_frame,
)


ROAD_ID = FINAL_CLASS_NAMES.index("road_surface")
LANE_ID = FINAL_CLASS_NAMES.index("lane_marking")
REGULATORY_ID = FINAL_CLASS_NAMES.index("regulatory_road_marking")
VEHICLE_ID = FINAL_CLASS_NAMES.index("vehicle")
TWO_WHEELER_ID = FINAL_CLASS_NAMES.index("two_wheeler")
PEDESTRIAN_ID = FINAL_CLASS_NAMES.index("pedestrian")
BOUNDARY_ID = FINAL_CLASS_NAMES.index("road_boundary_or_obstacle")
OTHER_ID = FINAL_CLASS_NAMES.index("other_environment")

LINE_CLASS_IDS = np.asarray([LANE_ID, REGULATORY_ID], dtype=np.uint16)
DYNAMIC_CLASS_IDS = np.asarray(
    [VEHICLE_ID, TWO_WHEELER_ID, PEDESTRIAN_ID], dtype=np.uint16)
NONROAD_BARRIER_IDS = np.asarray(
    [VEHICLE_ID, TWO_WHEELER_ID, PEDESTRIAN_ID, *INTERNAL_CLASS_IDS],
    dtype=np.uint16)

FINAL_PROVENANCE = {
    0: "invalid_unassigned",
    1: "current_model",
    2: "temporal_evidence",
    3: "morphological_link",
    4: "final_fill",
}

FINAL_OUTPUT_DIRS = (
    "final_masks", "final_confidence", "final_provenance",
    "final_support", "metadata",
)


@dataclass
class ComponentGeometry:
    label: int
    area: int
    length: float
    width: float
    aspect: float
    angle_deg: float
    endpoints: tuple[tuple[int, int], tuple[int, int]]


@dataclass
class LineRefinement:
    mask: np.ndarray
    morphological_added: np.ndarray
    removed: np.ndarray
    stats: dict[str, Any]


@dataclass
class FinalPassResult:
    mask: np.ndarray
    confidence: np.ndarray
    provenance: np.ndarray
    support_count: np.ndarray
    road_support: np.ndarray
    stats: dict[str, Any]

    def validate(self) -> None:
        shape = self.mask.shape
        values = (
            self.confidence, self.provenance, self.support_count,
            self.road_support)
        if any(value.shape != shape for value in values):
            raise ValueError("final-pass output geometry mismatch")
        if np.any(self.mask == 0) or np.any(
                self.mask >= len(FINAL_CLASS_NAMES)):
            raise ValueError("final-pass output is not a dense valid mask")
        if np.any(self.provenance == 0):
            raise ValueError("final-pass provenance is incomplete")


def _read_u8(path: Path) -> np.ndarray:
    value = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if value is None:
        raise FileNotFoundError(path)
    return np.asarray(value, np.float32) / 255.0


def _write_u8(path: Path, value: np.ndarray) -> None:
    ok, encoded = cv2.imencode(".png", np.asarray(value, np.uint8))
    if not ok:
        raise RuntimeError(f"could not encode {path}")
    atomic_write_bytes(path, encoded.tobytes())


def _resize_mask(value: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    return cv2.resize(
        value, size, interpolation=cv2.INTER_NEAREST).astype(np.uint16)


def _resize_float(value: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    return cv2.resize(
        np.asarray(value, np.float32), size,
        interpolation=cv2.INTER_LINEAR).astype(np.float32)


def _odd(value: int) -> int:
    value = max(1, int(value))
    return value if value % 2 else value + 1


def _angle_difference(first: float, second: float) -> float:
    difference = abs(first - second) % 180.0
    return min(difference, 180.0 - difference)


def _oriented_kernel(length: int, angle_deg: float) -> np.ndarray:
    size = _odd(length)
    kernel = np.zeros((size, size), np.uint8)
    center = (size - 1) / 2.0
    radius = center
    angle = math.radians(angle_deg)
    dx, dy = radius * math.cos(angle), radius * math.sin(angle)
    first = (int(round(center - dx)), int(round(center - dy)))
    second = (int(round(center + dx)), int(round(center + dy)))
    cv2.line(kernel, first, second, 1, 1, cv2.LINE_8)
    return kernel


def _component_geometries(
        binary: np.ndarray,
) -> tuple[np.ndarray, list[ComponentGeometry]]:
    count, labels, stats, _ = cv2.connectedComponentsWithStats(
        binary.astype(np.uint8), 8)
    result: list[ComponentGeometry] = []
    for label in range(1, count):
        area = int(stats[label, cv2.CC_STAT_AREA])
        yy, xx = np.nonzero(labels == label)
        if area < 2:
            continue
        points = np.column_stack([xx, yy]).astype(np.float32)
        centered = points - points.mean(0, keepdims=True)
        covariance = centered.T @ centered / max(1, len(points) - 1)
        eigenvalues, eigenvectors = np.linalg.eigh(covariance)
        direction = eigenvectors[:, int(np.argmax(eigenvalues))]
        if direction[0] < 0:
            direction = -direction
        projection = points @ direction
        perpendicular = points @ np.asarray(
            [-direction[1], direction[0]], np.float32)
        length = float(projection.max() - projection.min() + 1.0)
        width = float(perpendicular.max() - perpendicular.min() + 1.0)
        endpoint_values = (
            points[int(np.argmin(projection))],
            points[int(np.argmax(projection))],
        )
        endpoints = tuple(
            (int(round(point[0])), int(round(point[1])))
            for point in endpoint_values)
        result.append(ComponentGeometry(
            label=label,
            area=area,
            length=length,
            width=width,
            aspect=length / max(width, 1.0),
            angle_deg=float(
                math.degrees(math.atan2(direction[1], direction[0]))
                % 180.0),
            endpoints=endpoints,
        ))
    return labels, result


def _component_seed_mask(
        labels: np.ndarray,
        components: Sequence[ComponentGeometry],
        minimum_area: int,
) -> np.ndarray:
    selected = np.zeros(labels.shape, bool)
    for component in components:
        if component.area >= minimum_area:
            selected |= labels == component.label
    return selected


def _path_is_allowed(
        first: tuple[int, int],
        second: tuple[int, int],
        allowed: np.ndarray,
        forbidden: np.ndarray,
) -> tuple[bool, np.ndarray]:
    path = np.zeros(allowed.shape, np.uint8)
    cv2.line(path, first, second, 1, 1, cv2.LINE_8)
    pixels = path > 0
    return bool(
        pixels.any() and np.all(allowed[pixels]) and
        not np.any(forbidden[pixels])), pixels


def _bridge_component_gaps(
        binary: np.ndarray,
        allowed: np.ndarray,
        forbidden: np.ndarray,
        cfg: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray, int]:
    """Connect only close, collinear component endpoints along an allowed path."""
    labels, components = _component_geometries(binary)
    minimum_area = int(cfg.get("seed_minimum_area_px", 3))
    components = [
        value for value in components if value.area >= minimum_area]
    output = _component_seed_mask(labels, components, minimum_area)
    added = np.zeros(output.shape, bool)
    maximum_gap = float(cfg.get("maximum_gap_px", 28))
    orientation_tolerance = float(
        cfg.get("orientation_tolerance_deg", 18))
    endpoint_tolerance = float(
        cfg.get("endpoint_alignment_tolerance_deg", 24))
    candidates = []
    for first_index, first in enumerate(components):
        for second in components[first_index + 1:]:
            if _angle_difference(
                    first.angle_deg, second.angle_deg) > orientation_tolerance:
                continue
            for first_endpoint in first.endpoints:
                for second_endpoint in second.endpoints:
                    dx = second_endpoint[0] - first_endpoint[0]
                    dy = second_endpoint[1] - first_endpoint[1]
                    distance = math.hypot(dx, dy)
                    if not 1.0 < distance <= maximum_gap:
                        continue
                    bridge_angle = math.degrees(math.atan2(dy, dx)) % 180.0
                    if _angle_difference(
                            bridge_angle,
                            first.angle_deg) > endpoint_tolerance:
                        continue
                    if _angle_difference(
                            bridge_angle,
                            second.angle_deg) > endpoint_tolerance:
                        continue
                    valid, pixels = _path_is_allowed(
                        first_endpoint, second_endpoint, allowed, forbidden)
                    if valid:
                        candidates.append(
                            (distance, first.label, second.label, pixels))
    linked_labels: set[tuple[int, int]] = set()
    bridge_count = 0
    for _, first_label, second_label, pixels in sorted(
            candidates, key=lambda item: (
                item[0], item[1], item[2])):
        pair = tuple(sorted((first_label, second_label)))
        if pair in linked_labels:
            continue
        output[pixels] = True
        added[pixels & ~binary] = True
        linked_labels.add(pair)
        bridge_count += 1
    return output, added, bridge_count


def _orientation_aware_closing(
        binary: np.ndarray,
        allowed: np.ndarray,
        forbidden: np.ndarray,
        cfg: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray]:
    """Close gaps within orientation bins, never across semantic barriers."""
    labels, components = _component_geometries(binary)
    minimum_area = int(cfg.get("seed_minimum_area_px", 3))
    base = _component_seed_mask(labels, components, minimum_area)
    result = base.copy()
    tolerance = float(cfg.get("orientation_tolerance_deg", 18))
    length = _odd(int(cfg.get("closing_kernel_length_px", 17)))
    for angle in cfg.get("closing_orientations_deg", [0, 30, 60, 90, 120, 150]):
        selected = np.zeros(binary.shape, bool)
        for component in components:
            if component.area >= minimum_area and _angle_difference(
                    component.angle_deg, float(angle)) <= tolerance:
                selected |= labels == component.label
        if not selected.any():
            continue
        closed = cv2.morphologyEx(
            selected.astype(np.uint8), cv2.MORPH_CLOSE,
            _oriented_kernel(length, float(angle))) > 0
        result |= closed & allowed & ~forbidden
    return result, result & ~binary


def skeletonize_binary(binary: np.ndarray) -> np.ndarray:
    """Return a deterministic one-pixel skeleton."""
    source = (binary.astype(np.uint8) * 255)
    if hasattr(cv2, "ximgproc") and hasattr(cv2.ximgproc, "thinning"):
        return cv2.ximgproc.thinning(
            source, thinningType=cv2.ximgproc.THINNING_ZHANGSUEN) > 0
    skeleton = np.zeros_like(source)
    element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    current = source.copy()
    while cv2.countNonZero(current):
        opened = cv2.morphologyEx(current, cv2.MORPH_OPEN, element)
        skeleton = cv2.bitwise_or(
            skeleton, cv2.subtract(current, opened))
        current = cv2.erode(current, element)
    return skeleton > 0


def _remove_incompatible_components(
        binary: np.ndarray,
        cfg: dict[str, Any],
) -> np.ndarray:
    labels, components = _component_geometries(binary)
    minimum_length = float(cfg.get("minimum_length_px", 10))
    minimum_aspect = float(cfg.get("minimum_aspect_ratio", 2.0))
    result = np.zeros(binary.shape, bool)
    for component in components:
        if component.length >= minimum_length and \
                component.aspect >= minimum_aspect:
            result |= labels == component.label
    return result


def refine_line_class(
        class_mask: np.ndarray,
        road_support: np.ndarray,
        forbidden: np.ndarray,
        probability: np.ndarray,
        cfg: dict[str, Any],
) -> LineRefinement:
    """Repair line gaps with orientation and road-geometry constraints."""
    original = np.asarray(class_mask, bool)
    probability = np.asarray(probability, np.float32)
    if original.shape != road_support.shape or original.shape != forbidden.shape \
            or original.shape != probability.shape:
        raise ValueError("line-refinement geometry mismatch")
    probability_support = probability >= float(
        cfg.get("probability_support_floor", .25))
    allowed = (road_support | probability_support) & ~forbidden
    seeded = original & allowed
    closed, closing_added = _orientation_aware_closing(
        seeded, allowed, forbidden, cfg)
    linked, bridge_added, bridge_count = _bridge_component_gaps(
        closed, allowed, forbidden, cfg)
    linked &= allowed

    thickness = 1
    if bool(cfg.get("skeletonize_after_linking", True)) and linked.any():
        skeleton = skeletonize_binary(linked)
        distance = cv2.distanceTransform(
            original.astype(np.uint8), cv2.DIST_L2, 5)
        supported_skeleton = skeleton & (distance > 0)
        if supported_skeleton.any():
            thickness = int(round(
                2.0 * float(np.median(distance[supported_skeleton]))))
        thickness = max(
            1, min(int(cfg.get("maximum_thickness_px", 7)), thickness))
        radius = max(0, (thickness - 1) // 2)
        if radius:
            kernel = cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1))
            linked = cv2.dilate(
                skeleton.astype(np.uint8), kernel) > 0
        else:
            linked = skeleton
        linked &= allowed

    refined = _remove_incompatible_components(linked, cfg)
    morphological_added = refined & ~original
    removed = original & ~refined
    _, before_components = _component_geometries(original)
    _, after_components = _component_geometries(refined)
    stats = {
        "components_before": len(before_components),
        "components_after": len(after_components),
        "mean_component_length_before_px": float(np.mean([
            value.length for value in before_components]))
            if before_components else 0.0,
        "mean_component_length_after_px": float(np.mean([
            value.length for value in after_components]))
            if after_components else 0.0,
        "closing_added_pixels": int(closing_added.sum()),
        "bridge_added_pixels": int(bridge_added.sum()),
        "morphological_added_pixels": int(morphological_added.sum()),
        "removed_pixels": int(removed.sum()),
        "bridges": bridge_count,
        "estimated_original_thickness_px": thickness,
        "added_outside_road_support_pixels": int(
            (morphological_added & ~road_support).sum()),
    }
    return LineRefinement(
        refined, morphological_added, removed, stats)


def _fill_small_holes(binary: np.ndarray, maximum_area: int) -> np.ndarray:
    inverse = (~binary).astype(np.uint8)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(inverse, 8)
    result = binary.copy()
    h, w = binary.shape
    for label in range(1, count):
        x = int(stats[label, cv2.CC_STAT_LEFT])
        y = int(stats[label, cv2.CC_STAT_TOP])
        width = int(stats[label, cv2.CC_STAT_WIDTH])
        height = int(stats[label, cv2.CC_STAT_HEIGHT])
        area = int(stats[label, cv2.CC_STAT_AREA])
        touches_border = (
            x == 0 or y == 0 or x + width == w or y + height == h)
        if not touches_border and area <= maximum_area:
            result[labels == label] = True
    return result


def refine_road_boundary(
        boundary_mask: np.ndarray,
        road_support: np.ndarray,
        probability: np.ndarray,
        forbidden: np.ndarray,
        cfg: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    """Keep connected curb/boundary regions close to the road edge."""
    boundary = np.asarray(boundary_mask, bool)
    edge_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    road_edge = cv2.morphologyEx(
        road_support.astype(np.uint8), cv2.MORPH_GRADIENT,
        edge_kernel) > 0
    radius = max(1, int(cfg.get("road_edge_radius_px", 16)))
    edge_band = cv2.dilate(
        road_edge.astype(np.uint8),
        cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1))) > 0
    probability_support = np.asarray(probability, np.float32) >= float(
        cfg.get("probability_support_floor", .28))
    allowed = (edge_band | probability_support) & ~forbidden
    candidate = boundary & allowed
    close_size = _odd(int(cfg.get("closing_kernel_px", 5)))
    closed = cv2.morphologyEx(
        candidate.astype(np.uint8), cv2.MORPH_CLOSE,
        cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (close_size, close_size))) > 0
    closed &= allowed
    filled = _fill_small_holes(
        closed, int(cfg.get("maximum_hole_area_px", 80))) & allowed
    labels, components = _component_geometries(filled)
    minimum_area = int(cfg.get("minimum_component_area_px", 18))
    refined = np.zeros(boundary.shape, bool)
    for component in components:
        component_mask = labels == component.label
        near_edge_fraction = float(
            (component_mask & edge_band).sum() / max(1, component.area))
        if component.area >= minimum_area and (
                near_edge_fraction >= float(
                    cfg.get("minimum_road_edge_fraction", .25))
                or np.any(component_mask & probability_support)):
            refined |= component_mask
    morphological = (closed & ~candidate) & refined
    final_fill = (refined & ~closed)
    removed = boundary & ~refined
    stats = {
        "components_before": len(_component_geometries(boundary)[1]),
        "components_after": len(_component_geometries(refined)[1]),
        "morphological_added_pixels": int(morphological.sum()),
        "final_fill_pixels": int(final_fill.sum()),
        "removed_isolated_pixels": int(removed.sum()),
        "added_outside_road_edge_band_pixels": int(
            ((morphological | final_fill) & ~edge_band).sum()),
    }
    return refined, morphological, final_fill, stats


def _build_road_support(
        mask: np.ndarray,
        cfg: dict[str, Any],
) -> np.ndarray:
    base = np.isin(mask, [ROAD_ID, LANE_ID, REGULATORY_ID])
    close_size = _odd(int(cfg.get("road_closing_kernel_px", 11)))
    base = cv2.morphologyEx(
        base.astype(np.uint8), cv2.MORPH_CLOSE,
        cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (close_size, close_size))) > 0
    radius = max(1, int(cfg.get("road_support_radius_px", 8)))
    return cv2.dilate(
        base.astype(np.uint8),
        cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1))) > 0


def finalize_presentation_frame(
        temporal: PresentationResult,
        current_mask: np.ndarray,
        external_mask: np.ndarray,
        external_confidence: np.ndarray,
        cfg: dict[str, Any],
) -> FinalPassResult:
    """Apply constrained line/boundary geometry to one temporally fused frame."""
    temporal.validate()
    shape = temporal.mask.shape
    if any(value.shape != shape for value in (
            current_mask, external_mask, external_confidence)):
        raise ValueError("final-pass evidence geometry mismatch")
    output = temporal.mask.copy()
    confidence = temporal.confidence.copy()
    support = temporal.support_count.copy()
    provenance = np.full(shape, 1, np.uint8)
    provenance[np.isin(temporal.provenance, [2, 3, 4])] = 2
    forbidden = np.isin(output, NONROAD_BARRIER_IDS)
    barrier_radius = max(0, int(cfg.get("barrier_dilation_px", 2)))
    if barrier_radius:
        forbidden = cv2.dilate(
            forbidden.astype(np.uint8),
            cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE,
                (2 * barrier_radius + 1, 2 * barrier_radius + 1))) > 0
    road_support = _build_road_support(output, cfg)
    class_stats: dict[str, Any] = {}
    total_morphological = np.zeros(shape, bool)
    total_fill = np.zeros(shape, bool)
    total_outside_road = 0

    line_cfg_root = cfg.get("line_filter", {})
    for class_id in LINE_CLASS_IDS:
        name = FINAL_CLASS_NAMES[int(class_id)]
        class_cfg = {
            **line_cfg_root.get("common", {}),
            **line_cfg_root.get(name, {}),
        }
        probability = np.where(
            external_mask == class_id, external_confidence, 0.0)
        probability = np.maximum(
            probability,
            np.where(output == class_id, confidence, 0.0))
        refined = refine_line_class(
            output == class_id, road_support, forbidden,
            probability, class_cfg)
        removed = (output == class_id) & refined.removed
        output[removed & road_support] = ROAD_ID
        output[removed & ~road_support] = OTHER_ID
        confidence[removed] = np.maximum(
            confidence[removed],
            float(cfg.get("fill_confidence", .30)))
        provenance[removed] = 4
        support[removed] = np.maximum(support[removed], 1)
        can_add = (
            refined.morphological_added
            & np.isin(output, [ROAD_ID, OTHER_ID])
            & ~forbidden)
        output[can_add] = class_id
        confidence[can_add] = np.maximum(
            confidence[can_add],
            float(class_cfg.get("added_confidence", .48)))
        provenance[can_add] = 3
        support[can_add] = np.maximum(support[can_add], 2)
        total_morphological |= can_add
        total_fill |= removed
        total_outside_road += int((can_add & ~road_support).sum())
        class_stats[name] = refined.stats

    boundary_cfg = cfg.get("road_boundary_filter", {})
    boundary_probability = np.maximum(
        np.where(external_mask == BOUNDARY_ID, external_confidence, 0.0),
        np.where(output == BOUNDARY_ID, confidence, 0.0))
    boundary, boundary_morph, boundary_fill, boundary_stats = \
        refine_road_boundary(
            output == BOUNDARY_ID, road_support, boundary_probability,
            forbidden, boundary_cfg)
    boundary_removed = (output == BOUNDARY_ID) & ~boundary
    output[boundary_removed & road_support] = ROAD_ID
    output[boundary_removed & ~road_support] = OTHER_ID
    provenance[boundary_removed] = 4
    confidence[boundary_removed] = np.maximum(
        confidence[boundary_removed],
        float(cfg.get("fill_confidence", .30)))
    boundary_add = (
        (boundary_morph | boundary_fill)
        & np.isin(output, [ROAD_ID, OTHER_ID])
        & ~forbidden)
    output[boundary_add] = BOUNDARY_ID
    confidence[boundary_add] = np.maximum(
        confidence[boundary_add],
        float(boundary_cfg.get("added_confidence", .46)))
    provenance[boundary_morph & boundary_add] = 3
    provenance[boundary_fill & boundary_add] = 4
    support[boundary_add] = np.maximum(support[boundary_add], 2)
    total_morphological |= boundary_morph & boundary_add
    total_fill |= boundary_fill & boundary_add
    total_fill |= boundary_removed
    class_stats["road_boundary_or_obstacle"] = boundary_stats

    invalid = (output == 0) | (output >= len(FINAL_CLASS_NAMES))
    output[invalid] = OTHER_ID
    confidence[invalid] = float(cfg.get("fill_confidence", .30))
    provenance[invalid] = 4
    support[invalid] = 1
    total_fill |= invalid
    stats = {
        "dense_coverage": float((output > 0).mean()),
        "invalid_id_pixels": int(
            ((output == 0) | (output >= len(FINAL_CLASS_NAMES))).sum()),
        "changed_from_previous_pixels": int(
            (output != current_mask).sum()),
        "temporal_evidence_pixels": int((provenance == 2).sum()),
        "morphological_link_pixels": int((provenance == 3).sum()),
        "final_fill_pixels": int((provenance == 4).sum()),
        "filter_added_pixels": int(
            (total_morphological | total_fill).sum()),
        "filter_added_outside_road_support_pixels": total_outside_road,
        "classes": class_stats,
    }
    result = FinalPassResult(
        output, np.clip(confidence, 0, 1), provenance,
        support.astype(np.uint8), road_support, stats)
    result.validate()
    return result


def compute_line_component_metrics(
        masks: Sequence[np.ndarray],
) -> dict[str, Any]:
    component_counts = []
    lengths = []
    per_class: dict[str, dict[str, float]] = {}
    for class_id in LINE_CLASS_IDS:
        class_counts = []
        class_lengths = []
        for mask in masks:
            components = _component_geometries(mask == class_id)[1]
            class_counts.append(len(components))
            class_lengths.extend(value.length for value in components)
        name = FINAL_CLASS_NAMES[int(class_id)]
        per_class[name] = {
            "mean_components_per_frame": float(np.mean(class_counts)),
            "mean_component_length_px": float(np.mean(class_lengths))
            if class_lengths else 0.0,
        }
        component_counts.extend(class_counts)
        lengths.extend(class_lengths)
    return {
        "mean_broken_components_per_class_frame": float(
            np.mean(component_counts)),
        "mean_component_length_px": float(np.mean(lengths))
        if lengths else 0.0,
        "per_class": per_class,
    }


def compute_internal_flicker(
        masks: Sequence[np.ndarray],
) -> dict[str, Any]:
    if not masks:
        raise ValueError("internal flicker requires masks")
    pixels = 0
    denominator = max(
        1, (len(masks) - 2) * masks[0].shape[0] * masks[0].shape[1])
    for index in range(1, len(masks) - 1):
        previous = masks[index - 1]
        current = masks[index]
        following = masks[index + 1]
        stable_neighbors = previous == following
        internal_neighbors = np.isin(previous, INTERNAL_CLASS_IDS)
        pixels += int(
            (stable_neighbors & internal_neighbors &
             (current != previous)).sum())
    return {
        "isolated_internal_flicker_pixels": pixels,
        "isolated_internal_flicker_rate": pixels / denominator,
    }


def _tree_contract(root: Path) -> str:
    """Hash all scientific raw files that the final stage promises read-only."""
    digest = hashlib.sha256()
    paths = [root / "manifest.json"]
    for directory in (
            "final_masks", "final_confidence", "external_masks",
            "external_confidence", "internal_masks", "internal_confidence"):
        paths.extend(sorted((root / directory).glob("*")))
    for path in paths:
        digest.update(str(path.relative_to(root)).encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def _presentation_complete(out: Path, stem: str) -> bool:
    return all((out / directory / f"{stem}{suffix}").exists()
               for directory, suffix in (
                   ("final_masks", ".png"),
                   ("final_confidence", ".png"),
                   ("final_provenance", ".png"),
                   ("final_support", ".png"),
                   ("metadata", ".json")))


def _full_resolution_result(
        previous_mask: np.ndarray,
        previous_confidence: np.ndarray,
        working_previous: np.ndarray,
        working_result: FinalPassResult,
) -> FinalPassResult:
    h, w = previous_mask.shape
    size = (w, h)
    previous_up = _resize_mask(working_previous, size)
    candidate_up = _resize_mask(working_result.mask, size)
    change = candidate_up != previous_up
    mask = previous_mask.copy()
    mask[change] = candidate_up[change]
    confidence = previous_confidence.copy()
    confidence_up = _resize_float(working_result.confidence, size)
    confidence[change] = confidence_up[change]
    provenance = np.full(mask.shape, 1, np.uint8)
    provenance_up = cv2.resize(
        working_result.provenance, size,
        interpolation=cv2.INTER_NEAREST).astype(np.uint8)
    provenance[change] = provenance_up[change]
    support = np.ones(mask.shape, np.uint8)
    support_up = cv2.resize(
        working_result.support_count, size,
        interpolation=cv2.INTER_NEAREST).astype(np.uint8)
    support[change] = support_up[change]
    road_support = cv2.resize(
        working_result.road_support.astype(np.uint8), size,
        interpolation=cv2.INTER_NEAREST) > 0
    stats = {
        **working_result.stats,
        "full_resolution_changed_pixels": int(change.sum()),
        "full_resolution_changed_fraction": float(change.mean()),
        "full_resolution_dense_coverage": float((mask > 0).mean()),
    }
    result = FinalPassResult(
        mask, np.clip(confidence, 0, 1), provenance,
        support, road_support, stats)
    result.validate()
    return result


def run_semantic_camera_final_pass(
        input_dir: str | Path,
        semantic_camera_dir: str | Path,
        previous_presentation_dir: str | Path,
        cfg: Config,
        output_dir: str | Path,
        resume: bool = True,
        force: bool = False,
        pair_flow_provider=compute_bidirectional_pair_flow,
) -> int:
    """Run the final pass using existing frames and raw predictions only."""
    root = Path(input_dir)
    scientific = Path(semantic_camera_dir)
    previous = Path(previous_presentation_dir)
    out = Path(output_dir)
    for directory in FINAL_OUTPUT_DIRS:
        (out / directory).mkdir(parents=True, exist_ok=True)
    final_cfg = cfg.get("semantic_camera_final", {})
    scientific_manifest = json.loads(
        (scientific / "manifest.json").read_text())
    previous_manifest = json.loads(
        (previous / "manifest.json").read_text())
    frames = iter_frames(root)
    maximum = cfg.get("frames.max_frames")
    if maximum is not None:
        frames = frames[:int(maximum)]
    if not frames:
        raise RuntimeError("final-pass input contains no frames")
    frame_contract = [
        (value.frame_index, value.capture_timestamp_ns) for value in frames]
    fingerprint = stable_hash([
        final_cfg, scientific_manifest["fingerprint"],
        previous_manifest["fingerprint"], frame_contract,
        list(FINAL_CLASS_NAMES)])
    manifest_path = out / "manifest.json"
    old_manifest = (
        json.loads(manifest_path.read_text())
        if manifest_path.exists() else None)
    if old_manifest and old_manifest.get("fingerprint") != fingerprint:
        if resume and not force:
            raise RuntimeError("final-pass resume fingerprint is incompatible")
        old_manifest = None
    raw_contract_before = _tree_contract(scientific)
    manifest = old_manifest or {
        "stage": "article1_semantic_camera_final_pass_v1",
        "fingerprint": fingerprint,
        "scientific_source": str(scientific.resolve()),
        "scientific_source_fingerprint": scientific_manifest["fingerprint"],
        "previous_presentation_source": str(previous.resolve()),
        "previous_presentation_fingerprint": previous_manifest["fingerprint"],
        "raw_scientific_output_preserved": True,
        "raw_contract_sha256_before": raw_contract_before,
        "presentation_is_non_causal": True,
        "uses_future_frames": True,
        "frame_independent_baseline_unchanged": True,
        "frame_contract": frame_contract,
        "done": {},
        "provenance_codes": {
            str(code): name for code, name in FINAL_PROVENANCE.items()},
    }
    if force:
        manifest["done"] = {}
    if manifest["raw_contract_sha256_before"] != raw_contract_before:
        raise RuntimeError("scientific raw contract changed before final pass")

    first_bgr = cv2.imread(
        str(frames[0].rectified_path), cv2.IMREAD_COLOR)
    if first_bgr is None:
        raise FileNotFoundError(frames[0].rectified_path)
    full_h, full_w = first_bgr.shape[:2]
    scale = float(final_cfg.get("processing_scale", .5))
    work_size = (
        max(2, round(full_w * scale)),
        max(2, round(full_h * scale)))
    working_previous: list[np.ndarray] = []
    working_confidence: list[np.ndarray] = []
    working_external: list[np.ndarray] = []
    working_external_confidence: list[np.ndarray] = []
    working_rgb: list[np.ndarray] = []
    for frame in frames:
        stem = f"frame_{frame.frame_index:06d}"
        bgr = cv2.imread(str(frame.rectified_path), cv2.IMREAD_COLOR)
        if bgr is None:
            raise FileNotFoundError(frame.rectified_path)
        if bgr.shape[:2] != (full_h, full_w):
            raise RuntimeError("final-pass RGB geometry changed")
        working_rgb.append(cv2.cvtColor(
            cv2.resize(bgr, work_size, interpolation=cv2.INTER_AREA),
            cv2.COLOR_BGR2RGB))
        working_previous.append(_resize_mask(
            read_mask_u16(
                previous / "presentation_masks" / f"{stem}.png"),
            work_size))
        working_confidence.append(_resize_float(
            _read_u8(
                previous / "presentation_confidence" / f"{stem}.png"),
            work_size))
        working_external.append(_resize_mask(
            read_mask_u16(
                scientific / "external_masks" / f"{stem}.png"),
            work_size))
        working_external_confidence.append(_resize_float(
            _read_u8(
                scientific / "external_confidence" / f"{stem}.png"),
            work_size))

    pair_flows: list[PairFlow] = []
    flow_rows = []
    flow_cfg = final_cfg.get("optical_flow", {})
    for index in range(len(frames) - 1):
        first = frames[index].frame_index
        second = frames[index + 1].frame_index
        cached_path = (
            previous / "flow_cache" /
            f"flow_{first:06d}_{second:06d}.npz")
        started = time.perf_counter()
        if cached_path.exists():
            pair = _load_pair_flow(cached_path)
            cache_hit = True
        else:
            pair = pair_flow_provider(
                working_rgb[index], working_rgb[index + 1], flow_cfg)
            cache_hit = False
        if pair.forward.shape[:2] != (work_size[1], work_size[0]):
            raise RuntimeError("cached flow geometry is incompatible")
        pair_flows.append(pair)
        flow_rows.append({
            "first_frame_index": first,
            "second_frame_index": second,
            "cache_hit": cache_hit,
            "valid_forward_fraction": float(pair.valid_forward.mean()),
            "valid_backward_fraction": float(pair.valid_backward.mean()),
            "elapsed_ms": (time.perf_counter() - started) * 1000,
        })

    temporal_cfg = final_cfg.get("temporal", {})
    geometry_cfg = final_cfg.get("geometry", {})
    final_working_masks: list[np.ndarray] = []
    filter_totals = {
        "filter_added_pixels": 0,
        "filter_added_outside_road_support_pixels": 0,
        "temporal_evidence_pixels": 0,
        "morphological_link_pixels": 0,
        "final_fill_pixels": 0,
    }
    timing_rows = []
    for index, frame in enumerate(frames):
        stem = f"frame_{frame.frame_index:06d}"
        key = str(frame.frame_index)
        started = time.perf_counter()
        if resume and not force and key in manifest["done"] and \
                _presentation_complete(out, stem):
            final_full = read_mask_u16(
                out / "final_masks" / f"{stem}.png")
            final_working_masks.append(
                _resize_mask(final_full, work_size))
            stats = manifest["done"][key]
            for metric in filter_totals:
                filter_totals[metric] += int(stats.get(metric, 0))
            timing_rows.append({
                "frame_index": frame.frame_index,
                "cache_hit": True,
                "elapsed_ms": (time.perf_counter() - started) * 1000,
            })
            continue
        temporal = stabilize_presentation_frame(
            index, working_previous, working_confidence,
            working_external, working_external_confidence,
            pair_flows, temporal_cfg)
        working_final = finalize_presentation_frame(
            temporal, working_previous[index], working_external[index],
            working_external_confidence[index], geometry_cfg)
        previous_full = read_mask_u16(
            previous / "presentation_masks" / f"{stem}.png")
        previous_confidence_full = _read_u8(
            previous / "presentation_confidence" / f"{stem}.png")
        full_result = _full_resolution_result(
            previous_full, previous_confidence_full,
            working_previous[index], working_final)
        write_mask_u16(
            out / "final_masks" / f"{stem}.png", full_result.mask)
        _write_u8(
            out / "final_confidence" / f"{stem}.png",
            np.clip(full_result.confidence * 255, 0, 255))
        _write_u8(
            out / "final_provenance" / f"{stem}.png",
            full_result.provenance)
        _write_u8(
            out / "final_support" / f"{stem}.png",
            full_result.support_count)
        metadata = {
            "frame_index": frame.frame_index,
            "capture_timestamp_ns": frame.capture_timestamp_ns,
            "scientific_source_fingerprint":
                scientific_manifest["fingerprint"],
            "previous_presentation_fingerprint":
                previous_manifest["fingerprint"],
            "raw_scientific_output_preserved": True,
            "presentation_is_non_causal": True,
            "uses_future_frames": True,
            "window_radius_frames": temporal_cfg.get(
                "window_radius_frames", 0),
            "provenance": {
                "current_model": int(
                    (full_result.provenance == 1).sum()),
                "temporal_evidence": int(
                    (full_result.provenance == 2).sum()),
                "morphological_link": int(
                    (full_result.provenance == 3).sum()),
                "final_fill": int(
                    (full_result.provenance == 4).sum()),
            },
            "provenance_codes": {
                str(code): name for code, name
                in FINAL_PROVENANCE.items()},
            "stats": full_result.stats,
        }
        atomic_write_json(
            out / "metadata" / f"{stem}.json", metadata)
        manifest["done"][key] = full_result.stats
        atomic_write_json(manifest_path, manifest)
        final_working_masks.append(working_final.mask)
        for metric in filter_totals:
            filter_totals[metric] += int(
                full_result.stats.get(metric, 0))
        timing_rows.append({
            "frame_index": frame.frame_index,
            "cache_hit": False,
            "elapsed_ms": (time.perf_counter() - started) * 1000,
        })

    before_metrics = compute_presentation_metrics(working_previous)
    after_metrics = compute_presentation_metrics(final_working_masks)
    before_metrics["flow_aligned"] = compute_flow_aligned_metrics(
        working_previous, pair_flows)
    after_metrics["flow_aligned"] = compute_flow_aligned_metrics(
        final_working_masks, pair_flows)
    before_metrics["line_components"] = compute_line_component_metrics(
        working_previous)
    after_metrics["line_components"] = compute_line_component_metrics(
        final_working_masks)
    before_metrics["internal_flicker"] = compute_internal_flicker(
        working_previous)
    after_metrics["internal_flicker"] = compute_internal_flicker(
        final_working_masks)
    total_pixels = len(frames) * work_size[0] * work_size[1]
    metrics = {
        "before_previous_presentation": before_metrics,
        "after_final_pass": after_metrics,
        "filter": {
            **filter_totals,
            "pixels_evaluated": total_pixels,
            "filter_added_fraction": (
                filter_totals["filter_added_pixels"]
                / max(1, total_pixels)),
            "added_outside_road_support_fraction": (
                filter_totals[
                    "filter_added_outside_road_support_pixels"]
                / max(1, filter_totals["filter_added_pixels"])),
        },
        "comparison": {
            "switch_rate_change": (
                after_metrics["class_switch_rate"]
                - before_metrics["class_switch_rate"]),
            "lane_continuity_change": (
                after_metrics["lane_and_marking_continuity"]
                - before_metrics["lane_and_marking_continuity"]),
            "broken_components_change": (
                after_metrics["line_components"][
                    "mean_broken_components_per_class_frame"]
                - before_metrics["line_components"][
                    "mean_broken_components_per_class_frame"]),
            "mean_component_length_change_px": (
                after_metrics["line_components"][
                    "mean_component_length_px"]
                - before_metrics["line_components"][
                    "mean_component_length_px"]),
            "internal_flicker_change": (
                after_metrics["internal_flicker"][
                    "isolated_internal_flicker_rate"]
                - before_metrics["internal_flicker"][
                    "isolated_internal_flicker_rate"]),
        },
        "coverage": {
            "dense_fraction": float(np.mean([
                np.all(mask > 0) for mask in final_working_masks])),
            "invalid_id_pixels": int(sum(
                ((mask == 0) | (mask >= len(FINAL_CLASS_NAMES))).sum()
                for mask in final_working_masks)),
        },
        "disclaimer": (
            "Pre-GT continuity and geometry diagnostics only; these values "
            "do not establish segmentation accuracy."),
    }
    atomic_write_json(out / "metrics.json", metrics)
    raw_contract_after = _tree_contract(scientific)
    if raw_contract_after != raw_contract_before:
        raise RuntimeError("scientific raw output changed during final pass")
    manifest["raw_contract_sha256_after"] = raw_contract_after
    manifest["raw_scientific_output_preserved"] = True
    atomic_write_json(manifest_path, manifest)
    elapsed_values = [row["elapsed_ms"] for row in timing_rows]
    summary = {
        "stage": manifest["stage"],
        "fingerprint": fingerprint,
        "frame_count": len(frames),
        "frame_range": [
            frames[0].frame_index, frames[-1].frame_index],
        "processing_geometry": [work_size[1], work_size[0]],
        "full_geometry": [full_h, full_w],
        "window_radius_frames": temporal_cfg.get(
            "window_radius_frames", 0),
        "presentation_is_non_causal": True,
        "raw_scientific_output_preserved": True,
        "raw_contract_sha256": raw_contract_after,
        "flow_cache_hits": int(sum(
            row["cache_hit"] for row in flow_rows)),
        "flow_pairs": len(flow_rows),
        "mean_ms_per_frame": float(np.mean(elapsed_values)),
        "peak_ram_mb": (
            resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024),
        "metrics": metrics,
    }
    atomic_write_json(out / "summary.json", summary)
    return 0
