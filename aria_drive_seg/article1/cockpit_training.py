"""Preparation of the shared car/motorcycle cockpit training stage.

Nothing here trains anything. It defines the contract that the reviewed annotations
must satisfy, the leakage-safe grouping rules, the augmentation policy and the
metrics that will be reported, so that the moment reviewed annotations exist the
training run is a configuration change rather than a design exercise.

Two invariants the rest of the project depends on:

* the cockpit model is **one shared model** for both vehicles. A car-only or
  motorcycle-only model is rejected at construction time, because the scientific
  claim requires the two domains to be measured with the same instrument;
* splits are **grouped**, never per frame. Consecutive frames of one drive are
  near-duplicates; putting one in train and its neighbour in validation would
  measure memorisation.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

import numpy as np

from .cockpit_segformer import COCKPIT_CLASSES
from .splits import assert_no_group_leakage

# Grouping units that are safe to split on. Every one of them is a unit that a
# whole continuous stretch of footage belongs to.
SAFE_SPLIT_UNITS = ("participant_id", "session_id", "recording_id",
                    "route_segment_id", "pairing_group_id", "matched_pair_id")

# Explicitly forbidden: these identify an individual frame, so splitting on them
# puts neighbouring frames of the same sequence on both sides.
FORBIDDEN_SPLIT_UNITS = ("frame_index", "source_frame_index", "sample_id",
                         "timestamp_ns", "image_path")

# Article 1 macro classes that map into the cockpit model's training space.
MACRO_TO_COCKPIT = {
    "mirror": "mirror",
    "instrument_display": "instrument_display",
    "control_and_ego_vehicle": "control_and_ego_vehicle",
}


@dataclass
class AugmentationPolicy:
    """Augmentations valid for a left/right-asymmetric cockpit.

    A horizontal flip is allowed only together with swapping every left/right hand
    attribute; flipping the image while keeping `left_hand_visible` would teach the
    model a false association.
    """

    horizontal_flip: bool = True
    horizontal_flip_swaps_side_attributes: bool = True
    color_jitter: Dict[str, float] = field(default_factory=lambda: {
        "brightness": 0.25, "contrast": 0.25, "saturation": 0.20, "hue": 0.03})
    mild_blur_sigma: List[float] = field(default_factory=lambda: [0.0, 1.2])
    # Vibration is a real motorcycle artefact; simulating a little of it helps the
    # shared model rather than biasing it toward one domain.
    motion_blur_kernel_px: List[int] = field(default_factory=lambda: [0, 7])
    random_resized_crop_scale: List[float] = field(default_factory=lambda: [0.7, 1.0])
    # Never applied: they would change the semantics of the cockpit geometry.
    forbidden: List[str] = field(default_factory=lambda: [
        "vertical_flip", "rotation_beyond_15_degrees", "channel_shuffle",
        "cutout_over_cockpit_classes"])

    def to_dict(self) -> Dict[str, Any]:
        from dataclasses import asdict
        return asdict(self)


SIDE_ATTRIBUTE_PAIRS = (
    ("left_hand_visible", "right_hand_visible"),
    ("left_hand_partially_visible", "right_hand_partially_visible"),
    ("left_hand_occluded", "right_hand_occluded"),
    ("left_hand_out_of_frame", "right_hand_out_of_frame"),
    ("left_hand_motion_blurred", "right_hand_motion_blurred"),
    ("left_arm_visible", "right_arm_visible"),
)


def swap_side_attributes(attributes: Dict[str, Any]) -> Dict[str, Any]:
    """Swap left/right attributes to accompany a horizontal flip."""
    out = dict(attributes)
    for a, b in SIDE_ATTRIBUTE_PAIRS:
        if a in attributes or b in attributes:
            out[a] = attributes.get(b)
            out[b] = attributes.get(a)
    return out


def validate_shared_domains(rows) -> None:
    """Reject a single-domain cockpit dataset."""
    domains = set(str(d) for d in rows["domain"]) if "domain" in rows else set()
    if domains != {"car", "motorcycle"}:
        raise RuntimeError(
            "the cockpit model is shared between car and motorcycle; the dataset "
            f"provides {sorted(domains) or 'no domain'}. A car-only or "
            "motorcycle-only cockpit model is not an acceptable final solution.")


def validate_split_unit(unit: str) -> None:
    if unit in FORBIDDEN_SPLIT_UNITS:
        raise ValueError(
            f"{unit!r} identifies a single frame; splitting on it would place "
            "neighbouring frames of the same sequence in train and validation")
    if unit not in SAFE_SPLIT_UNITS:
        raise ValueError(f"split unit must be one of {SAFE_SPLIT_UNITS}")


def grouped_split(rows, unit: str, validation_groups: Iterable[str]):
    """Split by a whole group, then verify no group appears on both sides."""
    validate_split_unit(unit)
    if unit not in rows:
        raise ValueError(f"missing split column {unit}")
    validate_shared_domains(rows)
    groups = {str(g) for g in validation_groups}
    mask = rows[unit].astype(str).isin(groups)
    valid, train = rows[mask].copy(), rows[~mask].copy()
    assert_no_group_leakage(train, valid, unit)
    return train, valid


def assert_no_temporal_leakage(train, valid, min_separation_s: float = 5.0) -> None:
    """No train frame may sit within `min_separation_s` of a validation frame.

    Frame-index distance is not enough across recordings sampled at different rates:
    10 frames apart is 1.0 s at 10 fps but 0.67 s at 15 fps. The guarantee is
    therefore expressed in seconds.
    """
    if "recording_id" not in train or "timestamp_ns" not in train:
        raise ValueError("temporal leakage check needs recording_id and timestamp_ns")
    limit_ns = float(min_separation_s) * 1e9
    shared = set(train["recording_id"].astype(str)) & set(valid["recording_id"].astype(str))
    for rec in shared:
        a = train.loc[train["recording_id"].astype(str) == rec, "timestamp_ns"].to_numpy(np.int64)
        b = valid.loc[valid["recording_id"].astype(str) == rec, "timestamp_ns"].to_numpy(np.int64)
        if a.size and b.size:
            closest = float(np.min(np.abs(a[:, None] - b[None, :])))
            if closest < limit_ns:
                raise ValueError(
                    f"temporal leakage in recording {rec}: a train frame is "
                    f"{closest / 1e9:.2f} s from a validation frame, below the "
                    f"{min_separation_s} s guarantee")


def planned_metrics() -> Dict[str, Any]:
    """Metrics that will be reported once reviewed annotations exist."""
    return {
        "primary": [
            "per-class IoU for mirror, instrument_display, control_and_ego_vehicle",
            "mean IoU over the three cockpit classes",
            "per-domain IoU reported separately for car and motorcycle",
            "the gap between the two domains, as the domain-shift indicator",
        ],
        "secondary": [
            "boundary F-score on the cockpit classes",
            "per-class precision and recall",
            "calibration: reliability of the predicted confidence",
            "normalised predictive entropy on correct vs incorrect pixels",
        ],
        "hands": [
            "reported only on frames a reviewer marked visible or "
            "partially_visible",
            "on frames marked out_of_frame, occluded, not_visible or uncertain a "
            "missing mask is NOT counted as an error",
            "false hand masks in those frames ARE counted and reported separately",
        ],
        "reporting_rules": [
            "no metric is computed against pseudo-labels",
            "every number is reported per domain as well as pooled",
            "the unit of inference is the recording or the matched pair, never the "
            "individual frame",
        ],
    }


def training_readiness(dataset_dir: str | Path) -> Dict[str, Any]:
    """Report exactly what is still missing before training may start."""
    root = Path(dataset_dir)
    marker = root / "REVIEWED_ANNOTATIONS.json"
    blockers: List[str] = []
    if not marker.exists():
        blockers.append(
            f"{marker} is absent: no human-reviewed annotation has been delivered")
    if not (root / "masks").exists():
        blockers.append("no reviewed mask directory")
    return {
        "ready_to_train": not blockers,
        "blockers": blockers,
        "gate": ("training is blocked until reviewed car AND motorcycle cockpit "
                 "annotations exist; pseudo-labels are never promoted to ground "
                 "truth"),
        "classes": list(COCKPIT_CLASSES),
        "split_units_allowed": list(SAFE_SPLIT_UNITS),
        "split_units_forbidden": list(FORBIDDEN_SPLIT_UNITS),
        "augmentation": AugmentationPolicy().to_dict(),
        "metrics": planned_metrics(),
    }


def future_fusion_contract() -> Dict[str, Any]:
    """How the trained cockpit model will replace the current proxy."""
    return {
        "replaces": "grounded_sam2_cockpit_proxy + geometric_cockpit_proxy",
        "internal_provider": "SegFormerInternalProvider",
        "activation": ("semantic_camera.internal.mode=auto already prefers a "
                       "reviewed checkpoint when weights/segformer-b2-article1-"
                       "cockpit exists; no fusion code change is required"),
        "output_contract": {
            "probabilities": "4xHxW normalised over " + ", ".join(COCKPIT_CLASSES),
            "mask": "HxW uint16 in the cockpit training space",
            "confidence": "HxW float32",
            "entropy": "HxW normalised predictive entropy",
        },
        "mapping_to_macro_taxonomy": MACRO_TO_COCKPIT,
        "fallback_policy": (
            "if the reviewed checkpoint is absent the pipeline must keep using the "
            "explicitly-marked proxy and keep reporting internal_is_fallback=true"),
        "revalidation_required": [
            "internal_min_confidence thresholds must be recalibrated against the "
            "trained model, not inherited from the proxy",
            "the geometric bottom-fraction prior must be removed for the motorcycle "
            "once the trained model covers the handlebar",
        ],
    }
