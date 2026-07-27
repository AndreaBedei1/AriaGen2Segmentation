"""SegFormer-B2 cockpit interface for Article 1.

Training is intentionally not started until manually reviewed annotations from both
car and motorcycle domains are available. The schema lives here so annotation and
downstream fusion code can target one stable contract without fabricating weights.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


COCKPIT_CLASSES = (
    "background_internal", "mirror", "instrument_display",
    "control_and_ego_vehicle",
)


@dataclass
class CockpitOutput:
    probabilities: np.ndarray  # 4×H×W, normalized
    mask: np.ndarray           # H×W uint16 training-space ids
    confidence: np.ndarray     # H×W float32
    domain: str                # car | motorcycle
    checkpoint: str

    def validate(self) -> None:
        if self.domain not in {"car", "motorcycle"}:
            raise ValueError("domain must be car or motorcycle")
        if self.probabilities.ndim != 3 or self.probabilities.shape[0] != len(COCKPIT_CLASSES):
            raise ValueError("probabilities must have shape 4xHxW")
        if self.mask.shape != self.probabilities.shape[1:]:
            raise ValueError("mask/probability geometry mismatch")
        if not np.allclose(self.probabilities.sum(axis=0), 1, atol=2e-3):
            raise ValueError("cockpit probabilities must sum to one")


def require_reviewed_annotations(dataset_dir: str | Path) -> None:
    """Fail closed rather than treating pseudo-labels as ground truth."""
    marker = Path(dataset_dir) / "REVIEWED_ANNOTATIONS.json"
    if not marker.exists():
        raise RuntimeError(
            "SegFormer training blocked: manually reviewed car+motorcycle "
            "cockpit annotations are required.")
