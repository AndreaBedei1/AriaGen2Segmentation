"""Portable causal state for Article 1 temporal stabilization."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..io_utils import atomic_write


@dataclass
class TemporalState:
    probabilities: np.ndarray
    mask: np.ndarray
    confidence: np.ndarray
    class_age: np.ndarray
    propagation_age: np.ndarray
    candidate_class: np.ndarray
    candidate_age: np.ndarray
    thin_mask: np.ndarray
    thin_confidence: np.ndarray
    previous_rgb: np.ndarray
    frame_index: int
    timestamp_ns: int
    reset_count: int
    config_fingerprint: str
    static_policy_fingerprint: str

    def validate(self) -> None:
        shape = self.mask.shape
        if self.probabilities.ndim != 3 or self.probabilities.shape[1:] != shape:
            raise ValueError("temporal state probability geometry mismatch")
        for array in (
            self.confidence, self.class_age, self.propagation_age,
            self.candidate_class, self.candidate_age, self.thin_mask,
            self.thin_confidence,
        ):
            if array.shape != shape:
                raise ValueError("temporal state map geometry mismatch")
        if self.previous_rgb.shape[:2] != shape:
            raise ValueError("temporal state RGB geometry mismatch")
        if not np.allclose(self.probabilities.sum(0), 1, atol=2e-3):
            raise ValueError("temporal probabilities are not normalized")

    def save(self, path: str | Path) -> None:
        self.validate()
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with atomic_write(path, "wb") as handle:
            np.savez_compressed(
                handle,
                probabilities=self.probabilities.astype(np.float16),
                mask=self.mask.astype(np.uint16),
                confidence=self.confidence.astype(np.float16),
                class_age=self.class_age.astype(np.uint16),
                propagation_age=self.propagation_age.astype(np.uint16),
                candidate_class=self.candidate_class.astype(np.uint16),
                candidate_age=self.candidate_age.astype(np.uint16),
                thin_mask=self.thin_mask.astype(np.uint16),
                thin_confidence=self.thin_confidence.astype(np.float16),
                previous_rgb=self.previous_rgb.astype(np.uint8),
                frame_index=np.int64(self.frame_index),
                timestamp_ns=np.int64(self.timestamp_ns),
                reset_count=np.int64(self.reset_count),
                config_fingerprint=np.asarray(self.config_fingerprint),
                static_policy_fingerprint=np.asarray(
                    self.static_policy_fingerprint),
            )

    @classmethod
    def load(cls, path: str | Path, config_fingerprint: str,
             static_policy_fingerprint: str) -> "TemporalState":
        with np.load(path) as data:
            stored_config = str(data["config_fingerprint"].item())
            stored_static = str(data["static_policy_fingerprint"].item())
            if stored_config != config_fingerprint:
                raise RuntimeError("incompatible temporal state config fingerprint")
            if stored_static != static_policy_fingerprint:
                raise RuntimeError("incompatible static policy fingerprint")
            state = cls(
                probabilities=data["probabilities"].astype(np.float32),
                mask=data["mask"].astype(np.uint16),
                confidence=data["confidence"].astype(np.float32),
                class_age=data["class_age"].astype(np.uint16),
                propagation_age=data["propagation_age"].astype(np.uint16),
                candidate_class=data["candidate_class"].astype(np.uint16),
                candidate_age=data["candidate_age"].astype(np.uint16),
                thin_mask=data["thin_mask"].astype(np.uint16),
                thin_confidence=data["thin_confidence"].astype(np.float32),
                previous_rgb=data["previous_rgb"].astype(np.uint8),
                frame_index=int(data["frame_index"]),
                timestamp_ns=int(data["timestamp_ns"]),
                reset_count=int(data["reset_count"]),
                config_fingerprint=stored_config,
                static_policy_fingerprint=stored_static,
            )
        state.validate()
        return state
