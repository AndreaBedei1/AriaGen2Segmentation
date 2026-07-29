"""Dense Article 1 semantic-camera fusion.

The external road-scene stream and the cockpit stream remain independent inputs.
The cockpit stream can be a reviewed SegFormer checkpoint or an explicitly marked
Grounded-SAM2/geometric proxy.  Fusion is per-frame and never mutates either source.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
import math
import resource
import time
from pathlib import Path
from typing import Any, Protocol

import cv2
import numpy as np

from ..config import Config
from ..hashing import stable_hash
from ..io_utils import (
    atomic_write, atomic_write_bytes, atomic_write_json, read_mask_u16,
    write_mask_u16,
)
from ..segmentation.base import FrameOutput, iter_frames
from ..taxonomy import Taxonomy
from .cockpit_segformer import COCKPIT_CLASSES, CockpitSegFormer


FINAL_CLASS_NAMES = (
    "unknown", "road_surface", "lane_marking",
    "regulatory_road_marking", "vehicle", "two_wheeler", "pedestrian",
    "traffic_light", "traffic_sign", "road_boundary_or_obstacle", "mirror",
    "instrument_display", "control_and_ego_vehicle", "other_environment",
)

PROVENANCE = {
    0: "invalid_unassigned",
    1: "external_mask2former",
    2: "cockpit_segformer",
    3: "grounded_sam2_cockpit_proxy",
    4: "geometric_cockpit_proxy",
    5: "dense_other_environment_fill",
    6: "conflict_external_wins",
    7: "conflict_internal_wins",
}

INTERNAL_TO_ARTICLE1 = np.asarray([0, 10, 11, 12], dtype=np.uint16)

MIRROR_NAMES = {
    "rear_view_mirror", "side_mirror", "left_side_mirror",
    "right_side_mirror",
}
DISPLAY_NAMES = {
    "instrument_cluster", "infotainment_screen", "speedometer_display",
    "warning_indicator", "navigation_display",
}
CONTROL_NAMES = {
    "steering_wheel", "dashboard", "a_pillar", "a_pillar_left",
    "a_pillar_right", "driver_hand", "driver_arm", "smartphone",
    "other_cockpit", "center_console", "climate_controls", "gear_selector",
    "left_air_vent", "right_air_vent", "sun_visor", "door_panel",
    "phone_mount", "passenger", "headrest",
}


@dataclass
class ExternalEvidence:
    mask: np.ndarray
    confidence: np.ndarray
    entropy: np.ndarray
    probabilities: np.ndarray | None
    source_kind: str
    entropy_kind: str

    def validate(self) -> None:
        shape = self.mask.shape
        if self.mask.ndim != 2:
            raise ValueError("external mask must be HxW")
        if self.confidence.shape != shape or self.entropy.shape != shape:
            raise ValueError("external evidence geometry mismatch")
        if self.probabilities is not None:
            if self.probabilities.ndim != 3 or \
                    self.probabilities.shape[1:] != shape:
                raise ValueError("external probability geometry mismatch")
            if not np.allclose(
                    self.probabilities.sum(0), 1.0, atol=3e-3):
                raise ValueError("external probabilities are not normalized")
        if np.any(self.mask < 0) or np.any(self.mask >= len(FINAL_CLASS_NAMES)):
            raise ValueError("external mask id outside Article 1 taxonomy")


@dataclass
class InternalEvidence:
    probabilities: np.ndarray  # background, mirror, display, control
    mask: np.ndarray           # ids 0..3
    confidence: np.ndarray
    entropy: np.ndarray
    evidence_kind: np.ndarray  # 0 background, 1 model/detection, 2 geometry
    source_kind: str
    checkpoint: str
    fallback: bool

    def validate(self) -> None:
        if self.probabilities.ndim != 3 or \
                self.probabilities.shape[0] != len(COCKPIT_CLASSES):
            raise ValueError("internal probabilities must have shape 4xHxW")
        shape = self.probabilities.shape[1:]
        if any(value.shape != shape for value in (
                self.mask, self.confidence, self.entropy,
                self.evidence_kind)):
            raise ValueError("internal evidence geometry mismatch")
        if np.any(self.mask < 0) or np.any(self.mask >= len(COCKPIT_CLASSES)):
            raise ValueError("internal mask id outside cockpit schema")
        if not np.allclose(self.probabilities.sum(0), 1.0, atol=3e-3):
            raise ValueError("internal probabilities are not normalized")


@dataclass
class SemanticCameraResult:
    mask: np.ndarray
    confidence: np.ndarray
    entropy: np.ndarray
    provenance: np.ndarray
    conflict: np.ndarray
    internal_article1_mask: np.ndarray
    stats: dict[str, Any]

    def validate(self) -> None:
        shape = self.mask.shape
        if any(value.shape != shape for value in (
                self.confidence, self.entropy, self.provenance,
                self.conflict, self.internal_article1_mask)):
            raise ValueError("semantic-camera result geometry mismatch")
        if np.any(self.mask == 0):
            raise ValueError("semantic-camera output is not dense")
        allowed = np.arange(1, len(FINAL_CLASS_NAMES), dtype=np.uint16)
        if not np.isin(self.mask, allowed).all():
            raise ValueError("semantic-camera output contains an invalid class")
        if np.any(self.provenance == 0):
            raise ValueError("semantic-camera provenance is incomplete")


class InternalProvider(Protocol):
    source_kind: str
    fallback: bool

    def descriptor(self) -> dict[str, Any]: ...

    def infer(self, image_rgb: np.ndarray, frame_index: int,
              timestamp_ns: int, vehicle_type: str) -> InternalEvidence: ...


def _normalized_entropy(probabilities: np.ndarray) -> np.ndarray:
    count = probabilities.shape[0]
    return (
        -(probabilities * np.log(np.maximum(probabilities, 1e-8))).sum(0)
        / math.log(count)
    ).astype(np.float32)


def _top1_entropy_proxy(confidence: np.ndarray, class_count: int) -> np.ndarray:
    """Maximum normalized entropy compatible with a supplied top-1 score."""
    top = np.clip(np.asarray(confidence, np.float32), 1e-8, 1.0)
    residual = np.maximum(1.0 - top, 1e-8)
    tail = residual / max(1, class_count - 1)
    value = -(top * np.log(top) + (class_count - 1) * tail * np.log(tail))
    return np.clip(value / math.log(class_count), 0, 1).astype(np.float32)


def _read_float_map(path: Path) -> np.ndarray:
    if path.suffix == ".npz":
        with np.load(path) as data:
            if not data.files:
                raise RuntimeError(f"empty diagnostic array: {path}")
            value = data[data.files[0]]
        return np.asarray(value, np.float32)
    value = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if value is None:
        raise FileNotFoundError(path)
    value = np.asarray(value, np.float32)
    if np.issubdtype(value.dtype, np.integer) or value.max(initial=0) > 1:
        value /= 255.0
    return value


class ExternalEvidenceReader:
    """Read real Article 1 Mask2Former evidence without changing its source."""

    def __init__(self, root: str | Path, use_probabilities: bool = True):
        self.root = Path(root)
        self.use_probabilities = bool(use_probabilities)
        if (self.root / "static_masks").is_dir():
            self.mask_dir = self.root / "static_masks"
            self.confidence_dir = self.root / "static_confidence"
            self.probability_dir = self.root / "diagnostic_probabilities"
            self.entropy_dir = None
            self.kind = "mask2former_temporal_static_raw"
        elif (self.root / "masks").is_dir():
            self.mask_dir = self.root / "masks"
            self.confidence_dir = (
                self.root / "top1_probability"
                if (self.root / "top1_probability").is_dir()
                else self.root / "confidence"
            )
            self.probability_dir = self.root / "probabilities"
            self.entropy_dir = (
                self.root / "normalized_entropy"
                if (self.root / "normalized_entropy").is_dir()
                else self.root / "entropy"
            )
            self.kind = "mask2former_article1_external"
        else:
            raise RuntimeError(
                f"unsupported external Article 1 layout: {self.root}")
        if not self.mask_dir.is_dir() or not self.confidence_dir.is_dir():
            raise RuntimeError(
                f"external mask/confidence source is incomplete: {self.root}")

    def descriptor(self) -> dict[str, Any]:
        manifest = self.root / "manifest.json"
        manifest_data = json.loads(manifest.read_text()) if manifest.exists() else {}
        return {
            "kind": self.kind,
            "root": str(self.root.resolve()),
            "manifest_fingerprint": (
                manifest_data.get("fingerprint")
                or manifest_data.get("static_policy_fingerprint")
                or stable_hash(manifest_data)
            ),
            "use_probabilities": self.use_probabilities,
        }

    @staticmethod
    def _candidate(directory: Path | None, stem: str) -> Path | None:
        if directory is None:
            return None
        for suffix in (".npz", ".png"):
            path = directory / f"{stem}{suffix}"
            if path.exists():
                return path
        return None

    def read(self, frame_index: int) -> ExternalEvidence:
        stem = f"frame_{frame_index:06d}"
        mask_path = self.mask_dir / f"{stem}.png"
        confidence_path = self._candidate(self.confidence_dir, stem)
        if confidence_path is None:
            raise RuntimeError(
                f"missing external confidence for frame {frame_index}")
        mask = read_mask_u16(mask_path)
        confidence = _read_float_map(confidence_path)
        probabilities = None
        probability_path = self._candidate(self.probability_dir, stem)
        if self.use_probabilities and probability_path is not None:
            with np.load(probability_path) as data:
                key = (
                    "static_probabilities"
                    if "static_probabilities" in data.files
                    else "probabilities"
                )
                probabilities = np.asarray(data[key], np.float32)
            probabilities /= np.maximum(
                probabilities.sum(0, keepdims=True), 1e-8)
            entropy = _normalized_entropy(probabilities)
            entropy_kind = "exact_macro_probabilities"
        else:
            entropy_path = self._candidate(self.entropy_dir, stem)
            if entropy_path is not None:
                entropy = _read_float_map(entropy_path)
                entropy_kind = "saved_normalized_entropy"
            else:
                entropy = _top1_entropy_proxy(
                    confidence, len(FINAL_CLASS_NAMES))
                entropy_kind = "top1_maximum_entropy_proxy"
        result = ExternalEvidence(
            mask.astype(np.uint16), np.clip(confidence, 0, 1),
            np.clip(entropy, 0, 1), probabilities, self.kind, entropy_kind)
        result.validate()
        return result


class SegFormerInternalProvider:
    source_kind = "segformer_cockpit_reviewed"
    fallback = False

    def __init__(self, checkpoint: str | Path, device: str = "cuda"):
        self.checkpoint = Path(checkpoint)
        self.model = CockpitSegFormer(self.checkpoint, device=device)

    def descriptor(self) -> dict[str, Any]:
        files = sorted(
            (str(path.relative_to(self.checkpoint)), path.stat().st_size)
            for path in self.checkpoint.rglob("*") if path.is_file())
        return {
            "kind": self.source_kind,
            "checkpoint": str(self.checkpoint.resolve()),
            "checkpoint_layout_fingerprint": stable_hash(files),
            "fallback": False,
        }

    def infer(self, image_rgb: np.ndarray, frame_index: int,
              timestamp_ns: int, vehicle_type: str) -> InternalEvidence:
        output = self.model.infer(image_rgb, vehicle_type)
        entropy = (
            output.entropy if output.entropy is not None
            else _normalized_entropy(output.probabilities)
        )
        evidence = (output.mask > 0).astype(np.uint8)
        result = InternalEvidence(
            output.probabilities.astype(np.float32),
            output.mask.astype(np.uint16), output.confidence.astype(np.float32),
            entropy.astype(np.float32), evidence, self.source_kind,
            output.checkpoint, False)
        result.validate()
        return result


class GroundedSAM2CockpitProxy:
    """Real local open-vocabulary inference, but not a trained cockpit model."""

    source_kind = "grounded_sam2_cockpit_proxy"
    fallback = True

    def __init__(self, cfg: Config, geometric_cfg: dict[str, Any]):
        self.cfg = cfg
        self.geometric_cfg = geometric_cfg
        self._segmenter = None
        self._taxonomy = Taxonomy.load(
            cfg.resolve(cfg.get("project.classes")))

    def descriptor(self) -> dict[str, Any]:
        weights_manifest = self.cfg.resolve("weights/manifest.json")
        prompt_path = self.cfg.resolve(
            self.cfg.get("project.grounded_prompts"))
        return {
            "kind": self.source_kind,
            "fallback": True,
            "grounded_sam2_config": self.cfg.get("grounded_sam2"),
            "prompt_fingerprint": stable_hash(prompt_path.read_text()),
            "weights_manifest_fingerprint": stable_hash(
                weights_manifest.read_text()
                if weights_manifest.exists() else "missing"),
            "geometric_proxy": self.geometric_cfg,
        }

    def _load(self) -> None:
        if self._segmenter is not None:
            return
        from ..segmentation.grounded_sam2 import GroundedSAM2Segmenter
        self._segmenter = GroundedSAM2Segmenter(self.cfg, self._taxonomy)
        self._segmenter.load()

    def infer(self, image_rgb: np.ndarray, frame_index: int,
              timestamp_ns: int, vehicle_type: str) -> InternalEvidence:
        self._load()
        output = self._segmenter.segment(
            image_rgb, frame_index, timestamp_ns, gaze_xy=None)
        return grounded_output_to_internal(
            output, self._taxonomy, self.geometric_cfg)


def grounded_output_to_internal(
        output: FrameOutput, source_taxonomy: Taxonomy,
        geometric_cfg: dict[str, Any] | None = None) -> InternalEvidence:
    """Roll Grounded-SAM2 cockpit detections into the stable 4-class schema."""
    geometric_cfg = geometric_cfg or {}
    h, w = output.canonical_mask.shape
    internal_mask = np.zeros((h, w), np.uint16)
    evidence_kind = np.zeros((h, w), np.uint8)
    detected_confidence = np.asarray(
        output.confidence
        if output.confidence is not None else np.zeros((h, w)),
        np.float32)
    combined = np.zeros((h, w), np.uint16)
    for layer in ("cockpit", "mirror"):
        value = output.layers.get(layer)
        if value is not None:
            active = value > 0
            combined[active] = value[active]
    for cid in np.unique(combined):
        if cid == 0:
            continue
        name = source_taxonomy.name_of(int(cid))
        if name in MIRROR_NAMES:
            target = 1
        elif name in DISPLAY_NAMES:
            target = 2
        elif name in CONTROL_NAMES:
            target = 3
        else:
            continue
        pixels = combined == cid
        internal_mask[pixels] = target
        evidence_kind[pixels] = 1

    confidence = np.zeros((h, w), np.float32)
    detected = internal_mask > 0
    confidence[detected] = np.clip(detected_confidence[detected], 0, 1)
    use_geometry = bool(geometric_cfg.get("enabled", True))
    if use_geometry:
        start = int(round(
            h * float(geometric_cfg.get("bottom_start_fraction", .74))))
        geometry = np.zeros((h, w), bool)
        geometry[max(0, min(start, h)):] = True
        geometry &= internal_mask == 0
        internal_mask[geometry] = 3
        evidence_kind[geometry] = 2
        confidence[geometry] = float(
            geometric_cfg.get("confidence", .35))

    probabilities = np.zeros((4, h, w), np.float32)
    probabilities[0] = 1.0
    active = internal_mask > 0
    if active.any():
        rows, cols = np.where(active)
        target = internal_mask[active]
        probabilities[0, rows, cols] = 1.0 - confidence[active]
        probabilities[target, rows, cols] = confidence[active]
    result = InternalEvidence(
        probabilities, internal_mask, confidence,
        _normalized_entropy(probabilities), evidence_kind,
        "grounded_sam2_cockpit_proxy",
        "local Grounding DINO + SAM2.1; no reviewed cockpit training", True)
    result.validate()
    return result


def resolve_internal_provider(cfg: Config) -> InternalProvider:
    semantic_cfg = cfg.get("semantic_camera", {})
    internal_cfg = semantic_cfg.get("internal", {})
    mode = internal_cfg.get("mode", "auto")
    checkpoint = cfg.resolve(internal_cfg.get(
        "segformer_checkpoint", "weights/segformer-b2-article1-cockpit"))
    if mode in {"auto", "segformer"} and checkpoint.exists():
        return SegFormerInternalProvider(
            checkpoint, device=internal_cfg.get("device", "cuda"))
    if mode == "segformer":
        raise RuntimeError(
            "reviewed cockpit SegFormer requested but checkpoint is absent")
    fallback = internal_cfg.get("fallback", "grounded_sam2_proxy")
    if fallback != "grounded_sam2_proxy":
        raise RuntimeError(
            "no reviewed cockpit model and no authorized internal fallback")
    return GroundedSAM2CockpitProxy(
        cfg, internal_cfg.get("geometric_proxy", {}))


def _priority_map(mask: np.ndarray, priorities: dict[str, Any]) -> np.ndarray:
    values = np.zeros(mask.shape, np.float32)
    for cid, name in enumerate(FINAL_CLASS_NAMES):
        values[mask == cid] = float(priorities.get(name, 0))
    return values


def fuse_semantic_camera(
        external: ExternalEvidence, internal: InternalEvidence,
        cfg: dict[str, Any]) -> SemanticCameraResult:
    """Fuse two same-frame evidence streams into a dense Article 1 mask."""
    external.validate()
    internal.validate()
    if external.mask.shape != internal.mask.shape:
        raise ValueError("external/internal fusion geometry mismatch")
    ext_mask = external.mask.astype(np.uint16)
    int_article = INTERNAL_TO_ARTICLE1[internal.mask]
    final = ext_mask.copy()
    confidence = external.confidence.astype(np.float32).copy()
    entropy = external.entropy.astype(np.float32).copy()
    provenance = np.full(ext_mask.shape, 1, np.uint8)
    conflict = np.zeros(ext_mask.shape, np.uint8)

    ext_priorities = cfg.get("external_priorities", {})
    int_priorities = cfg.get("internal_priorities", {})
    entropy_penalty = float(cfg.get("entropy_penalty", .20))
    ext_score = (
        external.confidence
        - entropy_penalty * external.entropy
        + _priority_map(ext_mask, ext_priorities) / 1000.0
    )
    int_score = (
        internal.confidence
        - entropy_penalty * internal.entropy
        + _priority_map(int_article, int_priorities) / 1000.0
    )
    minimums = cfg.get("internal_min_confidence", {})
    minimum_map = np.zeros(ext_mask.shape, np.float32)
    for cid in (10, 11, 12):
        minimum_map[int_article == cid] = float(
            minimums.get(FINAL_CLASS_NAMES[cid], .35))
    internal_candidate = (
        (int_article > 0) & (internal.confidence >= minimum_map))

    # Open-vocabulary detections compete with external evidence using confidence,
    # entropy and class priority. The geometric proxy has a much narrower gate.
    detected = internal_candidate & (internal.evidence_kind == 1)
    geometry = internal_candidate & (internal.evidence_kind == 2)
    allowed_geometry_names = set(cfg.get(
        "geometric_override_classes",
        ["unknown", "vehicle", "road_boundary_or_obstacle",
         "other_environment"]))
    allowed_geometry_ids = [
        FINAL_CLASS_NAMES.index(name) for name in allowed_geometry_names]
    geometry &= np.isin(ext_mask, allowed_geometry_ids)
    geometry &= external.confidence <= float(
        cfg.get("geometric_external_max_confidence", .65))

    margin = float(cfg.get("conflict_margin", .03))
    model_wins = detected & (int_score >= ext_score + margin)
    geometry_wins = geometry & (int_score >= ext_score - float(
        cfg.get("geometric_score_relaxation", .20)))
    internal_wins = model_wins | geometry_wins
    conflicts = internal_candidate & (ext_mask != int_article) & (ext_mask != 0)
    conflict[conflicts & ~internal_wins] = 1
    conflict[conflicts & internal_wins] = 2
    provenance[conflicts & ~internal_wins] = 6

    if internal_wins.any():
        final[internal_wins] = int_article[internal_wins]
        confidence[internal_wins] = internal.confidence[internal_wins]
        entropy[internal_wins] = internal.entropy[internal_wins]
        provenance[model_wins] = (
            2 if internal.source_kind == "segformer_cockpit_reviewed" else 3)
        provenance[geometry_wins] = 4
        provenance[conflicts & internal_wins] = 7

    dense_fill = final == 0
    final[dense_fill] = FINAL_CLASS_NAMES.index("other_environment")
    confidence[dense_fill] = float(cfg.get("dense_fill_confidence", 0.0))
    entropy[dense_fill] = 1.0
    provenance[dense_fill] = 5

    stats = {
        "pixels": int(final.size),
        "dense_coverage": float((final > 0).mean()),
        "external_selected_fraction": float(
            ((~internal_wins) & ~dense_fill).mean()),
        "internal_model_selected_fraction": float(model_wins.mean()),
        "geometric_proxy_selected_fraction": float(geometry_wins.mean()),
        "dense_fill_fraction": float((provenance == 5).mean()),
        "conflict_fraction": float((conflict > 0).mean()),
        "internal_conflict_win_fraction": float((conflict == 2).mean()),
        "unknown_external_fraction": float((ext_mask == 0).mean()),
        "mean_final_confidence": float(confidence.mean()),
        "mean_final_entropy": float(entropy.mean()),
    }
    result = SemanticCameraResult(
        final, np.clip(confidence, 0, 1), np.clip(entropy, 0, 1),
        provenance, conflict, int_article, stats)
    result.validate()
    return result


OUTPUT_DIRS = (
    "final_masks", "overlays", "final_confidence", "final_entropy",
    "provenance", "conflicts", "external_masks", "external_confidence",
    "internal_masks", "internal_confidence", "metadata",
    "diagnostic_probabilities", "videos",
)


def _write_u8(path: Path, value: np.ndarray) -> None:
    ok, encoded = cv2.imencode(".png", np.asarray(value, np.uint8))
    if not ok:
        raise RuntimeError(f"could not encode {path}")
    atomic_write_bytes(path, encoded.tobytes())


def render_dense_overlay(
        image_rgb: np.ndarray, result: SemanticCameraResult,
        taxonomy: Taxonomy, frame_index: int, timestamp_ns: int,
        internal_label: str, alpha: float = .45) -> np.ndarray:
    color = taxonomy.colorize(result.mask)
    overlay = cv2.addWeighted(
        image_rgb, 1.0 - alpha, color, alpha, 0)
    overlay = cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR)
    cv2.rectangle(overlay, (0, 0), (overlay.shape[1], 82), (12, 15, 18), -1)
    lines = [
        f"SEMANTIC CAMERA | frame {frame_index} | ts {timestamp_ns}",
        f"dense=100.00% | internal={internal_label} | "
        f"proxy={100 * (result.stats['internal_model_selected_fraction'] + result.stats['geometric_proxy_selected_fraction']):.2f}% "
        f"| fill={100 * result.stats['dense_fill_fraction']:.2f}%",
    ]
    for line_no, text in enumerate(lines):
        cv2.putText(
            overlay, text, (14, 29 + line_no * 34),
            cv2.FONT_HERSHEY_SIMPLEX, .72, (255, 255, 255), 2,
            cv2.LINE_AA)
    return overlay


def _write_frame(
        out: Path, stem: str, image_rgb: np.ndarray,
        external: ExternalEvidence, internal: InternalEvidence,
        result: SemanticCameraResult, metadata: dict[str, Any],
        taxonomy: Taxonomy, save_probabilities: bool, alpha: float) -> None:
    write_mask_u16(out / "final_masks" / f"{stem}.png", result.mask)
    write_mask_u16(
        out / "external_masks" / f"{stem}.png", external.mask)
    write_mask_u16(
        out / "internal_masks" / f"{stem}.png",
        result.internal_article1_mask)
    _write_u8(
        out / "final_confidence" / f"{stem}.png",
        np.clip(result.confidence * 255, 0, 255))
    _write_u8(
        out / "final_entropy" / f"{stem}.png",
        np.clip(result.entropy * 255, 0, 255))
    _write_u8(out / "provenance" / f"{stem}.png", result.provenance)
    _write_u8(out / "conflicts" / f"{stem}.png", result.conflict)
    _write_u8(
        out / "external_confidence" / f"{stem}.png",
        np.clip(external.confidence * 255, 0, 255))
    _write_u8(
        out / "internal_confidence" / f"{stem}.png",
        np.clip(internal.confidence * 255, 0, 255))
    overlay = render_dense_overlay(
        image_rgb, result, taxonomy, metadata["frame_index"],
        metadata["capture_timestamp_ns"], internal.source_kind, alpha)
    ok, encoded = cv2.imencode(
        ".jpg", overlay, [cv2.IMWRITE_JPEG_QUALITY, 92])
    if not ok:
        raise RuntimeError("could not encode semantic-camera overlay")
    atomic_write_bytes(out / "overlays" / f"{stem}.jpg", encoded.tobytes())
    if save_probabilities:
        with atomic_write(
                out / "diagnostic_probabilities" / f"{stem}.npz", "wb") as handle:
            arrays = {
                "internal_probabilities": internal.probabilities.astype(
                    np.float16),
                "external_top1_confidence": external.confidence.astype(
                    np.float16),
                "external_entropy": external.entropy.astype(np.float16),
                "final_confidence": result.confidence.astype(np.float16),
                "final_entropy": result.entropy.astype(np.float16),
            }
            if external.probabilities is not None:
                arrays["external_probabilities"] = \
                    external.probabilities.astype(np.float16)
            np.savez_compressed(handle, **arrays)
    atomic_write_json(out / "metadata" / f"{stem}.json", metadata)


def _peak_vram_mb() -> float | None:
    try:
        import torch
        if torch.cuda.is_available():
            return float(torch.cuda.max_memory_allocated() / 1e6)
    except ImportError:
        pass
    return None


def _output_complete(out: Path, stem: str) -> bool:
    return all((out / directory / f"{stem}{suffix}").exists() for directory, suffix in (
        ("final_masks", ".png"), ("overlays", ".jpg"),
        ("provenance", ".png"), ("metadata", ".json")))


def run_semantic_camera(
        input_dir: str | Path, external_dir: str | Path, cfg: Config,
        vehicle_type: str, output_dir: str | Path | None = None,
        resume: bool = True, force: bool = False,
        internal_provider: InternalProvider | None = None) -> int:
    """Fuse saved external evidence with a cockpit stream, frame by frame."""
    if vehicle_type not in {"car", "motorcycle"}:
        raise ValueError("vehicle_type must be car or motorcycle")
    root = Path(input_dir)
    semantic_cfg = cfg.get("semantic_camera", {})
    out = (
        Path(output_dir) if output_dir is not None
        else root / semantic_cfg.get("output_subdir", "semantic_camera")
    )
    for directory in OUTPUT_DIRS:
        (out / directory).mkdir(parents=True, exist_ok=True)
    taxonomy_path = cfg.resolve(semantic_cfg.get(
        "classes", "configs/article1/classes_article1.yaml"))
    taxonomy = Taxonomy.load(taxonomy_path)
    if tuple(taxonomy.names()) != FINAL_CLASS_NAMES:
        raise RuntimeError("semantic-camera taxonomy contract changed")
    external_reader = ExternalEvidenceReader(
        external_dir, use_probabilities=semantic_cfg.get(
            "external", {}).get("use_probabilities_if_available", True))
    provider = internal_provider or resolve_internal_provider(cfg)
    descriptor = provider.descriptor()
    fingerprint = stable_hash([
        semantic_cfg, taxonomy_path.read_text(),
        external_reader.descriptor(), descriptor, vehicle_type])
    manifest_path = out / "manifest.json"
    existing = (
        json.loads(manifest_path.read_text())
        if manifest_path.exists() else None)
    if existing and existing.get("fingerprint") != fingerprint:
        if resume and not force:
            raise RuntimeError(
                "semantic-camera resume fingerprint is incompatible")
        existing = None
    done = {} if force or not existing else existing.get("done", {})
    frames = iter_frames(root)
    max_frames = cfg.get("frames.max_frames")
    if max_frames is not None:
        frames = frames[:int(max_frames)]
    if not frames:
        raise RuntimeError("semantic-camera input contains no frames")
    rows: list[dict[str, Any]] = []
    for position, ref in enumerate(frames):
        stem = f"frame_{ref.frame_index:06d}"
        if resume and not force and str(ref.frame_index) in done and \
                _output_complete(out, stem):
            metadata_path = out / "metadata" / f"{stem}.json"
            rows.append(json.loads(metadata_path.read_text()))
            continue
        frame_start = time.perf_counter()
        bgr = cv2.imread(str(ref.rectified_path), cv2.IMREAD_COLOR)
        if bgr is None:
            raise FileNotFoundError(ref.rectified_path)
        image_rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        external_start = time.perf_counter()
        external = external_reader.read(ref.frame_index)
        external_ms = (time.perf_counter() - external_start) * 1000
        if external.mask.shape != image_rgb.shape[:2]:
            raise RuntimeError(
                f"external/RGB geometry mismatch at frame {ref.frame_index}")
        internal_start = time.perf_counter()
        internal = provider.infer(
            image_rgb, ref.frame_index, ref.capture_timestamp_ns,
            vehicle_type)
        internal_ms = (time.perf_counter() - internal_start) * 1000
        fusion_start = time.perf_counter()
        result = fuse_semantic_camera(
            external, internal, semantic_cfg.get("fusion", {}))
        fusion_ms = (time.perf_counter() - fusion_start) * 1000
        save_probability = position in {
            0, len(frames) // 2, len(frames) - 1}
        metadata = {
            "frame_index": ref.frame_index,
            "capture_timestamp_ns": ref.capture_timestamp_ns,
            "image_size": [image_rgb.shape[0], image_rgb.shape[1]],
            "external_source": external.source_kind,
            "external_entropy_kind": external.entropy_kind,
            "internal_source": internal.source_kind,
            "internal_checkpoint": internal.checkpoint,
            "internal_is_fallback": internal.fallback,
            "final_is_dense": True,
            "provenance_codes": {str(k): v for k, v in PROVENANCE.items()},
            "timings_ms": {
                "external_read": external_ms,
                "internal_inference": internal_ms,
                "fusion": fusion_ms,
            },
            **result.stats,
        }
        write_start = time.perf_counter()
        _write_frame(
            out, stem, image_rgb, external, internal, result, metadata,
            taxonomy, save_probability,
            float(semantic_cfg.get("render", {}).get("overlay_alpha", .45)))
        metadata["timings_ms"]["output_writing"] = (
            time.perf_counter() - write_start) * 1000
        metadata["timings_ms"]["total"] = (
            time.perf_counter() - frame_start) * 1000
        atomic_write_json(out / "metadata" / f"{stem}.json", metadata)
        rows.append(metadata)
        done[str(ref.frame_index)] = {
            "dense_coverage": result.stats["dense_coverage"],
            "internal_source": internal.source_kind,
            "total_ms": metadata["timings_ms"]["total"],
        }
        manifest = {
            "stage": "article1_semantic_camera_v1",
            "fingerprint": fingerprint,
            "causal": False,
            "temporal_fusion": False,
            "frame_independent": True,
            "vehicle_type": vehicle_type,
            "taxonomy": list(FINAL_CLASS_NAMES[1:]),
            "external": external_reader.descriptor(),
            "internal": descriptor,
            "internal_is_fallback": bool(provider.fallback),
            "available_outputs": list(OUTPUT_DIRS),
            "done": done,
        }
        atomic_write_json(manifest_path, manifest)
    _write_summary(out, rows, fingerprint, external_reader, provider)
    return 0


def _write_summary(
        out: Path, rows: list[dict[str, Any]], fingerprint: str,
        external_reader: ExternalEvidenceReader,
        provider: InternalProvider) -> None:
    keys = (
        "dense_coverage", "external_selected_fraction",
        "internal_model_selected_fraction",
        "geometric_proxy_selected_fraction", "dense_fill_fraction",
        "conflict_fraction", "internal_conflict_win_fraction",
        "unknown_external_fraction", "mean_final_confidence",
        "mean_final_entropy",
    )
    timing_keys = (
        "external_read", "internal_inference", "fusion",
        "output_writing", "total",
    )
    summary = {
        "stage": "article1_semantic_camera_v1",
        "fingerprint": fingerprint,
        "frame_count": len(rows),
        "external": external_reader.descriptor(),
        "internal": provider.descriptor(),
        "internal_is_fallback": bool(provider.fallback),
        "dense_class_ids": list(range(1, len(FINAL_CLASS_NAMES))),
        "mean_metrics": {
            key: float(np.mean([row[key] for row in rows]))
            for key in keys
        },
        "mean_timings_ms": {
            key: float(np.mean([
                row["timings_ms"].get(key, 0.0) for row in rows]))
            for key in timing_keys
        },
        "peak_ram_mb": resource.getrusage(
            resource.RUSAGE_SELF).ru_maxrss / 1024,
        "peak_vram_mb": _peak_vram_mb(),
    }
    size = sum(
        path.stat().st_size for path in out.rglob("*") if path.is_file())
    summary["storage_bytes"] = size
    summary["storage_bytes_per_frame"] = size / max(1, len(rows))
    atomic_write_json(out / "summary.json", summary)
