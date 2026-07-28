"""Streaming runner for causal Article 1 temporal stabilization."""
from __future__ import annotations

import json
import time
from pathlib import Path

import cv2
import numpy as np

from ..config import Config
from ..hashing import stable_hash
from ..io_utils import atomic_write, atomic_write_json, write_mask_u16
from ..segmentation.base import iter_frames
from ..segmentation.oneformer import OneFormerMapillarySegmenter
from ..taxonomy import Taxonomy
from .external import (
    Article1Mapper, apply_article1_policy, infer_native_probabilities)
from .optical_flow import compute_optical_flow
from .temporal import (
    PROVENANCE, RESET_REASON, SWITCH_REASON, TemporalResult, stabilize_frame)
from .temporal_state import TemporalState


OUTPUT_DIRS = (
    "static_masks", "temporal_masks", "static_thin", "temporal_thin",
    "static_confidence",
    "temporal_confidence", "temporal_entropy", "provenance",
    "propagation_age", "class_age", "flow_validity", "occlusion",
    "switch_reason", "reset_reason", "unknown_raw", "unknown_temporal",
    "unknown_recovered", "unknown_recovery_source", "unknown_recovery_age",
    "metadata", "temporal_state", "diagnostic_probabilities", "diagnostic_flow",
    "videos",
)
MODES = ("T0", "T1", "T2", "T3", "T4")


def _resize_channels(probabilities: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    resized = np.moveaxis(cv2.resize(
        np.moveaxis(probabilities, 0, -1), size,
        interpolation=cv2.INTER_AREA), -1, 0)
    resized = np.asarray(resized, np.float32)
    return resized / np.maximum(resized.sum(0, keepdims=True), 1e-8)


def _write_npz(path: Path, **arrays) -> None:
    with atomic_write(path, "wb") as handle:
        np.savez_compressed(handle, **arrays)


def _up(array: np.ndarray, size: tuple[int, int], nearest=True) -> np.ndarray:
    return cv2.resize(
        array, size, interpolation=cv2.INTER_NEAREST if nearest else cv2.INTER_LINEAR)


def _write_frame(out: Path, stem: str, static: dict, temporal: TemporalResult,
                 full_size: tuple[int, int], metadata: dict,
                 save_probability: bool, flow=None) -> None:
    static_mask = static["aggregate"]["mask"]
    static_thin = static["thin"]["filtered_thin_mask"]
    temporal_mask = _up(temporal.mask, full_size).astype(np.uint16)
    temporal_thin = _up(temporal.thin_mask, full_size).astype(np.uint16)
    write_mask_u16(out / "static_masks" / f"{stem}.png", static_mask)
    write_mask_u16(out / "temporal_masks" / f"{stem}.png", temporal_mask)
    write_mask_u16(out / "static_thin" / f"{stem}.png", static_thin)
    write_mask_u16(out / "temporal_thin" / f"{stem}.png", temporal_thin)
    uint8_maps = {
        "static_confidence": np.clip(
            static["aggregate"]["probabilities"].max(0) * 255, 0, 255),
        "temporal_confidence": np.clip(_up(
            temporal.confidence, full_size, False) * 255, 0, 255),
        "temporal_entropy": np.clip(_up(
            -(temporal.probabilities * np.log(np.maximum(
                temporal.probabilities, 1e-8))).sum(0) /
            np.log(temporal.probabilities.shape[0]), full_size, False) * 255, 0, 255),
        "provenance": _up(temporal.provenance, full_size),
        "propagation_age": _up(temporal.propagation_age, full_size),
        "class_age": np.minimum(_up(temporal.class_age, full_size), 255),
        "flow_validity": _up(temporal.flow_validity.astype(np.uint8), full_size) * 255,
        "occlusion": _up(temporal.occlusion.astype(np.uint8), full_size) * 255,
        "switch_reason": _up(temporal.switch_reason, full_size),
        "reset_reason": np.full((full_size[1], full_size[0]),
                                temporal.reset_reason, np.uint8),
        "unknown_raw": (static_mask == 0).astype(np.uint8) * 255,
        "unknown_temporal": (temporal_mask == 0).astype(np.uint8) * 255,
        "unknown_recovered": _up(
            temporal.unknown_recovered.astype(np.uint8), full_size) * 255,
        "unknown_recovery_source": _up(
            (temporal.unknown_recovered.astype(np.uint8) * 2), full_size),
        "unknown_recovery_age": _up(
            np.where(temporal.unknown_recovered, temporal.propagation_age, 0),
            full_size),
    }
    for directory, array in uint8_maps.items():
        cv2.imwrite(str(out / directory / f"{stem}.png"),
                    np.asarray(array, np.uint8))
    if save_probability:
        _write_npz(
            out / "diagnostic_probabilities" / f"{stem}.npz",
            temporal_probabilities=temporal.probabilities.astype(np.float16),
            static_probabilities=static["aggregate"]["probabilities"].astype(np.float16))
        if flow is not None:
            _write_npz(
                out / "diagnostic_flow" / f"{stem}.npz",
                forward=flow.forward.astype(np.float16),
                backward=flow.backward.astype(np.float16),
                valid=flow.valid)
    atomic_write_json(out / "metadata" / f"{stem}.json", metadata)


def _frame_metrics(mode: str, static_mask: np.ndarray, result: TemporalResult,
                   prior_static: np.ndarray | None,
                   prior_temporal: np.ndarray | None, elapsed_ms: float) -> dict:
    temporal_mask = result.mask
    static_small = cv2.resize(
        static_mask, (temporal_mask.shape[1], temporal_mask.shape[0]),
        interpolation=cv2.INTER_NEAREST)
    propagated = np.isin(result.provenance, [2, 3, 5, 6])
    record = {
        "mode": mode,
        "raw_unknown_rate": float((static_small == 0).mean()),
        "temporal_unknown_rate": float((temporal_mask == 0).mean()),
        "unknown_recovery_rate": float(result.unknown_recovered.mean()),
        "propagated_pixels": int(propagated.sum()),
        "mean_propagation_age": (
            float(result.propagation_age[propagated].mean()) if propagated.any() else 0.0),
        "pixel_difference_raw_temporal": float(
            (static_small != temporal_mask).mean()),
        "flow_valid_fraction": float(result.flow_validity.mean()),
        "reset": int(result.reset_reason != 0),
        "thin_pixels": int((result.thin_mask > 0).sum()),
        "elapsed_ms": float(elapsed_ms),
        "raw_switch_rate": (
            float((static_small != prior_static).mean())
            if prior_static is not None else 0.0),
        "temporal_switch_rate": (
            float((temporal_mask != prior_temporal).mean())
            if prior_temporal is not None else 0.0),
        "current_evidence_free_propagation": int(
            (propagated & (static_small != temporal_mask)).sum()),
    }
    return record


def _latest_resume_state(out: Path, modes: tuple[str, ...], temporal_fp: str,
                         static_fp: str, frame_indices: list[int],
                         manifest: dict) -> tuple[dict[str, TemporalState], int]:
    done = [int(x) for x in manifest.get("processed_frame_indices", [])]
    if done and done != frame_indices[:len(done)]:
        raise RuntimeError("temporal resume failed: missing or non-contiguous frame")
    common = None
    for mode in modes:
        files = {int(path.stem.split("_")[-1]): path for path in
                 (out / "temporal_state" / mode).glob("frame_*.npz")}
        common = set(files) if common is None else common & set(files)
    if not common:
        return {}, -1
    last_frame = max(common)
    if last_frame not in frame_indices:
        raise RuntimeError("temporal resume state frame is absent from input frames")
    states = {
        mode: TemporalState.load(
            out / "temporal_state" / mode / f"frame_{last_frame:06d}.npz",
            temporal_fp, static_fp)
        for mode in modes
    }
    return states, frame_indices.index(last_frame)


def run_temporal(input_dir: str, cfg: Config, resume=True, force=False,
                 static_output: str | None = None) -> int:
    """Run direct static+temporal streaming, or replay saved static probabilities."""
    root = Path(input_dir)
    article_cfg = cfg.get("article1", {})
    temporal_cfg = cfg.get("temporal", {})
    out = root / temporal_cfg.get("output_subdir", "article1_temporal")
    for name in OUTPUT_DIRS:
        (out / name).mkdir(parents=True, exist_ok=True)
    tax = Taxonomy.load(cfg.resolve(article_cfg["classes"]))
    class_names = tax.names()
    temporal_fp = stable_hash(temporal_cfg)
    static_fp = stable_hash(article_cfg)
    frames = list(iter_frames(root))
    if not frames:
        raise RuntimeError("temporal pipeline requires extracted consecutive frames")
    frame_indices = [ref.frame_index for ref in frames]
    manifest_path = out / "manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    if manifest and (
            manifest.get("config_fingerprint") != temporal_fp or
            manifest.get("static_policy_fingerprint") != static_fp):
        if resume and not force:
            raise RuntimeError("temporal cache fingerprint is incompatible")
        manifest = {}
    modes = tuple(temporal_cfg.get("experiment_modes", MODES))
    states, resume_position = ({}, -1)
    if resume and not force and manifest:
        states, resume_position = _latest_resume_state(
            out, modes, temporal_fp, static_fp, frame_indices, manifest)
    if force:
        states, resume_position, manifest = {}, -1, {}

    mapper = None
    segmenter = None
    source = Path(static_output) if static_output else None
    if source is None:
        segmenter = OneFormerMapillarySegmenter(cfg, tax)
        segmenter.load()
        mapper = Article1Mapper(
            segmenter._model.config.id2label,
            cfg.resolve(article_cfg["mapillary_mapping"]), tax)
    else:
        model_cfg = cfg.resolve(
            cfg.get("oneformer_mapillary.mask2former_id")) / "config.json"
        id2label = json.loads(model_cfg.read_text())["id2label"]
        mapper = Article1Mapper(
            id2label, cfg.resolve(article_cfg["mapillary_mapping"]), tax)

    scale = float(temporal_cfg.get("processing_scale", .5))
    checkpoint_interval = int(
        temporal_cfg.get("state_checkpoint_interval", 50))
    records = manifest.get("records", {mode: [] for mode in modes})
    prior_static = {mode: None for mode in modes}
    prior_temporal = {mode: None for mode in modes}
    processed = frame_indices[:resume_position + 1]
    for position, ref in enumerate(frames):
        if position <= resume_position:
            continue
        frame_start = time.perf_counter()
        bgr = cv2.imread(str(ref.rectified_path))
        if bgr is None:
            raise FileNotFoundError(ref.rectified_path)
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        h, w = rgb.shape[:2]
        if source is None:
            static_start = time.perf_counter()
            native = infer_native_probabilities(segmenter, rgb)
            static = apply_article1_policy(native, mapper, article_cfg)
            static_ms = (time.perf_counter() - static_start) * 1000
        else:
            stem = f"frame_{ref.frame_index:06d}"
            probability_path = source / "probabilities" / f"{stem}.npz"
            if not probability_path.exists():
                raise RuntimeError(
                    f"stabilize-temporal missing static probability map: {probability_path}")
            probabilities = np.load(probability_path)["probabilities"].astype(np.float32)
            base_mask = cv2.imread(
                str(source / "base_masks" / f"{stem}.png"),
                cv2.IMREAD_UNCHANGED)
            thin_mask = cv2.imread(
                str(source / "filtered_thin_masks" / f"{stem}.png"),
                cv2.IMREAD_UNCHANGED)
            final_mask = cv2.imread(
                str(source / "masks" / f"{stem}.png"), cv2.IMREAD_UNCHANGED)
            if any(x is None for x in (base_mask, thin_mask, final_mask)):
                raise RuntimeError(f"stabilize-temporal static masks missing for {stem}")
            static = {
                "aggregate": {"probabilities": probabilities,
                              "mask": base_mask.astype(np.uint16)},
                "thin": {
                    "filtered_thin_mask": thin_mask.astype(np.uint16),
                    "composite": final_mask.astype(np.uint16),
                    "road_support_final": base_mask == 1,
                },
            }
            static_ms = 0.0
        grid_size = (max(2, round(w * scale)), max(2, round(h * scale)))
        rgb_small = cv2.resize(rgb, grid_size, interpolation=cv2.INTER_AREA)
        p_small = _resize_channels(
            static["aggregate"]["probabilities"], grid_size)
        thin_small = cv2.resize(
            static["thin"]["filtered_thin_mask"], grid_size,
            interpolation=cv2.INTER_NEAREST).astype(np.uint16)
        road_small = cv2.resize(
            static["thin"]["road_support_final"].astype(np.uint8), grid_size,
            interpolation=cv2.INTER_NEAREST).astype(bool)
        flow = None
        flow_ms = 0.0
        prior = states.get("T4") or next(iter(states.values()), None)
        if prior is not None and prior.previous_rgb.shape == rgb_small.shape:
            flow_start = time.perf_counter()
            flow = compute_optical_flow(
                prior.previous_rgb, rgb_small, temporal_cfg.get("optical_flow", {}))
            flow_ms = (time.perf_counter() - flow_start) * 1000
        mode_results = {}
        for mode in modes:
            start = time.perf_counter()
            mode_results[mode] = stabilize_frame(
                p_small, thin_small, road_small, rgb_small, ref.frame_index,
                ref.capture_timestamp_ns, class_names, temporal_cfg,
                temporal_fp, static_fp, states.get(mode), flow, mode=mode)
            elapsed = (time.perf_counter() - start) * 1000
            records.setdefault(mode, []).append(_frame_metrics(
                mode, static["thin"]["composite"], mode_results[mode],
                prior_static[mode], prior_temporal[mode], elapsed))
            prior_static[mode] = cv2.resize(
                static["thin"]["composite"], grid_size,
                interpolation=cv2.INTER_NEAREST)
            prior_temporal[mode] = mode_results[mode].mask.copy()
            states[mode] = mode_results[mode].state
        primary = mode_results[temporal_cfg.get("mode", "T4")]
        stem = f"frame_{ref.frame_index:06d}"
        difference = records[temporal_cfg.get("mode", "T4")][-1][
            "pixel_difference_raw_temporal"]
        save_probability = (
            position == 0 or position == len(frames) // 2 or
            position == len(frames) - 1 or primary.reset_reason != 0 or
            difference >= float(temporal_cfg.get(
                "large_difference_threshold", .15)))
        metadata = {
            "frame_index": ref.frame_index,
            "capture_timestamp_ns": ref.capture_timestamp_ns,
            "static_ms": static_ms, "optical_flow_ms": flow_ms,
            "total_ms": (time.perf_counter() - frame_start) * 1000,
            "flow_valid_fraction": float(primary.flow_validity.mean()),
            "reset_reason": RESET_REASON[primary.reset_reason],
            "reset_reason_code": primary.reset_reason,
            "propagated_pixels": int(np.isin(
                primary.provenance, [2, 3, 5, 6]).sum()),
            "mean_propagation_age": float(primary.propagation_age.mean()),
            "raw_temporal_difference": difference,
            "causal": True, "gaze_assisted": False,
            "provenance_codes": {str(k): v for k, v in PROVENANCE.items()},
            "switch_reason_codes": {str(k): v for k, v in SWITCH_REASON.items()},
        }
        _write_frame(
            out, stem, static, primary, (w, h), metadata,
            save_probability, flow)
        processed.append(ref.frame_index)
        if (position + 1) % checkpoint_interval == 0 or position == len(frames) - 1:
            for mode, state in states.items():
                state.save(
                    out / "temporal_state" / mode / f"{stem}.npz")
        manifest = {
            "stage": "article1_temporal_v1",
            "config_fingerprint": temporal_fp,
            "static_policy_fingerprint": static_fp,
            "flow_backend": temporal_cfg.get("optical_flow", {}).get(
                "backend", "opencv_dis"),
            "processing_scale": scale, "causal": True, "gaze_assisted": False,
            "processed_frame_indices": processed,
            "frame_count": len(processed),
            "timestamp_range_ns": [
                frames[0].capture_timestamp_ns, ref.capture_timestamp_ns],
            "reset_count": primary.state.reset_count,
            "available_outputs": list(OUTPUT_DIRS),
            "records": records,
        }
        atomic_write_json(manifest_path, manifest)
    _write_summary(out, manifest, tax)
    return 0


def _write_summary(out: Path, manifest: dict, taxonomy: Taxonomy) -> None:
    summary = {
        "disclaimer": (
            "Pre-GT temporal diagnostics: lower flicker/unknown does not imply "
            "higher accuracy; persistence can propagate errors."),
        "frame_count": manifest.get("frame_count", 0),
        "reset_count": manifest.get("reset_count", 0),
        "modes": {},
    }
    for mode, rows in manifest.get("records", {}).items():
        if not rows:
            continue
        numeric = [key for key, value in rows[0].items()
                   if isinstance(value, (int, float)) and key != "mode"]
        summary["modes"][mode] = {
            f"mean_{key}": float(np.mean([row[key] for row in rows]))
            for key in numeric}
    size = sum(path.stat().st_size for path in out.rglob("*") if path.is_file())
    summary["storage_bytes"] = size
    summary["storage_bytes_per_frame"] = (
        size / max(1, summary["frame_count"]))
    atomic_write_json(out / "summary.json", summary)
