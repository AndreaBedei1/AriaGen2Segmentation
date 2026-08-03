"""Small reproducible ablation for fast semantic-gaze rate and batching."""
from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Dict, Iterable, Sequence

import numpy as np

from .fast_io import nearest_indices, timestamp_grid_indices


def frequency_ablation(source_timestamps_ns: Sequence[int],
                       gaze_timestamps_ns: Sequence[int],
                       frequencies: Iterable[float | str]) -> list[Dict[str, Any]]:
    """Temporal coverage of each candidate using only observed timestamps."""
    source = np.asarray(source_timestamps_ns, np.int64)
    gaze = np.asarray(gaze_timestamps_ns, np.int64)
    rows = []
    for frequency in frequencies:
        selected = timestamp_grid_indices(source, frequency)
        _, dt = nearest_indices(source[selected], gaze)
        abs_ms = np.abs(dt) / 1e6
        rows.append({
            "frequency_hz": frequency, "selected_frames": int(selected.size),
            "reduction_vs_native": float(1.0 - selected.size / source.size),
            "median_gaze_to_mask_dt_ms": float(np.median(abs_ms)),
            "p95_gaze_to_mask_dt_ms": float(np.percentile(abs_ms, 95)),
            "max_gaze_to_mask_dt_ms": float(abs_ms.max()),
            "synthetic_frames": 0,
        })
    return rows


def temporal_reuse_agreement(native_masks: np.ndarray,
                             native_timestamps_ns: Sequence[int],
                             frequency: float | str) -> float:
    """Agreement with native inference when the nearest sampled mask is reused."""
    ts = np.asarray(native_timestamps_ns, np.int64)
    selected = timestamp_grid_indices(ts, frequency)
    near, _ = nearest_indices(ts[selected], ts)
    approximated = native_masks[selected[near]]
    return float(np.mean(approximated == native_masks))


def run_benchmark(cfg, input_roots: Dict[str, str | Path],
                  behavior_root: str | Path,
                  ingestion_gaze_root: str | Path,
                  window_s: float = 4.0,
                  batch_sizes: Sequence[int] = (1, 4, 8, 16),
                  repeats: int = 3) -> Dict[str, Any]:
    """Benchmark batches and a native/5/2.5 Hz temporal reuse ablation."""
    import pandas as pd
    import psutil
    import torch
    from ..segmentation.oneformer import OneFormerMapillarySegmenter
    from ..taxonomy import Taxonomy
    from .fast_semantic_gaze import (FastMapillaryMapper,
                                     aggregate_query_probabilities,
                                     dense_outputs, _load_batch)

    fast = cfg.get("semantic_gaze_fast_external", {})
    taxonomy = Taxonomy.load(cfg.resolve(fast["classes"]))
    model = OneFormerMapillarySegmenter(cfg, taxonomy); model.load()
    mapper = FastMapillaryMapper(
        model._model.config.id2label, cfg.resolve(fast["mapillary_mapping"]), taxonomy)
    matrix = mapper.matrix()
    windows = {}
    all_paths = []
    all_ts = []
    for domain, raw_root in input_roots.items():
        root = Path(raw_root)
        frames = pd.read_parquet(root / "frames" / "frames.parquet").sort_values(
            "capture_timestamp_ns").reset_index(drop=True)
        ts = frames.capture_timestamp_ns.values.astype(np.int64)
        centre = int(ts[len(ts) // 2])
        keep = np.abs(ts - centre) <= window_s * 1e9 / 2
        sample = frames[keep]
        windows[domain] = {
            "root": root, "frames": sample,
            "slice_duration_s": ((int(sample.capture_timestamp_ns.iloc[-1]) -
                                  int(sample.capture_timestamp_ns.iloc[0])) / 1e9
                                 if len(sample) > 1 else 0.0)}
        all_paths.extend([root / str(p) for p in sample.rectified_path])
        all_ts.extend(sample.capture_timestamp_ns.astype(np.int64).tolist())
    if not all_paths:
        raise ValueError("benchmark inputs contain no frames")

    output_size = (int(fast.get("output_height", 756)),
                   int(fast.get("output_width", 1008)))
    batch_rows = []
    process = psutil.Process()
    probe_paths = [all_paths[i % len(all_paths)] for i in range(max(batch_sizes))]
    for batch_size in batch_sizes:
        paths = probe_paths[:batch_size]
        # Warm up this shape once.
        images = _load_batch(paths)
        inputs = model._proc(images=images, return_tensors="pt")
        inputs = {k: v.to(model.device) if hasattr(v, "to") else v
                  for k, v in inputs.items()}
        with torch.inference_mode(), torch.autocast(
                model.device, dtype=model._amp_dtype, enabled=model.device == "cuda"):
            warm = model._model(**inputs)
            probabilities = aggregate_query_probabilities(
                warm.class_queries_logits, warm.masks_queries_logits, matrix, output_size)
            dense_outputs(probabilities)
        torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
        stage = {"load_rgb": [], "preprocess": [], "forward": [],
                 "aggregate_transfer": []}
        finite_outputs = True
        peak_rss = process.memory_info().rss
        for _ in range(repeats):
            t0 = time.perf_counter(); images = _load_batch(paths)
            stage["load_rgb"].append(time.perf_counter() - t0)
            t0 = time.perf_counter()
            inputs = model._proc(images=images, return_tensors="pt")
            stage["preprocess"].append(time.perf_counter() - t0)
            inputs = {k: v.to(model.device) if hasattr(v, "to") else v
                      for k, v in inputs.items()}
            torch.cuda.synchronize(); t0 = time.perf_counter()
            with torch.inference_mode(), torch.autocast(
                    model.device, dtype=model._amp_dtype, enabled=model.device == "cuda"):
                outputs = model._model(**inputs)
            torch.cuda.synchronize(); stage["forward"].append(time.perf_counter() - t0)
            t0 = time.perf_counter()
            probabilities = aggregate_query_probabilities(
                outputs.class_queries_logits, outputs.masks_queries_logits,
                matrix, output_size)
            finite_outputs = finite_outputs and bool(torch.isfinite(probabilities).all())
            dense_outputs(probabilities); torch.cuda.synchronize()
            stage["aggregate_transfer"].append(time.perf_counter() - t0)
            peak_rss = max(peak_rss, process.memory_info().rss)
        total = sum(np.mean(v) for v in stage.values())
        peak_vram = torch.cuda.max_memory_allocated() / 1e6
        batch_rows.append({
            "batch_size": batch_size,
            "load_rgb_ms_per_frame": 1000 * np.mean(stage["load_rgb"]) / batch_size,
            "preprocess_ms_per_frame": 1000 * np.mean(stage["preprocess"]) / batch_size,
            "forward_ms_per_frame": 1000 * np.mean(stage["forward"]) / batch_size,
            "aggregate_transfer_ms_per_frame":
                1000 * np.mean(stage["aggregate_transfer"]) / batch_size,
            "total_profiled_ms_per_frame": 1000 * total / batch_size,
            "batch_latency_ms": 1000 * total,
            "throughput_profiled_frames_s": batch_size / total,
            "peak_vram_mb": peak_vram,
            "peak_vram_fraction": peak_vram / (
                torch.cuda.get_device_properties(0).total_memory / 1e6),
            "peak_ram_mb": peak_rss / 1e6,
            "numerically_finite": finite_outputs,
        })

    # One native inference for each short real-time window.  Lower-rate rows reuse
    # only masks from real selected frames, so agreement measures temporal aliasing.
    chosen_batch = int(max(batch_rows, key=lambda row: row["throughput_profiled_frames_s"])
                       ["batch_size"])
    quality_rows = []
    cursor = 0
    for domain, spec in windows.items():
        frame_count = len(spec["frames"])
        paths = all_paths[cursor:cursor + frame_count]; cursor += frame_count
        native_masks = []
        for offset in range(0, frame_count, chosen_batch):
            images = _load_batch(paths[offset:offset + chosen_batch])
            inputs = model._proc(images=images, return_tensors="pt")
            inputs = {k: v.to(model.device) if hasattr(v, "to") else v
                      for k, v in inputs.items()}
            with torch.inference_mode(), torch.autocast(
                    model.device, dtype=model._amp_dtype, enabled=model.device == "cuda"):
                outputs = model._model(**inputs)
                probabilities = aggregate_query_probabilities(
                    outputs.class_queries_logits, outputs.masks_queries_logits,
                    matrix, output_size)
            masks, _, _ = dense_outputs(probabilities); native_masks.extend(masks)
        native_masks = np.asarray(native_masks)
        ts = spec["frames"].capture_timestamp_ns.values.astype(np.int64)
        for frequency in fast.get("ablation_frequencies_hz", ["native", 5.0, 2.5]):
            quality_rows.append({
                "domain": domain, "frequency_hz": frequency,
                "window_frames": frame_count,
                "window_duration_s": spec["slice_duration_s"],
                "pixel_agreement_with_native_inference":
                    temporal_reuse_agreement(native_masks, ts, frequency),
            })

    behavior_root = Path(behavior_root); gaze_root = Path(ingestion_gaze_root)
    timing_rows = []
    recording_ids = {"car": "car_2e84f0c3e245",
                     "motorcycle": "motorcycle_5ab8604a14df"}
    for domain, rec_id in recording_ids.items():
        timeline = pd.read_parquet(behavior_root / rec_id / "multimodal_timeline.parquet")
        gaze = pd.read_parquet(gaze_root / f"{rec_id}.parquet")
        for row in frequency_ablation(
                timeline.timestamp_ns.values, gaze.timestamp_ns.values,
                fast.get("ablation_frequencies_hz", ["native", 5.0, 2.5])):
            timing_rows.append({"domain": domain, **row})
    return {"batch_profiles": batch_rows, "temporal_quality": quality_rows,
            "frequency_coverage": timing_rows, "chosen_batch_size": chosen_batch,
            "sampled_frames_per_domain": {
                domain: int(len(spec["frames"])) for domain, spec in windows.items()},
            "benchmark_requirement_min_frames_per_domain": 300,
            "benchmark_requirement_met": all(
                len(spec["frames"]) >= 300 for spec in windows.values()),
            "window_s": window_s, "repeats": repeats,
            "model": str(model.model_id), "synthetic_frames": 0}
