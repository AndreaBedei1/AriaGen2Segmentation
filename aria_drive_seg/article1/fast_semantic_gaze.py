"""Batched Mask2Former/Mapillary full-run semantic gaze.

The fast path aggregates native class-query probabilities into the compact
taxonomy at the model mask resolution *before* one upsample.  It writes only the
dense id mask, confidence and normalized entropy.  Gaze is a read-only consumer
of those masks and never conditions segmentation.
"""
from __future__ import annotations

import json
import math
import time
from collections import OrderedDict
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

import numpy as np
import yaml

from ..behavior.gaze_semantics import (foveal_weights, identify_fixations,
                                       pixels_per_degree,
                                       sample_foveal_semantics,
                                       semantic_gaze_metrics)
from ..config import Config
from ..io_utils import atomic_write_bytes, atomic_write_json, config_fingerprint
from ..logging_utils import get_logger
from ..segmentation.oneformer import OneFormerMapillarySegmenter, _norm
from ..taxonomy import Taxonomy

log = get_logger("article1.fast_semantic_gaze")


class FastMapillaryMapper:
    """Validated native-to-fast mapping and its one-hot aggregation matrix."""

    def __init__(self, id2label: Mapping[int, str], mapping_path: str | Path,
                 taxonomy: Taxonomy):
        doc = yaml.safe_load(Path(mapping_path).read_text())
        raw = {_norm(k): v for k, v in doc["mapping"].items()}
        self.id2label = {int(k): str(v) for k, v in id2label.items()}
        self.taxonomy = taxonomy
        self.native_to_fast = np.zeros(max(self.id2label) + 1, dtype=np.int64)
        self.residual_native_labels: List[str] = []
        self.unmapped_native_labels: List[str] = []
        self.native_ego_ids: List[int] = []
        for native_id, label in self.id2label.items():
            entry = raw.get(_norm(label))
            if entry is None:
                self.unmapped_native_labels.append(label)
                continue
            name = str(entry["fast_class"])
            if name not in taxonomy.by_name:
                raise ValueError(f"mapped class {name!r} is absent from fast taxonomy")
            self.native_to_fast[native_id] = taxonomy.id_of(name)
            if entry.get("residual"):
                self.residual_native_labels.append(label)
            if entry.get("native_ego_region"):
                self.native_ego_ids.append(native_id)
        if self.unmapped_native_labels:
            raise ValueError(
                "all Mapillary labels must be mapped: " +
                ", ".join(self.unmapped_native_labels))
        if np.any(self.native_to_fast == 0):
            missing = [self.id2label[i] for i in np.flatnonzero(self.native_to_fast == 0)]
            raise ValueError("native labels may not map to unknown: " + ", ".join(missing))

    def matrix(self) -> np.ndarray:
        out = np.zeros((len(self.native_to_fast), self.taxonomy.max_id + 1),
                       dtype=np.float32)
        out[np.arange(len(self.native_to_fast)), self.native_to_fast] = 1.0
        return out


def aggregate_query_probabilities(class_queries, mask_queries, mapping_matrix,
                                  output_size: tuple[int, int]):
    """Return dense fast probabilities while avoiding a 65-channel upsample."""
    import torch
    import torch.nn.functional as F
    class_prob = class_queries.float().softmax(dim=-1)[..., :-1]
    matrix = torch.as_tensor(mapping_matrix, device=class_prob.device,
                             dtype=class_prob.dtype)
    if class_prob.shape[-1] != matrix.shape[0]:
        raise ValueError("model class channels and mapping matrix disagree")
    macro_queries = torch.einsum("bqc,ck->bqk", class_prob, matrix)
    mask_prob = mask_queries.float().sigmoid()
    low = torch.einsum("bqk,bqhw->bkhw", macro_queries, mask_prob)
    low = low / low.sum(dim=1, keepdim=True).clamp_min(1e-12)
    dense = F.interpolate(low, size=output_size, mode="bilinear", align_corners=False)
    return dense / dense.sum(dim=1, keepdim=True).clamp_min(1e-12)


def dense_outputs(probabilities):
    """Compact uint arrays used by storage, gaze and rendering."""
    import torch
    confidence, mask = probabilities.max(dim=1)
    entropy = -(probabilities * probabilities.clamp_min(1e-12).log()).sum(dim=1)
    entropy = entropy / math.log(max(2, probabilities.shape[1]))
    return (mask.to(torch.int16).cpu().numpy().astype(np.uint16),
            (confidence.clamp(0, 1) * 255).to(torch.uint8).cpu().numpy(),
            (entropy.clamp(0, 1) * 255).to(torch.uint8).cpu().numpy())


def apply_interior_bottom_contact(mask: np.ndarray, taxonomy: Taxonomy,
                                  policy: Mapping[str, Any]) -> int:
    """Reclassify only lower ego-like components that touch the image border.

    This deliberately cannot discover an interior object in the middle of the
    image.  It only resolves the common Mapillary failure where the rider's own
    fairing/tank is labelled as an external vehicle at the lower crop boundary.
    """
    import cv2
    if not policy.get("enabled", False):
        return 0
    eligible_ids = [taxonomy.id_of(name) for name in policy.get(
        "eligible_classes", ["vehicle", "two_wheeler"])]
    interior_id = taxonomy.id_of("interior_cockpit")
    h = mask.shape[0]
    start = int(round(h * float(policy.get("lower_start_fraction", .72))))
    candidate = np.isin(mask, eligible_ids)
    candidate[:start] = False
    count, labels = cv2.connectedComponents(candidate.astype(np.uint8), 8)
    if count <= 1:
        return 0
    contact_rows = max(1, int(policy.get("contact_rows", 4)))
    touching = set(int(v) for v in np.unique(labels[-contact_rows:]) if v > 0)
    changed = np.isin(labels, list(touching)) if touching else np.zeros_like(candidate)
    pixels = int(changed.sum())
    mask[changed] = interior_id
    return pixels


def apply_interior_upper_contact(mask: np.ndarray, taxonomy: Taxonomy,
                                 policy: Mapping[str, Any], domain: str) -> int:
    """Conservatively recover a car headliner from upper-border components.

    A component must both touch the top border and have most of its pixels in
    the upper qualification band.  Qualifying components are then reclassified
    in full, avoiding an artificial horizontal boundary through a pillar.  No
    RGB content or gaze signal is used, and the rule is disabled for motorcycles.
    """
    import cv2
    if domain not in policy.get("enabled_domains", []):
        return 0
    eligible_ids = [taxonomy.id_of(name) for name in policy.get(
        "eligible_classes", ["built_environment"])]
    h = mask.shape[0]
    end = min(h, max(1, int(round(
        h * float(policy.get("upper_end_fraction", .42))))))
    candidate = np.isin(mask, eligible_ids)
    count, labels = cv2.connectedComponents(candidate.astype(np.uint8), 8)
    if count <= 1:
        return 0
    contact_rows = max(1, int(policy.get("contact_rows", 4)))
    touching = set(int(v) for v in np.unique(labels[:contact_rows]) if v > 0)
    minimum_upper = float(policy.get("minimum_upper_fraction", .55))
    qualified = []
    for label in touching:
        total = int(np.count_nonzero(labels == label))
        upper = int(np.count_nonzero(labels[:end] == label))
        if total and upper / total >= minimum_upper:
            qualified.append(label)
    changed = (np.isin(labels, qualified) if qualified
               else np.zeros_like(candidate))
    pixels = int(changed.sum())
    mask[changed] = taxonomy.id_of("interior_cockpit")
    return pixels


def _png_bytes(array: np.ndarray) -> bytes:
    import cv2
    ok, buf = cv2.imencode(".png", array)
    if not ok:
        raise RuntimeError("PNG encode failed")
    return buf.tobytes()


def _write_layers(root: Path, stem: str, mask: np.ndarray,
                  confidence: np.ndarray, entropy: np.ndarray) -> float:
    t0 = time.perf_counter()
    atomic_write_bytes(root / "masks" / f"{stem}.png", _png_bytes(mask))
    atomic_write_bytes(root / "confidence" / f"{stem}.png", _png_bytes(confidence))
    atomic_write_bytes(root / "normalized_entropy" / f"{stem}.png",
                       _png_bytes(entropy))
    return (time.perf_counter() - t0) * 1000.0


def _atomic_parquet(path: Path, frame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp.parquet")
    frame.to_parquet(tmp, index=False)
    tmp.replace(path)


def _drain(pending: set[Future], block: bool, write_ms: List[float]) -> set[Future]:
    if not pending:
        return pending
    if block:
        done, left = wait(pending)
    else:
        done, left = wait(pending, return_when=FIRST_COMPLETED)
    for future in done:
        write_ms.append(float(future.result()))
    return set(left)


def _load_batch(paths: Sequence[Path]):
    from PIL import Image
    images = []
    for path in paths:
        with Image.open(path) as image:
            images.append(image.convert("RGB"))
    return images


def run_fast_external(input_dir: str | Path, cfg: Config,
                      resume: bool = True) -> Dict[str, Any]:
    """Run the compact batched segmentation stage over an extracted recording."""
    import pandas as pd
    import psutil
    import torch

    root = Path(input_dir)
    fast = cfg.get("semantic_gaze_fast_external", {})
    tax_path = cfg.resolve(fast["classes"])
    map_path = cfg.resolve(fast["mapillary_mapping"])
    taxonomy = Taxonomy.load(tax_path)
    frames = pd.read_parquet(root / "frames" / "frames.parquet").sort_values(
        "capture_timestamp_ns").reset_index(drop=True)
    if frames.empty:
        raise ValueError("fast external input has no frames")
    extraction = json.loads((root / "frames" / "extraction_summary.json").read_text())
    domain = str(extraction["domain"])
    seg_root = root / "segmentation"
    for name in ("masks", "confidence", "normalized_entropy"):
        (seg_root / name).mkdir(parents=True, exist_ok=True)

    segmenter = OneFormerMapillarySegmenter(cfg, taxonomy)
    segmenter.load()
    mapper = FastMapillaryMapper(segmenter._model.config.id2label, map_path, taxonomy)
    matrix = mapper.matrix()
    fp = config_fingerprint(
        "semantic_gaze_fast_external", cfg.get("oneformer_mapillary"), fast,
        tax_path.read_text(), map_path.read_text())
    index_path = seg_root / "segmentation_index.parquet"
    previous: Dict[int, Dict[str, Any]] = {}
    manifest_path = seg_root / "manifest.json"
    if resume and manifest_path.exists() and index_path.exists():
        manifest = json.loads(manifest_path.read_text())
        if manifest.get("fingerprint") == fp:
            old = pd.read_parquet(index_path)
            previous = {int(row["frame_index"]): row.to_dict()
                        for _, row in old.iterrows()}

    batch_size = max(1, int(cfg.get("hardware.batch_size",
                                    fast.get("batch_size", 8))))
    workers = max(1, int(fast.get("cpu_write_workers", 8)))
    max_pending = max(workers, int(fast.get("max_pending_writes", 32)))
    rows: List[Dict[str, Any]] = []
    pending_frames = []
    for _, row in frames.iterrows():
        fi = int(row.frame_index); stem = f"frame_{fi:06d}"
        required = [seg_root / d / f"{stem}.png"
                    for d in ("masks", "confidence", "normalized_entropy")]
        if resume and fi in previous and all(p.exists() for p in required):
            rows.append(previous[fi]); continue
        pending_frames.append(row)

    preprocess_ms: List[float] = []
    forward_ms: List[float] = []
    aggregate_ms: List[float] = []
    transfer_ms: List[float] = []
    load_ms: List[float] = []
    write_ms: List[float] = []
    futures: set[Future] = set()
    start = time.perf_counter()
    torch.cuda.reset_peak_memory_stats()
    process = psutil.Process()
    peak_rss = process.memory_info().rss
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for offset in range(0, len(pending_frames), batch_size):
            batch = pending_frames[offset:offset + batch_size]
            paths = [root / str(row.rectified_path) for row in batch]
            t0 = time.perf_counter(); images = _load_batch(paths)
            load_ms.append((time.perf_counter() - t0) * 1000.0)
            t0 = time.perf_counter()
            inputs = segmenter._proc(images=images, return_tensors="pt")
            preprocess_ms.append((time.perf_counter() - t0) * 1000.0)
            inputs = {k: (v.to(segmenter.device, non_blocking=True)
                          if hasattr(v, "to") else v) for k, v in inputs.items()}
            torch.cuda.synchronize(); t0 = time.perf_counter()
            with torch.inference_mode(), torch.autocast(
                    segmenter.device, dtype=segmenter._amp_dtype,
                    enabled=segmenter.device == "cuda"):
                outputs = segmenter._model(**inputs)
            torch.cuda.synchronize(); forward_ms.append(
                (time.perf_counter() - t0) * 1000.0)
            output_size = (int(images[0].height), int(images[0].width))
            t0 = time.perf_counter()
            probabilities = aggregate_query_probabilities(
                outputs.class_queries_logits, outputs.masks_queries_logits,
                matrix, output_size)
            torch.cuda.synchronize(); aggregate_ms.append(
                (time.perf_counter() - t0) * 1000.0)
            t0 = time.perf_counter()
            masks, confidences, entropies = dense_outputs(probabilities)
            transfer_ms.append((time.perf_counter() - t0) * 1000.0)
            del outputs, probabilities, inputs
            for j, row in enumerate(batch):
                fi = int(row.frame_index); stem = f"frame_{fi:06d}"
                interior_contact_pixels = apply_interior_bottom_contact(
                    masks[j], taxonomy,
                    fast.get("interior_cockpit", {}).get(
                        "bottom_contact_proxy", {}))
                interior_upper_pixels = apply_interior_upper_contact(
                    masks[j], taxonomy,
                    fast.get("interior_cockpit", {}).get(
                        "upper_border_proxy", {}), domain)
                hist = np.bincount(masks[j].ravel(),
                                   minlength=taxonomy.max_id + 1).astype(float)
                hist /= max(1.0, hist.sum())
                record = {
                    "frame_index": fi,
                    "capture_timestamp_ns": int(row.capture_timestamp_ns),
                    "mask_path": f"segmentation/masks/{stem}.png",
                    "confidence_path": f"segmentation/confidence/{stem}.png",
                    "normalized_entropy_path":
                        f"segmentation/normalized_entropy/{stem}.png",
                    "mean_confidence": float(confidences[j].mean() / 255.0),
                    "mean_normalized_entropy": float(entropies[j].mean() / 255.0),
                    "interior_bottom_contact_pixels": interior_contact_pixels,
                    "interior_upper_contact_pixels": interior_upper_pixels,
                }
                for klass in taxonomy.classes:
                    record[f"fraction_{klass.name}"] = float(hist[klass.id])
                rows.append(record)
                futures.add(pool.submit(
                    _write_layers, seg_root, stem, masks[j], confidences[j],
                    entropies[j]))
                if len(futures) >= max_pending:
                    futures = _drain(futures, False, write_ms)
            peak_rss = max(peak_rss, process.memory_info().rss)
            done = min(offset + len(batch), len(pending_frames))
            if done % max(batch_size, 200) < batch_size:
                log.info("fast external: %d/%d new frames", done, len(pending_frames))
        futures = _drain(futures, True, write_ms)

    result = pd.DataFrame(rows).sort_values("capture_timestamp_ns").reset_index(drop=True)
    _atomic_parquet(index_path, result)
    elapsed = time.perf_counter() - start
    processed = len(pending_frames)
    def per_frame(values: Sequence[float]) -> Optional[float]:
        return (float(np.sum(values) / processed) if processed else None)
    class_means = {klass.name: float(result[f"fraction_{klass.name}"].mean())
                   for klass in taxonomy.classes}
    summary = {
        "schema": "article1_fast_external_segmentation_v1",
        "result_status": cfg.get("result_status", "exploratory_pilot"),
        "frames": int(len(result)), "new_frames": int(processed),
        "elapsed_s": elapsed, "throughput_frames_s": processed / elapsed if elapsed else None,
        "batch_size": batch_size, "write_workers": workers,
        "peak_vram_mb": float(torch.cuda.max_memory_allocated() / 1e6),
        "peak_ram_mb": float(peak_rss / 1e6),
        "mean_ms_per_frame": {
            "load_rgb": per_frame(load_ms), "preprocess": per_frame(preprocess_ms),
            "forward": per_frame(forward_ms), "macro_aggregate_and_upsample":
                per_frame(aggregate_ms), "compact_transfer": per_frame(transfer_ms),
            "async_encode_write": per_frame(write_ms),
        },
        "model": str(segmenter.model_id), "model_source": segmenter.source,
        "native_labels_mapped": len(mapper.id2label),
        "unmapped_native_labels": mapper.unmapped_native_labels,
        "residual_native_labels": mapper.residual_native_labels,
        "class_mean_pixel_fraction": class_means,
        "other_environment_mean_pixel_fraction": class_means.get("other_environment"),
        "interior_source": fast.get("interior_cockpit", {}).get("source"),
        "interior_bottom_contact_pixels": (
            int(result["interior_bottom_contact_pixels"].sum())
            if "interior_bottom_contact_pixels" in result else 0),
        "interior_upper_contact_pixels": (
            int(result["interior_upper_contact_pixels"].sum())
            if "interior_upper_contact_pixels" in result else 0),
        "mirror_in_primary_mask": False,
        "grounded_models_used": False,
        "temporal_stabilization": fast.get("temporal_stabilization", {}),
        "config_fingerprint": fp,
    }
    atomic_write_json(seg_root / "summary.json", summary)
    atomic_write_json(manifest_path, {
        "stage": "semantic_gaze_fast_external", "fingerprint": fp,
        "frames": int(len(result)), "taxonomy": str(tax_path),
        "mapping": str(map_path)})
    return summary


def reclassify_saved_interior_contact(input_dir: str | Path, cfg: Config) -> Dict[str, Any]:
    """Apply the same compact ego-contact policy to saved masks, without inference."""
    import cv2
    import pandas as pd
    root = Path(input_dir)
    fast = cfg.get("semantic_gaze_fast_external", {})
    taxonomy = Taxonomy.load(cfg.resolve(fast["classes"]))
    cockpit = fast.get("interior_cockpit", {})
    bottom_policy = cockpit.get("bottom_contact_proxy", {})
    upper_policy = cockpit.get("upper_border_proxy", {})
    policy = {"bottom_contact_proxy": bottom_policy,
              "upper_border_proxy": upper_policy}
    extraction = json.loads((root / "frames" / "extraction_summary.json").read_text())
    domain = str(extraction["domain"])
    seg_root = root / "segmentation"
    index_path = seg_root / "segmentation_index.parquet"
    summary_path = seg_root / "summary.json"
    manifest_path = seg_root / "manifest.json"
    receipt_path = seg_root / "interior_contact_reclass.json"
    tax_path = cfg.resolve(fast["classes"])
    map_path = cfg.resolve(fast["mapillary_mapping"])
    fp = config_fingerprint(
        "semantic_gaze_fast_external", cfg.get("oneformer_mapillary"), fast,
        tax_path.read_text(), map_path.read_text())
    if receipt_path.exists():
        receipt = json.loads(receipt_path.read_text())
        if receipt.get("policy") == policy:
            summary = json.loads(summary_path.read_text())
            summary["config_fingerprint"] = fp
            atomic_write_json(summary_path, summary)
            manifest = json.loads(manifest_path.read_text())
            manifest["fingerprint"] = fp
            atomic_write_json(manifest_path, manifest)
            return {**receipt, "already_applied": True}
    index = pd.read_parquet(index_path)
    total_new = changed_frames = 0
    for i, row in index.iterrows():
        path = root / str(row.mask_path)
        mask = cv2.imread(str(path), cv2.IMREAD_UNCHANGED).astype(np.uint16)
        bottom_new = apply_interior_bottom_contact(mask, taxonomy, bottom_policy)
        upper_new = apply_interior_upper_contact(
            mask, taxonomy, upper_policy, domain)
        changed = bottom_new + upper_new
        if changed:
            atomic_write_bytes(path, _png_bytes(mask))
            changed_frames += 1
            total_new += changed
        hist = np.bincount(mask.ravel(), minlength=taxonomy.max_id + 1).astype(float)
        hist /= max(1.0, hist.sum())
        for klass in taxonomy.classes:
            index.loc[i, f"fraction_{klass.name}"] = hist[klass.id]
        index.loc[i, "interior_bottom_contact_pixels"] = (
            int(row.get("interior_bottom_contact_pixels", 0)) + bottom_new)
        index.loc[i, "interior_upper_contact_pixels"] = (
            int(row.get("interior_upper_contact_pixels", 0)) + upper_new)
    _atomic_parquet(index_path, index)
    summary = json.loads(summary_path.read_text())
    means = {klass.name: float(index[f"fraction_{klass.name}"].mean())
             for klass in taxonomy.classes}
    summary.update({
        "class_mean_pixel_fraction": means,
        "other_environment_mean_pixel_fraction": means["other_environment"],
        "interior_source": fast.get("interior_cockpit", {}).get("source"),
        "interior_bottom_contact_pixels": int(
            index["interior_bottom_contact_pixels"].sum()),
        "interior_upper_contact_pixels": int(
            index["interior_upper_contact_pixels"].sum()),
        "interior_contact_last_changed_frames": changed_frames,
    })
    summary["config_fingerprint"] = fp
    atomic_write_json(summary_path, summary)
    manifest = json.loads(manifest_path.read_text())
    manifest["fingerprint"] = fp
    atomic_write_json(manifest_path, manifest)
    result = {"frames": len(index), "changed_frames": changed_frames,
              "changed_pixels": total_new,
              "bottom_contact_pixels_cumulative": int(
                  index["interior_bottom_contact_pixels"].sum()),
              "upper_contact_pixels_cumulative": int(
                  index["interior_upper_contact_pixels"].sum()),
              "policy": policy}
    atomic_write_json(receipt_path, result)
    return result


class _LayerCache:
    def __init__(self, root: Path, max_items: int = 8):
        self.root = root; self.max_items = max_items
        self.cache: OrderedDict[int, Dict[str, np.ndarray]] = OrderedDict()

    def load(self, frame_index: int) -> Dict[str, np.ndarray]:
        import cv2
        fi = int(frame_index)
        if fi in self.cache:
            value = self.cache.pop(fi); self.cache[fi] = value; return value
        stem = f"frame_{fi:06d}.png"
        mask = cv2.imread(str(self.root / "segmentation" / "masks" / stem),
                          cv2.IMREAD_UNCHANGED)
        confidence = cv2.imread(
            str(self.root / "segmentation" / "confidence" / stem),
            cv2.IMREAD_UNCHANGED)
        entropy = cv2.imread(
            str(self.root / "segmentation" / "normalized_entropy" / stem),
            cv2.IMREAD_UNCHANGED)
        if mask is None or confidence is None or entropy is None:
            raise FileNotFoundError(f"incomplete segmentation layers for frame {fi}")
        value = {"mask": mask.astype(np.uint16),
                 "confidence": confidence.astype(np.float32) / 255.0,
                 "entropy": entropy.astype(np.float32) / 255.0,
                 "provenance": np.zeros(mask.shape, np.uint8)}
        self.cache[fi] = value
        while len(self.cache) > self.max_items:
            self.cache.popitem(last=False)
        return value


def possible_mirror_candidate(u: float, v: float, width: int, height: int,
                              confidence: float, entropy: float,
                              zones: Sequence[Sequence[float]]) -> bool:
    """Weak, spatial high-precision review proxy; never a semantic label."""
    if not (np.isfinite(u) and np.isfinite(v) and confidence >= .50 and entropy <= .70):
        return False
    x, y = u / width, v / height
    return any(x0 <= x <= x1 and y0 <= y <= y1 for x0, y0, x1, y1 in zones)


def run_fast_semantic_gaze(input_dir: str | Path, cfg: Config) -> Dict[str, Any]:
    """Associate every real gaze sample to its nearest real segmented frame."""
    import pandas as pd

    root = Path(input_dir)
    fast = cfg.get("semantic_gaze_fast_external", {})
    taxonomy = Taxonomy.load(cfg.resolve(fast["classes"]))
    names = taxonomy.names()
    projected = pd.read_parquet(root / "gaze" / "projected_gaze.parquet").sort_values(
        "timestamp_ns").reset_index(drop=True)
    if projected.empty:
        raise ValueError("projected gaze table is empty")
    calibration = json.loads((root / "frames" / "calibration.json").read_text())
    stored = calibration["stored_resolution"]
    focal = float(calibration["pinhole_focal_at_stored_resolution"])
    ppd = pixels_per_degree(focal)
    gaze_cfg = fast.get("gaze", {})
    sigma_px = float(gaze_cfg.get("foveal_sigma_deg", 1.5)) * ppd
    radius_px = int(round(float(gaze_cfg.get("foveal_radius_sigmas", 2.5)) * sigma_px))
    weights = foveal_weights(radius_px, sigma_px)
    kin, fixations = identify_fixations(
        projected.timestamp_ns.values, projected.yaw_rad.values,
        projected.pitch_rad.values, projected.combined_valid.values,
        velocity_threshold_deg_s=float(gaze_cfg.get("ivt_velocity_threshold_deg_s", 30.0)),
        min_duration_s=float(gaze_cfg.get("min_fixation_duration_s", .1)),
        max_gap_s=float(gaze_cfg.get("max_gap_within_fixation_s", .075)))
    cache = _LayerCache(root)
    zones = fast.get("mirror", {}).get("normalized_review_zones", {}).get(
        str(projected.iloc[0].get("domain", "")), [])
    if not zones:
        # Domain is stored in frames, not necessarily in the projected gaze table.
        frame_domain = pd.read_parquet(root / "frames" / "frames.parquet").iloc[0].domain
        zones = fast.get("mirror", {}).get("normalized_review_zones", {}).get(
            str(frame_domain), [])
    rows: List[Dict[str, Any]] = []
    mirror_candidates = 0
    start = time.perf_counter()
    for i, gaze in projected.iterrows():
        rec = gaze.to_dict()
        rec["rel_time_s"] = float(
            (int(gaze.timestamp_ns) - int(projected.timestamp_ns.iloc[0])) / 1e9)
        rec.update({"semantic_valid": False, "invalid_reason": None,
                    "fixation_id": int(kin.fixation_id[i]),
                    "is_fixation": bool(kin.is_fixation[i]),
                    "angular_velocity_deg_s": float(kin.angular_velocity_deg_s[i]),
                    "top1_class_id": -1, "top2_class_id": -1,
                    "top1_class": None, "top2_class": None,
                    "top1_probability": np.nan, "top2_probability": np.nan,
                    "foveal_entropy": np.nan, "model_entropy": np.nan,
                    "confidence": np.nan,
                    "possible_mirror_gaze_candidate": False,
                    "mirror_proxy_status": "weak_spatial_review_only"})
        for name in names:
            rec[f"p_{name}"] = np.nan
        if not bool(gaze.semantic_ready):
            rec["invalid_reason"] = "invalid gaze, off-image, or no nearby segmentation"
            rows.append(rec); continue
        layers = cache.load(int(gaze.segmentation_frame_index))
        sample = sample_foveal_semantics(
            layers, float(gaze.rect_u), float(gaze.rect_v), weights, len(names))
        if sample is None:
            rec["invalid_reason"] = "foveal window has no usable pixels"
            rows.append(rec); continue
        probs = sample.pop("foveal_probabilities")
        rec.update(sample)
        rec["top1_class"] = names[int(rec["top1_class_id"])]
        rec["top2_class"] = names[int(rec["top2_class_id"])]
        rec["semantic_valid"] = True
        for cid, name in enumerate(names):
            rec[f"p_{name}"] = float(probs[cid])
        candidate = possible_mirror_candidate(
            float(gaze.rect_u), float(gaze.rect_v), int(stored["width"]),
            int(stored["height"]), float(rec["confidence"]),
            float(rec["foveal_entropy"]), zones)
        rec["possible_mirror_gaze_candidate"] = candidate
        mirror_candidates += int(candidate)
        rows.append(rec)
    samples = pd.DataFrame(rows)
    _atomic_parquet(root / "gaze" / "semantic_gaze.parquet", samples)

    fix_rows = [f.to_dict() for f in fixations]
    for fix in fix_rows:
        subset = samples[(samples.fixation_id == fix["fixation_id"]) &
                         samples.semantic_valid.astype(bool)]
        if len(subset):
            fix["dominant_semantic_class"] = str(subset.top1_class.mode().iloc[0])
            fix["mean_semantic_confidence"] = float(subset.confidence.mean())
        else:
            fix["dominant_semantic_class"] = None
            fix["mean_semantic_confidence"] = None
    fix_df = pd.DataFrame(fix_rows)
    _atomic_parquet(root / "gaze" / "fixations.parquet", fix_df)
    if len(projected) > 1:
        sample_interval_s = float(np.median(np.diff(projected.timestamp_ns.values)) / 1e9)
    else:
        sample_interval_s = 0.0
    primary_road = ["road_surface", "lane_marking", "regulatory_road_marking",
                    "vehicle", "two_wheeler", "pedestrian", "traffic_sign",
                    "traffic_light", "road_boundary_or_sidewalk"]
    metrics = semantic_gaze_metrics(
        samples, names, fixations, sample_interval_s,
        off_road_classes=primary_road)
    valid = int(samples.semantic_valid.astype(bool).sum())
    summary = {
        "schema": "article1_fast_semantic_gaze_v1",
        "result_status": cfg.get("result_status", "exploratory_pilot"),
        "gaze_samples": int(len(samples)), "semantic_valid_samples": valid,
        "semantic_valid_fraction": valid / len(samples) if len(samples) else 0.0,
        "semantic_valid_time_s": valid * sample_interval_s,
        "sample_interval_s_measured_from_timestamps": sample_interval_s,
        "segmentation_frequency_hz": fast.get("segmentation_frequency_hz"),
        "foveal_sigma_deg": gaze_cfg.get("foveal_sigma_deg", 1.5),
        "foveal_radius_px": radius_px,
        "fixations": int(len(fixations)), "metrics": metrics,
        "possible_mirror_gaze_candidates": mirror_candidates,
        "mirror_proxy": {
            "status": "weak_spatial_review_only", "included_in_primary_metrics": False,
            "included_in_primary_masks": False,
            "note": "no reliable full-run mirror detector was enabled"},
        "segmentation_is_gaze_independent": True,
        "grounded_models_used": False,
        "elapsed_s": time.perf_counter() - start,
    }
    atomic_write_json(root / "gaze" / "semantic_gaze_summary.json", summary)
    class_rows = []
    for name in names:
        entry = metrics["per_class"][name]
        class_rows.append({"class_id": taxonomy.id_of(name), "class_name": name, **entry})
    pd.DataFrame(class_rows).to_csv(root / "gaze" / "class_metrics.csv", index=False)
    return summary


def run_semantic_gaze_fast_external(input_dir: str | Path, cfg: Config,
                                    resume: bool = True) -> Dict[str, Any]:
    """Explicit full-run mode: compact segmentation followed by semantic gaze."""
    segmentation = run_fast_external(input_dir, cfg, resume=resume)
    gaze = run_fast_semantic_gaze(input_dir, cfg)
    return {"mode": "semantic_gaze_fast_external", "segmentation": segmentation,
            "semantic_gaze": gaze}
