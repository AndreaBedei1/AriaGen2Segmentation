"""Probabilistic Mapillary → reduced Article 1 external segmentation."""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Dict, Mapping

import numpy as np
import yaml

from ..config import Config
from ..hashing import sha256_file
from ..io_utils import (Manifest, atomic_write, atomic_write_json,
                        config_fingerprint, write_mask_u16)
from ..segmentation.base import iter_frames
from ..segmentation.oneformer import OneFormerMapillarySegmenter, _norm
from ..taxonomy import Taxonomy


class Article1Mapper:
    """Map native channels to macroclasses while retaining probability mass."""

    def __init__(self, id2label: Mapping[int, str], mapping_path: str | Path,
                 taxonomy: Taxonomy):
        doc = yaml.safe_load(Path(mapping_path).read_text())
        raw = {_norm(k): v for k, v in doc["mapping"].items()}
        self.id2label = {int(k): str(v) for k, v in id2label.items()}
        self.taxonomy = taxonomy
        self.native_to_article = np.zeros(max(self.id2label) + 1, dtype=np.int64)
        self.attributes: Dict[int, dict] = {}
        self.unmapped = []
        for native_id, label in self.id2label.items():
            entry = raw.get(_norm(label))
            if entry is None:
                self.unmapped.append(label)
                article_name = "unknown"
                entry = {}
            else:
                article_name = entry["article1"]
            self.native_to_article[native_id] = taxonomy.id_of(article_name)
            self.attributes[native_id] = {
                k: v for k, v in entry.items() if k not in {"article1"}
            }
        self.mapillary_ego_ids = {
            native_id for native_id, attrs in self.attributes.items()
            if attrs.get("mapillary_ego_region") is True
        }

    def aggregate(self, native_probabilities: np.ndarray, confidence_threshold: float,
                  entropy_unknown_threshold: float | None = None,
                  unknown_policy: dict | None = None) -> dict:
        return aggregate_native_probabilities(
            native_probabilities, self.native_to_article, self.taxonomy.max_id + 1,
            confidence_threshold, entropy_unknown_threshold, unknown_policy)


def aggregate_native_probabilities(native_probabilities: np.ndarray,
                                   native_to_article: np.ndarray,
                                   num_article_classes: int,
                                   confidence_threshold: float = 0.38,
                                   entropy_unknown_threshold: float | None = None,
                                   unknown_policy: dict | None = None) -> dict:
    """Sum native probability channels into Article 1 channels.

    Input is C×H×W and is normalized defensively. Unsupported native channels map
    to class 0. Low confidence/high entropy transfers the complete pixel to unknown;
    this keeps ``unknown`` distinct from the known ``other_environment`` class.
    """
    p = np.asarray(native_probabilities, dtype=np.float32)
    if p.ndim != 3 or p.shape[0] != len(native_to_article):
        raise ValueError("native_probabilities must be CxHxW and match mapping length")
    p = np.clip(p, 0, None)
    denom = p.sum(axis=0, keepdims=True)
    p = np.divide(p, denom, out=np.full_like(p, 1.0 / p.shape[0]), where=denom > 0)
    out = np.zeros((num_article_classes, *p.shape[1:]), dtype=np.float32)
    for native_id, article_id in enumerate(native_to_article):
        out[int(article_id)] += p[native_id]
    out /= np.maximum(out.sum(axis=0, keepdims=True), 1e-12)
    entropy = -(out * np.log(np.maximum(out, 1e-12))).sum(axis=0)
    normalized_entropy = entropy / np.log(max(2, num_article_classes))
    known = out[1:]
    order = np.argsort(known, axis=0)
    top1_ids = order[-1].astype(np.uint16) + 1
    top2_ids = order[-2].astype(np.uint16) + 1
    top1 = np.take_along_axis(known, order[-1:], axis=0)[0]
    top2 = np.take_along_axis(known, order[-2:-1], axis=0)[0]
    margin = top1 - top2
    dominant_native = p.argmax(axis=0)
    unsupported = native_to_article[dominant_native] == 0
    reasons = np.zeros(p.shape[1:], dtype=np.uint8)
    policy = unknown_policy or {}
    if policy.get("enabled", bool(unknown_policy)):
        low_probability = top1 < float(policy.get("min_top1_probability", .45))
        low_margin = margin < float(policy.get("min_top1_top2_margin", .08))
        high_entropy = normalized_entropy > float(policy.get("max_normalized_entropy", .75))
        reasons[low_probability] |= 1
        reasons[low_margin] |= 2
        reasons[high_entropy] |= 4
        if policy.get("unsupported_native_to_unknown", True):
            reasons[unsupported] |= 8
        tests = np.stack([low_probability, low_margin, high_entropy,
                          unsupported if policy.get("unsupported_native_to_unknown", True)
                          else np.zeros_like(unsupported)])
        uncertain = tests.all(axis=0) if policy.get("combine_rule") == "all" else tests.any(axis=0)
    else:
        uncertain = top1 < float(confidence_threshold)
        reasons[uncertain] |= 1
        if entropy_unknown_threshold is not None:
            hi = entropy > float(entropy_unknown_threshold)
            uncertain |= hi
            reasons[hi] |= 4
    out[:, uncertain] = 0
    out[0, uncertain] = 1
    mask = out.argmax(axis=0).astype(np.uint16)
    confidence = out.max(axis=0).astype(np.float32)
    final_top2 = np.stack([top1_ids, top2_ids])
    return {"probabilities": out, "mask": mask, "confidence": confidence,
            "entropy": entropy.astype(np.float32),
            "normalized_entropy": normalized_entropy.astype(np.float32),
            "top2": final_top2, "top1_probability": top1.astype(np.float32),
            "top2_probability": top2.astype(np.float32),
            "margin": margin.astype(np.float32), "unknown_reason": reasons,
            "dominant_native": dominant_native.astype(np.uint16)}


def preserve_thin_markings(base_mask: np.ndarray, probabilities: np.ndarray,
                           lane_id: int = 2, regulatory_id: int = 3,
                           lane_threshold: float = .24,
                           regulatory_threshold: float = .20,
                           min_component_area: int = 18,
                           max_gap: int = 2,
                           morphology: str = "conservative_close") -> dict:
    """Legacy checkpoint-1 helper.

    Kept only to read/reproduce historical outputs. Production paths must use
    :func:`filter_thin_markings` through :func:`apply_article1_policy`.
    """
    import cv2

    def clean(binary):
        binary = binary.astype(np.uint8)
        if morphology == "conservative_close" and max_gap > 0:
            k = max(1, min(int(max_gap), 3))
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
            binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        n, labels, stats, _ = cv2.connectedComponentsWithStats(binary, 8)
        kept = np.zeros_like(binary, dtype=bool)
        for i in range(1, n):
            if int(stats[i, cv2.CC_STAT_AREA]) >= int(min_component_area):
                kept |= labels == i
        return kept

    lane = clean(probabilities[lane_id] >= lane_threshold)
    regulatory = clean(probabilities[regulatory_id] >= regulatory_threshold)
    composite = base_mask.copy()
    composite[lane] = lane_id
    composite[regulatory] = regulatory_id  # regulatory wins by explicit priority
    thin = np.zeros_like(base_mask, dtype=np.uint16)
    thin[lane] = lane_id
    thin[regulatory] = regulatory_id
    return {"composite": composite, "thin_mask": thin, "lane_mask": lane,
            "regulatory_mask": regulatory}


THIN_REASON = {
    0: "none", 1: "low_probability", 2: "low_margin", 3: "outside_road_support",
    4: "conflicts_with_nonroad", 5: "component_too_small",
    6: "component_too_large", 7: "accepted",
}


def filter_thin_markings(base_mask: np.ndarray, probabilities: np.ndarray,
                         cfg: dict) -> dict:
    """Filter thin components against probabilistic road support.

    Components are formed *before* road clipping, so ``min_road_overlap`` is the
    actual supported fraction of each original candidate component.
    """
    import cv2
    h, w = base_mask.shape
    lane_id, regulatory_id, road_id = 2, 3, 1
    lane_t = float(cfg.get("lane_threshold", .40))
    reg_t = float(cfg.get("regulatory_threshold", .45))
    floor = float(cfg.get("candidate_probability_floor", .10))
    known = probabilities[1:]
    ordered = np.sort(known, axis=0)
    margin = ordered[-1] - ordered[-2]
    min_margin = float(cfg.get("min_top1_top2_margin", .05))
    road_probability = probabilities[road_id]
    road_probability_support = (
        road_probability >= float(cfg.get("road_probability_threshold", .25))
        if cfg.get("use_road_probability_support", True)
        else np.zeros((h, w), dtype=bool)
    )
    base_support = (
        base_mask == road_id if cfg.get("use_base_mask_support", True)
        else np.zeros((h, w), dtype=bool)
    )
    road_support_raw = road_probability_support | base_support
    dilation = int(cfg.get("road_support_dilation_px", 12))
    if dilation > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * dilation + 1,) * 2)
        road_support_dilated = cv2.dilate(
            road_support_raw.astype(np.uint8), k).astype(bool)
    else:
        road_support_dilated = road_support_raw.copy()
    nonroad_ids = cfg.get(
        "nonroad_class_ids", [4, 6, 7, 8, 13, 10, 11, 12])
    nonroad_confidence = probabilities[np.asarray(nonroad_ids, dtype=int)].max(axis=0)
    nonroad = nonroad_confidence >= float(cfg.get("nonroad_conflict_threshold", .55))
    if not cfg.get("suppress_on_nonroad_classes", True):
        nonroad = np.zeros_like(nonroad)
    road_support_final = road_support_dilated & ~nonroad
    raw = np.zeros((h, w), np.uint16)
    raw[probabilities[lane_id] >= floor] = lane_id
    raw[probabilities[regulatory_id] >= floor] = regulatory_id
    filtered = np.zeros_like(raw)
    reasons = np.zeros((h, w), np.uint8)
    stats = {name: 0 for name in THIN_REASON.values()}
    max_area = float(cfg.get("max_component_area_frac", .08)) * h * w
    min_area = int(cfg.get("min_component_area_px", 25))
    min_overlap = float(cfg.get("min_road_overlap", .60))
    accepted_policy = cfg.get("accepted_component_policy", "supported_pixels_only")
    if accepted_policy not in {"supported_pixels_only", "full_component_if_supported"}:
        raise ValueError(f"unsupported accepted_component_policy: {accepted_policy}")
    for cid, threshold in ((lane_id, lane_t), (regulatory_id, reg_t)):
        candidate = probabilities[cid] >= floor
        lowp = candidate & (probabilities[cid] < threshold)
        reasons[lowp] = 1
        lowm = candidate & ~lowp & (margin < min_margin)
        reasons[lowm] = 2
        eligible = candidate & ~lowp & ~lowm
        binary = eligible.astype(np.uint8)
        op = int(cfg.get("morphology_open_kernel", 1))
        cl = int(cfg.get("morphology_close_kernel", 1))
        if op > 1:
            binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, np.ones((op, op), np.uint8))
        if cl > 1:
            binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, np.ones((cl, cl), np.uint8))
        n, labels, comp_stats, _ = cv2.connectedComponentsWithStats(binary, 8)
        for i in range(1, n):
            comp = labels == i
            area = int(comp_stats[i, cv2.CC_STAT_AREA])
            overlap = float(road_support_final[comp].sum() / area) if area else 0
            if area < min_area:
                reasons[comp] = 5
            elif area > max_area:
                reasons[comp] = 6
            elif overlap < min_overlap:
                reasons[comp] = 3
            else:
                accepted = (comp if accepted_policy == "full_component_if_supported"
                            else comp & road_support_final)
                if cfg.get("suppress_on_nonroad_classes", True):
                    reasons[comp & nonroad] = 4
                    accepted &= ~nonroad
                unsupported = comp & ~road_support_final & ~nonroad
                reasons[unsupported] = 3
                filtered[accepted] = cid
                reasons[accepted] = 7
    composite = base_mask.copy()
    composite[filtered == lane_id] = lane_id
    composite[filtered == regulatory_id] = regulatory_id
    values, counts = np.unique(reasons, return_counts=True)
    for value, count in zip(values, counts):
        stats[THIN_REASON[int(value)]] = int(count)
    return {"raw_thin_mask": raw, "filtered_thin_mask": filtered,
            "reason_map": reasons, "composite": composite, "reason_stats": stats,
            "road_support_raw": road_support_raw,
            "road_support_dilated": road_support_dilated,
            "road_support_final": road_support_final}


def _atomic_npz(path: Path, **arrays) -> None:
    with atomic_write(path, "wb") as handle:
        np.savez_compressed(handle, **arrays)


ARTICLE1_OUTPUT_DIRS = (
    "native_masks", "native_probabilities", "probabilities", "base_masks",
    "raw_thin_masks", "filtered_thin_masks", "masks", "confidence", "entropy",
    "normalized_entropy", "top2", "top1_probability", "top2_probability",
    "margin", "unknown_reason", "thin_rejection_reason", "provenance",
    "mapillary_ego_region", "metadata",
)


def infer_native_probabilities(segmenter, image_rgb: np.ndarray) -> np.ndarray:
    """Run Mask2Former and return normalized C×H×W native probabilities."""
    import torch
    import torch.nn.functional as F
    from PIL import Image

    inputs = segmenter._proc(images=Image.fromarray(image_rgb), return_tensors="pt")
    inputs = {k: (v.to(segmenter.device) if hasattr(v, "to") else v)
              for k, v in inputs.items()}
    with torch.inference_mode(), torch.autocast(
            segmenter.device, dtype=segmenter._amp_dtype,
            enabled=segmenter.device == "cuda"):
        outputs = segmenter._model(**inputs)
    class_queries = outputs.class_queries_logits.float().softmax(dim=-1)[..., :-1]
    mask_queries = F.interpolate(
        outputs.masks_queries_logits.float(), size=image_rgb.shape[:2],
        mode="bilinear", align_corners=False).sigmoid()
    scores = torch.einsum("bqc,bqhw->bchw", class_queries, mask_queries)[0]
    probabilities = scores / scores.sum(dim=0, keepdim=True).clamp_min(1e-12)
    return probabilities.cpu().numpy().astype(np.float32)


def apply_article1_policy(native_probabilities: np.ndarray, mapper: Article1Mapper,
                          article1_cfg: dict) -> dict:
    """Single production implementation of aggregation, unknown and thin policy."""
    if article1_cfg.get("probability_dtype", "float16") == "float16":
        # Apply policy to the exact persisted representation. Reprocessing the
        # stored tensor is therefore bit-equivalent to the direct path.
        native_probabilities = np.asarray(
            native_probabilities, dtype=np.float16).astype(np.float32)
    else:
        native_probabilities = np.asarray(native_probabilities, dtype=np.float32)
    aggregate = mapper.aggregate(
        native_probabilities,
        article1_cfg.get("confidence_threshold", .38),
        article1_cfg.get("entropy_unknown_threshold"),
        article1_cfg.get("unknown"),
    )
    thin = filter_thin_markings(
        aggregate["mask"], aggregate["probabilities"],
        article1_cfg.get("thin_markings", {}))
    native_mask = aggregate["dominant_native"]
    return {
        "aggregate": aggregate,
        "thin": thin,
        "native_mask": native_mask,
        "mapillary_ego_region": np.isin(
            native_mask, list(mapper.mapillary_ego_ids)),
    }


def build_article1_metadata(result: dict, frame_index: int,
                            capture_timestamp_ns: int, article1_cfg: dict,
                            elapsed_ms: float) -> dict:
    aggregate, thin = result["aggregate"], result["thin"]
    reason_counts = {
        str(int(value)): int((aggregate["unknown_reason"] == value).sum())
        for value in np.unique(aggregate["unknown_reason"])
    }
    return {
        "frame_index": int(frame_index),
        "capture_timestamp_ns": int(capture_timestamp_ns),
        "vehicle_type": article1_cfg.get("vehicle_type"),
        "session_id": article1_cfg.get("session_id"),
        "participant_id": article1_cfg.get("participant_id"),
        "total_ms": round(float(elapsed_ms), 1),
        "coverage": float((thin["composite"] > 0).mean()),
        "unknown_rate": float((thin["composite"] == 0).mean()),
        "mean_confidence": float(aggregate["confidence"].mean()),
        "mean_normalized_entropy": float(aggregate["normalized_entropy"].mean()),
        "unknown_reason_bitmask_counts": reason_counts,
        "unknown_reason_bits": {
            "1": "low_probability", "2": "low_margin",
            "4": "high_entropy", "8": "unsupported_native",
        },
        "thin_reason_counts": thin["reason_stats"],
        "thin_reason_codes": {str(k): v for k, v in THIN_REASON.items()},
        "thin_raw_pixels": int((thin["raw_thin_mask"] > 0).sum()),
        "thin_filtered_pixels": int((thin["filtered_thin_mask"] > 0).sum()),
        "mapillary_ego_region": True,
        "mapillary_ego_region_pixels": int(result["mapillary_ego_region"].sum()),
        "provenance_codes": {"1": "mapillary"},
        "policy_version": "article1_static_v2",
    }


def write_article1_outputs(out: Path, stem: str, result: dict,
                           metadata: dict, native_probabilities: np.ndarray | None,
                           article1_cfg: dict) -> None:
    """Write the canonical storage contract used by direct and reprocess paths."""
    import cv2

    for name in ARTICLE1_OUTPUT_DIRS:
        (out / name).mkdir(parents=True, exist_ok=True)
    aggregate, thin, native_mask = (
        result["aggregate"], result["thin"], result["native_mask"])
    write_mask_u16(out / "native_masks" / f"{stem}.png", native_mask)
    write_mask_u16(out / "base_masks" / f"{stem}.png", aggregate["mask"])
    write_mask_u16(out / "raw_thin_masks" / f"{stem}.png", thin["raw_thin_mask"])
    write_mask_u16(out / "filtered_thin_masks" / f"{stem}.png",
                   thin["filtered_thin_mask"])
    write_mask_u16(out / "masks" / f"{stem}.png", thin["composite"])
    write_mask_u16(out / "provenance" / f"{stem}.png",
                   np.ones_like(native_mask, dtype=np.uint16))
    cv2.imwrite(str(out / "mapillary_ego_region" / f"{stem}.png"),
                result["mapillary_ego_region"].astype(np.uint8) * 255)
    cv2.imwrite(str(out / "confidence" / f"{stem}.png"),
                np.clip(aggregate["confidence"] * 255, 0, 255).astype(np.uint8))
    cv2.imwrite(str(out / "unknown_reason" / f"{stem}.png"),
                aggregate["unknown_reason"])
    cv2.imwrite(str(out / "thin_rejection_reason" / f"{stem}.png"),
                thin["reason_map"])
    if native_probabilities is not None and article1_cfg.get(
            "save_native_probabilities", True):
        _atomic_npz(out / "native_probabilities" / f"{stem}.npz",
                    probabilities=native_probabilities.astype(np.float16))
    if article1_cfg.get("save_article1_probabilities", True):
        _atomic_npz(out / "probabilities" / f"{stem}.npz",
                    probabilities=aggregate["probabilities"].astype(np.float16))
    _atomic_npz(out / "entropy" / f"{stem}.npz",
                entropy=aggregate["entropy"].astype(np.float16))
    _atomic_npz(out / "normalized_entropy" / f"{stem}.npz",
                normalized_entropy=aggregate["normalized_entropy"].astype(np.float16))
    _atomic_npz(out / "top2" / f"{stem}.npz", class_ids=aggregate["top2"])
    _atomic_npz(out / "top1_probability" / f"{stem}.npz",
                probability=aggregate["top1_probability"].astype(np.float16))
    _atomic_npz(out / "top2_probability" / f"{stem}.npz",
                probability=aggregate["top2_probability"].astype(np.float16))
    _atomic_npz(out / "margin" / f"{stem}.npz",
                margin=aggregate["margin"].astype(np.float16))
    if article1_cfg.get("diagnostic_outputs", False):
        for name in ("road_support_raw", "road_support_dilated", "road_support_final"):
            directory = out / name
            directory.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(directory / f"{stem}.png"),
                        thin[name].astype(np.uint8) * 255)
    atomic_write_json(out / "metadata" / f"{stem}.json", metadata)


def run_external(input_dir: str, cfg: Config, resume: bool = True,
                 force: bool = False) -> int:
    import cv2
    import torch

    root = Path(input_dir)
    article1_cfg = cfg.get("article1", {})
    out = root / article1_cfg.get("output_subdir", "article1_external")
    tax_path = cfg.resolve(cfg.get("article1.classes"))
    map_path = cfg.resolve(cfg.get("article1.mapillary_mapping"))
    tax = Taxonomy.load(tax_path)
    base = OneFormerMapillarySegmenter(cfg, tax)
    base.load()
    mapper = Article1Mapper(base._model.config.id2label, map_path, tax)
    fp = config_fingerprint("article1_external", cfg.get("oneformer_mapillary"),
                            cfg.get("article1"), tax_path.read_text(), map_path.read_text())
    manifest = Manifest.load_or_new(out / "manifest.json", "article1_external", fp,
                                    meta={"taxonomy_sha256": sha256_file(tax_path),
                                          "mapping_sha256": sha256_file(map_path)})
    if force:
        manifest.done.clear()
    timings = []
    for ref in iter_frames(input_dir):
        fi = ref.frame_index
        required = out / "masks" / f"frame_{fi:06d}.png"
        if resume and not force and manifest.is_done(fi) and required.exists():
            continue
        t0 = time.time()
        rgb = cv2.cvtColor(cv2.imread(str(ref.rectified_path)), cv2.COLOR_BGR2RGB)
        native_prob = infer_native_probabilities(base, rgb)
        result = apply_article1_policy(native_prob, mapper, article1_cfg)
        stem = f"frame_{fi:06d}"
        elapsed = (time.time() - t0) * 1000
        timings.append(elapsed)
        meta = build_article1_metadata(
            result, fi, ref.capture_timestamp_ns, article1_cfg, elapsed)
        write_article1_outputs(
            out, stem, result, meta, native_prob, article1_cfg)
        manifest.mark(fi, {"total_ms": elapsed})
        manifest.save()
    peak_vram_mb = None
    if torch.cuda.is_available():
        peak_vram_mb = round(torch.cuda.max_memory_allocated() / 1e6, 1)
    prior_summary = {}
    if (out / "summary.json").exists():
        prior_summary = json.loads((out / "summary.json").read_text())
    atomic_write_json(out / "summary.json",
                      {"frames": len(manifest.done),
                       "mean_ms": (float(np.mean(timings)) if timings
                                   else prior_summary.get("mean_ms")),
                       "peak_vram_mb": peak_vram_mb,
                       "unmapped_native_labels": mapper.unmapped,
                       "config_fingerprint": fp})
    return 0
