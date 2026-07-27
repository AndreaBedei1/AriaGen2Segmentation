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
    """Filter raw thin probabilities using road semantics and component geometry."""
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
    # Thin predictions must be supported by independently predicted road surface;
    # they cannot create their own support merely by winning the base argmax.
    road_seed = (base_mask == road_id).astype(np.uint8)
    dilation = int(cfg.get("road_support_dilation_px", 12))
    if dilation > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * dilation + 1,) * 2)
        road_support = cv2.dilate(road_seed, k).astype(bool)
    else:
        road_support = road_seed.astype(bool)
    nonroad = ~np.isin(base_mask, [road_id, lane_id, regulatory_id, 0])
    raw = np.zeros((h, w), np.uint16)
    raw[probabilities[lane_id] >= floor] = lane_id
    raw[probabilities[regulatory_id] >= floor] = regulatory_id
    filtered = np.zeros_like(raw)
    reasons = np.zeros((h, w), np.uint8)
    stats = {name: 0 for name in THIN_REASON.values()}
    max_area = float(cfg.get("max_component_area_frac", .08)) * h * w
    min_area = int(cfg.get("min_component_area_px", 25))
    min_overlap = float(cfg.get("min_road_overlap", .60))
    for cid, threshold in ((lane_id, lane_t), (regulatory_id, reg_t)):
        candidate = probabilities[cid] >= floor
        lowp = candidate & (probabilities[cid] < threshold)
        reasons[lowp] = 1
        lowm = candidate & ~lowp & (margin < min_margin)
        reasons[lowm] = 2
        eligible = candidate & ~lowp & ~lowm
        outside = eligible & ~road_support
        reasons[outside] = 3
        eligible &= road_support
        if cfg.get("suppress_on_nonroad_classes", True):
            conflict = eligible & nonroad
            reasons[conflict] = 4
            eligible &= ~nonroad
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
            overlap = float(road_support[comp].mean()) if area else 0
            if area < min_area:
                reasons[comp] = 5
            elif area > max_area:
                reasons[comp] = 6
            elif overlap < min_overlap:
                reasons[comp] = 3
            else:
                filtered[comp] = cid
                reasons[comp] = 7
    composite = base_mask.copy()
    composite[filtered == lane_id] = lane_id
    composite[filtered == regulatory_id] = regulatory_id
    values, counts = np.unique(reasons, return_counts=True)
    for value, count in zip(values, counts):
        stats[THIN_REASON[int(value)]] = int(count)
    return {"raw_thin_mask": raw, "filtered_thin_mask": filtered,
            "reason_map": reasons, "composite": composite, "reason_stats": stats}


def _atomic_npz(path: Path, **arrays) -> None:
    with atomic_write(path, "wb") as handle:
        np.savez_compressed(handle, **arrays)


def run_external(input_dir: str, cfg: Config, resume: bool = True,
                 force: bool = False) -> int:
    import cv2
    import torch
    import torch.nn.functional as F
    from PIL import Image

    root = Path(input_dir)
    out = root / "article1_external"
    for name in ("native_masks", "native_probabilities", "probabilities", "base_masks",
                 "thin_masks", "masks", "confidence", "entropy", "top2", "metadata",
                 "provenance", "mapillary_ego_region"):
        (out / name).mkdir(parents=True, exist_ok=True)
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
        inputs = base._proc(images=Image.fromarray(rgb), return_tensors="pt")
        inputs = {k: (v.to(base.device) if hasattr(v, "to") else v) for k, v in inputs.items()}
        with torch.inference_mode(), torch.autocast(base.device, dtype=base._amp_dtype,
                                                    enabled=base.device == "cuda"):
            outputs = base._model(**inputs)
        cq = outputs.class_queries_logits.float().softmax(dim=-1)[..., :-1]
        mq = F.interpolate(outputs.masks_queries_logits.float(), size=rgb.shape[:2],
                           mode="bilinear", align_corners=False).sigmoid()
        native_scores = torch.einsum("bqc,bqhw->bchw", cq, mq)[0]
        native_prob = (native_scores / native_scores.sum(dim=0, keepdim=True).clamp_min(1e-12))
        native_prob = native_prob.cpu().numpy().astype(np.float32)
        native_mask = native_prob.argmax(axis=0).astype(np.uint16)
        ego_region = np.isin(native_mask, list(mapper.mapillary_ego_ids))
        agg = mapper.aggregate(native_prob, cfg.get("article1.confidence_threshold", .38),
                               cfg.get("article1.entropy_unknown_threshold"))
        thin = preserve_thin_markings(
            agg["mask"], agg["probabilities"],
            lane_threshold=cfg.get("article1.lane_marking_threshold", .24),
            regulatory_threshold=cfg.get("article1.regulatory_marking_threshold", .20),
            min_component_area=cfg.get("article1.thin_class_min_component_area", 18),
            max_gap=cfg.get("article1.thin_class_max_gap", 2),
            morphology=cfg.get("article1.thin_class_morphology", "conservative_close"))
        stem = f"frame_{fi:06d}"
        write_mask_u16(out / "native_masks" / f"{stem}.png", native_mask)
        write_mask_u16(out / "base_masks" / f"{stem}.png", agg["mask"])
        write_mask_u16(out / "thin_masks" / f"{stem}.png", thin["thin_mask"])
        write_mask_u16(out / "masks" / f"{stem}.png", thin["composite"])
        # Provenance id 1 = Mapillary. Separate binary ego flag retains the native
        # region without turning it into a scientific cockpit class.
        write_mask_u16(out / "provenance" / f"{stem}.png",
                       np.ones_like(native_mask, dtype=np.uint16))
        cv2.imwrite(str(out / "mapillary_ego_region" / f"{stem}.png"),
                    ego_region.astype(np.uint8) * 255)
        cv2.imwrite(str(out / "confidence" / f"{stem}.png"),
                    np.clip(agg["confidence"] * 255, 0, 255).astype(np.uint8))
        _atomic_npz(out / "native_probabilities" / f"{stem}.npz",
                    probabilities=native_prob.astype(np.float16))
        _atomic_npz(out / "probabilities" / f"{stem}.npz",
                    probabilities=agg["probabilities"].astype(np.float16))
        _atomic_npz(out / "entropy" / f"{stem}.npz",
                    entropy=agg["entropy"].astype(np.float16))
        _atomic_npz(out / "top2" / f"{stem}.npz", class_ids=agg["top2"])
        elapsed = (time.time() - t0) * 1000
        timings.append(elapsed)
        meta = {"frame_index": fi, "capture_timestamp_ns": ref.capture_timestamp_ns,
                "vehicle_type": cfg.get("article1.vehicle_type"),
                "session_id": cfg.get("article1.session_id"),
                "participant_id": cfg.get("article1.participant_id"),
                "total_ms": round(elapsed, 1),
                "coverage": float((thin["composite"] > 0).mean()),
                "unknown_rate": float((thin["composite"] == 0).mean()),
                "thin_lane_pixels": int(thin["lane_mask"].sum()),
                "thin_regulatory_pixels": int(thin["regulatory_mask"].sum())}
        meta["mapillary_ego_region"] = True
        meta["mapillary_ego_region_pixels"] = int(ego_region.sum())
        meta["provenance_codes"] = {"1": "mapillary"}
        atomic_write_json(out / "metadata" / f"{stem}.json", meta)
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
