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

    def aggregate(self, native_probabilities: np.ndarray, confidence_threshold: float,
                  entropy_unknown_threshold: float | None = None) -> dict:
        return aggregate_native_probabilities(
            native_probabilities, self.native_to_article, self.taxonomy.max_id + 1,
            confidence_threshold, entropy_unknown_threshold)


def aggregate_native_probabilities(native_probabilities: np.ndarray,
                                   native_to_article: np.ndarray,
                                   num_article_classes: int,
                                   confidence_threshold: float = 0.38,
                                   entropy_unknown_threshold: float | None = None) -> dict:
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
    known_conf = out[1:].max(axis=0) if num_article_classes > 1 else np.zeros(p.shape[1:])
    uncertain = known_conf < float(confidence_threshold)
    if entropy_unknown_threshold is not None:
        uncertain |= entropy > float(entropy_unknown_threshold)
    out[:, uncertain] = 0
    out[0, uncertain] = 1
    mask = out.argmax(axis=0).astype(np.uint16)
    confidence = out.max(axis=0).astype(np.float32)
    top2 = np.argsort(out, axis=0)[-2:][::-1].astype(np.uint16)
    return {"probabilities": out, "mask": mask, "confidence": confidence,
            "entropy": entropy.astype(np.float32), "top2": top2}


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
                 "thin_masks", "masks", "confidence", "entropy", "top2", "metadata"):
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
