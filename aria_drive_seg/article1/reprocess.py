"""Re-run Article 1 policy from saved native probabilities without model inference."""
from __future__ import annotations

import json
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from ..config import Config
from ..io_utils import Manifest, atomic_write, atomic_write_json, config_fingerprint, write_mask_u16
from ..segmentation.oneformer import _norm
from ..taxonomy import Taxonomy
from .external import Article1Mapper, THIN_REASON, filter_thin_markings


def _npz(path, **arrays):
    with atomic_write(path, "wb") as handle:
        np.savez_compressed(handle, **arrays)


def run_reprocess_external(input_dir: str, cfg: Config, source_subdir="article1_external",
                           resume=True, force=False) -> int:
    root = Path(input_dir)
    source = root / source_subdir
    target = root / cfg.get("article1.output_subdir", "article1_external_checkpoint2")
    dirs = ("probabilities", "base_masks", "raw_thin_masks", "filtered_thin_masks",
            "masks", "confidence", "entropy", "normalized_entropy", "top2",
            "top1_probability", "top2_probability", "margin", "unknown_reason",
            "thin_rejection_reason", "provenance", "mapillary_ego_region", "metadata")
    for name in dirs:
        (target / name).mkdir(parents=True, exist_ok=True)
    tax = Taxonomy.load(cfg.resolve(cfg.get("article1.classes")))
    mapping_path = cfg.resolve(cfg.get("article1.mapillary_mapping"))
    # The native id2label is stable and recorded in the local checkpoint config.
    model_cfg = cfg.resolve(cfg.get("oneformer_mapillary.mask2former_id")) / "config.json"
    id2label = json.loads(model_cfg.read_text())["id2label"]
    mapper = Article1Mapper(id2label, mapping_path, tax)
    fp = config_fingerprint("article1_reprocess_v2", cfg.get("article1"),
                            mapping_path.read_text())
    manifest = Manifest.load_or_new(target / "manifest.json", "article1_reprocess_v2", fp)
    if force:
        manifest.done.clear()
    frames = pd.read_parquet(root / "frames" / "frames.parquet").sort_values("frame_index")
    elapsed_all = []
    for _, row in frames.iterrows():
        fi = int(row.frame_index)
        stem = f"frame_{fi:06d}"
        if resume and not force and manifest.is_done(fi) and (target / "masks" / f"{stem}.png").exists():
            continue
        start = time.perf_counter()
        native_prob = np.load(source / "native_probabilities" / f"{stem}.npz")["probabilities"].astype(np.float32)
        agg = mapper.aggregate(native_prob, .0, None, cfg.get("article1.unknown"))
        thin = filter_thin_markings(agg["mask"], agg["probabilities"],
                                    cfg.get("article1.thin_markings", {}))
        native_mask = agg["dominant_native"]
        ego = np.isin(native_mask, list(mapper.mapillary_ego_ids))
        write_mask_u16(target / "base_masks" / f"{stem}.png", agg["mask"])
        write_mask_u16(target / "raw_thin_masks" / f"{stem}.png", thin["raw_thin_mask"])
        write_mask_u16(target / "filtered_thin_masks" / f"{stem}.png", thin["filtered_thin_mask"])
        write_mask_u16(target / "masks" / f"{stem}.png", thin["composite"])
        write_mask_u16(target / "provenance" / f"{stem}.png",
                       np.ones_like(native_mask, dtype=np.uint16))
        cv2.imwrite(str(target / "mapillary_ego_region" / f"{stem}.png"), ego.astype(np.uint8) * 255)
        cv2.imwrite(str(target / "confidence" / f"{stem}.png"),
                    np.clip(agg["confidence"] * 255, 0, 255).astype(np.uint8))
        cv2.imwrite(str(target / "unknown_reason" / f"{stem}.png"), agg["unknown_reason"])
        cv2.imwrite(str(target / "thin_rejection_reason" / f"{stem}.png"), thin["reason_map"])
        _npz(target / "probabilities" / f"{stem}.npz",
             probabilities=agg["probabilities"].astype(np.float16))
        _npz(target / "entropy" / f"{stem}.npz", entropy=agg["entropy"].astype(np.float16))
        _npz(target / "normalized_entropy" / f"{stem}.npz",
             normalized_entropy=agg["normalized_entropy"].astype(np.float16))
        _npz(target / "top2" / f"{stem}.npz", class_ids=agg["top2"])
        _npz(target / "top1_probability" / f"{stem}.npz",
             probability=agg["top1_probability"].astype(np.float16))
        _npz(target / "top2_probability" / f"{stem}.npz",
             probability=agg["top2_probability"].astype(np.float16))
        _npz(target / "margin" / f"{stem}.npz", margin=agg["margin"].astype(np.float16))
        reason_counts = {str(i): int((agg["unknown_reason"] == i).sum())
                         for i in np.unique(agg["unknown_reason"])}
        elapsed = (time.perf_counter() - start) * 1000
        elapsed_all.append(elapsed)
        meta = {
            "frame_index": fi, "capture_timestamp_ns": int(row.capture_timestamp_ns),
            "vehicle_type": cfg.get("article1.vehicle_type"),
            "session_id": cfg.get("article1.session_id"),
            "participant_id": cfg.get("article1.participant_id"),
            "postprocess_ms": round(elapsed, 1),
            "unknown_rate": float((thin["composite"] == 0).mean()),
            "mean_confidence": float(agg["confidence"].mean()),
            "mean_normalized_entropy": float(agg["normalized_entropy"].mean()),
            "unknown_reason_bitmask_counts": reason_counts,
            "unknown_reason_bits": {"1": "low_probability", "2": "low_margin",
                                    "4": "high_entropy", "8": "unsupported_native"},
            "thin_reason_counts": thin["reason_stats"],
            "thin_reason_codes": {str(k): v for k, v in THIN_REASON.items()},
            "thin_raw_pixels": int((thin["raw_thin_mask"] > 0).sum()),
            "thin_filtered_pixels": int((thin["filtered_thin_mask"] > 0).sum()),
            "mapillary_ego_region": True,
            "mapillary_ego_region_pixels": int(ego.sum()),
            "provenance_codes": {"1": "mapillary"},
        }
        atomic_write_json(target / "metadata" / f"{stem}.json", meta)
        manifest.mark(fi, {"postprocess_ms": elapsed})
        manifest.save()
    atomic_write_json(target / "summary.json", {
        "frames": len(manifest.done),
        "mean_postprocess_ms": float(np.mean(elapsed_all)) if elapsed_all else None,
        "config_fingerprint": fp,
        "unmapped_native_labels": mapper.unmapped,
    })
    return 0
