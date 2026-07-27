"""Re-run Article 1 policy from saved native probabilities without model inference."""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from ..config import Config
from ..io_utils import Manifest, atomic_write_json, config_fingerprint
from ..taxonomy import Taxonomy
from .external import (
    Article1Mapper,
    apply_article1_policy,
    build_article1_metadata,
    write_article1_outputs,
)


def run_reprocess_external(input_dir: str, cfg: Config, source_subdir="article1_external",
                           resume=True, force=False) -> int:
    root = Path(input_dir)
    source = root / source_subdir
    target = root / cfg.get("article1.output_subdir", "article1_external_checkpoint2")
    article1_cfg = cfg.get("article1", {})
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
        result = apply_article1_policy(native_prob, mapper, article1_cfg)
        elapsed = (time.perf_counter() - start) * 1000
        elapsed_all.append(elapsed)
        meta = build_article1_metadata(
            result, fi, int(row.capture_timestamp_ns), article1_cfg, elapsed)
        meta["postprocess_ms"] = meta.pop("total_ms")
        write_article1_outputs(
            target, stem, result, meta, None, article1_cfg)
        manifest.mark(fi, {"postprocess_ms": elapsed})
        manifest.save()
    atomic_write_json(target / "summary.json", {
        "frames": len(manifest.done),
        "mean_postprocess_ms": float(np.mean(elapsed_all)) if elapsed_all else None,
        "config_fingerprint": fp,
        "unmapped_native_labels": mapper.unmapped,
    })
    return 0
