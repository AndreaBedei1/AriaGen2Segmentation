"""Segmentation runner: resumable iteration, per-frame error isolation, OOM
fallback, and aggregation (§13, §14)."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import numpy as np

from ..config import Config
from ..io_utils import (Manifest, append_jsonl, atomic_write_json,
                        config_fingerprint)
from ..logging_utils import get_logger
from ..taxonomy import Taxonomy
from .base import (SegLayout, aggregate_metadata_parquet, iter_frames,
                  write_frame_output)

log = get_logger("segment")


def _read_rgb(path: Path) -> np.ndarray:
    import cv2
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise IOError(f"cannot read frame {path}")
    return np.ascontiguousarray(bgr[:, :, ::-1])


def run_segment(method: str, input_dir: str, cfg: Config,
                resume: bool = True, force: bool = False) -> int:
    if method == "oneformer_mapillary":
        from .oneformer import run_oneformer
        return run_oneformer(input_dir, cfg, resume=resume, force=force)
    if method != "grounded_sam2":
        raise ValueError(f"unknown method {method}")

    tax = Taxonomy.load(cfg.resolve(cfg.get("project.classes")))
    layout = SegLayout(input_dir, method)
    layout.ensure()
    frames = iter_frames(input_dir)

    fp = config_fingerprint(
        "grounded_sam2",
        cfg.get("grounded_sam2"),
        Path(cfg.resolve(cfg.get("project.grounded_prompts"))).read_text(),
        cfg.get("rectify"),
    )
    manifest = Manifest.load_or_new(layout.root / "manifest.json", "grounded_sam2", fp,
                                    meta={"num_frames": len(frames),
                                          "grounding_dino": cfg.get("grounded_sam2.grounding_dino_id"),
                                          "sam2_ckpt": str(cfg.get("grounded_sam2.sam2_ckpt"))})
    if force:
        manifest.done.clear()

    from .grounded_sam2 import GroundedSAM2Segmenter
    seg = GroundedSAM2Segmenter(cfg, tax)
    seg.load()

    write_conf = bool(cfg.get("segmentation.write_confidence", True))
    n_done = n_err = 0
    for k, ref in enumerate(frames):
        i = ref.frame_index
        if resume and not force and manifest.is_done(i) and layout.is_frame_done(i):
            n_done += 1
            continue
        try:
            img = _read_rgb(ref.rectified_path)
            out = _segment_with_oom_guard(seg, img, i, ref.capture_timestamp_ns)
            meta = write_frame_output(layout, out, write_confidence=write_conf)
            manifest.mark(i, {"coverage": out.extra.get("coverage"),
                              "num_detections": len(out.detections),
                              "total_ms": out.timings_ms.get("total")})
            n_done += 1
            if (k + 1) % 10 == 0:
                manifest.save()
                log.info("  %d/%d frames (last: %d dets, %.0fms, cov=%.2f)",
                         k + 1, len(frames), len(out.detections),
                         out.timings_ms.get("total", 0), out.extra.get("coverage", 0))
        except Exception as e:
            n_err += 1
            append_jsonl(Path(input_dir) / "logs" / "grounded_sam2_errors.jsonl",
                         {"frame_index": i, "error": repr(e)})
            log.warning("frame %d failed: %s", i, e)
    manifest.save()
    parq = aggregate_metadata_parquet(layout)
    atomic_write_json(layout.root / "summary.json",
                      {"method": "grounded_sam2", "frames_done": n_done,
                       "errors": n_err, "peak_vram_mb": _peak_vram(),
                       "metadata_parquet": str(parq) if parq else None})
    log.info("grounded_sam2 done: %d frames, %d errors -> %s", n_done, n_err, layout.root)
    return 0


def _peak_vram():
    try:
        import torch
        if torch.cuda.is_available():
            return round(torch.cuda.max_memory_allocated() / 1e6, 1)
    except Exception:
        pass
    return None


def _segment_with_oom_guard(seg, img, i, ts):
    """Retry once on CUDA OOM after clearing cache (§14)."""
    import torch
    try:
        return seg.segment(img, i, ts)
    except torch.cuda.OutOfMemoryError:
        log.warning("CUDA OOM on frame %d; clearing cache and retrying", i)
        torch.cuda.empty_cache()
        return seg.segment(img, i, ts)
