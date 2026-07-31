#!/usr/bin/env python3
"""Sparse external-model scouting pass over the scouting frame subsample.

Runs the frozen Mask2Former external stage on the 1 Hz scouting frames and stores
per-frame Article 1 class pixel fractions. These are **proxy statistics used to
rank candidate segments and annotation frames**, not scientific results and not
ground truth: no accuracy figure is derived from them anywhere.

Runs in the ML environment (torch + transformers).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from aria_drive_seg.article1.external import Article1Mapper
from aria_drive_seg.config import Config
from aria_drive_seg.logging_utils import get_logger, setup_logging
from aria_drive_seg.segmentation.oneformer import OneFormerMapillarySegmenter
from aria_drive_seg.taxonomy import Taxonomy

log = get_logger("scout.semantics")


def _scout_native_mask(segmenter, image_rgb: np.ndarray) -> np.ndarray:
    """Dominant native Mapillary class per pixel, at the model's input resolution.

    The production path upsamples the full C x H x W probability tensor to the
    native image size before the argmax. For 66 classes at 2016 x 1512 that is a
    ~800 MB float tensor per frame and it dominates the runtime, without adding
    information: the processor resizes every image to a fixed 384 x 384 square, so
    384 x 384 is the finest geometry the model actually produces.

    The resize is anisotropic but is a pure scaling of both axes with no padding
    (the returned `pixel_mask` is fully valid), so relative class *areas* are
    preserved exactly. That is all a scouting ranking needs.

    This is a proxy statistic. It is never written into a mask, a metric or a
    result.
    """
    import torch
    import torch.nn.functional as F
    from PIL import Image

    inputs = segmenter._proc(images=Image.fromarray(image_rgb), return_tensors="pt")
    target_hw = tuple(inputs["pixel_values"].shape[-2:])
    inputs = {k: (v.to(segmenter.device) if hasattr(v, "to") else v)
              for k, v in inputs.items()}
    with torch.inference_mode(), torch.autocast(
            segmenter.device, dtype=segmenter._amp_dtype,
            enabled=segmenter.device == "cuda"):
        outputs = segmenter._model(**inputs)
    class_queries = outputs.class_queries_logits.float().softmax(dim=-1)[..., :-1]
    mask_queries = F.interpolate(outputs.masks_queries_logits.float(),
                                 size=target_hw, mode="bilinear",
                                 align_corners=False).sigmoid()
    scores = torch.einsum("bqc,bqhw->bchw", class_queries, mask_queries)[0]
    return scores.argmax(dim=0).cpu().numpy()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", required=True, help="scouting run directory")
    ap.add_argument("--output", required=True, help="output .npz")
    ap.add_argument("--config", default="configs/article1/external.yaml")
    ap.add_argument("--max-size", type=int, default=1024,
                    help="longest side for the scouting pass (speed over detail)")
    ap.add_argument("--log-level", default="INFO")
    args = ap.parse_args()
    setup_logging(args.log_level)

    cfg = Config.load(args.config)
    cfg.set("oneformer_mapillary.input_max_size", args.max_size)
    article_cfg = cfg.get("article1")
    taxonomy = Taxonomy.load(cfg.resolve(article_cfg["classes"]))

    root = Path(args.frames)
    frames = pd.read_parquet(root / "frames" / "frames.parquet") \
        .sort_values("source_frame_index")
    frames = frames[frames["valid"]]
    log.info("scouting %d frames from %s", len(frames), root)

    segmenter = OneFormerMapillarySegmenter(cfg, taxonomy)
    segmenter.load()
    mapper = Article1Mapper(segmenter._model.config.id2label,
                            cfg.resolve(article_cfg["mapillary_mapping"]), taxonomy)

    import cv2
    names = taxonomy.names() + ["mapillary_ego_region"]
    rows, fractions = [], []
    ego_ids = list(mapper.mapillary_ego_ids)
    for n, (_, row) in enumerate(frames.iterrows()):
        bgr = cv2.imread(str(root / row["rectified_path"]), cv2.IMREAD_COLOR)
        if bgr is None:
            log.warning("unreadable frame %s", row["rectified_path"])
            continue
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        native_mask = _scout_native_mask(segmenter, rgb)
        article_mask = mapper.native_to_article[native_mask]
        ego = np.isin(native_mask, ego_ids)

        total = article_mask.size
        frac = [float(np.count_nonzero(article_mask == taxonomy.id_of(c)) / total)
                for c in taxonomy.names()]
        frac.append(float(np.count_nonzero(ego) / total))
        fractions.append(frac)
        rows.append({
            "recording_id": row["recording_id"],
            "domain": row["domain"],
            "source_frame_index": int(row["source_frame_index"]),
            "timestamp_ns": int(row["timestamp_ns"]),
            "timestamp_s": float(row["timestamp_s"]),
        })
        if (n + 1) % 50 == 0:
            log.info("  %d/%d", n + 1, len(frames))

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out,
        class_names=np.array(names),
        class_fraction=np.array(fractions, dtype=np.float32),
        timestamp_ns=np.array([r["timestamp_ns"] for r in rows], dtype=np.int64),
        source_frame_index=np.array([r["source_frame_index"] for r in rows],
                                    dtype=np.int64),
        recording_id=np.array(rows[0]["recording_id"] if rows else ""),
        domain=np.array(rows[0]["domain"] if rows else ""),
        note=np.array("proxy scouting statistics; not ground truth, not a result"),
    )
    pd.DataFrame(rows).assign(**{
        n: [f[i] for f in fractions] for i, n in enumerate(names)
    }).to_csv(out.with_suffix(".csv"), index=False)
    log.info("wrote %s (%d frames)", out, len(rows))
    print(json.dumps({"frames": len(rows), "output": str(out)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
