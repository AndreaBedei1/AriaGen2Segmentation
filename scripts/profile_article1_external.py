#!/usr/bin/env python3
"""Profile P0–P3 Article 1 external inference on the same small frame set."""
from __future__ import annotations

import argparse
import json
import tempfile
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from aria_drive_seg.article1.external import Article1Mapper, filter_thin_markings
from aria_drive_seg.config import Config
from aria_drive_seg.hashing import stable_hash
from aria_drive_seg.io_utils import atomic_write_json, atomic_write_text
from aria_drive_seg.segmentation.oneformer import load_verified_mask2former
from aria_drive_seg.taxonomy import Taxonomy


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--config", default="configs/article1/external_segmentation.yaml")
    ap.add_argument("--output", default="reports")
    args = ap.parse_args()
    import torch
    import torch.nn.functional as F
    from transformers import AutoImageProcessor

    cfg, root, out = Config.load(args.config), Path(args.input), Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    model_path = cfg.resolve(cfg.get("oneformer_mapillary.mask2former_id"))
    proc = AutoImageProcessor.from_pretrained(model_path)
    model = load_verified_mask2former(str(model_path)).to("cuda").eval()
    tax = Taxonomy.load(cfg.resolve(cfg.get("article1.classes")))
    mapper = Article1Mapper(model.config.id2label,
                            cfg.resolve(cfg.get("article1.mapillary_mapping")), tax)
    frames = pd.read_parquet(root / "frames/frames.parquet").sort_values("frame_index")
    gaze_path = root / "gaze/aligned_gaze.parquet"
    gaze = pd.read_parquet(gaze_path).set_index("frame_index") if gaze_path.exists() else None
    profiles = {
        "P0_current": {"scale": 1.0, "crop": None, "gaze_assisted": False},
        "P1_reduced": {"scale": .35, "crop": None, "gaze_assisted": False},
        "P2_reduced_road_crop": {"scale": .35, "crop": "road", "gaze_assisted": False},
        "P3_reduced_gaze_crop": {"scale": .35, "crop": "gaze", "gaze_assisted": True},
    }
    records, p0_masks = {k: [] for k in profiles}, {}

    def infer(rgb, scale, no_default_resize=False):
        if scale != 1:
            rgb = cv2.resize(rgb, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        t = time.perf_counter()
        inputs = proc(images=rgb, return_tensors="pt", do_resize=not no_default_resize)
        preprocess = (time.perf_counter() - t) * 1000
        inputs = {k: v.to("cuda") for k, v in inputs.items()}
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        t = time.perf_counter()
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            result = model(**inputs)
        torch.cuda.synchronize()
        forward = (time.perf_counter() - t) * 1000
        cq = result.class_queries_logits.float().softmax(-1)[..., :-1]
        mq = result.masks_queries_logits.float().sigmoid()
        sem = torch.einsum("bqc,bqhw->bchw", cq, mq)[0]
        sem = sem / sem.sum(0, keepdim=True).clamp_min(1e-12)
        native = sem.cpu().numpy()
        torch.cuda.synchronize()
        return native, preprocess, forward, torch.cuda.max_memory_allocated() / 1e6

    for _, row in frames.iterrows():
        fi = int(row.frame_index)
        t = time.perf_counter()
        rgb = cv2.cvtColor(cv2.imread(str(root / row.rectified_path)), cv2.COLOR_BGR2RGB)
        load_ms = (time.perf_counter() - t) * 1000
        h, w = rgb.shape[:2]
        for name, spec in profiles.items():
            start = time.perf_counter()
            native, pre_ms, fw_ms, vram = infer(rgb, spec["scale"], spec["scale"] != 1)
            native = np.moveaxis(cv2.resize(np.moveaxis(native, 0, -1), (w, h),
                                            interpolation=cv2.INTER_LINEAR), -1, 0)
            crop_ms = 0.0
            if spec["crop"] == "road":
                y0, x0, x1 = h // 2, 0, w
                t = time.perf_counter()
                crop, _, crop_fw, crop_vram = infer(rgb[y0:, x0:x1], 1.0, True)
                crop = np.moveaxis(cv2.resize(np.moveaxis(crop, 0, -1), (x1-x0, h-y0),
                                              interpolation=cv2.INTER_LINEAR), -1, 0)
                native[:, y0:, x0:x1] = crop
                crop_ms = (time.perf_counter() - t) * 1000
                vram = max(vram, crop_vram)
            elif spec["crop"] == "gaze" and gaze is not None and fi in gaze.index:
                g = gaze.loc[fi]
                if bool(g.valid) and np.isfinite(g.rect_u) and np.isfinite(g.rect_v):
                    size = 512; cx, cy = int(g.rect_u), int(g.rect_v)
                    x0, y0 = max(0, cx-size//2), max(0, cy-size//2)
                    x1, y1 = min(w, x0+size), min(h, y0+size)
                    t = time.perf_counter()
                    crop, _, _, crop_vram = infer(rgb[y0:y1, x0:x1], 1.0, True)
                    crop = np.moveaxis(cv2.resize(np.moveaxis(crop, 0, -1), (x1-x0, y1-y0),
                                                  interpolation=cv2.INTER_LINEAR), -1, 0)
                    native[:, y0:y1, x0:x1] = crop
                    crop_ms = (time.perf_counter() - t) * 1000
                    vram = max(vram, crop_vram)
            t = time.perf_counter()
            agg = mapper.aggregate(native, 0, None, cfg.get("article1.unknown"))
            aggregate_ms = (time.perf_counter() - t) * 1000
            t = time.perf_counter()
            thin = filter_thin_markings(agg["mask"], agg["probabilities"],
                                        cfg.get("article1.thin_markings"))
            thin_ms = (time.perf_counter() - t) * 1000
            t = time.perf_counter()
            with tempfile.NamedTemporaryFile(suffix=".npz") as handle:
                np.savez_compressed(handle, probabilities=native.astype(np.float16))
            save_native_ms = (time.perf_counter() - t) * 1000
            t = time.perf_counter()
            with tempfile.NamedTemporaryFile(suffix=".npz") as handle:
                np.savez_compressed(handle, probabilities=agg["probabilities"].astype(np.float16))
            save_macro_ms = (time.perf_counter() - t) * 1000
            t = time.perf_counter()
            _ = cv2.addWeighted(rgb, .52, tax.colorize(thin["composite"]), .48, 0)
            render_ms = (time.perf_counter() - t) * 1000
            mask = thin["composite"]
            if name == "P0_current":
                p0_masks[fi] = mask
                agreement = 1.0
            else:
                agreement = float((p0_masks[fi] == mask).mean())
            records[name].append({
                "frame_index": fi, "load_image_ms": load_ms, "preprocess_ms": pre_ms,
                "forward_ms": fw_ms, "crop_total_ms": crop_ms,
                "aggregation_ms": aggregate_ms, "thin_ms": thin_ms,
                "save_native_probabilities_ms": save_native_ms,
                "save_macro_probabilities_ms": save_macro_ms, "render_ms": render_ms,
                "total_profiled_ms": (time.perf_counter() - start) * 1000,
                "peak_vram_mb": vram, "unknown_rate": float((mask == 0).mean()),
                "thin_pixels": int((thin["filtered_thin_mask"] > 0).sum()),
                "pixel_agreement_with_P0": agreement,
                "class_histogram": np.bincount(mask.ravel(), minlength=14).tolist(),
            })
    summary = {"disclaimer": "Pre-GT speed/behavior profiling; not accuracy.",
               "config_fingerprint": stable_hash(cfg.get("article1")), "profiles": {}}
    for name, rows in records.items():
        numeric = [k for k, v in rows[0].items() if isinstance(v, (int, float)) and k != "frame_index"]
        summary["profiles"][name] = {f"mean_{k}": float(np.mean([r[k] for r in rows]))
                                     for k in numeric}
        summary["profiles"][name]["gaze_assisted"] = profiles[name]["gaze_assisted"]
    atomic_write_json(out / "article1_external_profile.json",
                      {"summary": summary, "per_frame": records})
    lines = ["# Article 1 external profiling", "",
             "**Pre-GT speed/behavior profile; not an accuracy evaluation.**", "",
             "| profile | total s/frame | forward s | VRAM MB | unknown | thin px | gaze-assisted |",
             "|---|---:|---:|---:|---:|---:|---|"]
    for name, s in summary["profiles"].items():
        lines.append(f"| {name} | {s['mean_total_profiled_ms']/1000:.3f} | "
                     f"{s['mean_forward_ms']/1000:.3f} | {s['mean_peak_vram_mb']:.1f} | "
                     f"{s['mean_unknown_rate']:.2%} | {s['mean_thin_pixels']:.0f} | "
                     f"{s['gaze_assisted']} |")
    lines += ["", "P2 uses a high-resolution lower road crop. P3 is explicitly gaze-assisted "
              "and is not eligible for unbiased global comparison. No final profile is selected."]
    atomic_write_text(out / "article1_external_profile.md", "\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
