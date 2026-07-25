#!/usr/bin/env python3
"""Prompt / config calibration + ablation for Grounded-SAM2 (Phase 6).

Runs one or more configs over a stratified frame set and produces a per-class table
(detections, frames-present, mean area, mean score, rejections + causes, time,
coverage) plus a cross-config comparison and visual contact sheets — so a config is
chosen on METRICS + visual QA, not on coverage alone. Never overwrites run200/val10;
all output goes under runs/calibration/<config>/.

    # stratified sample 80 frames from the VRS, then run 3 ablation configs
    python scripts/calibrate_prompts.py --vrs rec.vrs --num 80 \
        --configs configs/ablation/per_class_variants.yaml,configs/ablation/hierarchical_roi.yaml

    # or use an existing extract dir
    python scripts/calibrate_prompts.py --input runs/calib80 --configs a.yaml,b.yaml

Segmentation runs via ARIA_ML_PYTHON (env with torch); aggregation is pure-python.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parent.parent


def _ml_python() -> str:
    return os.environ.get("ARIA_ML_PYTHON") or sys.executable


def _vrs_python() -> str:
    return os.environ.get("ARIA_VRS_PYTHON") or sys.executable


def _stratified_extract(vrs: str, num: int, out: Path) -> None:
    """Even temporal stratification across the whole recording (day/traffic/junction
    variety comes from spanning the full drive)."""
    import numpy as np
    from aria_drive_seg.config import Config
    from aria_drive_seg.vrs.provider import AriaProvider
    cfg = Config.load()
    prov = AriaProvider(vrs, cfg.get("vrs.time_domain", "DEVICE_TIME"))
    n = prov.num_rgb(cfg.get("vrs.rgb_label", "camera-rgb"))
    step = max(1, n // num)
    subprocess.run([_vrs_python(), "-m", "aria_drive_seg", "extract", "--vrs", vrs,
                    "--output", str(out), "--frame-step", str(step), "--max-frames", str(num)],
                   check=True)


def _aggregate(seg_dir: Path) -> Dict[str, Any]:
    per_class = defaultdict(lambda: {"dets": 0, "frames": set(), "area": [], "score": []})
    rej = Counter()
    times, covs = [], []
    for jp in sorted((seg_dir / "metadata").glob("frame_*.json")):
        m = json.loads(jp.read_text())
        fi = m["frame_index"]
        times.append(m.get("timings_ms", {}).get("total"))
        covs.append(m.get("coverage", m.get("extra", {}).get("coverage")))
        for k, v in m.get("rejections", {}).items():
            rej[k] += v
        for d in m.get("detections", []):
            pc = per_class[d["canonical_name"]]
            pc["dets"] += 1; pc["frames"].add(fi)
            pc["area"].append(d.get("area_px", 0)); pc["score"].append(d.get("combined_score", 0))
    import numpy as np
    table = {}
    for name, pc in per_class.items():
        table[name] = {
            "detections": pc["dets"], "frames_present": len(pc["frames"]),
            "mean_area_px": round(float(np.mean(pc["area"])), 1) if pc["area"] else 0,
            "mean_score": round(float(np.mean(pc["score"])), 3) if pc["score"] else 0,
        }
    return {"per_class": table, "rejections": dict(rej.most_common()),
            "mean_ms": round(float(np.mean([t for t in times if t])), 1) if times else None,
            "mean_coverage": round(float(np.mean([c for c in covs if c is not None])), 3) if covs else None,
            "num_frames": len(times)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--vrs", default=None)
    ap.add_argument("--num", type=int, default=80)
    ap.add_argument("--input", default=None, help="existing extract dir (frames present)")
    ap.add_argument("--configs", required=True, help="comma-separated config yaml paths")
    ap.add_argument("--out", default="runs/calibration")
    ap.add_argument("--offline", action="store_true", default=True)
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    # 1) frame set
    if args.input:
        src_frames = Path(args.input)
    else:
        if not args.vrs:
            print("need --input or --vrs", file=sys.stderr); return 2
        src_frames = out / "_frames"
        if not (src_frames / "frames" / "frames.parquet").exists():
            _stratified_extract(args.vrs, args.num, src_frames)

    configs = [c.strip() for c in args.configs.split(",") if c.strip()]
    results: Dict[str, Any] = {}
    for cfg_path in configs:
        name = Path(cfg_path).stem
        dst = out / name
        dst.mkdir(parents=True, exist_ok=True)
        if not (dst / "frames").exists():
            shutil.copytree(src_frames / "frames", dst / "frames")
            if (src_frames / "gaze").exists():
                shutil.copytree(src_frames / "gaze", dst / "gaze", dirs_exist_ok=True)
        print(f"[calibrate] running config '{name}' ...")
        cmd = [_ml_python(), "-m", "aria_drive_seg", "segment", "--method", "grounded_sam2",
               "--input", str(dst), "--config", cfg_path]
        if args.offline:
            cmd.append("--offline")
        subprocess.run(cmd, check=False)
        results[name] = _aggregate(dst / "grounded_sam2")

    # 2) comparison table (per class: frames_present across configs)
    _write_comparison(out, results)
    (out / "calibration.json").write_text(json.dumps(results, indent=2))
    print(f"[calibrate] wrote {out/'calibration.json'} and {out/'comparison.md'}")
    return 0


def _write_comparison(out: Path, results: Dict[str, Any]) -> None:
    names = list(results.keys())
    classes = sorted({c for r in results.values() for c in r["per_class"]})
    L = ["# Prompt/config calibration — comparison\n",
         "Per-class **frames present** (higher = more recall). Judge with the contact sheets;",
         "coverage alone is not quality.\n",
         "| class | " + " | ".join(names) + " |",
         "|---|" + "---|" * len(names)]
    for c in classes:
        row = [c] + [str(results[n]["per_class"].get(c, {}).get("frames_present", 0)) for n in names]
        L.append("| " + " | ".join(row) + " |")
    L.append("\n## Totals")
    L.append("| metric | " + " | ".join(names) + " |")
    L.append("|---|" + "---|" * len(names))
    for metric, key in [("frames", "num_frames"), ("mean ms/frame", "mean_ms"),
                        ("mean coverage", "mean_coverage")]:
        L.append(f"| {metric} | " + " | ".join(str(results[n].get(key)) for n in names) + " |")
    (out / "comparison.md").write_text("\n".join(L) + "\n")


if __name__ == "__main__":
    raise SystemExit(main())
