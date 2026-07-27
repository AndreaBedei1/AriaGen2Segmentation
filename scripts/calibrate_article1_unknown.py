#!/usr/bin/env python3
"""Pre-GT unknown-policy ablation from saved native probability maps."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import yaml

from aria_drive_seg.article1.external import Article1Mapper
from aria_drive_seg.config import Config, _deep_merge
from aria_drive_seg.io_utils import atomic_write_json, atomic_write_text
from aria_drive_seg.taxonomy import Taxonomy


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--source-subdir", default="article1_external")
    ap.add_argument("--config", default="configs/article1/external_segmentation.yaml")
    ap.add_argument("--output", default="reports")
    args = ap.parse_args()
    root, out = Path(args.input), Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    cfg = Config.load(args.config)
    tax = Taxonomy.load(cfg.resolve(cfg.get("article1.classes")))
    id2 = json.loads((cfg.resolve(cfg.get("oneformer_mapillary.mask2former_id")) /
                      "config.json").read_text())["id2label"]
    mapper = Article1Mapper(id2, cfg.resolve(cfg.get("article1.mapillary_mapping")), tax)
    names = ("permissive", "balanced", "conservative")
    policies = {
        name: yaml.safe_load((cfg.project_root / f"configs/article1/unknown_{name}.yaml").read_text())
        ["article1"]["unknown"] for name in names
    }
    accum = {n: {"unknown": [], "reasons": {}} for n in names}
    paths = sorted((root / args.source_subdir / "native_probabilities").glob("*.npz"))
    for path in paths:
        native = np.load(path)["probabilities"].astype(np.float32)
        for name, policy in policies.items():
            result = mapper.aggregate(native, 0, None, policy)
            accum[name]["unknown"].append(float((result["mask"] == 0).mean()))
            values, counts = np.unique(result["unknown_reason"], return_counts=True)
            for value, count in zip(values, counts):
                accum[name]["reasons"][str(int(value))] = (
                    accum[name]["reasons"].get(str(int(value)), 0) + int(count))
    report = {"disclaimer": "Pre-GT behavior ablation; not an accuracy evaluation.",
              "frames": len(paths), "profiles": {}}
    for name in names:
        report["profiles"][name] = {
            "policy": policies[name],
            "mean_unknown_rate": float(np.mean(accum[name]["unknown"])),
            "min_unknown_rate": float(np.min(accum[name]["unknown"])),
            "max_unknown_rate": float(np.max(accum[name]["unknown"])),
            "unknown_reason_bitmask_counts": accum[name]["reasons"],
        }
    atomic_write_json(out / "article1_unknown_precalibration.json", report)
    lines = ["# Article 1 unknown pre-calibration", "",
             "**This is a behavior ablation, not an accuracy evaluation.** No profile is "
             "selected as optimal before reviewed Article 1 GT.", "",
             "| profile | mean unknown | min | max |", "|---|---:|---:|---:|"]
    for name, row in report["profiles"].items():
        lines.append(f"| {name} | {row['mean_unknown_rate']:.3%} | "
                     f"{row['min_unknown_rate']:.3%} | {row['max_unknown_rate']:.3%} |")
    lines += ["", "Reason maps use a bitmask: 1 low probability, 2 low margin, "
              "4 high normalized entropy, 8 unsupported dominant native class.",
              "The balanced profile is used only to produce checkpoint-2 diagnostics."]
    atomic_write_text(out / "article1_unknown_precalibration.md", "\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
