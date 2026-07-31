#!/usr/bin/env python3
"""Generate REPORT_ANNOTATION_DATASET.md from the selection and the package."""
from __future__ import annotations

import argparse
import datetime as dt
import json
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reports", default="reports/article1_motorcycle_ingestion")
    ap.add_argument("--package", default="datasets/article1_annotation_package")
    args = ap.parse_args()

    reports, package = Path(args.reports), Path(args.package)
    selection = json.loads((reports / "annotation_selection.json").read_text())
    manifest = json.loads((package / "manifest.json").read_text()) \
        if (package / "manifest.json").exists() else {}
    labels = json.loads((package / "labels.json").read_text()) \
        if (package / "labels.json").exists() else []

    counts = selection["counts"]
    stats = selection["per_domain_group_stats"]
    coverage: Dict[str, int] = selection["stratum_coverage"]

    L: List[str] = [
        "# Article 1 — annotation dataset and CVAT package",
        "",
        f"Generated: {dt.datetime.now(dt.timezone.utc).isoformat()}",
        "",
        "## Purpose",
        "",
        "This is the first reviewed ground truth the project will have. It is built "
        "to support five things at once: validating the external scene model, "
        "training the shared car/motorcycle cockpit model, evaluating mirrors, "
        "instrument displays and controls, evaluating hands where they are actually "
        "visible, and measuring the domain shift between the two vehicles.",
        "",
        "Nothing in the package is ground truth yet. The pre-annotations are "
        "automatic model output, marked as such in the directory name, the manifest "
        "and the guide.",
        "",
        "## How the frames were chosen",
        "",
        "Not randomly. Selection is a greedy maximisation of stratum coverage: each "
        "pick is the candidate that adds the most conditions not yet represented, so "
        "rare situations survive instead of being drowned by ordinary driving.",
        "",
        "### Balancing",
        "",
    ] + [f"- {r}" for r in selection.get("quota_rationale", [])] + [
        "",
        "| domain | pool | selected |",
        "|---|---:|---:|",
    ]
    for domain, size in selection.get("pool_sizes", {}).items():
        L.append(f"| {domain} | {size} | {counts['per_domain'].get(domain, 0)} |")
    L += ["",
          f"**Total: {counts['total']} frames, "
          + ", ".join(f"{v} {k}" for k, v in counts["per_domain"].items())
          + ".** The two domains receive the same budget even though the motorcycle "
          "recording is 2.7x longer and sampled 1.5x more densely.",
          ""]

    L += ["### Groups", "",
          "| domain | group | requested | selected | note |",
          "|---|---|---:|---:|---|"]
    for domain, groups in stats.items():
        for group, s in groups.items():
            L.append(f"| {domain} | `{group}` | {s['requested']} | {s['selected']} | "
                     f"{s['shortfall_reason'] or ''} |")
    L += ["",
          "- `external_validation` — validates the Mask2Former external scene "
          "against human annotation",
          "- `cockpit_training` — trains the shared cockpit model; restricted to "
          "frames where ego structure is actually present, because a frame with no "
          "cockpit teaches that model nothing",
          "- `failure_mode_review` — frames where the frozen baseline behaved "
          "unusually; only available where the baseline actually ran",
          ""]

    dedup = selection["deduplication"]
    L += ["### Avoiding near duplicates", "",
          f"- minimum temporal separation: {dedup['min_temporal_separation_s']} s",
          f"- minimum perceptual-hash distance: "
          f"{dedup['min_perceptual_hash_distance']} of 64 bits",
          f"- visual deduplication window: {dedup.get('visual_dedup_window_s')} s",
          "",
          dedup.get("rule", ""),
          "",
          "The temporal separation is the binding constraint on the car, whose "
          "recording is only 376 s long. Quotas were sized so that the shorter "
          "recording can meet them; asking for more would have quietly unbalanced "
          "the dataset in the motorcycle's favour.",
          ""]

    L += ["## What the selection covers", "",
          f"{len(coverage)} distinct strata are represented.", "",
          "| stratum | frames |", "|---|---:|"]
    for stratum, n in sorted(coverage.items(), key=lambda kv: (-kv[1], kv[0])):
        L.append(f"| `{stratum}` | {n} |")
    L += ["", selection.get("uncovered_note", ""), ""]

    if manifest:
        L += ["## Package contents", "",
              f"- items: {manifest['counts']['items']}",
              f"- per domain: {manifest['counts']['per_domain']}",
              f"- per group: {manifest['counts']['per_group']}",
              ""]
        pre = manifest.get("preannotations", {})
        L += ["### Pre-annotations", "",
              f"- present: {pre.get('present')}",
              f"- status: `{pre.get('status')}`",
              f"- source: {pre.get('source')}",
              f"- instruction: {pre.get('instruction')}",
              "",
              "They live in `preannotations_not_ground_truth/`, in CVAT "
              "`Segmentation mask 1.1` layout, with their own README repeating that "
              "they are automatic output.",
              ""]
        git = manifest.get("git_policy", {})
        L += ["### What is committed", "",
              "Committed: " + ", ".join(f"`{p}`" for p in git.get("committed", [])),
              "",
              "Local only: " + ", ".join(f"`{p}`" for p in git.get("local_only", []))
              + ". Full-resolution images and mask PNGs stay out of git; the manifest "
              "carries a SHA-256 for each so the local copy can be verified.",
              ""]

    if labels:
        cockpit = [l for l in labels
                   if l["name"] in ("mirror", "instrument_display",
                                    "control_and_ego_vehicle")]
        L += ["## Label specification", "",
              f"{len(labels)} labels, ids 0-13, exported as CVAT mask labels.",
              "",
              "| id | label | attributes |", "|---:|---|---:|"]
        for l in labels:
            L.append(f"| {l['id']} | `{l['name']}` | {len(l['attributes'])} |")
        L += ["",
              f"The three cockpit labels carry the full hand attribute vocabulary "
              f"({len(cockpit[0]['attributes']) if cockpit else 0} attributes each): "
              "per-side `visible`, `partially_visible`, `occluded`, `out_of_frame`, "
              "`motion_blurred` and `arm_visible`, plus "
              "`hand_tracking_available`, `hand_tracking_valid` and "
              "`hand_segmentation_proxy_available`.",
              "",
              "Hands are **not** a class. They are painted as "
              "`control_and_ego_vehicle` (12) and described by those attributes.",
              ""]

    L += [
        "## Rules the guide fixes for the annotators",
        "",
        "The full text is `ANNOTATION_GUIDE.md` inside the package. The decisions "
        "that most affect the resulting ground truth:",
        "",
        "- **`unknown` may not survive.** It exists in the tool as a parking place "
        "for an unresolved region during the session; a delivered frame must contain "
        "none. Unresolvable regions become `other_environment` with "
        "`ambiguity_flag` set.",
        "- **`mirror`** is the reflecting surface plus housing and stalk when they "
        "read as one object. What is *inside* the mirror is mirror, not the class of "
        "the reflected object.",
        "- **`instrument_display`** is the readable or emissive area and its "
        "inseparable bezel; the surrounding binnacle is `control_and_ego_vehicle`.",
        "- **`control_and_ego_vehicle`** covers the wheel, the handlebar, levers, "
        "grips, switchgear, the non-display dashboard, fairing, tank and interior "
        "structure — and the visible hands and forearms of the driver or rider.",
        "- **`other_environment`** is for pixels genuinely not assignable to 1-12, "
        "not a dumping ground for difficult ones.",
        "- **Thin markings** follow the painted extent; a dashed line is reconnected "
        "only where paint is visible, and a line is never thickened to make it "
        "easier to see.",
        "- **Hands** are never invented and never propagated from a neighbouring "
        "frame. On the motorcycle an absent hand is the normal case.",
        "",
        "## Limits",
        "",
        "- The car recording is a provisional 10 fps baseline. Frames drawn from it "
        "are still useful cockpit and external training material, but a definitive "
        "paired comparison needs the re-recorded car.",
        "- The candidate pool is a 1 Hz scouting subsample, so the finest temporal "
        "structure of either recording is not represented in the annotation set. "
        "That is deliberate: annotating near-adjacent frames buys little and costs "
        "reviewer time.",
        "- Class expectations attached to each frame come from the external model "
        "and are a hint for the annotator, not a target to reproduce.",
        "",
        "## Next step",
        "",
        "Human review. When it is delivered, the reviewer drops "
        "`REVIEWED_ANNOTATIONS.json` into the dataset directory and the cockpit "
        "training gate opens. Nothing downstream may proceed before that.",
        "",
    ]

    out = reports / "REPORT_ANNOTATION_DATASET.md"
    out.write_text("\n".join(L) + "\n")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
