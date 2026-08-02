#!/usr/bin/env python3
"""Phase 7: solid-line crossing candidates and the human-review package.

Produces candidates in four states and a review package. No row here says a rule
was broken; that judgement needs a person, and the package exists to give them
what they need to make it.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd

from aria_drive_seg.behavior import EXPLORATORY_MARKER
from aria_drive_seg.behavior.gaze_semantics import SemanticBlock
from aria_drive_seg.behavior.solid_line import (CANDIDATE_STATES,
                                                classify_continuity,
                                                detect_candidates,
                                                observe_lane_markings, summarise,
                                                summarise_vision_offset,
                                                vision_lane_offset)
from aria_drive_seg.config import Config
from aria_drive_seg.io_utils import atomic_write_json, atomic_write_text
from aria_drive_seg.logging_utils import get_logger, setup_logging
from aria_drive_seg.taxonomy import Taxonomy

log = get_logger("behavior.solid_line")

REVIEW_FORM = """# Solid-line crossing review

One row per candidate in `review_form.csv`. Fill `human_label` and
`human_notes`; leave `human_label` empty for anything you cannot decide from the
material provided, and say why in the notes.

## Permitted values for `human_label`

| value | meaning |
|---|---|
| `compliant` | the manoeuvre was lawful (broken line, junction turn, directed around an obstacle) |
| `noncompliant` | a continuous line was crossed with no lawful reason visible |
| `uncertain` | the evidence genuinely does not settle it |
| `not_evaluable` | the material does not show what is needed (no clip, marking not visible, camera obscured) |

## Before labelling `noncompliant`

Check all of these, because each makes crossing a continuous line lawful:

* an obstacle, parked vehicle, roadworks or an emergency vehicle;
* a junction, private access or roundabout inside the manoeuvre;
* a police or marshal direction;
* the line being a different marking altogether (edge line, hatching, bus lane).

## What the automatic state means

`state` is the detector's four-way candidate label. It is **not** a verdict and
`candidate_noncompliant` in particular means only that the evidence pattern is
consistent with crossing a continuous line. The detector cannot see obstacles,
signage or instructions.

## Positioning caveat

Read `evidence_median_gps_accuracy_m` on every row. Where it exceeds the lateral
displacement a lane change produces, the crossing cannot be resolved from the
track at all and the honest label is `not_evaluable` however suggestive the
numbers look.
"""


def marking_runs_for_domain(spec, cfg, lane_marking_id: int,
                            timeline: pd.DataFrame) -> Dict[str, Any]:
    """Continuity classification for each dense frozen semantic block."""
    runs: List[Dict[str, Any]] = []
    for b in (spec.get("semantic_blocks") or []):
        root = cfg.resolve(b["root"])
        if not root.exists():
            continue
        block = SemanticBlock(root)
        frames = block.frames
        tl = timeline.set_index("frame_index")
        present = [f for f in frames if f in tl.index]
        if not present:
            continue
        ts = tl.loc[present, "timestamp_ns"].values
        obs = observe_lane_markings(block, present, ts, lane_marking_id)
        if not obs:
            continue
        interval = float(np.median(np.diff(ts)) / 1e9) if len(ts) > 1 else 0.0
        # Image-based lateral position: the only measurement in this pilot with
        # the resolution a lane crossing needs.
        vision = vision_lane_offset(block, present, ts, lane_marking_id)
        vision_summary = summarise_vision_offset(vision)
        left = classify_continuity(obs, "left", interval)
        right = classify_continuity(obs, "right", interval)
        # The more continuous side is the one a crossing would have to cross.
        chosen = left if left["duty_cycle"] >= right["duty_cycle"] else right
        runs.append({
            "start_ns": int(ts[0]), "end_ns": int(ts[-1]),
            "observed_frames": len(obs),
            "frame_interval_s": interval,
            "continuity": chosen["continuity"],
            "continuity_side": chosen["side"],
            "continuity_confidence": chosen["confidence"],
            "continuity_reason": chosen["reason"],
            "duty_cycle_left": left["duty_cycle"],
            "duty_cycle_right": right["duty_cycle"],
            "mean_marking_pixels": float(np.mean([o.marking_pixels for o in obs])),
            "vision_lateral_offset": vision_summary,
        })
    return {"runs": runs}


def build_review_package(candidates, dest: Path, cfg: Config,
                         spec_by_domain: Dict[str, Any]) -> Dict[str, Any]:
    """Clips, overlay frames and a review form for every candidate."""
    dest.mkdir(parents=True, exist_ok=True)
    rows = [c.to_dict() for c in candidates]
    df = pd.DataFrame(rows)
    if df.empty:
        df = pd.DataFrame(columns=["candidate_id", "domain", "state"])
    form = df.copy()
    form["human_label"] = ""
    form["human_notes"] = ""
    form["reviewer"] = ""
    form["review_date"] = ""
    form.to_csv(dest / "review_form.csv", index=False)
    df.to_csv(dest / "candidates.csv", index=False)
    atomic_write_text(dest / "README.md", REVIEW_FORM)
    atomic_write_json(dest / "manifest.json", {
        "schema": "article1_action_review_v1",
        "result_status": EXPLORATORY_MARKER,
        "candidates": len(rows),
        "permitted_human_labels": ["compliant", "noncompliant", "uncertain",
                                   "not_evaluable"],
        "automatic_states": list(CANDIDATE_STATES),
        "clips_present": False,
        "clip_note": ("clips and overlay frames are rendered into this directory "
                      "only for candidates whose window is covered by a dense "
                      "frozen semantic block; the media are local artefacts and "
                      "are not committed"),
    })
    return {"rows": len(rows), "path": str(dest)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/article1/behavior_analysis.yaml")
    ap.add_argument("--output", default="output/article1/behavior_analysis")
    ap.add_argument("--reports", default="reports/article1_behavior_analysis")
    ap.add_argument("--review", default="datasets/article1_action_review")
    ap.add_argument("--log-level", default="INFO")
    args = ap.parse_args()
    setup_logging(args.log_level)

    cfg = Config.load(args.config)
    scfg = cfg.get("solid_line") or {}
    taxonomy = Taxonomy.load(cfg.resolve(
        cfg.get("article1.classes", "configs/article1/classes_article1.yaml")))
    lane_id = taxonomy.id_of("lane_marking")
    out_root = Path(args.output)

    all_candidates, diags = [], {}
    for domain, spec in (cfg.get("recordings") or {}).items():
        rec_id = spec["recording_id"]
        dest = out_root / rec_id
        log.info("== %s", domain)
        matched = pd.read_parquet(dest / "map_matched.parquet")
        timeline = pd.read_parquet(dest / "multimodal_timeline.parquet")
        runs = marking_runs_for_domain(spec, cfg, lane_id, timeline)
        for r in runs["runs"]:
            log.info("  block %d frames: %s line (duty L=%.2f R=%.2f) %s",
                     r["observed_frames"], r["continuity"], r["duty_cycle_left"],
                     r["duty_cycle_right"], r["continuity_reason"] or "")
            v = r["vision_lateral_offset"]
            log.info("    vision lateral offset on %d/%d frames (%.0f%%), "
                     "p05..p95 = %s", v["frames_with_offset"], v["frames"],
                     100 * v["coverage_fraction"],
                     (f"{v['offset_m']['p05']:+.2f}..{v['offset_m']['p95']:+.2f} m"
                      if v["frames_with_offset"] else "n/a"))

        cands, diag = detect_candidates(
            matched, domain, rec_id, marking_runs=runs,
            min_evidence_sources=int(scfg.get("min_evidence_sources", 3)),
            min_lateral_displacement_m=float(
                scfg.get("min_lateral_displacement_m", 1.2)),
            min_time_beyond_line_s=float(scfg.get("min_time_beyond_line_s", 0.5)),
            junction_exclusion_radius_m=float(
                scfg.get("junction_exclusion_radius_m", 25.0)))
        s = summarise(cands, diag)
        diags[domain] = {**s, "marking_runs": runs["runs"]}
        all_candidates.extend(cands)
        log.info("  %d candidates: %s", len(cands), s["states"])
        if s.get("resolution_caveat"):
            log.warning("  %s", s["resolution_caveat"])

    reports = Path(args.reports) / "solid_line"
    reports.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame([c.to_dict() for c in all_candidates])
    df.to_csv(reports / "solid_line_candidates.csv", index=False)

    pkg = build_review_package(all_candidates, Path(args.review), cfg,
                               cfg.get("recordings") or {})

    report = {
        "schema": "article1_solid_line_candidates_v1",
        "result_status": EXPLORATORY_MARKER,
        "permitted_states": list(CANDIDATE_STATES),
        "definitive_label_requires_human_review": True,
        "classifier_training": {
            "attempted": False,
            "reason": ("no reviewed labels exist yet; training is gated on at "
                       "least 20 confirmed events per class and the pipeline "
                       "refuses to train below that"),
        },
        "detection_parameters": scfg,
        "domains": diags,
        "review_package": pkg,
    }
    atomic_write_json(reports / "solid_line_summary.json", report)
    print(json.dumps({d: {"candidates": v["candidates"], "states": v["states"],
                          "resolvable": v.get("displacement_is_resolvable")}
                      for d, v in diags.items()}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
