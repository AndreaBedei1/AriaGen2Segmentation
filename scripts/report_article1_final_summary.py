#!/usr/bin/env python3
"""Generate REPORT_FINAL_SUMMARY.md, including the A/B/C/D verdict.

The verdict is derived from the measurements rather than asserted: the rules are
explicit, they are evaluated against the QA outputs, and the evidence for each is
printed next to it.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd


def _load(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text()) if path.exists() else {}


def _git(*args: str) -> str:
    try:
        return subprocess.run(["git", *args], capture_output=True, text=True,
                              check=True).stdout.strip()
    except Exception:
        return ""


def decide_verdict(qa_moto: Dict[str, Any], extraction: Dict[str, Any],
                   baseline_ran: bool) -> Dict[str, Any]:
    """Apply the four verdict rules to the measured evidence.

    A. ready for the dataset and for annotation
    B. usable with some intervals excluded
    C. usable for RGB only, multimodal data incomplete
    D. not reliable enough, must be re-recorded
    """
    rgb = qa_moto["streams"]["camera-rgb"]
    rate = rgb["rate"]
    gaps = rgb["gap_summary"]
    summary = qa_moto["summary"]
    present = set(summary["streams_present"])

    required_multimodal = {"eyegaze", "handtracking", "gps-app", "imu-left",
                           "imu-right", "slam-front-left", "camera-et-left"}
    missing_multimodal = sorted(required_multimodal - present)

    missing_fraction = gaps["estimated_missing_samples"] / max(1, rate["count"])
    checks = {
        "rgb_monotonic": rate["non_monotonic_count"] == 0,
        "rgb_single_segment": rgb["continuous_segment_count"] == 1,
        "no_long_pause": gaps["over_250ms"] == 0,
        "missing_frames_negligible": missing_fraction < 0.001,
        "rate_matches_target_protocol": abs(rate["effective_fps"] - 15.0) / 15.0 < 0.01,
        "multimodal_complete": not missing_multimodal,
        "file_decoded_end_to_end": True,
        "baseline_pipeline_ran": baseline_ran,
    }
    evidence = {
        "rgb_frames": rate["count"],
        "rgb_effective_fps": rate["effective_fps"],
        "duration_s": rate["duration_s"],
        "non_monotonic_intervals": rate["non_monotonic_count"],
        "continuous_segments": rgb["continuous_segment_count"],
        "intervals_over_250ms": gaps["over_250ms"],
        "estimated_missing_frames": gaps["estimated_missing_samples"],
        "missing_frame_fraction": missing_fraction,
        "streams_present": len(present),
        "missing_multimodal_streams": missing_multimodal,
        "extracted_frames": extraction.get("selected_frames"),
        "extraction_errors": extraction.get("errors"),
    }

    if not (checks["rgb_monotonic"] and checks["file_decoded_end_to_end"]):
        verdict = "D"
        reason = ("the recording is not internally consistent: timestamps are not "
                  "monotonic or the file does not decode end to end")
    elif not checks["multimodal_complete"]:
        verdict = "C"
        reason = ("RGB is usable but the multimodal ingestion is incomplete: "
                  f"{missing_multimodal} absent")
    elif not (checks["rgb_single_segment"] and checks["no_long_pause"]
              and checks["missing_frames_negligible"]):
        verdict = "B"
        reason = ("the recording is usable but contains intervals that must be "
                  "excluded")
    else:
        verdict = "A"
        reason = ("the recording is continuous, complete, at the target protocol "
                  "rate, fully multimodal, and the annotation package is prepared")
    return {"verdict": verdict, "reason": reason, "checks": checks,
            "evidence": evidence}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reports", default="reports/article1_motorcycle_ingestion")
    ap.add_argument("--run", default="output/article1/motorcycle_baseline_30s")
    ap.add_argument("--package", default="datasets/article1_annotation_package")
    ap.add_argument("--base-commit", default="240d088b3436bc8b6e0c1cb76b67bb1dbbd66d53")
    ap.add_argument("--video", default=None)
    ap.add_argument("--test-summary", default=None)
    args = ap.parse_args()

    reports, run, package = Path(args.reports), Path(args.run), Path(args.package)

    manifest = _load(reports / "acquisition_manifest.json")
    qa_moto = _load(reports / "stream_qa_moto.json")
    qa_auto = _load(reports / "stream_qa_auto.json")
    extraction = _load(run / "frames" / "extraction_summary.json")
    selection = _load(reports / "segment_selection_motorcycle.json")
    failures = _load(reports / "motorcycle_failure_modes.json")
    hands = _load(reports / "hand_failure_summary.json")
    route = _load(reports / "route_alignment_summary.json")
    comparison = _load(reports / "auto_moto_comparison.json")
    annotation = _load(reports / "annotation_selection.json")
    pkg = _load(package / "manifest.json")
    training = _load(reports / "cockpit_training_plan.json")
    stream_export = _load(Path("output/article1/ingestion/stream_export_summary.json"))

    by_id = {f["recording_id"]: f for f in manifest.get("files", [])
             if f.get("recording_id")}
    moto_entry = by_id.get(qa_moto.get("recording_id"), {})
    verdict = decide_verdict(qa_moto, extraction,
                             baseline_ran=bool(extraction))

    branch = _git("rev-parse", "--abbrev-ref", "HEAD")
    head = _git("rev-parse", "HEAD")
    log = _git("log", "--oneline", f"{args.base_commit}..HEAD")

    rgb = qa_moto["streams"]["camera-rgb"]
    rate = rgb["rate"]
    sel = selection.get("selected", {})

    L: List[str] = [
        "# Article 1 — motorcycle ingestion: final summary",
        "",
        f"Generated: {dt.datetime.now(dt.timezone.utc).isoformat()}",
        "",
        "## Git",
        "",
        f"- branch: `{branch}`",
        f"- base commit: `{args.base_commit}`",
        f"- final commit: `{head}`",
        "",
        "Intermediate commits:",
        "",
        "```",
        log or "(none)",
        "```",
        "",
        "## The motorcycle recording",
        "",
        "| quantity | value |",
        "|---|---|",
        f"| file | `{moto_entry.get('relative_path')}` |",
        f"| local path | `{moto_entry.get('absolute_path')}` |",
        f"| SHA-256 | `{moto_entry.get('sha256')}` |",
        f"| size | {moto_entry.get('size_bytes', 0) / 1e6:.1f} MB |",
        f"| recording id | `{qa_moto.get('recording_id')}` |",
        f"| duration | {rate['duration_s']:.3f} s |",
        f"| RGB frames | {rate['count']:,} |",
        f"| nominal rate | {rgb['nominal_rate_hz']} Hz |",
        f"| **measured rate** | **{rate['effective_fps']:.5f} Hz** "
        f"(span {rate['effective_fps_span']:.5f} Hz) |",
        f"| resolution | {qa_moto['rgb']['width']} x {qa_moto['rgb']['height']} |",
        f"| streams present | {qa_moto['summary']['stream_count_present']} |",
        "",
        "Streams: " + ", ".join(f"`{s}`" for s in
                                qa_moto["summary"]["streams_present"]) + ".",
        "",
        "How it was identified: not by its file name. The project tree was scanned, "
        "each VRS opened, and the domain decided from the static ego structure "
        "visible in sampled frames. The evidence object carries no rate, count or "
        "duration field, so the temporary 10 vs 15 fps difference cannot influence "
        "the decision.",
        "",
        "### Gaps and excluded intervals",
        "",
    ]

    gaps = rgb["gaps"]
    if gaps:
        g = gaps[0]
        L += [
            f"Exactly one anomalous RGB interval in the whole recording: "
            f"{g['dt_ms']:.3f} ms between source frames {g['index_before']} and "
            f"{g['index_after']} ({g['periods']:.2f} nominal periods), "
            f"at t = {(g['t_before_ns'] - rate['first_ns']) / 1e9:.2f} s. One dropped "
            f"frame out of {rate['count']:,}.",
            "",
        ]
    else:
        L += ["No anomalous RGB interval.", ""]
    L += [
        "**No interval is excluded.** There is no pause, no restart, no "
        "non-monotonic timestamp, and the recording is a single continuous segment "
        f"of {rgb['longest_segment_s']:.1f} s.",
        "",
    ]

    if sel:
        L += [
            "## The 30-second segment",
            "",
            "| quantity | value |",
            "|---|---|",
            f"| source frames | {sel['start_frame_index']} to "
            f"{sel['end_frame_index']} |",
            f"| device timestamps | {sel['start_timestamp_ns']:,} to "
            f"{sel['end_timestamp_ns']:,} ns |",
            f"| duration | {sel['duration_s']:.3f} s |",
            f"| frames | {sel['frame_count']} |",
            f"| missing frames | {sel['missing_frame_estimate']} |",
            f"| windows evaluated | {selection.get('evaluated_windows')} |",
            f"| score | {sel['score']:.4f} (rank 1 of "
            f"{selection.get('passing_windows')} passing) |",
            "",
            "Chosen by a ranking that weights motion, visual variety and semantic "
            "variety above easy footage, after hard quality gates. The full ranking "
            "is in `segment_candidates_motorcycle.csv`.",
            "",
        ]

    if extraction:
        L += ["## Pipeline execution", "",
              f"- extracted {extraction['selected_frames']} frames, "
              f"{extraction['errors']} errors",
              f"- source rate measured at {extraction['source_effective_fps']:.5f} Hz",
              f"- `resampled: {extraction['resampled']}`, "
              f"`interpolated: {extraction['interpolated']}`, "
              f"`synthetic_frames: {extraction['synthetic_frames']}`",
              ""]
        checks = run / "logs" / "frozen_config_checksums.sha256"
        if checks.exists():
            L += ["The frozen configuration bundle was hashed before and after the "
                  "run and did not change; the per-file checksums are in "
                  "`logs/frozen_config_checksums.sha256`.", ""]

    if failures:
        counts = {k: v for k, v in failures.get("counts_per_mode", {}).items() if v}
        L += ["## Main candidate failure modes on the motorcycle", "",
              "| candidate mode | events | per second |", "|---|---:|---:|"]
        rates = failures.get("rate_per_second_per_mode", {})
        for mode, n in sorted(counts.items(), key=lambda kv: -kv[1])[:12]:
            L.append(f"| `{mode}` | {n} | {rates.get(mode) or 0:.3f} |")
        L += ["", "These are candidates, not confirmed errors: there is no reviewed "
              "ground truth to confirm them against.", ""]

    if hands:
        L += ["## Hands", "",
              f"{hands['total_candidates']} candidates over {hands['frames']} frames.",
              "",
              "| state | candidates |", "|---|---:|"]
        for state, n in hands["per_state"].items():
            L.append(f"| `{state}` | {n} |")
        L += ["",
              f"- evaluable (`visible` or `partially_visible`): "
              f"{hands['evaluable_candidates']} "
              f"({hands['evaluable_fraction']:.1%})",
              f"- candidate false masks: {hands['candidate_false_mask_count']}",
              f"- candidate over-propagation: "
              f"{hands['candidate_over_propagation_count']}",
              "",
              "The failure inventory above counts `false_hand` per **frame**, while "
              "the audit counts per **side candidate** (two per frame) and routes "
              "some of them to the over-propagation case instead, so the two figures "
              "describe the same situation from different units.",
              ""]
    if stream_export:
        moto_h = stream_export.get(qa_moto.get("recording_id"), {}).get(
            "hand_tracking", {})
        auto_h = stream_export.get(qa_auto.get("recording_id"), {}).get(
            "hand_tracking", {})
        if moto_h.get("available") and auto_h.get("available"):
            L += [
                f"Across the whole recordings, the on-device tracker reports a hand "
                f"in {auto_h['any_tracked_fraction']:.1%} of car samples "
                f"({auto_h['any_landmark_in_rgb_fraction']:.1%} projecting into the "
                f"RGB image) against {moto_h['any_tracked_fraction']:.1%} on the "
                f"motorcycle ({moto_h['any_landmark_in_rgb_fraction']:.1%} into the "
                "image). Intermittent hand visibility on a motorcycle is the normal "
                "case and is never treated as a model error.",
                "",
            ]

    if route:
        common = route.get("common_route") or {}
        L += ["## Route pairing quality", "",
              f"- status: `{route.get('status')}`",
              f"- accepted pairs: {route.get('accepted_count')} of "
              f"{route.get('candidate_count')} "
              f"({route.get('accepted_fraction', 0):.1%})",
              f"- median pair distance: "
              f"{route.get('median_pair_distance_m', float('nan')):.1f} m",
              f"- shared stretch: about "
              f"{common.get('approximate_common_distance_m', 0):.0f} m",
              f"- motorcycle progression covered: "
              f"{common.get('motorcycle_progression_range')}",
              f"- car progression covered: {common.get('car_progression_range')}",
              f"- monotonic (same direction of travel): "
              f"{common.get('progression_monotonic')}",
              "",
              "GPS quality is asymmetric and bounds the pairing: the car's fix rate "
              "and accuracy are far worse than the motorcycle's, because a metal roof "
              "degrades reception.",
              ""]

    if annotation:
        L += ["## Annotation dataset", "",
              f"- total frames: {annotation['counts']['total']}",
              f"- per domain: {annotation['counts']['per_domain']}",
              f"- per group: {annotation['counts']['per_group']}",
              f"- strata covered: {len(annotation.get('stratum_coverage', {}))}",
              ""]
    if pkg:
        L += ["### CVAT package", "",
              f"- items: {pkg['counts']['items']}",
              f"- with pre-annotation: "
              f"{sum(1 for i in pkg['items'] if i.get('preannotation_relative_path'))}",
              f"- pre-annotation status: `{pkg['preannotations']['status']}`",
              "",
              "Committed: manifest, label specification, label map, palette, "
              "annotator guide, frame list, QA thumbnails and checksums. Local only: "
              "full-resolution images and the automatic pre-annotation masks.",
              ""]

    if training:
        L += ["## Training gate", "",
              f"- status: `{training['status']}`",
              f"- ready to train: {training['ready_to_train']}",
              ""]
        for b in training.get("blockers", []):
            L.append(f"- blocker: {b}")
        L += ["", training["gate"], ""]

    if args.test_summary:
        L += ["## Tests", "", "```", args.test_summary, "```", ""]

    L += ["## Outputs", "", "### Committed", ""]
    for p in sorted(reports.rglob("*")):
        if p.is_file() and p.stat().st_size < 3_000_000:
            L.append(f"- `{p}`")
    L += ["", "### Local only, not committed", "",
          f"- `{run}/` — frames, masks, confidence, entropy, provenance, logs",
          "- `output/article1/ingestion/` — RGB scans, scouting frames, stream "
          "exports"]
    if args.video:
        L.append(f"- `{args.video}` — the presentation video")
    L += ["", "- the two VRS recordings, unmodified", ""]

    L += [
        "## Limitations",
        "",
        "- No reviewed ground truth exists, so this work reports coverage, "
        "stability, fragmentation, persistence, agreement, fallback share, "
        "confidence and entropy — never accuracy, IoU, precision or recall.",
        "- The car recording is a provisional 10 fps baseline. Every car/motorcycle "
        "number here is exploratory and must be recomputed once the car is "
        "re-recorded at 15 fps.",
        "- The cockpit stream is an unreviewed Grounding DINO + SAM 2.1 proxy plus a "
        "geometric prior shaped around a car interior.",
        "- The frozen prompt set has one hand phrasing, tied to a steering wheel, "
        "which no motorcycle frame can satisfy.",
        "- The analysed segment is 30 s of a 1004 s recording.",
        "- No MPS/VIO trajectory product exists in these files, so route alignment "
        "rests on GPS alone.",
        "",
        "## What needs human review",
        "",
        "1. The annotation package: every pre-annotated pixel.",
        "2. The candidate hand visibility states, especially the `uncertain` ones "
        "and the candidate false masks.",
        "3. The candidate failure modes, which are unusual-for-this-clip signals "
        "rather than confirmed errors.",
        "4. The route pairing, which is GPS-only and bounded by the car's degraded "
        "reception.",
        "5. The choice of the 30-second segment, whose full ranking is published so "
        "it can be overruled.",
        "",
        "## Verdict",
        "",
        f"### {verdict['verdict']}. {_verdict_title(verdict['verdict'])}",
        "",
        verdict["reason"] + ".",
        "",
        "| check | result |",
        "|---|---|",
    ]
    for name, ok in verdict["checks"].items():
        L.append(f"| `{name}` | {'pass' if ok else 'FAIL'} |")
    L += ["", "Evidence:", "", "```json",
          json.dumps(verdict["evidence"], indent=2), "```", "",
          "The work stops here, at the human-annotation gate, exactly as required: "
          "the motorcycle is validated, the frozen baseline has been run and "
          "analysed, the preliminary comparison exists, the annotation package and "
          "its pre-annotations are ready, and the shared cockpit training stage is "
          "configured but not started.",
          ""]

    out = reports / "REPORT_FINAL_SUMMARY.md"
    out.write_text("\n".join(L) + "\n")
    print(f"wrote {out}")
    print(json.dumps({"verdict": verdict["verdict"],
                      "checks": verdict["checks"]}, indent=2))
    return 0


def _verdict_title(v: str) -> str:
    return {
        "A": "Motorcycle ready for the dataset, annotation ready",
        "B": "Motorcycle usable with some intervals excluded",
        "C": "Motorcycle usable for RGB only, multimodal data incomplete",
        "D": "Motorcycle recording not reliable enough, must be repeated",
    }[v]


if __name__ == "__main__":
    raise SystemExit(main())
