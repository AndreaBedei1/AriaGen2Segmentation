#!/usr/bin/env python3
"""Generate REPORT_MOTORCYCLE_BASELINE.md from the frozen run's own outputs."""
from __future__ import annotations

import argparse
import datetime as dt
import json
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd


def _load(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text()) if path.exists() else {}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--reports", default="reports/article1_motorcycle_ingestion")
    ap.add_argument("--segment-selection", default=None)
    ap.add_argument("--video", default=None)
    args = ap.parse_args()

    run, reports = Path(args.run), Path(args.reports)
    extraction = _load(run / "frames" / "extraction_summary.json")
    camera = _load(run / "semantic_camera" / "summary.json")
    final = _load(run / "semantic_camera_final_pass" / "summary.json")
    final_metrics = _load(run / "semantic_camera_final_pass" / "metrics.json")
    stabilized = _load(run / "semantic_camera_video_stabilized" / "summary.json")
    failures = _load(reports / "motorcycle_failure_modes.json")
    selection = _load(Path(args.segment_selection)) if args.segment_selection else {}
    coverage_path = reports / "motorcycle_class_coverage.csv"
    coverage = pd.read_csv(coverage_path) if coverage_path.exists() else None

    sel = selection.get("selected", {})
    L: List[str] = [
        "# Article 1 — frozen baseline on the motorcycle recording",
        "",
        f"Generated: {dt.datetime.now(dt.timezone.utc).isoformat()}",
        "",
        "Branch: `feature/article1-motorcycle-ingestion`",
        "",
        "## What was run, and what `frozen` means",
        "",
        "The existing Article 1 pipeline was run on the motorcycle segment without "
        "changing a single checkpoint, prompt, threshold, class mapping, fusion "
        "rule, temporal parameter or fallback policy. The configuration bundle was "
        "hashed before and after the run and the two hashes are recorded in "
        "`logs/frozen_run.log`; a mismatch aborts the run.",
        "",
        "This is deliberate. The point of a frozen first run is to see what a "
        "car-tuned pipeline actually does on a motorcycle, before anything is "
        "adapted to make it look better.",
        "",
        "Stages, in order:",
        "",
        "1. timestamp-driven extraction of the selected window",
        "2. Mask2Former Swin-L (Mapillary Vistas) external segmentation plus the "
        "causal temporal stage",
        "3. dense semantic-camera fusion with the Grounding DINO + SAM 2.1 cockpit "
        "proxy and the geometric fallback",
        "4. causal presentation stabilisation",
        "5. bidirectional presentation final pass (presentation only, kept separate "
        "from the causal scientific output)",
        "6. semantic gaze alignment, strictly after segmentation",
        "",
    ]

    if sel:
        L += [
            "## Selected segment",
            "",
            "| quantity | value |",
            "|---|---|",
            f"| recording | `{selection.get('recording_id')}` |",
            f"| source frame indices | {sel['start_frame_index']} to "
            f"{sel['end_frame_index']} |",
            f"| device timestamps (ns) | {sel['start_timestamp_ns']:,} to "
            f"{sel['end_timestamp_ns']:,} |",
            f"| duration | {sel['duration_s']:.3f} s |",
            f"| frames | {sel['frame_count']} |",
            f"| estimated missing frames | {sel['missing_frame_estimate']} |",
            f"| largest interval | {sel['largest_gap_ms']:.2f} ms |",
            f"| score | {sel['score']:.4f} |",
            f"| windows evaluated | {selection.get('evaluated_windows')} |",
            f"| windows passing every quality gate | "
            f"{selection.get('passing_windows')} |",
            "",
            "Why this window:",
            "",
        ] + [f"- {r}" for r in sel.get("rationale", [])] + [""]

        shortlist = selection.get("shortlist", [])
        if len(shortlist) > 1:
            L += [
                "### Ranked alternatives",
                "",
                "| rank | start frame | start s | score | motion | visual diversity "
                "| semantic diversity | traffic | cockpit proxy | hand-tracking share |",
                "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
            ]
            for c in shortlist:
                L.append(
                    f"| {c['rank']} | {c['start_frame_index']} | "
                    f"{c['start_timestamp_ns'] / 1e9:.1f} | {c['score']:.4f} | "
                    f"{c['mean_frame_difference']:.4f} | "
                    f"{c['visual_diversity']:.3f} | "
                    f"{(c['semantic_diversity'] or 0):.3f} | "
                    f"{(c['traffic_class_fraction'] or 0):.4f} | "
                    f"{(c['cockpit_proxy_fraction'] or 0):.4f} | "
                    f"{(c['hand_tracked_fraction'] or 0):.2%} |")
            L += ["",
                  "The weighting favours motion, visual variety and semantic variety "
                  "over easy footage, so the ranking cannot be won by a static, "
                  "visually trivial window. Quality gates (missing frames, pauses, "
                  "exposure, sharpness) are applied as hard constraints before "
                  "scoring.", ""]

    if extraction:
        L += [
            "## Extraction",
            "",
            "| quantity | value |",
            "|---|---|",
            f"| recording | `{extraction['recording_id']}` |",
            f"| source SHA-256 | `{extraction['source_file_sha256']}` |",
            f"| source stream | `{extraction['source_stream_id']}` |",
            f"| measured source rate | {extraction['source_effective_fps']:.5f} Hz |",
            f"| frames extracted | {extraction['selected_frames']} |",
            f"| errors | {extraction['errors']} |",
            f"| window duration | {extraction['window_duration_s']:.3f} s |",
            f"| resampled | {extraction['resampled']} |",
            f"| interpolated | {extraction['interpolated']} |",
            f"| synthetic frames | {extraction['synthetic_frames']} |",
            "",
        ]
        gaze = extraction.get("sidecars", {}).get("gaze", {})
        hand = extraction.get("sidecars", {}).get("hand_tracking", {})
        if gaze.get("available"):
            L += [
                f"Gaze: {gaze['total_samples']} samples in the recording; "
                f"{gaze['frames_with_valid_sample_in_window']} of "
                f"{extraction['selected_frames']} frames have at least one valid "
                f"sample within +/-{gaze['window_s'] * 1000:.0f} ms. Association is "
                "by timestamp; several samples per frame are preserved.",
                "",
            ]
        if hand.get("available"):
            L += [
                f"Hand tracking: {hand['total_samples']} samples; the left hand is "
                f"tracked in {hand['frames_with_left_tracked']} and the right in "
                f"{hand['frames_with_right_tracked']} of "
                f"{extraction['selected_frames']} frames. Tracked is not the same as "
                "visible, and an untracked hand is not evidence of an absent hand.",
                "",
            ]

    if camera:
        L += ["## Semantic camera", "",
              f"- external evidence: `{camera['external']['kind']}`",
              f"- internal stream: `{camera['internal']['kind']}` "
              f"(fallback: {camera['internal_is_fallback']})",
              f"- frames: {camera['frame_count']}",
              ""]
        if "dense_coverage" in camera:
            L.append(f"- dense coverage: {camera['dense_coverage']}")
        L.append("")

    if final:
        L += ["## Presentation final pass", ""]
        for k, v in sorted(final.items()):
            if isinstance(v, (int, float, str, bool)):
                L.append(f"- `{k}`: {v}")
        L.append("")
    if final_metrics:
        L += ["### Final-pass diagnostics", "", "```json",
              json.dumps(final_metrics, indent=2)[:4000], "```", ""]
    if stabilized:
        L += ["### Causal stabilisation summary", "", "```json",
              json.dumps({k: v for k, v in stabilized.items()
                          if isinstance(v, (int, float, str, bool))},
                         indent=2)[:2000], "```", ""]

    if coverage is not None:
        L += [
            "## Per-class coverage on the motorcycle",
            "",
            "Coverage, not accuracy: these numbers say how often and how much of the "
            "frame each class occupies, not whether it is right.",
            "",
            "| class | frames present | mean pixel share | median share when present "
            "| mean components when present | max components |",
            "|---|---:|---:|---:|---:|---:|",
        ]
        for r in coverage.itertuples():
            L.append(
                f"| `{r._1}` | {r.presence_fraction:.1%} | "
                f"{r.mean_pixel_fraction:.4%} | "
                f"{r.median_pixel_fraction_when_present:.4%} | "
                f"{r.mean_components_when_present:.1f} | {r.max_components} |")
        L.append("")

    if failures:
        counts = {k: v for k, v in failures.get("counts_per_mode", {}).items() if v}
        rates = failures.get("rate_per_second_per_mode", {})
        L += [
            "## Candidate failure modes",
            "",
            f"Detected over {failures['frames']} frames "
            f"({failures['duration_s']:.2f} s at "
            f"{failures['effective_fps']:.3f} Hz). Counts are also given per second "
            "so they stay comparable with a run sampled at a different rate.",
            "",
            "These are **candidates**, not confirmed errors. Thresholds are relative "
            "to this run's own distribution, so a mode firing here means \"unusual "
            "for this clip\", not \"wrong\".",
            "",
            "| candidate failure mode | events | per second |",
            "|---|---:|---:|",
        ]
        for mode, n in sorted(counts.items(), key=lambda kv: -kv[1]):
            L.append(f"| `{mode}` | {n} | {rates.get(mode, 0) or 0:.3f} |")
        L += ["",
              "Modes checked and **not** triggered: "
              + ", ".join(f"`{m}`" for m in failures.get("counts_per_mode", {})
                          if not failures["counts_per_mode"][m]) + ".",
              "",
              f"Causality: {failures.get('causality_note', '')}",
              ""]

        seqs = failures.get("qa_sequences", [])
        if seqs:
            L += ["### QA sequences", "",
                  "Each directory holds consecutive frames with RGB, coloured mask, "
                  "overlay, a contact sheet and a description.", "",
                  "| directory | mode | frames | severity |", "|---|---|---:|---:|"]
            for s in seqs:
                L.append(f"| `qa/{s['directory']}` | `{s['mode']}` | "
                         f"{len(s['frames'])} | {s['severity']:.2f} |")
            L.append("")

    if args.video:
        L += ["## Local video output", "",
              f"- `{args.video}` (not committed)", ""]

    L += [
        "## Limits",
        "",
        "- No reviewed ground truth exists for this data, so nothing here is an "
        "accuracy statement. The vocabulary is deliberately coverage, stability, "
        "fragmentation, persistence, agreement, fallback share, confidence and "
        "entropy.",
        "- The cockpit stream is an unreviewed Grounding DINO + SAM 2.1 proxy plus a "
        "geometric bottom-of-frame prior. That prior was shaped around a car "
        "interior and does not describe a handlebar; its behaviour on the motorcycle "
        "is one of the things this run was meant to expose.",
        "- The frozen prompt set contains exactly one hand phrasing, \"human hand on "
        "steering wheel\", which no motorcycle frame can satisfy. Any conclusion "
        "about hands from the frozen run alone would be an artefact of that prompt.",
        "- A single 30-second window is not the recording. It was chosen to be "
        "representative and difficult, not to be exhaustive.",
        "",
    ]

    out = reports / "REPORT_MOTORCYCLE_BASELINE.md"
    out.write_text("\n".join(L) + "\n")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
