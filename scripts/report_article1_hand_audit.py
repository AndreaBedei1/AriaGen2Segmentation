#!/usr/bin/env python3
"""Generate REPORT_HAND_VISIBILITY_AUDIT.md from the audit's own outputs."""
from __future__ import annotations

import argparse
import datetime as dt
import json
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reports", default="reports/article1_motorcycle_ingestion")
    ap.add_argument("--stream-export",
                    default="output/article1/ingestion/stream_export_summary.json")
    ap.add_argument("--proxy-summary", default=None)
    args = ap.parse_args()

    reports = Path(args.reports)
    summary = json.loads((reports / "hand_failure_summary.json").read_text())
    candidates = pd.read_csv(reports / "hand_visibility_candidates.csv")
    agreement = pd.read_csv(reports / "hand_proxy_agreement.csv")
    export = json.loads(Path(args.stream_export).read_text()) \
        if Path(args.stream_export).exists() else {}
    proxy = json.loads(Path(args.proxy_summary).read_text()) \
        if args.proxy_summary and Path(args.proxy_summary).exists() else {}

    L: List[str] = [
        "# Article 1 — motorcycle hand visibility audit",
        "",
        f"Generated: {dt.datetime.now(dt.timezone.utc).isoformat()}",
        "",
        "## What this is, and what it is not",
        "",
        "Every state below is a **candidate proposed for human review**. There is no "
        "reviewed hand ground truth in this project yet, so this report contains no "
        "recall, no precision and no accuracy, and it must not be read as a "
        "measurement of how well anything detects hands.",
        "",
        "Intermittent hand visibility is **normal** on a motorcycle. A hand can be "
        "below the camera, hidden by the handlebar or the rider's own arm, smeared by "
        "vibration, or simply outside the field of view when the rider looks away. "
        "None of that is a model error, and the audit is built so that \"no mask "
        "because no hand is visible\" is a correct outcome.",
        "",
        "The failures that *are* worth counting are the opposite ones: a hand region "
        "proposed where no hand is visible, and a hand region that keeps living after "
        "the hand has gone.",
        "",
        "Hands are not a taxonomy class. They stay inside `control_and_ego_vehicle` "
        "(12) and are described through separate attributes.",
        "",
    ]

    if export:
        L += ["## Recording-level hand tracking", "",
              "The on-device tracker gives the first, cheapest signal. It is a proxy: "
              "it can track a hand that is outside the RGB field of view, and it can "
              "fail on a perfectly visible one.",
              "",
              "| recording | domain | samples | left tracked | right tracked | "
              "either tracked | any landmark inside the RGB image |",
              "|---|---|---:|---:|---:|---:|---:|"]
        for rec, v in export.items():
            h = v.get("hand_tracking", {})
            if not h.get("available"):
                continue
            L.append(
                f"| `{rec}` | {v['domain']} | {h['samples']:,} | "
                f"{h['left_tracked_fraction']:.1%} | "
                f"{h['right_tracked_fraction']:.1%} | "
                f"{h['any_tracked_fraction']:.1%} | "
                f"{h['any_landmark_in_rgb_fraction']:.1%} |")
        L += ["",
              "The contrast between the two domains is the central observation of "
              "this audit, and it is a property of the acquisition geometry rather "
              "than of any model: in the car the wheel sits inside the camera's view, "
              "on the motorcycle the grips sit mostly below it.",
              ""]

    L += [
        "## Candidate states over the analysed segment",
        "",
        f"{summary['total_candidates']} candidates over {summary['frames']} frames "
        f"({summary['duration_s']:.2f} s at {summary['effective_fps']:.3f} Hz), two "
        "per frame (left and right).",
        "",
        "| state | candidates | share | per second |",
        "|---|---:|---:|---:|",
    ]
    total = max(1, summary["total_candidates"])
    for state, n in summary["per_state"].items():
        rate = summary.get("per_state_per_second", {}).get(state)
        L.append(f"| `{state}` | {n} | {n / total:.1%} | "
                 f"{(rate if rate is not None else 0):.3f} |")
    L += ["",
          f"Evaluable candidates (`visible` or `partially_visible`): "
          f"{summary['evaluable_candidates']} "
          f"({summary['evaluable_fraction']:.1%}). **A model may be scored on hands "
          "only on frames a reviewer confirms as one of those two states.** In the "
          "other states a missing mask is not an error and must not be penalised; a "
          "present mask still deserves inspection.",
          ""]

    if "per_state_by_side" in summary:
        L += ["### By side", "", "| state | left | right |", "|---|---:|---:|"]
        left = summary["per_state_by_side"]["left"]
        right = summary["per_state_by_side"]["right"]
        for state in left:
            L.append(f"| `{state}` | {left[state]} | {right.get(state, 0)} |")
        L.append("")

    L += ["## Agreement between the signals", "",
          "The ten cases the audit is required to separate:",
          "",
          "| case | meaning | candidates |", "|---:|---|---:|"]
    for key, v in sorted(summary["per_agreement_case"].items(), key=lambda kv: int(kv[0])):
        L.append(f"| {key} | {v['label']} | {v['count']} |")
    L += ["",
          f"- candidate false masks (case 4): "
          f"**{summary['candidate_false_mask_count']}**",
          f"- candidate over-propagation (case 9): "
          f"**{summary['candidate_over_propagation_count']}**",
          ""]

    if proxy:
        L += [
            "## The segmentation proxy used here",
            "",
            "The frozen cockpit proxy carries exactly one hand phrasing, \"human hand "
            "on steering wheel\", which no motorcycle frame can satisfy. Reading the "
            "frozen run alone would therefore confuse \"the prompt cannot match\" "
            "with \"there is no hand\".",
            "",
            "This audit adds a separate diagnostic probe with domain-appropriate "
            "phrasings. It is not part of the frozen pipeline and does not modify it.",
            "",
            f"- caption: `{proxy.get('caption')}`",
            f"- box / text thresholds: {proxy.get('box_threshold')} / "
            f"{proxy.get('text_threshold')}",
            f"- frames probed: {proxy.get('frames')}",
            f"- frames with a proposed hand region: "
            f"{proxy.get('frames_with_hand_region')}",
            f"- mean proposed area share: "
            f"{proxy.get('mean_hand_area_fraction', 0):.4%}",
            f"- status: `{proxy.get('status')}`",
            "",
        ]

    seq_index = reports / "hand_qa_sequences" / "index.json"
    if seq_index.exists():
        seqs = json.loads(seq_index.read_text())["sequences"]
        L += ["## QA sequences", "",
              "| directory | found | centre frame |", "|---|---|---:|"]
        for name, v in seqs.items():
            L.append(f"| `hand_qa_sequences/{name}` | "
                     f"{'yes' if v.get('found') else 'no'} | "
                     f"{v.get('centre_frame_index', '—')} |")
        L.append("")

    L += [
        "## Attributes carried into annotation",
        "",
        "Per side: `*_hand_visible`, `*_hand_partially_visible`, `*_hand_occluded`, "
        "`*_hand_out_of_frame`, `*_hand_motion_blurred`, `*_arm_visible`. Shared: "
        "`hand_tracking_available`, `hand_tracking_valid`, "
        "`hand_segmentation_proxy_available`.",
        "",
        "`*_arm_visible` is deliberately left empty by the automatic pass: it is not "
        "inferable from the available signals and guessing it would give the "
        "reviewer a wrong default to accept.",
        "",
        "## Outputs",
        "",
        "- `hand_visibility_candidates.csv` — one row per frame and side, with the "
        "proposed state, its rationale and the annotation attributes",
        "- `hand_proxy_agreement.csv` — the raw signals behind each candidate",
        "- `hand_failure_summary.json` — aggregate counts",
        "- `hand_qa_sequences/` — consecutive-frame examples of the main situations",
        "",
        "## What a reviewer must decide",
        "",
        "1. Confirm or overturn each proposed state, especially every `uncertain`.",
        "2. Confirm the candidate false masks: those are the only cases where the "
        "cockpit stream is doing something actively wrong about hands.",
        "3. Mark the frames where a hand is visible but neither signal found it; "
        "those are the cases that will teach the shared cockpit model most.",
        "",
    ]

    out = reports / "REPORT_HAND_VISIBILITY_AUDIT.md"
    out.write_text("\n".join(L) + "\n")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
