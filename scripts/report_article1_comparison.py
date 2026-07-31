#!/usr/bin/env python3
"""Generate REPORT_PRELIMINARY_AUTO_MOTO_COMPARISON.md."""
from __future__ import annotations

import argparse
import datetime as dt
import json
from pathlib import Path
from typing import Any, Dict, List


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reports", default="reports/article1_motorcycle_ingestion")
    args = ap.parse_args()

    reports = Path(args.reports)
    doc = json.loads((reports / "auto_moto_comparison.json").read_text())
    car, moto = doc["car"], doc["motorcycle"]
    findings = doc["findings"]

    L: List[str] = [
        "# Article 1 — preliminary car vs motorcycle comparison",
        "",
        f"Generated: {dt.datetime.now(dt.timezone.utc).isoformat()}",
        "",
        f"## Status: `{doc['status']}`",
        "",
        "**This is not a scientific result.** It exists to expose failure modes and "
        "domain shift before annotation. Nothing here supports a claim about how the "
        "vehicle changes visual attention.",
        "",
    ] + [f"- {c}" for c in doc["caveats"]] + [
        "",
        "## What is being compared",
        "",
        "| | car | motorcycle |",
        "|---|---:|---:|",
        f"| recording | `{car['recording_id']}` | `{moto['recording_id']}` |",
        f"| frames | {car['frame_count']} | {moto['frame_count']} |",
        f"| duration | {car['duration_s']:.2f} s | {moto['duration_s']:.2f} s |",
        f"| measured rate | {car['effective_fps']:.4f} Hz | "
        f"{moto['effective_fps']:.4f} Hz |",
        "",
        "The two clips have different frame counts because they have different "
        "sampling rates, not because they cover different amounts of time. Every "
        "temporal quantity below is per second for exactly that reason.",
        "",
        "## Semantic coverage",
        "",
        "Coverage, not accuracy. There is no ground truth to be accurate against.",
        "",
        "| class | car pixel share | motorcycle pixel share | difference | "
        "car presence | motorcycle presence |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for name, v in doc["per_class"].items():
        L.append(
            f"| `{name}` | {v['car_pixel_fraction']:.4%} | "
            f"{v['motorcycle_pixel_fraction']:.4%} | "
            f"{v['difference']:+.4%} | {v['car_presence_fraction']:.1%} | "
            f"{v['motorcycle_presence_fraction']:.1%} |")

    L += [
        "",
        "## Confidence, entropy, provenance",
        "",
        "| quantity | car | motorcycle |",
        "|---|---:|---:|",
        f"| mean final confidence | {car['mean_confidence']:.4f} | "
        f"{moto['mean_confidence']:.4f} |",
        f"| mean final entropy | {car['mean_entropy']:.4f} | "
        f"{moto['mean_entropy']:.4f} |",
        f"| dense coverage | {car['dense_coverage']:.4f} | "
        f"{moto['dense_coverage']:.4f} |",
        f"| invalid class ids | {car['invalid_id_count']} | "
        f"{moto['invalid_id_count']} |",
        f"| cockpit pixel share | {car['cockpit_fraction']:.4%} | "
        f"{moto['cockpit_fraction']:.4%} |",
        f"| unreviewed-fallback frames | {car['fallback_fraction']:.1%} | "
        f"{moto['fallback_fraction']:.1%} |",
        "",
        "### Provenance shares",
        "",
        "| source | car | motorcycle |",
        "|---|---:|---:|",
    ]
    labels = sorted(set(car["provenance_fraction"]) | set(moto["provenance_fraction"]))
    for label in labels:
        L.append(f"| `{label}` | {car['provenance_fraction'].get(label, 0):.2%} | "
                 f"{moto['provenance_fraction'].get(label, 0):.2%} |")

    temporal = doc["temporal_per_second"]
    L += [
        "",
        "## Temporal stability, per second",
        "",
        "| quantity | car | motorcycle |",
        "|---|---:|---:|",
        f"| class switching per second | "
        f"{(temporal['car_class_switch_rate_per_second'] or 0):.4f} | "
        f"{(temporal['motorcycle_class_switch_rate_per_second'] or 0):.4f} |",
        "",
        temporal["note"] + ".",
        "",
        f"**Caveat:** {temporal['caveat']}.",
        "",
        "## Fragmentation",
        "",
        "| class | car components per class-frame | motorcycle components per class-frame |",
        "|---|---:|---:|",
    ]
    for name, v in doc["per_class"].items():
        if v["car_components_per_class_frame"] or v["motorcycle_components_per_class_frame"]:
            L.append(f"| `{name}` | {v['car_components_per_class_frame']:.2f} | "
                     f"{v['motorcycle_components_per_class_frame']:.2f} |")

    for key, title in (("gaze", "Gaze"), ("hands", "Hands")):
        L += ["", f"## {title}", ""]
        for domain, m in (("car", car), ("motorcycle", moto)):
            block = m.get(key, {})
            if not block.get("available"):
                L.append(f"- {domain}: not available "
                         f"({block.get('reason', 'no data')})")
                continue
            L.append(f"- **{domain}**:")
            for k, v in block.items():
                if k in ("available", "note"):
                    continue
                L.append(f"  - `{k}`: {json.dumps(v) if isinstance(v, dict) else v}")
        if car.get(key, {}).get("note"):
            L += ["", car[key]["note"] + "."]

    if "route_pairing_quality" in doc:
        r = doc["route_pairing_quality"]
        L += ["", "## Route pairing quality", "",
              f"- status: `{r.get('status')}`",
              f"- accepted pairs: {r.get('accepted_count')} "
              f"({r.get('accepted_fraction', 0):.1%})",
              f"- median pair distance: "
              f"{r.get('median_pair_distance_m') or float('nan'):.1f} m",
              f"- median pair quality: "
              f"{r.get('median_pair_quality') or float('nan'):.3f}",
              ""]

    L += ["", "## Where the differences come from", ""]
    titles = {
        "common_problems": "Problems common to both domains",
        "motorcycle_specific": "Specific to the motorcycle",
        "car_specific": "Specific to the car",
        "external_model": "Attributable to the external model",
        "cockpit_proxy": "Attributable to the cockpit proxy",
        "fusion": "Attributable to the fusion",
        "temporal": "Temporal",
        "probably_due_to_sampling_rate": "Probably due to the different sampling rate",
        "probably_due_to_domain_shift": "Probably due to domain shift",
    }
    for key, title in titles.items():
        items = findings.get(key, [])
        L += [f"### {title}", ""]
        L += [f"- {i}" for i in items] if items else ["- nothing identified"]
        L.append("")

    L += [
        "## What this comparison may not be used for",
        "",
        "- It may not support any statement about how the vehicle changes visual "
        "attention. That requires the re-recorded car and reviewed ground truth.",
        "- The sampling-rate difference is not a feature and no vehicle classifier "
        "is trained here. A classifier given this data could learn the rate instead "
        "of the vehicle.",
        "- No accuracy, IoU, precision or recall appears anywhere above, by design.",
        "",
        "## What it is genuinely useful for",
        "",
        "- It shows where the car-tuned pipeline breaks on a motorcycle, which is "
        "what the frozen run was for.",
        "- It sizes the cockpit problem: the motorcycle exposes far less ego "
        "structure to the camera, and the geometric prior built for a dashboard does "
        "not describe a handlebar.",
        "- It tells the annotation stage which classes and conditions need the most "
        "human attention.",
        "",
    ]

    out = reports / "REPORT_PRELIMINARY_AUTO_MOTO_COMPARISON.md"
    out.write_text("\n".join(L) + "\n")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
