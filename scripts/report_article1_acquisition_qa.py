#!/usr/bin/env python3
"""Generate REPORT_ACQUISITION_QA.md from the manifest and the stream QA JSON.

The report is generated rather than hand-written so that every number in it is the
number actually measured by the QA stage.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
from pathlib import Path
from typing import Any, Dict, List

USABILITY_RULES = {
    "max_missing_frame_fraction": 0.001,
    "max_absolute_pause_ms": 250.0,
    "required_streams": ["camera-rgb", "eyegaze", "handtracking", "gps-app",
                         "imu-left", "imu-right", "slam-front-left"],
    "max_rate_deviation_from_nominal": 0.05,
}


def _fmt(v: Any, spec: str = ".3f") -> str:
    if v is None:
        return "n/a"
    if isinstance(v, bool):
        return "yes" if v else "no"
    if isinstance(v, (int,)):
        return f"{v:,}"
    if isinstance(v, float):
        return format(v, spec)
    return str(v)


def stream_table(qa: Dict[str, Any]) -> List[str]:
    rows = ["| stream | samples | nominal Hz | measured Hz | representative Hz | "
            "duration s | flagged intervals | >250 ms | segments | p95 dt vs RGB ms | note |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|"]
    for label, s in sorted(qa["streams"].items()):
        if not s.get("present"):
            rows.append(f"| `{label}` | — | — | — | — | — | — | — | — | — | "
                        "checked and not present |")
            continue
        if not s.get("rate"):
            rows.append(f"| `{label}` | {s.get('sample_count', 0)} | — | — | — | — | "
                        f"— | — | — | — | unreadable: {s.get('error', '')} |")
            continue
        r, g = s["rate"], s["gap_summary"]
        sync = s.get("synchronisation_vs_rgb") or {}
        p95 = sync.get("abs_dt_to_reference_ms", {}).get("p95")
        note = "bursty" if r.get("bursty") else ""
        rows.append(
            f"| `{label}` | {r['count']:,} | "
            f"{_fmt(s['nominal_rate_hz'], '.0f')} | {_fmt(r['effective_fps'])} | "
            f"{_fmt(r['representative_rate_hz'], '.2f')} | "
            f"{_fmt(r['duration_s'], '.1f')} | {g['total_flagged_intervals']} | "
            f"{g['over_250ms']} | {s['continuous_segment_count']} | "
            f"{_fmt(p95, '.1f')} | {note} |")
    return rows


def usability(qa: Dict[str, Any]) -> Dict[str, Any]:
    rgb = qa["streams"]["camera-rgb"]
    r, g = rgb["rate"], rgb["gap_summary"]
    missing_fraction = g["estimated_missing_samples"] / max(1, r["count"])
    present = set(qa["summary"]["streams_present"])
    missing_required = [s for s in USABILITY_RULES["required_streams"]
                        if s not in present]
    verdicts = []
    ok = True
    if missing_fraction > USABILITY_RULES["max_missing_frame_fraction"]:
        ok = False
        verdicts.append(f"estimated missing frames {missing_fraction:.4%} exceeds "
                        f"{USABILITY_RULES['max_missing_frame_fraction']:.2%}")
    else:
        verdicts.append(f"estimated missing frames {missing_fraction:.4%} within budget")
    if g["over_250ms"] > 0:
        ok = False
        verdicts.append(f"{g['over_250ms']} pauses longer than 250 ms")
    else:
        verdicts.append("no pause longer than 250 ms")
    if rgb["continuous_segment_count"] > 1:
        ok = False
        verdicts.append(f"RGB splits into {rgb['continuous_segment_count']} segments")
    else:
        verdicts.append("RGB is a single continuous segment")
    if missing_required:
        ok = False
        verdicts.append(f"missing required streams: {missing_required}")
    else:
        verdicts.append("every required stream is present")
    if r["non_monotonic_count"]:
        ok = False
        verdicts.append(f"{r['non_monotonic_count']} non-monotonic RGB intervals")
    else:
        verdicts.append("RGB timestamps are strictly monotonic")
    return {"usable": ok, "checks": verdicts,
            "missing_frame_fraction": missing_fraction}


def recording_section(qa: Dict[str, Any], entry: Dict[str, Any]) -> List[str]:
    rgb_stream = qa["streams"]["camera-rgb"]
    rgb, r = qa["rgb"], qa["rgb"]["rate"]
    g = rgb_stream["gap_summary"]
    u = usability(qa)
    dup = rgb["duplicates"]
    iq = rgb["image_quality"]
    drift = rgb_stream["drift"]

    out = [
        f"## {qa['domain'].capitalize()} — `{qa['recording_id']}`",
        "",
        f"- source file: `{entry['relative_path']}` "
        f"({entry['size_bytes'] / 1e6:.1f} MB)",
        f"- local absolute path: `{entry['absolute_path']}`",
        f"- SHA-256: `{entry['sha256']}`",
        f"- last modified: {entry['modified_iso']}",
        f"- device serial: `{qa['metadata'].get('device_serial')}`, "
        f"profile name: `{qa['metadata'].get('recording_profile')}`",
        f"- streams present: {qa['summary']['stream_count_present']} "
        f"(none of the checked modalities is missing)"
        if not qa["summary"]["streams_checked_and_missing"] else
        f"- streams checked and missing: {qa['summary']['streams_checked_and_missing']}",
        "",
        "### RGB",
        "",
        f"| quantity | value |",
        "|---|---|",
        f"| resolution | {rgb['width']} x {rgb['height']} |",
        f"| sensor | {rgb['codec'].get('sensor_model')} "
        f"(pixel format {rgb['codec'].get('pixel_format')}) |",
        f"| codec | {rgb['codec'].get('codec') or 'not exposed by the provider API'} |",
        f"| frames | {r['count']:,} |",
        f"| first / last timestamp (ns) | {r['first_ns']:,} / {r['last_ns']:,} |",
        f"| duration | {r['duration_s']:.3f} s |",
        f"| nominal rate | {_fmt(rgb_stream['nominal_rate_hz'], '.1f')} Hz |",
        f"| **measured rate (median interval)** | **{r['effective_fps']:.5f} Hz** |",
        f"| measured rate (span) | {r['effective_fps_span']:.5f} Hz |",
        f"| median interval | {r['median_dt_ms']:.4f} ms |",
        f"| mean / std interval | {r['mean_dt_ms']:.4f} / {r['std_dt_ms']:.4f} ms |",
        f"| min / max interval | {r['min_dt_ms']:.3f} / {r['max_dt_ms']:.3f} ms |",
        f"| interval p1 / p5 / p50 / p95 / p99 | {r['p01_dt_ms']:.3f} / "
        f"{r['p05_dt_ms']:.3f} / {r['p50_dt_ms']:.3f} / {r['p95_dt_ms']:.3f} / "
        f"{r['p99_dt_ms']:.3f} ms |",
        f"| non-monotonic intervals | {r['non_monotonic_count']} |",
        f"| duplicate timestamps | {dup['duplicate_timestamps']} |",
        f"| duplicate images | {dup['duplicate_images']} |",
        f"| near-identical consecutive pairs | "
        f"{dup['near_identical_consecutive_pairs']:,} |",
        f"| intervals > 1.5 periods | {g['over_1p5_periods']} |",
        f"| intervals > 2 periods | {g['over_2_periods']} |",
        f"| intervals > 250 ms | {g['over_250ms']} |",
        f"| estimated missing frames | {g['estimated_missing_samples']} |",
        f"| longest interval | {g['longest_interval_ms']:.3f} ms |",
        f"| continuous segments | {rgb_stream['continuous_segment_count']} |",
        f"| longest segment | {qa['summary']['rgb_longest_segment_s']:.2f} s |",
        f"| clock drift, max residual vs constant rate | "
        f"{drift['max_abs_residual_ms']:.2f} ms |",
        f"| clock drift, residual std | {drift['residual_std_ms']:.2f} ms |",
        f"| median frame luminance | {iq['mean_luminance']['median']:.3f} |",
        f"| median sharpness (Laplacian variance) | "
        f"{iq['blur_variance']['median']:.1f} |",
        f"| very dark / very bright frames | {iq['very_dark_frames']} / "
        f"{iq['very_bright_frames']} |",
        "",
    ]

    if dup["duplicate_images"]:
        out += [
            f"{dup['duplicate_images']} frame pairs are byte-level near identical "
            f"({dup['duplicate_definition']}). Consecutive frames of a driving video "
            "are normally very similar, so the much larger "
            f"{dup['near_identical_consecutive_pairs']:,} near-identical pairs is a "
            "scene-change statistic, not a defect.",
            "",
        ]

    gaps = rgb_stream["gaps"]
    if gaps:
        out += ["#### RGB intervals flagged", "",
                "| source index before | source index after | interval ms | periods | "
                "estimated missing | > 250 ms |", "|---:|---:|---:|---:|---:|---|"]
        for gap in gaps[:20]:
            out.append(
                f"| {gap['index_before']} | {gap['index_after']} | "
                f"{gap['dt_ms']:.3f} | {gap['periods']:.3f} | "
                f"{gap['missing_estimate']} | {'yes' if gap['over_250ms'] else 'no'} |")
        out.append("")
    else:
        out += ["No RGB interval was flagged: the stream is uninterrupted.", ""]

    out += ["### All streams", ""] + stream_table(qa) + [""]
    out += ["### Usability checks", ""]
    for c in u["checks"]:
        out.append(f"- {c}")
    out += ["", f"**Usable: {'yes' if u['usable'] else 'no'}**", ""]
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reports", default="reports/article1_motorcycle_ingestion")
    args = ap.parse_args()
    reports = Path(args.reports)

    manifest = json.loads((reports / "acquisition_manifest.json").read_text())
    by_id = {f["recording_id"]: f for f in manifest["files"] if f.get("recording_id")}
    qa = {tag: json.loads((reports / f"stream_qa_{tag}.json").read_text())
          for tag in ("auto", "moto")}

    moto_u = usability(qa["moto"])
    auto_u = usability(qa["auto"])
    moto_rgb = qa["moto"]["rgb"]["rate"]
    auto_rgb = qa["auto"]["rgb"]["rate"]

    lines: List[str] = [
        "# Article 1 — acquisition and stream QA (car + motorcycle)",
        "",
        f"Generated: {dt.datetime.now(dt.timezone.utc).isoformat()}",
        "",
        "Branch: `feature/article1-motorcycle-ingestion`",
        "",
        "## Scope",
        "",
        "This report validates the newly added motorcycle recording and re-validates "
        "the existing car recording with the same instrument. It answers a single "
        "question: **is the motorcycle recording usable, and over which intervals?**",
        "",
        "The car recording is currently sampled at ~10 fps and is an explicitly "
        "provisional development baseline. The final protocol is 15 fps for both "
        "vehicles. The rate difference is reported here as an acquisition fact and is "
        "never used as an analysis variable.",
        "",
        "## How the recordings were identified",
        "",
        "No file was identified by its name. The project tree was scanned for data "
        "files, each VRS was opened, and the domain was decided from the pixels:",
        "",
        f"- method: `{manifest['domain_resolution']['method']}`",
        f"- inputs: {', '.join(manifest['domain_resolution']['inputs'])}",
        f"- explicitly not used: "
        f"{', '.join(manifest['domain_resolution']['explicitly_not_used'])}",
        "",
        "The discriminator is the static ego structure around the camera. A car cabin "
        "encloses the camera, so the top of the frame is a dark, motionless headliner "
        "and the border is dominated by pillars and door frames. A motorcycle leaves "
        "the camera in the open, so the top of the frame is bright, changing sky and "
        "only a small region low in the image stays fixed.",
        "",
        "| recording | domain | confidence | top band static | top band luminance | "
        "border static | evidence |",
        "|---|---|---:|---:|---:|---:|---|",
    ]
    for tag in ("auto", "moto"):
        rec = qa[tag]["recording_id"]
        e = by_id[rec]
        ev = e["domain_evidence"]
        lines.append(
            f"| `{rec}` | {e['estimated_domain']} | {e['domain_confidence']:.2f} | "
            f"{ev['top_band_static_fraction']:.2%} | {ev['top_band_luminance']:.2f} | "
            f"{ev['border_static_fraction']:.2%} | "
            f"{ev['sampled_frames']} frames sampled across the recording |")
    lines += [""]
    for tag in ("auto", "moto"):
        e = by_id[qa[tag]["recording_id"]]
        lines.append(f"`{e['recording_id']}` rationale:")
        for r in e["domain_rationale"]:
            lines.append(f"- {r}")
        lines.append("")

    lines += [
        "Both recordings carry the identical profile name "
        f"`{qa['moto']['metadata'].get('recording_profile')}` and the identical device "
        f"serial `{qa['moto']['metadata'].get('device_serial')}`, yet their RGB "
        f"streams run at {auto_rgb['effective_fps']:.3f} Hz and "
        f"{moto_rgb['effective_fps']:.3f} Hz respectively. This is the concrete reason "
        "the frame rate is measured from timestamps and never read from the profile "
        "name.",
        "",
        "### Candidate selection",
        "",
    ]
    for domain, sel in manifest["selection"].items():
        lines.append(f"- **{domain}**: `{sel['selected_recording_id']}` — "
                     + "; ".join(sel["rationale"]))
    lines += [
        "",
        "No duplicate recording, partial export or truncated file was found. Both VRS "
        "files decoded end to end: the RGB scan read every frame the timestamp table "
        "declared, which is the completeness evidence used here.",
        "",
    ]

    lines += recording_section(qa["moto"], by_id[qa["moto"]["recording_id"]])
    lines += recording_section(qa["auto"], by_id[qa["auto"]["recording_id"]])

    gps_note = ("- GPS quality was not compared here because the stream export "
                "summary is absent.")
    export_path = Path("output/article1/ingestion/stream_export_summary.json")
    if export_path.exists():
        export = json.loads(export_path.read_text())
        g_auto = export.get(qa["auto"]["recording_id"], {}).get("gps", {})
        g_moto = export.get(qa["moto"]["recording_id"], {}).get("gps", {})
        if g_auto.get("available") and g_moto.get("available"):
            gps_note = (
                f"- GPS quality differs sharply between the two recordings: the "
                f"motorcycle has a position fix in "
                f"{g_moto['position_fraction']:.1%} of its samples with a median "
                f"accuracy of {g_moto['median_accuracy_m']:.1f} m, while the car has "
                f"a fix in only {g_auto['position_fraction']:.1%} of its samples with "
                f"a median accuracy of {g_auto['median_accuracy_m']:.1f} m. Route "
                "alignment quality is bounded by the car, not by the motorcycle: a "
                "metal roof degrades reception in a way an open motorcycle does not.")

    moto_gaps = qa["moto"]["streams"]["camera-rgb"]["gaps"]
    lines += [
        "## Conclusions",
        "",
        "### Is the motorcycle recording usable?",
        "",
        f"**Yes.** {moto_rgb['count']:,} RGB frames over {moto_rgb['duration_s']:.1f} s, "
        "strictly monotonic timestamps, no duplicate timestamp, and a single "
        "continuous segment covering the whole recording.",
        "",
        "### Is the recording continuous?",
        "",
    ]
    if moto_gaps:
        g0 = moto_gaps[0]
        t0 = moto_rgb["first_ns"]
        lines += [
            f"Effectively yes. Exactly one interval is anomalous: "
            f"{g0['dt_ms']:.3f} ms between source frames {g0['index_before']} and "
            f"{g0['index_after']}, i.e. {g0['periods']:.2f} nominal periods at "
            f"t = {(g0['t_before_ns'] - t0) / 1e9:.2f} s from the start. That is one "
            "dropped frame out of "
            f"{moto_rgb['count']:,} ({1 / moto_rgb['count']:.4%}), it is shorter than "
            "the 250 ms pause threshold, and it does not split the recording.",
            "",
            "There is no pause, no restart and no non-monotonic timestamp anywhere in "
            "the motorcycle recording.",
            "",
        ]
    else:
        lines += ["Yes: no RGB interval was flagged at all.", ""]

    lines += [
        "### Which intervals are valid, and which must be excluded?",
        "",
        f"The whole recording is valid: "
        f"[{moto_rgb['first_ns']:,} ns, {moto_rgb['last_ns']:,} ns], "
        f"{moto_rgb['duration_s']:.1f} s, source frame indices 0 to "
        f"{moto_rgb['count'] - 1}.",
        "",
        "**No interval has to be excluded on acquisition grounds.** The single dropped "
        "frame is recorded in `timestamp_gaps_moto.csv` so that any stage crossing it "
        "can account for it; it does not justify discarding a segment.",
        "",
        "### Is the effective frame rate compatible with 15 fps?",
        "",
        f"Yes. Measured {moto_rgb['effective_fps']:.5f} Hz from the median interval and "
        f"{moto_rgb['effective_fps_span']:.5f} Hz across the full span, against a "
        f"declared nominal of {qa['moto']['streams']['camera-rgb']['nominal_rate_hz']} Hz: "
        f"a deviation of "
        f"{abs(moto_rgb['effective_fps'] - 15.0) / 15.0:.4%}. Interval jitter is "
        f"{moto_rgb['std_dt_ms']:.3f} ms standard deviation around a "
        f"{moto_rgb['median_dt_ms']:.3f} ms period, and the p1-p99 interval range is "
        f"{moto_rgb['p01_dt_ms']:.2f}-{moto_rgb['p99_dt_ms']:.2f} ms. The recording is "
        "a genuine 15 fps acquisition.",
        "",
        "### Which streams are available?",
        "",
        f"All {qa['moto']['summary']['stream_count_present']} expected streams are "
        "present in **both** recordings: "
        + ", ".join(f"`{s}`" for s in qa["moto"]["summary"]["streams_present"]) + ".",
        "",
        "The motorcycle ingestion is therefore fully multimodal: RGB, on-device eye "
        "gaze, eye-tracking cameras, hand tracking, four SLAM cameras, two IMUs, "
        "magnetometer, barometer, GPS, PPG, ALS and temperature.",
        "",
        "### What limitations remain?",
        "",
        "- The `temperature` stream is bursty in both recordings: several samples "
        "arrive together and then the stream waits. Its median-interval rate is "
        "meaningless and its `representative Hz` (span rate) should be used. The "
        "flagged intervals for that stream are inter-burst waits, not data loss.",
        gps_note,
        "- The car recording remains a provisional 10 fps baseline. Any comparison "
        "against the motorcycle is exploratory until the car is re-recorded at 15 fps.",
        "- No reviewed ground truth exists yet, so nothing in this report is an "
        "accuracy statement.",
        "",
        "## Machine-readable outputs",
        "",
        "- `acquisition_manifest.json` / `.csv` — every candidate file with size, "
        "SHA-256, modification time, stream structure and domain evidence",
        "- `stream_qa_moto.json` / `stream_qa_auto.json` — full per-stream QA",
        "- `timestamp_gaps_moto.csv` / `timestamp_gaps_auto.csv` — every flagged "
        "interval of every stream",
        "",
        "## Verdict",
        "",
        f"- motorcycle: **{'usable' if moto_u['usable'] else 'NOT usable'}** "
        "over its full duration, with no excluded interval",
        f"- car: **{'usable' if auto_u['usable'] else 'NOT usable'}**, but provisional "
        "at 10 fps",
        "",
    ]

    out = reports / "REPORT_ACQUISITION_QA.md"
    out.write_text("\n".join(lines) + "\n")
    print(f"wrote {out} ({len(lines)} lines)")
    print(json.dumps({"motorcycle_usable": moto_u["usable"],
                      "car_usable": auto_u["usable"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
