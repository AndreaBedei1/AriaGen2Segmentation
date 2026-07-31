#!/usr/bin/env python3
"""Phase 1-2: inventory the acquisitions and QA every stream (car + motorcycle).

Discovers candidate recordings by scanning the project tree, decides the domain
from the pixels (never from the file name and never from the frame rate), scans the
RGB stream once, and writes the acquisition manifest plus per-recording stream QA.

    python scripts/ingest_article1_acquisitions.py --root . \
        --reports reports/article1_motorcycle_ingestion \
        --work output/article1/ingestion
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd

from aria_drive_seg.config import Config
from aria_drive_seg.ingestion.inventory import (HashCache, classify_domain,
                                                collect_domain_evidence,
                                                recording_id_for, scan_files)
from aria_drive_seg.ingestion.rgb_scan import (RgbScan, sample_frames_for_domain,
                                               scan_rgb_stream)
from aria_drive_seg.ingestion.stream_qa import gaps_to_rows, run_stream_qa
from aria_drive_seg.ingestion.timeline import measure_rate
from aria_drive_seg.io_utils import atomic_write_json
from aria_drive_seg.logging_utils import get_logger, setup_logging
from aria_drive_seg.vrs.provider import AriaProvider

log = get_logger("ingest")


def _probe_vrs(record, work: Path, cfg: Config, rescan: bool) -> Dict[str, Any]:
    """Open a VRS, decide its domain, scan RGB once, and QA all streams."""
    provider = AriaProvider(record.absolute_path, cfg.get("vrs.time_domain", "DEVICE_TIME"))
    rgb_label = cfg.get("vrs.rgb_label", "camera-rgb")

    streams = provider.list_streams()
    record.num_streams = len(streams)
    record.stream_labels = sorted(str(s.label) for s in streams)

    meta = {}
    try:
        md = provider._dp.get_metadata()
        meta = {
            "device_serial": str(getattr(md, "device_serial", "")),
            "recording_profile": str(getattr(md, "recording_profile", "")),
            "start_time_epoch_sec": int(getattr(md, "start_time_epoch_sec", 0) or 0),
        }
    except Exception:
        pass
    record.device_serial = meta.get("device_serial")
    record.recording_profile = meta.get("recording_profile")
    record.start_time_epoch_sec = meta.get("start_time_epoch_sec")

    if not provider.has_label(rgb_label):
        record.notes.append("no RGB stream: cannot resolve the domain from content")
        record.complete = False
        return {"record": record, "provider": provider, "qa": None, "scan": None}

    # --- domain from content -------------------------------------------------
    frames = sample_frames_for_domain(provider, count=14, rgb_label=rgb_label)
    evidence = collect_domain_evidence(frames)
    decision = classify_domain(evidence)
    record.estimated_domain = decision.domain
    record.domain_confidence = decision.confidence
    record.domain_rationale = decision.rationale
    record.domain_evidence = decision.evidence
    record.recording_id = recording_id_for(record)

    rgb_ts = provider.rgb_timestamps_ns(rgb_label)
    rate = measure_rate(rgb_ts)
    conf = provider.rgb_config(rgb_label)
    record.duration_s = rate.duration_s
    record.rgb_frame_count = rate.count
    record.rgb_effective_fps = rate.effective_fps
    record.rgb_width, record.rgb_height = conf["width"], conf["height"]

    # --- single RGB decode pass (cached) ------------------------------------
    # Keyed by content hash, not by recording_id: the identifier carries the domain
    # label, and a corrected domain must not silently invalidate a decode pass.
    scan_path = work / "rgb_scan" / f"{record.sha256[:12]}.npz"
    if scan_path.exists() and not rescan:
        log.info("reusing cached RGB scan %s", scan_path)
        scan = RgbScan.load(scan_path)
        if int(scan.frame_index.size) != rate.count:
            log.warning("cached scan has %d frames, stream has %d: rescanning",
                        scan.frame_index.size, rate.count)
            scan = scan_rgb_stream(provider, record.recording_id, record.sha256, rgb_label)
            scan.save(scan_path)
        # The cache holds image features only; identity always comes from the
        # current domain decision, never from whatever was stored earlier.
        scan.recording_id = record.recording_id
    else:
        scan = scan_rgb_stream(provider, record.recording_id, record.sha256, rgb_label)
        scan.save(scan_path)

    # A file that ends mid-write usually leaves the last frames undecodable; the
    # scan completing over the full stream is our completeness evidence.
    record.complete = bool(scan.frame_index.size == rate.count)
    if not record.complete:
        record.notes.append(
            f"RGB scan decoded {scan.frame_index.size}/{rate.count} frames")

    qa = run_stream_qa(provider, record.recording_id, record.estimated_domain,
                       record.absolute_path, record.sha256 or "", scan, rgb_label)
    return {"record": record, "provider": provider, "qa": qa, "scan": scan}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=".")
    ap.add_argument("--reports", default="reports/article1_motorcycle_ingestion")
    ap.add_argument("--work", default="output/article1/ingestion")
    ap.add_argument("--rescan", action="store_true",
                    help="ignore the cached RGB scan and decode again")
    ap.add_argument("--log-level", default="INFO")
    args = ap.parse_args()
    setup_logging(args.log_level)

    root = Path(args.root).resolve()
    reports = Path(args.reports)
    work = Path(args.work)
    reports.mkdir(parents=True, exist_ok=True)
    work.mkdir(parents=True, exist_ok=True)

    cfg = Config.load()
    cache = HashCache(work / "hash_cache.json")

    log.info("scanning %s for acquisition files", root)
    records = scan_files(root, hash_cache=cache)
    # Do not inventory this run's own products: a manifest that lists itself gets a
    # different hash on every run and says nothing about the acquisitions.
    own = reports.resolve()
    records = [r for r in records if own not in Path(r.absolute_path).parents]
    log.info("found %d candidate data files", len(records))

    probed: List[Dict[str, Any]] = []
    for rec in records:
        if rec.kind != "vrs_recording":
            continue
        log.info("probing %s (%.1f MB)", rec.relative_path, rec.size_bytes / 1e6)
        try:
            probed.append(_probe_vrs(rec, work, cfg, args.rescan))
        except Exception as exc:
            rec.notes.append(f"probe failed: {exc}")
            rec.complete = False
            log.error("probe failed for %s: %s", rec.relative_path, exc)

    # --- candidate selection -------------------------------------------------
    by_domain: Dict[str, List] = {}
    for p in probed:
        by_domain.setdefault(p["record"].estimated_domain, []).append(p)

    selection: Dict[str, Any] = {}
    for domain in ("car", "motorcycle"):
        cands = by_domain.get(domain, [])
        entries = [{
            "recording_id": c["record"].recording_id,
            "relative_path": c["record"].relative_path,
            "sha256": c["record"].sha256,
            "size_bytes": c["record"].size_bytes,
            "modified_iso": c["record"].modified_iso,
            "duration_s": c["record"].duration_s,
            "rgb_frame_count": c["record"].rgb_frame_count,
            "rgb_effective_fps": c["record"].rgb_effective_fps,
            "num_streams": c["record"].num_streams,
            "domain_confidence": c["record"].domain_confidence,
            "complete": c["record"].complete,
        } for c in cands]
        chosen = None
        rationale: List[str] = []
        if not cands:
            rationale.append(f"no {domain} recording found")
        elif len(cands) == 1:
            chosen = cands[0]["record"].recording_id
            rationale.append(f"exactly one {domain} candidate: no ambiguity")
        else:
            # Prefer complete recordings, then the richest stream set, then the
            # longest. Recency alone is never the deciding factor.
            ranked = sorted(cands, key=lambda c: (
                bool(c["record"].complete), c["record"].num_streams or 0,
                c["record"].duration_s or 0.0), reverse=True)
            chosen = ranked[0]["record"].recording_id
            rationale.append(
                f"{len(cands)} {domain} candidates; selected the complete recording "
                "with the richest stream set and the longest duration")
            for c in ranked[1:]:
                rationale.append(
                    f"not selected: {c['record'].relative_path} "
                    f"(complete={c['record'].complete}, "
                    f"streams={c['record'].num_streams}, "
                    f"duration={c['record'].duration_s})")
        selection[domain] = {"selected_recording_id": chosen,
                             "candidates": entries, "rationale": rationale}

    # --- manifest ------------------------------------------------------------
    manifest = {
        "schema": "article1_acquisition_manifest_v1",
        "generated_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "project_root_local": str(root),
        "domain_resolution": {
            "method": "static_ego_structure_v1",
            "inputs": ["sampled RGB frames"],
            "explicitly_not_used": [
                "file name", "rgb frame rate", "frame count", "recording profile name",
            ],
            "note": ("The car is currently recorded at ~10 fps and the motorcycle at "
                     "~15 fps. That difference is a temporary acquisition artefact and "
                     "is never an input to domain resolution."),
        },
        "selection": selection,
        "files": [p["record"].to_dict() for p in probed]
                 + [r.to_dict() for r in records if r.kind != "vrs_recording"],
    }
    atomic_write_json(reports / "acquisition_manifest.json", manifest)

    flat = []
    for r in manifest["files"]:
        flat.append({k: (json.dumps(v) if isinstance(v, (list, dict)) else v)
                     for k, v in r.items()})
    pd.DataFrame(flat).to_csv(reports / "acquisition_manifest.csv", index=False)
    log.info("wrote acquisition manifest with %d files", len(flat))

    # --- stream QA + gap CSVs ------------------------------------------------
    for p in probed:
        rec, qa = p["record"], p["qa"]
        if qa is None:
            continue
        domain = rec.estimated_domain
        suffix = {"car": "auto", "motorcycle": "moto"}.get(domain, domain)
        if selection.get(domain, {}).get("selected_recording_id") != rec.recording_id:
            suffix = f"{suffix}_{rec.recording_id[-6:]}"
        atomic_write_json(reports / f"stream_qa_{suffix}.json", qa)
        rows = gaps_to_rows(qa)
        pd.DataFrame(rows if rows else [{
            "recording_id": rec.recording_id, "domain": domain,
            "stream_label": "", "note": "no flagged interval in any stream"}]
        ).to_csv(reports / f"timestamp_gaps_{suffix}.csv", index=False)
        log.info("wrote stream QA for %s (%s)", rec.recording_id, suffix)

    print(json.dumps({d: v["selected_recording_id"] for d, v in selection.items()},
                     indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
