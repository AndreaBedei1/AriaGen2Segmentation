#!/usr/bin/env python3
"""Phase 11: choose the frames a human will annotate.

Balanced by duration and scientific need, never by how many frames each recording
happens to contain: the motorcycle is sampled 1.5x more densely than the car and
must not receive 1.5x the annotation budget for that reason alone.

Runs in the VRS I/O environment.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from aria_drive_seg.ingestion.annotation_select import (FrameCandidate, GROUPS,
                                                        build_selection)
from aria_drive_seg.ingestion.rgb_scan import RgbScan
from aria_drive_seg.ingestion.route_align import build_track, progression_of_timestamp
from aria_drive_seg.ingestion.streams import GpsSample, valid_gps
from aria_drive_seg.io_utils import atomic_write_json
from aria_drive_seg.logging_utils import get_logger, setup_logging

log = get_logger("annotation.select")


def _load_scout(path: Path) -> Optional[Dict[str, Any]]:
    if not path.exists():
        return None
    d = np.load(path, allow_pickle=False)
    return {"timestamp_ns": d["timestamp_ns"],
            "source_frame_index": d["source_frame_index"],
            "class_fraction": d["class_fraction"],
            "class_names": [str(x) for x in d["class_names"]]}


def _route_track(work: Path, recording_id: str, domain: str):
    path = work / "gps" / f"{recording_id}.parquet"
    if not path.exists():
        return None
    df = pd.read_parquet(path)
    samples = [GpsSample(int(r.Index), int(r.timestamp_ns), r.latitude, r.longitude,
                         r.altitude, r.accuracy, r.speed, r.utc_time_ms)
               for r in df.itertuples()]
    good = valid_gps(samples)
    if len(good) < 2:
        return None
    return build_track(recording_id, domain, [s.timestamp_ns for s in good],
                       [s.latitude for s in good], [s.longitude for s in good],
                       [s.accuracy for s in good], [s.speed for s in good])


def _hand_states(work: Path, recording_id: str) -> Dict[int, Dict[str, str]]:
    """Coarse per-timestamp hand state from tracking alone (proxy, not GT)."""
    path = work / "hand_tracking" / f"{recording_id}.parquet"
    if not path.exists():
        return {}
    df = pd.read_parquet(path)
    out: Dict[int, Dict[str, str]] = {}
    for r in df.itertuples():
        states = {}
        for side in ("left", "right"):
            tracked = bool(getattr(r, f"{side}_hand_tracked"))
            in_frame = int(getattr(r, f"{side}_landmarks_in_frame"))
            total = int(getattr(r, f"{side}_landmarks_total"))
            if not tracked:
                states[side] = "not_visible"
            elif total and in_frame == 0:
                states[side] = "out_of_frame"
            elif total and in_frame < 0.85 * total:
                states[side] = "partially_visible"
            else:
                states[side] = "visible"
        out[int(r.timestamp_ns)] = states
    return out


class NearestByTimestamp:
    """Nearest-sample lookup over a fixed timestamp index.

    The key array is built once; rebuilding it per frame would make the lookup
    quadratic in the number of samples.
    """

    def __init__(self, mapping: Dict[int, Any], tolerance_ns: int):
        self.mapping = mapping
        self.tolerance_ns = tolerance_ns
        self.keys = np.sort(np.fromiter(mapping.keys(), dtype=np.int64,
                                        count=len(mapping))) if mapping else None

    def __call__(self, ts: int) -> Optional[Any]:
        if self.keys is None or self.keys.size == 0:
            return None
        pos = int(np.searchsorted(self.keys, ts))
        best = None
        for candidate in (pos - 1, pos):
            if 0 <= candidate < self.keys.size:
                delta = abs(int(self.keys[candidate]) - ts)
                if best is None or delta < best[0]:
                    best = (delta, int(self.keys[candidate]))
        if best is None or best[0] > self.tolerance_ns:
            return None
        return self.mapping[best[1]]


def _segment_diagnostics(path: Optional[str]) -> Dict[int, Dict[str, Any]]:
    """Per-frame semantic-camera diagnostics from the frozen baseline run."""
    if not path or not Path(path).exists():
        return {}
    out: Dict[int, Dict[str, Any]] = {}
    for meta_path in sorted((Path(path) / "metadata").glob("*.json")):
        m = json.loads(meta_path.read_text())
        out[int(m["frame_index"])] = {
            "confidence": float(m.get("mean_final_confidence", float("nan"))),
            "entropy": float(m.get("mean_final_entropy", float("nan"))),
            "conflict_fraction": float(m.get("conflict_fraction", 0.0)),
            "fallback": bool(m.get("internal_is_fallback", False)),
            "provenance": {
                "external_mask2former": float(m.get("external_selected_fraction", 0.0)),
                "cockpit_proxy": float(m.get("internal_model_selected_fraction", 0.0)),
                "geometric_cockpit_proxy": float(
                    m.get("geometric_proxy_selected_fraction", 0.0)),
                "dense_other_environment_fill": float(m.get("dense_fill_fraction", 0.0)),
            },
        }
    return out


def _failure_lookup(path: Optional[str]) -> Dict[int, str]:
    if not path or not Path(path).exists():
        return {}
    doc = json.loads(Path(path).read_text())
    best: Dict[int, tuple] = {}
    for e in doc.get("events", []):
        fi = int(e["frame_index"])
        if fi not in best or e["severity"] > best[fi][0]:
            best[fi] = (e["severity"], e["mode"])
    return {k: v[1] for k, v in best.items()}


def build_pool(domain: str, recording_id: str, sha12: str, work: Path,
               scout_dir: Path, segment_dir: Optional[str],
               failure_json: Optional[str]) -> List[FrameCandidate]:
    scan = RgbScan.load(work / "rgb_scan" / f"{sha12}.npz")
    by_index = {int(fi): i for i, fi in enumerate(scan.frame_index)}
    scout = _load_scout(scout_dir / "semantics.npz")
    track = _route_track(work, recording_id, domain)
    hands = NearestByTimestamp(_hand_states(work, recording_id), 150_000_000)
    diagnostics = _segment_diagnostics(segment_dir)
    failures = _failure_lookup(failure_json)

    frames = pd.read_parquet(scout_dir / "frames" / "frames.parquet")
    frames = frames[frames["valid"]].sort_values("source_frame_index")

    scout_by_index: Dict[int, Dict[str, float]] = {}
    if scout:
        names = scout["class_names"]
        for k, fi in enumerate(scout["source_frame_index"]):
            scout_by_index[int(fi)] = {
                n: float(scout["class_fraction"][k][j]) for j, n in enumerate(names)}

    pool: List[FrameCandidate] = []
    for _, row in frames.iterrows():
        fi = int(row["source_frame_index"])
        ts = int(row["timestamp_ns"])
        si = by_index.get(fi)
        diag = diagnostics.get(fi, {})
        progression = segment = None
        if track is not None:
            progression, dt_s = progression_of_timestamp(track, ts)
            if progression is not None:
                segment = f"q{min(3, int(progression * 4))}"
        pool.append(FrameCandidate(
            domain=domain, recording_id=recording_id,
            source_frame_index=fi, timestamp_ns=ts,
            timestamp_s=float(row["timestamp_s"]),
            image_path=str(Path(scout_dir) / row["rectified_path"]),
            mean_luminance=float(scan.mean_luminance[si]) if si is not None else 0.0,
            blur_variance=float(scan.blur_variance[si]) if si is not None else 0.0,
            frame_difference=float(scan.frame_difference[si]) if si is not None else 0.0,
            dhash=int(scan.dhash[si]) if si is not None else 0,
            class_fraction=scout_by_index.get(fi, {}),
            confidence=diag.get("confidence"),
            entropy=diag.get("entropy"),
            provenance=diag.get("provenance", {}),
            fallback=bool(diag.get("fallback", False)),
            conflict_fraction=diag.get("conflict_fraction"),
            failure_mode_candidate=failures.get(fi),
            hand_visibility_candidate=hands(ts) or {},
            route_progression=progression,
            route_segment=segment,
            has_semantic_camera_output=bool(diag),
        ))
    log.info("%s pool: %d candidate frames (scout semantics for %d, "
             "semantic-camera diagnostics for %d)",
             domain, len(pool), len(scout_by_index), len(diagnostics))
    return pool


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest",
                    default="reports/article1_motorcycle_ingestion/acquisition_manifest.json")
    ap.add_argument("--work", default="output/article1/ingestion")
    ap.add_argument("--reports", default="reports/article1_motorcycle_ingestion")
    ap.add_argument("--segment-dir", default=None,
                    help="frozen baseline semantic_camera directory (motorcycle)")
    ap.add_argument("--failure-json", default=None)
    ap.add_argument("--external-validation", type=int, default=30)
    ap.add_argument("--cockpit-training", type=int, default=40)
    ap.add_argument("--failure-mode-review", type=int, default=15)
    # 2 s at the 1 Hz scouting density is two distinct scout frames, about 19 m of
    # road at the car's average speed. Larger values cannot be met by the 376 s car
    # recording without starving it relative to the 1004 s motorcycle, which would
    # unbalance the very thing the quotas exist to balance.
    ap.add_argument("--min-separation-s", type=float, default=2.0)
    ap.add_argument("--min-hamming", type=int, default=8)
    ap.add_argument("--log-level", default="INFO")
    args = ap.parse_args()
    setup_logging(args.log_level)

    manifest = json.loads(Path(args.manifest).read_text())
    work, reports = Path(args.work), Path(args.reports)
    by_id = {f["recording_id"]: f for f in manifest["files"] if f.get("recording_id")}

    pools: Dict[str, List[FrameCandidate]] = {}
    for domain, sel in manifest["selection"].items():
        rec = sel["selected_recording_id"]
        if not rec:
            continue
        entry = by_id[rec]
        pools[domain] = build_pool(
            domain, rec, entry["sha256"][:12], work,
            work / "scout" / rec,
            args.segment_dir if domain == "motorcycle" else None,
            args.failure_json if domain == "motorcycle" else None)

    # Equal quotas per domain. The car is a shorter recording and the motorcycle is
    # sampled more densely; neither fact should buy annotation budget.
    quota = {"external_validation": args.external_validation,
             "cockpit_training": args.cockpit_training,
             "failure_mode_review": args.failure_mode_review}
    per_domain = {d: dict(quota) for d in pools}
    # the failure-mode group only exists where the frozen baseline actually ran
    for d in per_domain:
        if not any(c.has_semantic_camera_output for c in pools[d]):
            per_domain[d]["failure_mode_review"] = 0

    result = build_selection(pools, per_domain, args.min_separation_s, args.min_hamming)
    result["quota_rationale"] = [
        "equal per-domain quotas: the annotation budget follows scientific need, "
        "not the number of frames a recording happens to contain",
        "the motorcycle is sampled at 15 fps and the car at 10 fps; selecting per "
        "frame would have given the motorcycle 1.5x the budget for no reason",
        "candidates are drawn from a 1 Hz scouting subsample of each recording, so "
        "the two pools are already time-uniform rather than frame-uniform",
        "diversity constraints are expressed in seconds, so they mean the same thing "
        "in both recordings",
    ]
    result["pool_sizes"] = {d: len(p) for d, p in pools.items()}

    atomic_write_json(reports / "annotation_selection.json", result)
    rows = []
    for s in result["selected"]:
        rows.append({
            **{k: v for k, v in s.items()
               if k not in ("strata_covered", "expected_classes",
                            "hand_visibility_candidate", "provenance")},
            "strata_covered": ";".join(s["strata_covered"]),
            "expected_classes": ";".join(s["expected_classes"]),
            "hand_visibility_candidate": json.dumps(s["hand_visibility_candidate"]),
            "provenance": json.dumps(s["provenance"]),
        })
    pd.DataFrame(rows).to_csv(reports / "annotation_selection.csv", index=False)

    print(json.dumps({"counts": result["counts"],
                      "pool_sizes": result["pool_sizes"],
                      "per_domain_group_stats": result["per_domain_group_stats"],
                      "strata": len(result["stratum_coverage"])}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
