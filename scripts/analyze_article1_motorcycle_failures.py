#!/usr/bin/env python3
"""Phase 6: visual QA and candidate failure modes of the frozen motorcycle run.

Produces per-class coverage statistics, a candidate failure-mode inventory and
consecutive-frame QA sequences for the main ones.

Everything here is pre-ground-truth: the vocabulary is coverage, stability,
fragmentation, persistence, agreement, fallback share, confidence and entropy. No
accuracy, IoU, precision or recall is computed, because there is nothing to compute
them against yet.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from aria_drive_seg.config import Config
from aria_drive_seg.ingestion.failure_modes import (FAILURE_MODES, detect,
                                                    load_diagnostics, pick_sequences)
from aria_drive_seg.ingestion.rgb_scan import RgbScan
from aria_drive_seg.io_utils import atomic_write_json, read_mask_u16
from aria_drive_seg.logging_utils import get_logger, setup_logging
from aria_drive_seg.taxonomy import Taxonomy

log = get_logger("moto.failures")

# Sequences are produced for the modes that matter most for the motorcycle domain.
SEQUENCE_MODES = (
    "cockpit_absorbed_by_external",
    "mirror_not_detected",
    "instrument_display_not_detected",
    "excessive_fallback",
    "fragmentation",
    "flicker",
    "high_entropy",
    "blur_related_failure",
    "vibration_related_failure",
    "internal_external_fusion_error",
)


def _scan_lookup(work: Path, sha12: str) -> Dict[int, Dict[str, float]]:
    path = work / "rgb_scan" / f"{sha12}.npz"
    if not path.exists():
        return {}
    scan = RgbScan.load(path)
    return {int(fi): {"blur_variance": float(scan.blur_variance[i]),
                      "frame_difference": float(scan.frame_difference[i])}
            for i, fi in enumerate(scan.frame_index)}


def _hand_lookup(path: Optional[str], agreement_path: Optional[str] = None):
    """Per-frame hand states and whether a hand region was actually proposed.

    Both are needed: without the mask-presence side the inventory would report zero
    false hand regions while the audit reports many, because a false region is
    precisely a mask with no visibility support.
    """
    if not path or not Path(path).exists():
        return None, None
    df = pd.read_csv(path)
    states: Dict[int, Dict[str, str]] = {}
    for r in df.itertuples():
        states.setdefault(int(r.source_frame_index), {})[r.side] = r.state_candidate

    masks: Dict[int, bool] = {}
    if agreement_path and Path(agreement_path).exists():
        agreement = pd.read_csv(agreement_path)
        if "proxy_mask_present" in agreement.columns:
            for r in agreement.itertuples():
                fi = int(r.source_frame_index)
                masks[fi] = bool(masks.get(fi, False) or bool(r.proxy_mask_present))
    return states, (masks or None)


def _overlay(rgb, mask, taxonomy: Taxonomy, alpha: float = 0.45):
    import cv2
    colour = taxonomy.colorize(mask)
    out = rgb.copy()
    sel = mask > 0
    out[sel] = (out[sel] * (1 - alpha) + colour[sel] * alpha).astype(np.uint8)
    edges = cv2.Canny((mask.astype(np.uint16) % 256).astype(np.uint8), 0, 1)
    out[edges > 0] = 255
    return out


def _write_sequences(sequences: List[Dict[str, Any]], frames_root: Path,
                     semantic_dir: Path, taxonomy: Taxonomy, out: Path,
                     mask_subdir: str) -> Dict[str, Any]:
    import cv2

    out.mkdir(parents=True, exist_ok=True)
    frames = pd.read_parquet(frames_root / "frames" / "frames.parquet") \
        .set_index("source_frame_index")
    written = []
    for n, seq in enumerate(sequences, start=1):
        name = f"{n:02d}_{seq['mode']}"
        d = out / name
        d.mkdir(parents=True, exist_ok=True)
        tiles = []
        used = []
        for fi in seq["frame_indices"]:
            if fi not in frames.index:
                continue
            row = frames.loc[fi]
            rgb_path = frames_root / row["rectified_path"]
            mask_path = semantic_dir / mask_subdir / f"frame_{fi:06d}.png"
            if not rgb_path.exists() or not mask_path.exists():
                continue
            bgr = cv2.imread(str(rgb_path), cv2.IMREAD_COLOR)
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            mask = read_mask_u16(mask_path)
            over = _overlay(rgb, mask, taxonomy)
            small = (rgb.shape[1] // 3, rgb.shape[0] // 3)
            cv2.imwrite(str(d / f"frame_{fi:06d}_rgb.jpg"),
                        cv2.resize(bgr, small), [cv2.IMWRITE_JPEG_QUALITY, 80])
            cv2.imwrite(str(d / f"frame_{fi:06d}_mask.png"),
                        cv2.resize(taxonomy.colorize(mask)[:, :, ::-1], small,
                                   interpolation=cv2.INTER_NEAREST))
            cv2.imwrite(str(d / f"frame_{fi:06d}_overlay.jpg"),
                        cv2.resize(over[:, :, ::-1], small),
                        [cv2.IMWRITE_JPEG_QUALITY, 80])
            tiles.append(cv2.resize(over[:, :, ::-1], (small[0] // 2, small[1] // 2)))
            used.append({"source_frame_index": int(fi),
                         "timestamp_ns": int(row["timestamp_ns"]),
                         "timestamp_s": float(row["timestamp_s"])})
        if tiles:
            cv2.imwrite(str(d / "contact_sheet.jpg"), np.hstack(tiles),
                        [cv2.IMWRITE_JPEG_QUALITY, 82])
        (d / "README.md").write_text(
            f"# {seq['mode']}\n\n"
            f"{len(used)} consecutive frames centred on source frame "
            f"{seq['centre_frame_index']}.\n\n"
            f"**Candidate** failure mode, severity {seq['severity']:.2f} relative to "
            "this run's own distribution. Not a confirmed error: no reviewed ground "
            "truth exists.\n\n"
            f"Detector evidence: `{json.dumps(seq['evidence'])}`\n\n"
            f"{seq['description']}\n\n"
            + "\n".join(f"- frame {u['source_frame_index']} "
                        f"(t={u['timestamp_s']:.3f} s)" for u in used) + "\n")
        written.append({"directory": name, "mode": seq["mode"],
                        "frames": used, "severity": seq["severity"]})
    return {"sequences": written}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", required=True)
    ap.add_argument("--semantic-camera", required=True)
    ap.add_argument("--mask-subdir", default="final_masks")
    ap.add_argument("--work", default="output/article1/ingestion")
    ap.add_argument("--source-sha12", required=True)
    ap.add_argument("--hand-candidates", default=None)
    ap.add_argument("--hand-agreement", default=None,
                    help="hand_proxy_agreement.csv, for mask presence")
    ap.add_argument("--reports", default="reports/article1_motorcycle_ingestion")
    ap.add_argument("--sequence-length", type=int, default=5)
    ap.add_argument("--config", default="configs/article1/semantic_camera.yaml")
    ap.add_argument("--log-level", default="INFO")
    args = ap.parse_args()
    setup_logging(args.log_level)

    cfg = Config.load(args.config)
    taxonomy = Taxonomy.load(cfg.resolve(cfg.get("semantic_camera.classes")))
    reports = Path(args.reports)
    reports.mkdir(parents=True, exist_ok=True)

    diagnostics = load_diagnostics(
        Path(args.semantic_camera), taxonomy, args.mask_subdir,
        _scan_lookup(Path(args.work), args.source_sha12))
    log.info("loaded diagnostics for %d frames", len(diagnostics))

    hand_states, hand_masks = _hand_lookup(args.hand_candidates,
                                          args.hand_agreement)
    result = detect(diagnostics, taxonomy, hand_states=hand_states,
                    hand_masks_present=hand_masks)
    result["mask_subdir"] = args.mask_subdir
    result["causality_note"] = (
        "this is an offline diagnostic over an already-produced run; it looks at "
        "neighbouring frames in both directions to describe flicker and trailing, "
        "and never feeds back into any mask")

    per_class = []
    for name in taxonomy.names():
        fractions = np.array([d.class_fraction[name] for d in diagnostics])
        components = np.array([d.component_count[name] for d in diagnostics])
        present = fractions > 0
        per_class.append({
            "class": name,
            "presence_fraction": float(present.mean()),
            "mean_pixel_fraction": float(fractions.mean()),
            "median_pixel_fraction_when_present": (
                float(np.median(fractions[present])) if present.any() else 0.0),
            "mean_components_when_present": (
                float(components[present].mean()) if present.any() else 0.0),
            "max_components": int(components.max()) if components.size else 0,
        })
    pd.DataFrame(per_class).to_csv(reports / "motorcycle_class_coverage.csv",
                                   index=False)
    result["per_class_coverage"] = per_class

    sequences = pick_sequences(result, SEQUENCE_MODES, args.sequence_length)
    written = _write_sequences(sequences, Path(args.frames),
                               Path(args.semantic_camera), taxonomy,
                               reports / "qa", args.mask_subdir)
    result["qa_sequences"] = written["sequences"]
    (reports / "qa" / "index.json").write_text(json.dumps({
        "status": "candidate_failure_modes_pre_ground_truth",
        "is_confirmed_error_list": False,
        "sequences": written["sequences"]}, indent=2) + "\n")

    atomic_write_json(reports / "motorcycle_failure_modes.json", result)
    print(json.dumps({
        "frames": result["frames"],
        "duration_s": result["duration_s"],
        "effective_fps": result["effective_fps"],
        "counts_per_mode": {k: v for k, v in result["counts_per_mode"].items() if v},
        "sequences": len(written["sequences"]),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
