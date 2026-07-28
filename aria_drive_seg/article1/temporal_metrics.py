"""Pre-GT temporal and post-hoc gaze diagnostics."""
from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from ..io_utils import atomic_write_json, atomic_write_text
from ..taxonomy import Taxonomy


def _read(path: Path) -> np.ndarray:
    value = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if value is None:
        raise FileNotFoundError(path)
    return value


def compute_temporal_metrics(run_dir: str | Path, taxonomy: Taxonomy,
                             report_dir: str | Path = "reports") -> dict:
    run, report_dir = Path(run_dir), Path(report_dir)
    metadata_files = sorted((run / "metadata").glob("frame_*.json"))
    rows = [json.loads(path.read_text()) for path in metadata_files]
    frame_ids = [int(row["frame_index"]) for row in rows]
    class_stats = {
        name: {"appearances": 0, "disappearances": 0, "switches": 0,
               "propagated_pixels": 0, "average_ttl": 0.0, "maximum_ttl": 0,
               "supported_by_current_evidence_pct": 0.0,
               "_ttl_sum": 0, "_supported": 0}
        for name in taxonomy.names()
    }
    previous_raw = previous_temporal = previous_thin = None
    raw_switch = temporal_switch = thin_continuity_sum = 0.0
    thin_pairs = 0
    dropout_1 = dropout_2 = 0
    masks_raw, masks_temporal = [], []
    for frame_id in frame_ids:
        stem = f"frame_{frame_id:06d}.png"
        raw = _read(run / "static_masks" / stem)
        temporal = _read(run / "temporal_masks" / stem)
        thin = _read(run / "temporal_thin" / stem)
        provenance = _read(run / "provenance" / stem)
        age = _read(run / "propagation_age" / stem)
        masks_raw.append(raw)
        masks_temporal.append(temporal)
        propagated = np.isin(provenance, [2, 3, 5, 6])
        for cid, name in enumerate(taxonomy.names()):
            stats = class_stats[name]
            current = temporal == cid
            class_propagated = current & propagated
            stats["propagated_pixels"] += int(class_propagated.sum())
            stats["_ttl_sum"] += int(age[class_propagated].sum())
            stats["maximum_ttl"] = max(
                stats["maximum_ttl"],
                int(age[class_propagated].max()) if class_propagated.any() else 0)
            stats["_supported"] += int((class_propagated & (raw == cid)).sum())
            if previous_temporal is not None:
                previous = previous_temporal == cid
                stats["appearances"] += int((current & ~previous).sum())
                stats["disappearances"] += int((previous & ~current).sum())
                stats["switches"] += int(((previous_temporal != temporal) &
                                         (current | previous)).sum())
        if previous_raw is not None:
            raw_switch += float((raw != previous_raw).mean())
            temporal_switch += float((temporal != previous_temporal).mean())
            union = (thin > 0) | (previous_thin > 0)
            if union.any():
                thin_continuity_sum += float(
                    ((thin == previous_thin) & union).sum() / union.sum())
                thin_pairs += 1
        previous_raw, previous_temporal, previous_thin = raw, temporal, thin
    for sequence in (masks_raw,):
        for i in range(1, len(sequence) - 1):
            dropout_1 += int(((sequence[i - 1] == sequence[i + 1]) &
                              (sequence[i] != sequence[i - 1])).sum())
        for i in range(1, len(sequence) - 2):
            dropout_2 += int(((sequence[i - 1] == sequence[i + 2]) &
                              (sequence[i] != sequence[i - 1]) &
                              (sequence[i + 1] != sequence[i - 1])).sum())
    for stats in class_stats.values():
        count = stats["propagated_pixels"]
        stats["average_ttl"] = stats.pop("_ttl_sum") / max(1, count)
        stats["supported_by_current_evidence_pct"] = (
            100 * stats.pop("_supported") / max(1, count))
    n_pairs = max(1, len(frame_ids) - 1)
    summary_file = json.loads((run / "summary.json").read_text())
    result = {
        "disclaimer": (
            "Pre-GT only. Reduced flicker/unknown is not accuracy; persistence "
            "can propagate errors."),
        "frame_count": len(frame_ids),
        "raw_class_switch_rate": raw_switch / n_pairs,
        "temporal_class_switch_rate": temporal_switch / n_pairs,
        "raw_unknown_rate": float(np.mean([np.mean(mask == 0) for mask in masks_raw])),
        "temporal_unknown_rate": float(
            np.mean([np.mean(mask == 0) for mask in masks_temporal])),
        "thin_temporal_continuity": thin_continuity_sum / max(1, thin_pairs),
        "dropout_events_1_frame_pixels": dropout_1,
        "dropout_events_2_frame_pixels": dropout_2,
        "reset_count": sum(int(row["reset_reason_code"] != 0) for row in rows),
        "storage_bytes": summary_file.get("storage_bytes"),
        "modes": summary_file.get("modes", {}),
        "classes": class_stats,
    }
    report_dir.mkdir(parents=True, exist_ok=True)
    atomic_write_json(report_dir / "article1_temporal_pre_gt_metrics.json", result)
    lines = [
        "# Article 1 temporal pre-GT diagnostics", "",
        "**No reviewed GT: lower flicker or unknown is not evidence of accuracy. "
        "Persistence can propagate errors.**", "",
        f"- Frames: {len(frame_ids)}",
        f"- Raw switch rate: {result['raw_class_switch_rate']:.4%}",
        f"- Temporal switch rate: {result['temporal_class_switch_rate']:.4%}",
        f"- Raw unknown: {result['raw_unknown_rate']:.4%}",
        f"- Temporal unknown: {result['temporal_unknown_rate']:.4%}",
        f"- Thin continuity: {result['thin_temporal_continuity']:.4%}",
        f"- Resets: {result['reset_count']}", "",
        "## T0–T4", "",
        "| mode | switch | unknown | propagated px | age | thin px | reset | ms/frame |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for mode, values in result["modes"].items():
        lines.append(
            f"| {mode} | {values.get('mean_temporal_switch_rate', 0):.4%} | "
            f"{values.get('mean_temporal_unknown_rate', 0):.4%} | "
            f"{values.get('mean_propagated_pixels', 0):.0f} | "
            f"{values.get('mean_mean_propagation_age', 0):.3f} | "
            f"{values.get('mean_thin_pixels', 0):.0f} | "
            f"{values.get('mean_reset', 0):.3f} | "
            f"{values.get('mean_elapsed_ms', 0):.1f} |")
    atomic_write_text(
        report_dir / "article1_temporal_pre_gt_metrics.md",
        "\n".join(lines) + "\n")
    return result


def compute_gaze_diagnostics(run_dir: str | Path, frames_dir: str | Path) -> Path | None:
    """Post-hoc only: segmentation has already completed before this function runs."""
    run, frames_dir = Path(run_dir), Path(frames_dir)
    gaze_path = frames_dir / "gaze" / "aligned_gaze.parquet"
    if not gaze_path.exists():
        return None
    gaze = pd.read_parquet(gaze_path).set_index("frame_index")
    records = []
    for metadata_path in sorted((run / "metadata").glob("frame_*.json")):
        metadata = json.loads(metadata_path.read_text())
        frame_id = int(metadata["frame_index"])
        if frame_id not in gaze.index:
            continue
        sample = gaze.loc[frame_id]
        valid = bool(sample.get("valid", False))
        x = int(round(float(sample.get("rect_u", -1))))
        y = int(round(float(sample.get("rect_v", -1))))
        stem = f"frame_{frame_id:06d}.png"
        raw = _read(run / "static_masks" / stem)
        temporal = _read(run / "temporal_masks" / stem)
        inside = valid and 0 <= x < raw.shape[1] and 0 <= y < raw.shape[0]
        raw_conf = _read(run / "static_confidence" / stem)
        temporal_conf = _read(run / "temporal_confidence" / stem)
        provenance = _read(run / "provenance" / stem)
        age = _read(run / "propagation_age" / stem)
        records.append({
            "frame_index": frame_id,
            "capture_timestamp_ns": metadata["capture_timestamp_ns"],
            "valid": inside, "rect_u": x, "rect_v": y,
            "raw_class": int(raw[y, x]) if inside else -1,
            "temporal_class": int(temporal[y, x]) if inside else -1,
            "raw_top1_probability": float(raw_conf[y, x] / 255) if inside else np.nan,
            "temporal_top1_probability": (
                float(temporal_conf[y, x] / 255) if inside else np.nan),
            "raw_unknown_at_gaze": bool(raw[y, x] == 0) if inside else False,
            "temporal_unknown_at_gaze": (
                bool(temporal[y, x] == 0) if inside else False),
            "propagation_age_at_gaze": int(age[y, x]) if inside else 0,
            "provenance_at_gaze": int(provenance[y, x]) if inside else 0,
        })
    if not records:
        return None
    previous_raw = previous_temporal = None
    for record in records:
        if record["valid"]:
            record["raw_gaze_class_switch"] = (
                previous_raw is not None and record["raw_class"] != previous_raw)
            record["temporal_gaze_class_switch"] = (
                previous_temporal is not None and
                record["temporal_class"] != previous_temporal)
            previous_raw = record["raw_class"]
            previous_temporal = record["temporal_class"]
        else:
            record["raw_gaze_class_switch"] = False
            record["temporal_gaze_class_switch"] = False
    output = run / "gaze" / "article1_temporal_gaze_diagnostics.parquet"
    output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(records).to_parquet(output, index=False)
    return output
