#!/usr/bin/env python3
"""Phase 7: lightweight hand-visibility audit for the motorcycle.

Produces review candidates, an agreement table and QA sequences. It never reports
recall, precision or accuracy, because no reviewed hand ground truth exists.

Absence of a hand is a normal outcome on a motorcycle and is never counted as a
model failure; the failures that are counted are masks without visual support and
masks that outlive the hand.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from aria_drive_seg.ingestion.hand_audit import (AGREEMENT_CASES, HandSignals,
                                                 attributes_for, local_patch_metrics,
                                                 propose_state, summarise)
from aria_drive_seg.ingestion.streams import associate_by_timestamp
from aria_drive_seg.ingestion.timeline import frames_for_seconds, measure_rate
from aria_drive_seg.io_utils import atomic_write_json
from aria_drive_seg.logging_utils import get_logger, setup_logging

log = get_logger("hand.audit")


# A hand on a grip, seen from a head-mounted camera, occupies a small but not
# negligible part of the frame. The open-vocabulary probe regularly returns a box
# covering a whole quadrant and calls it "a hand" — on this segment the largest
# such region was 19.7% of the frame, which is the fairing, not a hand. Detections
# outside this band are kept in the record but not treated as hand evidence.
MIN_PLAUSIBLE_HAND_AREA = 0.0005     # ~1500 px at 2016x1512
MAX_PLAUSIBLE_HAND_AREA = 0.05       # 5% of the frame


def _load_proxy(path: Optional[str]) -> Dict[int, Dict[str, Any]]:
    if not path or not Path(path).exists():
        return {}
    df = pd.read_parquet(path)
    out: Dict[int, Dict[str, Any]] = {}
    for r in df.itertuples():
        detections = json.loads(r.detections_json)
        plausible = [d for d in detections
                     if MIN_PLAUSIBLE_HAND_AREA <= d.get("area_fraction", 0.0)
                     <= MAX_PLAUSIBLE_HAND_AREA]
        out[int(r.source_frame_index)] = {
            "present": bool(plausible),
            "raw_present": bool(r.hand_mask_present),
            "score": max((d["score"] for d in plausible), default=0.0),
            "area_fraction": float(sum(d.get("area_fraction", 0.0)
                                       for d in plausible)),
            "raw_area_fraction": float(r.hand_area_fraction),
            "detections": plausible,
            "raw_detections": len(detections),
            "implausible_detections": len(detections) - len(plausible),
        }
    return out


def _proxy_overlap(detections: List[Dict[str, Any]],
                   centroid: Optional[List[float]], radius_px: float = 160.0
                   ) -> Optional[float]:
    """Crude agreement between the proxy region and the tracked hand position."""
    if centroid is None or not detections:
        return None
    best = 0.0
    for d in detections:
        cx, cy = d.get("centroid_x"), d.get("centroid_y")
        if cx is None or cy is None:
            continue
        dist = float(np.hypot(cx - centroid[0], cy - centroid[1]))
        best = max(best, max(0.0, 1.0 - dist / radius_px))
    return best


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", required=True, help="segment run containing frames/")
    ap.add_argument("--hand-tracking", required=True,
                    help="whole-recording hand tracking parquet")
    ap.add_argument("--proxy", default=None, help="hand proxy parquet")
    ap.add_argument("--reports", default="reports/article1_motorcycle_ingestion")
    ap.add_argument("--association-window-s", type=float, default=0.10)
    ap.add_argument("--sequence-length", type=int, default=5)
    ap.add_argument("--log-level", default="INFO")
    args = ap.parse_args()
    setup_logging(args.log_level)

    import cv2

    root = Path(args.frames)
    reports = Path(args.reports)
    reports.mkdir(parents=True, exist_ok=True)

    frames = pd.read_parquet(root / "frames" / "frames.parquet") \
        .sort_values("source_frame_index")
    frames = frames[frames["valid"]].reset_index(drop=True)
    ht = pd.read_parquet(args.hand_tracking)
    proxy = _load_proxy(args.proxy)
    log.info("%d frames, %d hand-tracking samples, proxy for %d frames",
             len(frames), len(ht), len(proxy))

    frame_ts = frames["timestamp_ns"].to_numpy(np.int64)
    rate = measure_rate(frame_ts)
    fps = rate.effective_fps or 0.0
    # the persistence window is declared in seconds, converted with this run's rate
    persistence_frames = frames_for_seconds(0.8, fps) if fps > 0 else 5

    assoc = associate_by_timestamp(
        frame_ts, ht["timestamp_ns"].to_numpy(np.int64), args.association_window_s,
        frames["source_frame_index"].tolist())

    candidates: List[Any] = []
    rows: List[Dict[str, Any]] = []
    agreement_rows: List[Dict[str, Any]] = []
    persisted = {"left": 0, "right": 0}

    for i, (_, frame) in enumerate(frames.iterrows()):
        a = assoc[i]
        sample = ht.iloc[a.nearest_index] if a.nearest_index is not None else None
        p = proxy.get(int(frame["source_frame_index"]), {})
        gray = None
        reference_blur = None
        image_path = root / frame["rectified_path"]

        for side in ("left", "right"):
            tracked = bool(sample[f"{side}_hand_tracked"]) if sample is not None else False
            in_frame = int(sample[f"{side}_landmarks_in_frame"]) if sample is not None else 0
            total = int(sample[f"{side}_landmarks_total"]) if sample is not None else 0
            cx = sample[f"{side}_centroid_x"] if sample is not None else None
            cy = sample[f"{side}_centroid_y"] if sample is not None else None
            centroid = [float(cx), float(cy)] if (cx is not None and not pd.isna(cx)) else None

            local_blur = None
            if centroid is not None:
                if gray is None:
                    img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
                    if img is not None:
                        gray = img
                        reference_blur = float(cv2.Laplacian(gray, cv2.CV_32F).var())
                if gray is not None:
                    local_blur = local_patch_metrics(gray, centroid)["blur_variance"]

            mask_present = bool(p.get("present", False))
            if mask_present and not tracked:
                persisted[side] += 1
            elif tracked or not mask_present:
                persisted[side] = 0

            signals = HandSignals(
                tracking_available=True,
                tracking_valid=tracked,
                tracking_confidence=(float(sample[f"{side}_hand_confidence"])
                                     if sample is not None
                                     and not pd.isna(sample[f"{side}_hand_confidence"])
                                     else None),
                tracking_dt_ms=a.nearest_dt_ms,
                landmarks_total=total, landmarks_in_frame=in_frame,
                centroid_xy=centroid,
                proxy_available=bool(p),
                proxy_mask_present=mask_present,
                proxy_score=p.get("score"),
                proxy_area_fraction=float(p.get("area_fraction", 0.0)),
                proxy_iou_with_tracking_region=_proxy_overlap(
                    p.get("detections", []), centroid),
                local_blur_variance=local_blur,
                reference_blur_variance=reference_blur,
                proxy_persisted_frames=persisted[side],
            )
            c = propose_state(signals)
            c.recording_id = str(frame["recording_id"])
            c.domain = str(frame["domain"])
            c.source_frame_index = int(frame["source_frame_index"])
            c.timestamp_ns = int(frame["timestamp_ns"])
            c.side = side
            candidates.append(c)

            rows.append({
                "recording_id": c.recording_id, "domain": c.domain,
                "source_frame_index": c.source_frame_index,
                "timestamp_ns": c.timestamp_ns,
                "timestamp_s": float(frame["timestamp_s"]),
                "side": side,
                "state_candidate": c.state_candidate,
                "state_confidence": c.state_confidence,
                "evaluable": c.evaluable,
                "missing_mask_penalised": c.missing_mask_penalised,
                "is_ground_truth": False,
                "review_required": True,
                "rationale": " | ".join(c.rationale),
                **{k: v for k, v in attributes_for(c).items()},
            })
            agreement_rows.append({
                "recording_id": c.recording_id,
                "source_frame_index": c.source_frame_index,
                "timestamp_ns": c.timestamp_ns,
                "side": side,
                "agreement_case": c.agreement_case,
                "agreement_label": c.agreement_label,
                "state_candidate": c.state_candidate,
                "hand_tracking_valid": tracked,
                "landmarks_in_frame": in_frame,
                "landmarks_total": total,
                "tracking_dt_ms": a.nearest_dt_ms,
                "proxy_available": bool(p),
                "proxy_mask_present": mask_present,
                "proxy_mask_present_before_plausibility": p.get("raw_present"),
                "proxy_implausible_detections": p.get("implausible_detections"),
                "proxy_score": p.get("score"),
                "proxy_area_fraction": p.get("area_fraction"),
                "proxy_raw_area_fraction": p.get("raw_area_fraction"),
                "proxy_tracking_overlap": signals.proxy_iou_with_tracking_region,
                "local_blur_variance": local_blur,
                "reference_blur_variance": reference_blur,
                "proxy_persisted_frames": persisted[side],
            })

    pd.DataFrame(rows).to_csv(reports / "hand_visibility_candidates.csv", index=False)
    pd.DataFrame(agreement_rows).to_csv(reports / "hand_proxy_agreement.csv", index=False)

    summary = summarise(candidates)
    summary["frames"] = len(frames)
    summary["effective_fps"] = fps
    summary["duration_s"] = rate.duration_s
    summary["persistence_window_s"] = 0.8
    summary["persistence_window_frames"] = persistence_frames
    summary["association_window_s"] = args.association_window_s
    summary["proxy_available"] = bool(proxy)
    summary["proxy_plausibility_bounds"] = {
        "min_area_fraction": MIN_PLAUSIBLE_HAND_AREA,
        "max_area_fraction": MAX_PLAUSIBLE_HAND_AREA,
        "rationale": ("the open-vocabulary probe regularly returns a whole-quadrant "
                      "box labelled 'a hand'; detections outside this area band are "
                      "kept in the record but are not treated as hand evidence"),
    }
    summary["proxy_frames_with_raw_region"] = int(
        sum(1 for v in proxy.values() if v["raw_present"]))
    summary["proxy_frames_with_plausible_region"] = int(
        sum(1 for v in proxy.values() if v["present"]))
    summary["proxy_implausible_detections"] = int(
        sum(v["implausible_detections"] for v in proxy.values()))
    summary["per_state_per_second"] = {
        k: (v / rate.duration_s if rate.duration_s > 0 else None)
        for k, v in summary["per_state"].items()}
    atomic_write_json(reports / "hand_failure_summary.json", summary)

    _write_sequences(frames, rows, root, reports / "hand_qa_sequences",
                     args.sequence_length)

    print(json.dumps({"per_state": summary["per_state"],
                      "per_agreement_case": {
                          k: v["count"] for k, v in
                          summary["per_agreement_case"].items()},
                      "evaluable_fraction": summary["evaluable_fraction"]}, indent=2))
    return 0


def _write_sequences(frames: pd.DataFrame, rows: List[Dict[str, Any]],
                     root: Path, out: Path, length: int) -> None:
    """Save consecutive-frame sequences for the most informative hand situations."""
    import cv2

    out.mkdir(parents=True, exist_ok=True)
    by_frame: Dict[int, Dict[str, str]] = {}
    for r in rows:
        by_frame.setdefault(r["source_frame_index"], {})[r["side"]] = r["state_candidate"]

    wanted = {
        "01_both_hands_absent": lambda s: set(s.values()) == {"not_visible"},
        "02_tracked_but_out_of_frame": lambda s: "out_of_frame" in s.values(),
        "03_hand_visible_candidate": lambda s: any(
            v in ("visible", "partially_visible") for v in s.values()),
        "04_uncertain": lambda s: "uncertain" in s.values(),
        "05_motion_blurred": lambda s: "motion_blurred" in s.values(),
    }

    index = frames.set_index("source_frame_index")
    order = list(frames["source_frame_index"])
    written = {}
    for name, predicate in wanted.items():
        centre = next((f for f in order if predicate(by_frame.get(f, {}))), None)
        if centre is None:
            written[name] = {"found": False,
                             "note": "no frame in this segment matches this situation"}
            continue
        pos = order.index(centre)
        chosen = order[max(0, pos - length // 2): max(0, pos - length // 2) + length]
        d = out / name
        d.mkdir(parents=True, exist_ok=True)
        details = []
        for fi in chosen:
            row = index.loc[fi]
            src = root / row["rectified_path"]
            img = cv2.imread(str(src), cv2.IMREAD_COLOR)
            if img is None:
                continue
            small = cv2.resize(img, (img.shape[1] // 2, img.shape[0] // 2))
            cv2.imwrite(str(d / f"frame_{fi:06d}.jpg"), small,
                        [cv2.IMWRITE_JPEG_QUALITY, 82])
            details.append({"source_frame_index": int(fi),
                            "timestamp_ns": int(row["timestamp_ns"]),
                            "states": by_frame.get(fi, {})})
        (d / "README.md").write_text(
            f"# {name}\n\n"
            f"{length} consecutive frames centred on source frame {centre}.\n\n"
            "These are **candidate** hand states proposed for review, not ground "
            "truth. On a motorcycle an absent hand is a normal observation.\n\n"
            + "\n".join(f"- frame {d_['source_frame_index']} "
                        f"(t={d_['timestamp_ns']}): {d_['states']}" for d_ in details)
            + "\n")
        written[name] = {"found": True, "centre_frame_index": int(centre),
                         "frames": [d_["source_frame_index"] for d_ in details]}

    (out / "index.json").write_text(json.dumps(
        {"status": "review_candidates_only", "is_ground_truth": False,
         "sequences": written}, indent=2) + "\n")


if __name__ == "__main__":
    raise SystemExit(main())
