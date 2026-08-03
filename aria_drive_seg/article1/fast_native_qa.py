"""Deterministic temporal QA cases for native versus 5 Hz semantic gaze."""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any, Dict

import numpy as np

from ..config import Config
from ..io_utils import atomic_write_json
from ..taxonomy import Taxonomy
from .fast_render import _draw_gaze, _overlay


def _spaced_top(frame, score: str, n: int = 10, gap_s: float = .5):
    chosen = []
    for row in frame.sort_values(score, ascending=False).itertuples(index=False):
        ts = int(row.timestamp_ns)
        if all(abs(ts - int(other.timestamp_ns)) >= gap_s * 1e9 for other in chosen):
            chosen.append(row)
        if len(chosen) >= n:
            break
    return chosen


def _paired_gaze(baseline: Path, native: Path, domain: str):
    import pandas as pd
    columns = ["timestamp_ns", "semantic_valid", "segmentation_frame_index",
               "segmentation_dt_ms", "rect_u", "rect_v", "top1_class",
               "foveal_entropy", "possible_mirror_gaze_candidate",
               "p_traffic_sign", "p_vehicle", "p_pedestrian", "p_lane_marking"]
    five = pd.read_parquet(baseline / domain / "gaze" / "semantic_gaze.parquet")
    native_gaze = pd.read_parquet(native / domain / "gaze" / "semantic_gaze.parquet")
    return native_gaze[columns].merge(
        five[columns], on="timestamp_ns", suffixes=("_native", "_5hz"),
        validate="one_to_one")


def select_temporal_qa_cases(baseline_root: str | Path, native_root: str | Path,
                             events_path: str | Path, per_category: int = 10):
    import pandas as pd
    baseline = Path(baseline_root); native = Path(native_root)
    pairs = {domain: _paired_gaze(baseline, native, domain)
             for domain in ("car", "motorcycle")}
    cases = []

    def add(category: str, domain: str, row, score: float,
            event_id: str | None = None):
        cases.append({
            "qa_case_id": f"{category}_{len(cases):04d}", "category": category,
            "domain": domain, "timestamp_ns": int(row.timestamp_ns),
            "event_id": event_id, "selection_score": float(score),
            "native_frame_index": int(row.segmentation_frame_index_native),
            "five_hz_frame_index": int(row.segmentation_frame_index_5hz),
            "native_class": row.top1_class_native,
            "five_hz_class": row.top1_class_5hz,
            "native_dt_ms": float(row.segmentation_dt_ms_native),
            "five_hz_dt_ms": float(row.segmentation_dt_ms_5hz),
            "class_changed": row.top1_class_native != row.top1_class_5hz,
            "manual_review_status": "pending",
        })

    combined = pd.concat([frame.assign(domain=domain)
                          for domain, frame in pairs.items()], ignore_index=True)
    valid = combined[combined.semantic_valid_native.astype(bool)].copy()
    valid["brief_sign_score"] = valid.p_traffic_sign_native * (
        1.0 + (valid.top1_class_native != valid.top1_class_5hz).astype(float))
    for row in _spaced_top(valid, "brief_sign_score", per_category, .35):
        add("brief_traffic_sign", row.domain, row, row.brief_sign_score)

    ordered = valid.sort_values(["domain", "timestamp_ns"]).copy()
    previous = ordered.groupby("domain").top1_class_native.shift(1)
    transition = (((previous == "road_surface") &
                   (ordered.top1_class_native == "vehicle")) |
                  ((previous == "vehicle") &
                   (ordered.top1_class_native == "road_surface")))
    transitions = ordered[transition].assign(transition_score=1.0)
    for row in _spaced_top(transitions, "transition_score", per_category, .25):
        add("road_vehicle_transition", row.domain, row, 1.0)

    for category, score, gap in (
            ("pedestrian", "p_pedestrian_native", .5),
            ("moving_vehicle", "p_vehicle_native", .6),
            ("near_lane_marking", "p_lane_marking_native", .4)):
        for row in _spaced_top(valid, score, per_category, gap):
            add(category, row.domain, row, getattr(row, score))

    events = pd.read_csv(events_path)
    for category, kind in (("roundabout_event", "roundabout_traverse"),
                           ("junction_event", "junction_crossing")):
        subset = events[events.kind == kind].sort_values(
            ["domain", "start_ns"]).head(per_category)
        for event in subset.itertuples(index=False):
            frame = pairs[event.domain]
            midpoint = int((int(event.start_ns) + int(event.end_ns)) // 2)
            position = int(np.argmin(np.abs(frame.timestamp_ns.to_numpy(np.int64) -
                                            midpoint)))
            row = frame.iloc[position]
            add(category, event.domain, row, 1.0, str(event.event_id))

    mirror = combined[(combined.possible_mirror_gaze_candidate_native.astype(bool)) |
                      (combined.possible_mirror_gaze_candidate_5hz.astype(bool))]
    for row in mirror.sort_values(["domain", "timestamp_ns"]).itertuples(index=False):
        add("possible_mirror_all", row.domain, row, 1.0)
    return pd.DataFrame(cases)


@lru_cache(maxsize=4)
def _load_indices(root: Path):
    import pandas as pd
    frames = pd.read_parquet(root / "frames" / "frames.parquet").set_index("frame_index")
    seg = pd.read_parquet(root / "segmentation" / "segmentation_index.parquet").set_index(
        "frame_index")
    return frames, seg


def _panel(root: Path, frame_index: int, u: float, v: float, label: str,
           taxonomy: Taxonomy, alpha: float):
    import cv2
    frames, seg = _load_indices(root)
    frame_row = frames.loc[int(frame_index)]; seg_row = seg.loc[int(frame_index)]
    image = cv2.imread(str(root / str(frame_row.rectified_path)), cv2.IMREAD_COLOR)
    mask = cv2.imread(str(root / str(seg_row.mask_path)), cv2.IMREAD_UNCHANGED)
    rendered = _overlay(image, mask, taxonomy, alpha)
    _draw_gaze(rendered, u, v, 6)
    rendered = cv2.resize(rendered, (480, 360), interpolation=cv2.INTER_AREA)
    cv2.putText(rendered, label, (12, 26), cv2.FONT_HERSHEY_SIMPLEX, .52,
                (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(rendered, label, (12, 26), cv2.FONT_HERSHEY_SIMPLEX, .52,
                (255, 255, 255), 1, cv2.LINE_AA)
    return rendered


def render_temporal_qa_sheets(cases, baseline_root: str | Path,
                              native_root: str | Path, output_dir: str | Path,
                              cfg: Config) -> Dict[str, str]:
    import cv2
    import pandas as pd
    baseline = Path(baseline_root); native = Path(native_root); output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    taxonomy = Taxonomy.load(cfg.resolve(
        cfg.get("semantic_gaze_fast_external.classes")))
    alpha = float(cfg.get("semantic_gaze_fast_external.render.overlay_alpha", .62))
    gaze_tables = {}
    for domain in ("car", "motorcycle"):
        gaze_tables[("5hz", domain)] = pd.read_parquet(
            baseline / domain / "gaze" / "semantic_gaze.parquet").set_index("timestamp_ns")
        gaze_tables[("native", domain)] = pd.read_parquet(
            native / domain / "gaze" / "semantic_gaze.parquet").set_index("timestamp_ns")
    manifests = {}
    for category, group in cases.groupby("category", sort=True):
        tiles = []
        for case in group.itertuples(index=False):
            panels = []
            for run, root, frame_attr, class_attr, dt_attr in (
                    ("5 Hz", baseline / case.domain, "five_hz_frame_index",
                     "five_hz_class", "five_hz_dt_ms"),
                    ("Native", native / case.domain, "native_frame_index",
                     "native_class", "native_dt_ms")):
                key = "5hz" if run == "5 Hz" else "native"
                gaze = gaze_tables[(key, case.domain)].loc[int(case.timestamp_ns)]
                label = (f"{run} | {case.domain} | {getattr(case, class_attr)} | "
                         f"dt={abs(getattr(case, dt_attr)):.1f} ms")
                panels.append(_panel(
                    root, int(getattr(case, frame_attr)), float(gaze.rect_u),
                    float(gaze.rect_v), label, taxonomy, alpha))
            tiles.append(np.hstack(panels))
        columns = 2; rows = int(np.ceil(len(tiles) / columns))
        blank = np.full_like(tiles[0], 245)
        while len(tiles) < rows * columns:
            tiles.append(blank.copy())
        sheet = np.vstack([np.hstack(tiles[i:i + columns])
                           for i in range(0, len(tiles), columns)])
        path = output / f"qa_{category}.jpg"
        cv2.imwrite(str(path), sheet, [cv2.IMWRITE_JPEG_QUALITY, 92])
        manifests[category] = str(path)
    return manifests


def build_temporal_qa(baseline_root: str | Path, native_root: str | Path,
                      output_dir: str | Path, cfg: Config,
                      events_path: str | Path =
                      "reports/article1_behavior_analysis/events/route_events.csv"
                      ) -> Dict[str, Any]:
    import pandas as pd
    output = Path(output_dir); output.mkdir(parents=True, exist_ok=True)
    cases = select_temporal_qa_cases(
        baseline_root, native_root, events_path, per_category=10)
    cases.to_csv(output / "temporal_qa_cases.csv", index=False)
    sheets = render_temporal_qa_sheets(
        cases, baseline_root, native_root, output / "contact_sheets", cfg)
    counts = {str(k): int(v) for k, v in cases.groupby("category").size().items()}
    result = {"schema": "article1_native_temporal_qa_v1",
              "manual_review_status": "pending", "case_counts": counts,
              "contact_sheets": sheets,
              "all_mirror_candidates_included": True}
    atomic_write_json(output / "temporal_qa_manifest.json", result)
    return result
