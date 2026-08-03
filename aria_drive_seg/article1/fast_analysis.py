"""Structured route/event tables derived from full-recording fast semantic gaze."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Sequence

import numpy as np

from ..config import Config
from ..io_utils import atomic_write_json
from ..taxonomy import Taxonomy

RECORDING_IDS = {"car": "car_2e84f0c3e245",
                 "motorcycle": "motorcycle_5ab8604a14df"}
ROAD_CLASSES = ("road_surface", "lane_marking", "regulatory_road_marking",
                "vehicle", "two_wheeler", "pedestrian", "traffic_sign",
                "traffic_light", "road_boundary_or_sidewalk")


def _atomic_parquet(path: Path, frame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp.parquet")
    frame.to_parquet(tmp, index=False); tmp.replace(path)


def build_paired_route_bins(run_dirs: Dict[str, str | Path], cfg: Config,
                            behavior_root: str | Path,
                            route_report: str | Path):
    import pandas as pd
    fast = cfg.get("semantic_gaze_fast_external", {})
    taxonomy = Taxonomy.load(cfg.resolve(fast["classes"]))
    class_names = [c.name for c in taxonomy.classes
                   if c.name not in {"unknown", "mirror"}]
    behavior_root = Path(behavior_root); route_report = Path(route_report)
    route_summary = json.loads(
        (route_report / "route_alignment_summary.json").read_text())
    bin_size = float(route_summary["analysis_bin_size_m"])
    pairs = pd.read_csv(route_report / "paired_route_segments.csv")
    pairs = pairs[(pairs.bin_size_m == bin_size) & pairs.paired.astype(bool)].copy()
    pairs = pairs.sort_values("first_timestamp_ns_a").reset_index(drop=True)
    order = {key: i for i, key in enumerate(pairs.bin_key)}
    rows = []
    for domain, raw_root in run_dirs.items():
        root = Path(raw_root)
        gaze = pd.read_parquet(root / "gaze" / "semantic_gaze.parquet")
        gaze = gaze[gaze.semantic_valid.astype(bool)].copy()
        bins = pd.read_parquet(
            behavior_root / RECORDING_IDS[domain] / "frame_route_bins.parquet")
        bins = bins[["frame_index", "bin_key", "bin_valid"]].drop_duplicates(
            "frame_index")
        joined = gaze.merge(
            bins, left_on="segmentation_frame_index", right_on="frame_index", how="left")
        joined = joined[joined.bin_valid.fillna(False).astype(bool) &
                        joined.bin_key.isin(order)]
        dt_s = float(np.median(np.diff(gaze.timestamp_ns.values)) / 1e9)
        for key, group in joined.groupby("bin_key"):
            base = {"domain": domain, "bin_key": key, "bin_order": order[key],
                    "bin_size_m": bin_size, "gaze_samples": int(len(group)),
                    "gaze_time_s": float(len(group) * dt_s)}
            for name in class_names:
                rows.append({**base, "class_name": name,
                             "foveal_mass_percent": float(group[f"p_{name}"].mean() * 100),
                             "top1_time_percent":
                                 float((group.top1_class == name).mean() * 100)})
    result = pd.DataFrame(rows).sort_values(
        ["bin_order", "domain", "class_name"]).reset_index(drop=True)
    return result


def build_event_metrics(run_dirs: Dict[str, str | Path], events_path: str | Path):
    import pandas as pd
    events = pd.read_csv(events_path)
    kind_map = {"roundabout_traverse": "roundabout",
                "junction_crossing": "junction", "curve": "curve"}
    events = events[events.kind.isin(kind_map)].copy()
    rows = []
    for domain, raw_root in run_dirs.items():
        gaze = pd.read_parquet(Path(raw_root) / "gaze" / "semantic_gaze.parquet")
        gaze = gaze[gaze.semantic_valid.astype(bool)].copy()
        dt_s = float(np.median(np.diff(gaze.timestamp_ns.values)) / 1e9)
        domain_events = events[events.domain == domain]
        for event in domain_events.itertuples(index=False):
            windows = {"pre": (int(event.start_ns - 3e9), int(event.start_ns)),
                       "event": (int(event.start_ns), int(event.end_ns)),
                       "post": (int(event.end_ns), int(event.end_ns + 3e9))}
            for phase, (start, end) in windows.items():
                sample = gaze[(gaze.timestamp_ns >= start) & (gaze.timestamp_ns < end)]
                if sample.empty:
                    continue
                road_mass = sum(sample[f"p_{name}"] for name in ROAD_CLASSES)
                rows.append({
                    "event_id": str(event.event_id), "domain": domain,
                    "event_type": kind_map[event.kind], "phase": phase,
                    "samples": int(len(sample)), "observed_time_s": len(sample) * dt_s,
                    "road_relevant_mass_percent": float(road_mass.mean() * 100),
                    "mean_foveal_entropy": float(sample.foveal_entropy.mean()),
                    "fixation_time_percent": float(sample.is_fixation.mean() * 100),
                })
    return pd.DataFrame(rows)


def build_fast_analysis(run_dirs: Dict[str, str | Path], output_dir: str | Path,
                        cfg: Config,
                        behavior_root: str | Path = "output/article1/behavior_analysis",
                        reports_root: str | Path = "reports/article1_behavior_analysis"):
    import pandas as pd
    out = Path(output_dir); out.mkdir(parents=True, exist_ok=True)
    reports = Path(reports_root)
    paired = build_paired_route_bins(
        run_dirs, cfg, behavior_root, reports / "route")
    events = build_event_metrics(run_dirs, reports / "events" / "route_events.csv")
    _atomic_parquet(out / "paired_route_bins.parquet", paired)
    paired.to_csv(out / "paired_route_bins.csv", index=False)
    _atomic_parquet(out / "event_metrics.parquet", events)
    events.to_csv(out / "event_metrics.csv", index=False)
    summaries = {}
    for domain, root in run_dirs.items():
        root = Path(root)
        summaries[domain] = {
            "extraction": json.loads((root / "frames" / "extraction_summary.json").read_text()),
            "segmentation": json.loads((root / "segmentation" / "summary.json").read_text()),
            "semantic_gaze": json.loads((root / "gaze" /
                                          "semantic_gaze_summary.json").read_text()),
        }
    summary = {"schema": "article1_fast_full_analysis_v1",
               "result_status": "exploratory_pilot", "domains": summaries,
               "paired_route_rows": int(len(paired)),
               "event_metric_rows": int(len(events)),
               "frame_rate_used_as_feature": False,
               "frames_are_independent_samples": False,
               "comparison_units": ["seconds", "percent", "spatial_bins", "events"]}
    atomic_write_json(out / "full_run_summary.json", summary)
    return summary
