#!/usr/bin/env python3
"""Phase 2: fetch the OSM road network and map-match both GPS tracks.

The Overpass payload is cached under the local output tree and reused, so the
match is reproducible without network access once the cache is warm. Precise
coordinates never leave that tree: the committed report gets route progress in
metres and a summary of match quality, not positions.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd

from aria_drive_seg.behavior import EXPLORATORY_MARKER
from aria_drive_seg.behavior.mapmatch import match_track
from aria_drive_seg.behavior.osm import bbox_for_track, build_network, fetch_osm
from aria_drive_seg.config import Config
from aria_drive_seg.io_utils import atomic_write_json
from aria_drive_seg.logging_utils import get_logger, setup_logging

log = get_logger("behavior.mapmatch")


def load_tracks(out_root: Path, cfg: Config) -> Dict[str, pd.DataFrame]:
    tracks = {}
    for domain, spec in (cfg.get("recordings") or {}).items():
        path = out_root / spec["recording_id"] / "gps_raw.parquet"
        if not path.exists():
            raise SystemExit(f"missing {path}; run build_article1_behavior_timeline first")
        df = pd.read_parquet(path).dropna(subset=["latitude", "longitude"])
        df = df[(df.latitude.abs() > 1e-7) | (df.longitude.abs() > 1e-7)]
        df = df.sort_values("timestamp_ns").reset_index(drop=True)
        tracks[domain] = df
        log.info("%s: %d GPS samples with coordinates", domain, len(df))
    return tracks


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/article1/behavior_analysis.yaml")
    ap.add_argument("--output", default="output/article1/behavior_analysis")
    ap.add_argument("--reports", default="reports/article1_behavior_analysis")
    ap.add_argument("--no-network", action="store_true",
                    help="use only the on-disk OSM cache")
    ap.add_argument("--log-level", default="INFO")
    args = ap.parse_args()
    setup_logging(args.log_level)

    cfg = Config.load(args.config)
    out_root = Path(args.output)
    tracks = load_tracks(out_root, cfg)

    # One network for both domains: the shared-route analysis needs both tracks
    # matched against the *same* graph, or "same road" would be undefined.
    all_lat = np.concatenate([t.latitude.values for t in tracks.values()])
    all_lon = np.concatenate([t.longitude.values for t in tracks.values()])
    bbox = bbox_for_track(all_lat, all_lon,
                          margin_m=float(cfg.get("map.bbox_margin_m", 300.0)))
    log.info("bbox (s,w,n,e) spans %.3f x %.3f deg",
             bbox[2] - bbox[0], bbox[3] - bbox[1])

    cache_dir = cfg.resolve(cfg.get("map.cache_dir",
                                    "output/article1/behavior_analysis/osm_cache"))
    payload = fetch_osm(bbox, cache_dir,
                        endpoint=cfg.get("map.endpoint"),
                        timeout_s=int(cfg.get("map.timeout_s", 180)),
                        allow_network=not args.no_network)
    log.info("OSM payload: %d elements", len(payload.get("elements", [])))

    lat0, lon0 = float(np.mean(all_lat)), float(np.mean(all_lon))
    net = build_network(payload, lat0, lon0)
    log.info("network: %s", json.dumps(net.summary()["highway_classes"]))

    mm = cfg.get("map.matching") or {}
    summaries: Dict[str, Any] = {}
    for domain, df in tracks.items():
        log.info("matching %s (%d samples)", domain, len(df))
        res = match_track(
            net,
            timestamp_ns=df.timestamp_ns.values,
            latitude=df.latitude.values,
            longitude=df.longitude.values,
            accuracy_m=df.accuracy.values if "accuracy" in df else None,
            speed_mps=df.speed.values if "speed" in df else None,
            max_snap_distance_m=float(mm.get("max_snap_distance_m", 30.0)),
            max_heading_error_deg=float(mm.get("max_heading_error_deg", 60.0)),
            heading_valid_min_speed_mps=float(
                mm.get("heading_valid_min_speed_mps", 2.0)),
        )
        rec_id = (cfg.get("recordings") or {})[domain]["recording_id"]
        dest = out_root / rec_id
        dest.mkdir(parents=True, exist_ok=True)
        res.to_frame().to_parquet(dest / "map_matched.parquet", index=False)
        s = res.summary()
        s["recording_id"] = rec_id
        s["domain"] = domain
        summaries[domain] = s
        log.info("  matched %d/%d (%.1f%%), median snap %.1f m",
                 s["matched"], s["samples"], 100 * s["matched_fraction"],
                 s["snap_distance_m"]["median"] or float("nan"))

    report = {
        "schema": "article1_route_matching_v1",
        "result_status": EXPLORATORY_MARKER,
        "network": net.summary(),
        "network_origin_note": ("projection origin and the OSM cache hold absolute "
                                "coordinates and stay in the local output tree"),
        "matching_parameters": mm,
        "domains": summaries,
    }
    reports = Path(args.reports) / "route"
    reports.mkdir(parents=True, exist_ok=True)
    atomic_write_json(reports / "map_matching_summary.json", report)
    atomic_write_json(out_root / "map_matching_summary.json", report)
    print(json.dumps({d: {"matched": s["matched"], "of": s["samples"],
                          "median_snap_m": s["snap_distance_m"]["median"]}
                      for d, s in summaries.items()}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
