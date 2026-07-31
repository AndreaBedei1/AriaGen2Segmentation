#!/usr/bin/env python3
"""Phase 9: preliminary spatial alignment of the car and motorcycle routes.

Alignment is spatial. Neither absolute time nor frame index is used: the two drives
happened two months apart, at different speeds, at different sampling rates.

The car recording is a provisional baseline, so this pairing is exploratory. The
code is written to be re-run unchanged once the car is re-recorded at 15 fps.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from aria_drive_seg.ingestion.route_align import (build_track, detect_route_events,
                                                  pair_tracks)
from aria_drive_seg.ingestion.streams import GpsSample, valid_gps
from aria_drive_seg.io_utils import atomic_write_json
from aria_drive_seg.logging_utils import get_logger, setup_logging

log = get_logger("route.align")


def load_track(path: Path, recording_id: str, domain: str, max_accuracy_m: float):
    df = pd.read_parquet(path)
    samples = [GpsSample(index=int(r.index_), timestamp_ns=int(r.timestamp_ns),
                         latitude=r.latitude, longitude=r.longitude,
                         altitude=r.altitude, accuracy=r.accuracy, speed=r.speed,
                         utc_time_ms=r.utc_time_ms)
               for r in df.rename(columns={"index": "index_"}).itertuples()]
    good = valid_gps(samples, max_accuracy_m=max_accuracy_m)
    log.info("%s: %d/%d GPS samples usable (accuracy <= %.0f m)",
             recording_id, len(good), len(samples), max_accuracy_m)
    track = build_track(
        recording_id, domain,
        [s.timestamp_ns for s in good], [s.latitude for s in good],
        [s.longitude for s in good], [s.accuracy for s in good],
        [s.speed for s in good])
    track.warnings.append(
        f"{len(samples) - len(good)} of {len(samples)} GPS samples discarded: no fix "
        f"or accuracy worse than {max_accuracy_m:.0f} m")
    return track, len(samples), len(good)


def _plot(car, moto, pairs, out: Path) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        log.warning("no plot: %s", exc)
        return
    out.mkdir(parents=True, exist_ok=True)
    accepted = [p for p in pairs if p["accepted"]]

    fig, ax = plt.subplots(1, 3, figsize=(19, 6))

    # full extent: the motorcycle route dwarfs the car route, so draw the car last
    ax[0].plot(moto.longitude, moto.latitude, "-", lw=1.2, color="tab:orange",
               label="motorcycle (15 fps)")
    ax[0].plot(car.longitude, car.latitude, "-", lw=2.4, color="tab:blue",
               label="car (10 fps, provisional)")
    ax[0].set_title("full GPS extent")

    # zoom on the region the car actually covers
    pad = 0.004
    ax[1].plot(moto.longitude, moto.latitude, "-", lw=1.4, color="tab:orange",
               label="motorcycle")
    ax[1].plot(car.longitude, car.latitude, "-", lw=2.4, color="tab:blue", label="car")
    if accepted:
        ax[1].scatter([p["longitude_motorcycle"] for p in accepted],
                      [p["latitude_motorcycle"] for p in accepted],
                      s=14, color="tab:green", zorder=5,
                      label=f"accepted pairs ({len(accepted)})")
    ax[1].set_xlim(car.longitude.min() - pad, car.longitude.max() + pad)
    ax[1].set_ylim(car.latitude.min() - pad, car.latitude.max() + pad)
    ax[1].set_title("zoom on the shared route")

    for a in ax[:2]:
        a.set_xlabel("longitude"); a.set_ylabel("latitude")
        a.legend(fontsize=8); a.grid(alpha=.3)

    if accepted:
        ax[2].scatter([p["progression_motorcycle"] for p in accepted],
                      [p["progression_car"] for p in accepted],
                      s=10, c=[p["pairing_quality"] for p in accepted], cmap="viridis")
        ax[2].set_xlabel("motorcycle route progression")
        ax[2].set_ylabel("car route progression")
        ax[2].set_title("accepted pairings (colour = quality)")
        ax[2].set_xlim(0, 1); ax[2].set_ylim(0, 1); ax[2].grid(alpha=.3)
    else:
        ax[2].text(.5, .5, "no accepted pairing", ha="center")

    fig.suptitle("EXPLORATORY PRELIMINARY — the car recording is a provisional "
                 "10 fps baseline; recompute when both are at 15 fps")
    fig.tight_layout()
    fig.savefig(out / "route_tracks.png", dpi=110)
    plt.close(fig)
    log.info("wrote %s", out / "route_tracks.png")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest",
                    default="reports/article1_motorcycle_ingestion/acquisition_manifest.json")
    ap.add_argument("--work", default="output/article1/ingestion")
    ap.add_argument("--reports", default="reports/article1_motorcycle_ingestion")
    ap.add_argument("--max-accuracy-m", type=float, default=50.0)
    ap.add_argument("--max-distance-m", type=float, default=40.0)
    ap.add_argument("--max-heading-deg", type=float, default=45.0)
    ap.add_argument("--log-level", default="INFO")
    args = ap.parse_args()
    setup_logging(args.log_level)

    manifest = json.loads(Path(args.manifest).read_text())
    work, reports = Path(args.work), Path(args.reports)
    reports.mkdir(parents=True, exist_ok=True)

    ids = {d: v["selected_recording_id"] for d, v in manifest["selection"].items()}
    tracks, counts = {}, {}
    for domain in ("car", "motorcycle"):
        rec = ids.get(domain)
        path = work / "gps" / f"{rec}.parquet"
        if not rec or not path.exists():
            log.error("no GPS export for %s", domain)
            return 2
        tracks[domain], total, good = load_track(path, rec, domain, args.max_accuracy_m)
        counts[domain] = {"total_gps_samples": total, "usable_gps_samples": good}

    car, moto = tracks["car"], tracks["motorcycle"]
    events = {d: detect_route_events(t) for d, t in tracks.items()}
    result = pair_tracks(car, moto, args.max_distance_m, args.max_heading_deg)

    rows = result["pairs"]
    df = pd.DataFrame([{**r, "sensors_used": ",".join(r["sensors_used"]),
                        "warnings": " | ".join(r["warnings"])} for r in rows])
    df.to_csv(reports / "preliminary_route_alignment.csv", index=False)

    summary = {
        "status": "exploratory_preliminary",
        "is_scientific_conclusion": False,
        "alignment_priority_used": [
            "1. GPS — available in both recordings and used here",
            "2. SLAM/VIO trajectory — not used: no MPS/VIO trajectory product is "
            "present in these VRS files, only raw SLAM camera streams",
            "3. relative spatial progression — derived from the GPS track",
            "4. heading — derived from consecutive GPS positions and used as an "
            "acceptance gate",
            "5. cumulative distance — used for the normalised progression",
            "6. visual landmarks — not used",
            "7. scene similarity — not used",
        ],
        "not_used": ["absolute wall-clock time", "frame index", "frame rate"],
        "gps_counts": counts,
        "route_events": events,
        **result["summary"],
        "caveat": (
            "the car recording is a provisional 10 fps development baseline; this "
            "pairing is exploratory and must be recomputed once the car is "
            "re-recorded at the final protocol rate"),
    }
    atomic_write_json(reports / "route_alignment_summary.json", summary)
    _plot(car, moto, rows, reports / "route_alignment_qa")

    print(json.dumps({
        "car_distance_m": car.total_distance_m,
        "motorcycle_distance_m": moto.total_distance_m,
        "candidate_pairs": summary["candidate_count"],
        "accepted_pairs": summary["accepted_count"],
        "accepted_fraction": summary["accepted_fraction"],
        "paired": summary["paired"],
        "common_route": summary.get("common_route"),
        "car_warnings": car.warnings,
        "motorcycle_warnings": moto.warnings,
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
