#!/usr/bin/env python3
"""Phase 13: figures, maps and the local HTML dashboard.

Every committed figure uses relative coordinates in metres and has both ends of
the track trimmed. Absolute positions stay in the local output tree.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from aria_drive_seg.behavior import EXPLORATORY_MARKER
from aria_drive_seg.behavior.indices import road_complexity_index
from aria_drive_seg.behavior.privacy import redact_track
from aria_drive_seg.config import Config
from aria_drive_seg.io_utils import atomic_write_json, atomic_write_text
from aria_drive_seg.logging_utils import get_logger, setup_logging

log = get_logger("behavior.visualise")

DOMAIN_COLOUR = {"car": "#1f77b4", "motorcycle": "#d62728"}
FOOTER = f"exploratory pilot - one participant, one session per vehicle"

#: The twenty figures the analysis plan asks for, in its own order. Numbers 21
#: and up are supplementary. A run that cannot produce one of these fails rather
#: than quietly shipping a short gallery: a missing figure is a missing result.
REQUIRED_FIGURES = (
    "01_route_map",
    "02_shared_route_map",
    "03_junction_roundabout_map",
    "04_route_dominant_gaze",
    "05_route_heart_rate",
    "06_route_gps_speed_mps",
    "07_route_acceleration",
    "08_solid_line_candidates",
    "09_gaze_heatmap_car",
    "10_gaze_heatmap_motorcycle",
    "11_gaze_transitions",
    "12_gaze_class_distribution",
    "13_event_heart_rate",
    "14_event_gaze",
    "15_scatter_heart_rate_bpm_complexity",
    "16_scatter_gaze_entropy_complexity",
    "17_ppg_quality_timeline",
    "18_speed_acceleration_timeline",
    "19_head_motion_timeline",
    "20_paired_comparison",
)


def missing_required(produced: List[str]) -> List[str]:
    """Required figures that a run did not write."""
    names = {Path(p).name for p in produced}
    return [stem for stem in REQUIRED_FIGURES
            if not any(n.startswith(stem) for n in names)]


def _save(fig, path: Path, produced: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.text(0.99, 0.01, FOOTER, ha="right", va="bottom", fontsize=6,
             color="#666666")
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    produced.append(str(path))


def _nearest_index(source_ns, target_ns, tolerance_s: float):
    """For each target timestamp, the nearest source row and whether it is close
    enough to be used. No interpolation and no fabricated sample: a target with
    no source inside the tolerance is dropped, never filled."""
    src = np.asarray(source_ns, dtype=np.int64)
    tgt = np.asarray(target_ns, dtype=np.int64)
    if src.size == 0 or tgt.size == 0:
        return np.zeros(tgt.size, int), np.zeros(tgt.size, bool)
    order = np.argsort(src)
    ssrc = src[order]
    if ssrc.size == 1:
        pick = np.zeros(tgt.size, int)
    else:
        idx = np.clip(np.searchsorted(ssrc, tgt), 1, ssrc.size - 1)
        pick = np.where(tgt - ssrc[idx - 1] <= ssrc[idx] - tgt, idx - 1, idx)
    ok = np.abs(ssrc[pick] - tgt) <= tolerance_s * 1e9
    return order[pick], ok


def _gaze_on_track(entry: Dict[str, Any], tolerance_s: float = 1.0):
    """Semantically valid gaze samples placed on the redacted track."""
    g = entry.get("gaze")
    if g is None or g.empty:
        return None
    v = g[g["semantic_valid"].astype(bool)]
    if v.empty:
        return None
    track = entry["track"]
    pick, ok = _nearest_index(track["rows"]["timestamp_ns"].values,
                              v["timestamp_ns"].values, tolerance_s)
    if not ok.any():
        return None
    return {"x": track["x"][pick[ok]], "y": track["y"][pick[ok]],
            "rows": v[ok].reset_index(drop=True)}


def load_all(cfg: Config, out_root: Path) -> Dict[str, Dict[str, Any]]:
    data: Dict[str, Dict[str, Any]] = {}
    for domain, spec in (cfg.get("recordings") or {}).items():
        d = out_root / spec["recording_id"]
        entry: Dict[str, Any] = {"recording_id": spec["recording_id"]}
        for key, name in (("matched", "map_matched.parquet"),
                          ("vehicle", "vehicle_dynamics.parquet"),
                          ("head", "head_dynamics_per_frame.parquet"),
                          ("beats", "ppg_beats.parquet"),
                          ("quality", "ppg_quality.parquet"),
                          ("gaze", "semantic_gaze.parquet"),
                          ("bins", "frame_route_bins.parquet")):
            p = d / name
            entry[key] = pd.read_parquet(p) if p.exists() else None
        m = entry["matched"]
        ok = m[m["matched"].astype(bool)].reset_index(drop=True)
        # Endpoint trimming is per track — each has its own ends to hide — but the
        # coordinates must share ONE origin, or the two tracks land on top of each
        # other and a shared-route map shows the shared bins kilometres away from
        # the vehicle that also drove them.
        red = redact_track(ok["matched_x_m"].values, ok["matched_y_m"].values,
                           float(cfg.get("privacy.endpoint_redaction_m", 250.0)))
        entry["_keep"] = red.keep
        entry["_raw"] = ok
        entry["_redaction"] = red.to_dict()
        data[domain] = entry

    # Common origin: the first retained point of whichever track is longest, so
    # the published frame is still relative and still carries no absolute fix.
    longest = max(data, key=lambda d: int(data[d]["_keep"].sum()))
    ref = data[longest]["_raw"][data[longest]["_keep"]]
    ox = float(ref["matched_x_m"].values[0])
    oy = float(ref["matched_y_m"].values[0])
    for domain, entry in data.items():
        keep = entry.pop("_keep")
        raw = entry.pop("_raw")
        entry["track"] = {
            "x": raw["matched_x_m"].values[keep] - ox,
            "y": raw["matched_y_m"].values[keep] - oy,
            "rows": raw[keep].reset_index(drop=True),
            "redaction": {**entry.pop("_redaction"),
                          "shared_origin_domain": longest},
        }
    return data


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/article1/behavior_analysis.yaml")
    ap.add_argument("--output", default="output/article1/behavior_analysis")
    ap.add_argument("--figures", default="output/article1/behavior_analysis/figures")
    ap.add_argument("--reports", default="reports/article1_behavior_analysis")
    ap.add_argument("--log-level", default="INFO")
    args = ap.parse_args()
    setup_logging(args.log_level)

    cfg = Config.load(args.config)
    out_root = Path(args.output)
    figs = Path(args.figures)
    data = load_all(cfg, out_root)
    produced: List[str] = []

    # 1-2. Route maps, and the shared stretch ---------------------------------
    fig, ax = plt.subplots(figsize=(9, 7))
    for domain, e in data.items():
        ax.plot(e["track"]["x"], e["track"]["y"], lw=1.6, alpha=.85,
                color=DOMAIN_COLOUR[domain], label=f"{domain} (redacted ends)")
    ax.set_aspect("equal"); ax.grid(alpha=.3)
    ax.set_xlabel("east (m, relative)"); ax.set_ylabel("north (m, relative)")
    ax.set_title("Route of both recordings\nrelative coordinates, endpoints trimmed")
    ax.legend()
    _save(fig, figs / "01_route_map.png", produced)

    pairs = pd.read_csv(Path(args.reports) / "route" / "paired_route_segments.csv")
    bin_size = float(json.loads((Path(args.reports) / "route" /
                                 "route_alignment_summary.json").read_text()
                                )["analysis_bin_size_m"])
    paired_keys = set(pairs[(pairs.bin_size_m == bin_size) &
                            pairs.paired.astype(bool)]["bin_key"])
    moto = data["motorcycle"]
    xs, ys = [], []
    if moto["bins"] is not None:
        tr = moto["track"]["rows"]
        keys = tr["timestamp_ns"].values
        sel = moto["bins"][moto["bins"]["bin_key"].isin(paired_keys)]
        for t in sel["timestamp_ns"].values:
            if keys.size == 0:
                break
            k = int(np.argmin(np.abs(keys - t)))
            if abs(keys[k] - t) < 2e9:
                xs.append(moto["track"]["x"][k]); ys.append(moto["track"]["y"][k])

    # The car drove only the shared stretch, so at whole-route scale it hides
    # under the markers. A zoom panel is the only way to see both.
    fig, (ax, axz) = plt.subplots(1, 2, figsize=(15, 6.5))
    for a in (ax, axz):
        a.scatter(xs, ys, s=26, color="#2ca02c", alpha=.55, zorder=2,
                  label=f"shared bins ({len(paired_keys)} x {bin_size:.0f} m)")
        for domain, e in data.items():
            a.plot(e["track"]["x"], e["track"]["y"], lw=1.6, alpha=.9,
                   color=DOMAIN_COLOUR[domain], label=domain,
                   zorder=3 if domain == "car" else 1)
        a.set_aspect("equal"); a.grid(alpha=.3)
        a.set_xlabel("east (m, relative)"); a.set_ylabel("north (m, relative)")
    ax.legend(fontsize=8)
    ax.set_title("Whole route")
    car = data["car"]["track"]
    pad = 250
    axz.set_xlim(car["x"].min() - pad, car["x"].max() + pad)
    axz.set_ylim(car["y"].min() - pad, car["y"].max() + pad)
    axz.set_title("Zoom on the shared stretch")
    fig.suptitle("Stretches driven by both vehicles in the same direction\n"
                 "the car drove only this stretch; the motorcycle drove far beyond it")
    _save(fig, figs / "02_shared_route_map.png", produced)

    # 3. Junctions and roundabouts along the route ----------------------------
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    for ax, (domain, e) in zip(axes, data.items()):
        rows = e["track"]["rows"]
        ax.plot(e["track"]["x"], e["track"]["y"], lw=1, color="#999999")
        rb = rows["road_is_roundabout"].astype(bool).values
        ax.scatter(e["track"]["x"][rb], e["track"]["y"][rb], s=22,
                   color="#ff7f0e", label="roundabout", zorder=3)
        near = rows["distance_to_junction_m"].values <= 30
        ax.scatter(e["track"]["x"][near], e["track"]["y"][near], s=6,
                   color="#9467bd", label="within 30 m of a junction", zorder=2)
        ax.set_aspect("equal"); ax.set_title(domain); ax.grid(alpha=.3); ax.legend()
        ax.set_xlabel("east (m)"); ax.set_ylabel("north (m)")
    fig.suptitle("Mapped junctions and roundabouts encountered")
    _save(fig, figs / "03_junction_roundabout_map.png", produced)

    # 4. Dominant semantic-gaze class along the route -------------------------
    # The frozen semantic block is ~30 s of a much longer drive, so this map is
    # two things at once: where gaze fell, and how little of the route that is.
    gaze_tracks = {d: _gaze_on_track(e) for d, e in data.items()}
    present = sorted({c for gt in gaze_tracks.values() if gt is not None
                      for c in gt["rows"]["top1_class"].unique()})
    palette = dict(zip(present, plt.get_cmap("tab20").colors))
    fig, axes = plt.subplots(len(data), 2, figsize=(14, 6 * len(data)),
                             squeeze=False)
    for (ctx, zoom), (domain, e) in zip(axes, data.items()):
        gt = gaze_tracks[domain]
        for a in (ctx, zoom):
            a.plot(e["track"]["x"], e["track"]["y"], lw=1.2, color="#cccccc",
                   zorder=1)
            a.set_aspect("equal"); a.grid(alpha=.3)
            a.set_xlabel("east (m)"); a.set_ylabel("north (m)")
        if gt is None:
            ctx.set_title(f"{domain}: no semantically valid gaze on the track")
            zoom.set_axis_off()
            continue
        ctx.scatter(gt["x"], gt["y"], s=30, color="#111111", zorder=3,
                    label="frozen semantic block")
        ctx.legend(fontsize=8)
        ctx.set_title(f"{domain}: where the block sits on the whole route")
        for cls in present:
            m = (gt["rows"]["top1_class"] == cls).values
            if m.any():
                zoom.scatter(gt["x"][m], gt["y"][m], s=34, alpha=.85,
                             color=palette[cls], label=f"{cls} ({int(m.sum())})",
                             zorder=3)
        pad = 60
        zoom.set_xlim(gt["x"].min() - pad, gt["x"].max() + pad)
        zoom.set_ylim(gt["y"].min() - pad, gt["y"].max() + pad)
        zoom.legend(fontsize=7, loc="best")
        zoom.set_title(f"{domain}: dominant gaze class ({len(gt['x'])} samples)")
    fig.suptitle("Dominant semantic-gaze class along the route\n"
                 "coloured only where the frozen semantic block exists")
    _save(fig, figs / "04_route_dominant_gaze.png", produced)

    # 6, 22. Value-along-route maps -------------------------------------------
    for idx, (col, label, cmap) in [
            (6, ("gps_speed_mps", "speed (m/s)", "viridis")),
            (22, ("curvature_1_per_m", "|curvature| (1/m)", "magma"))]:
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        for ax, (domain, e) in zip(axes, data.items()):
            rows = e["track"]["rows"]
            v = np.abs(rows[col].values) if "curv" in col else rows[col].values
            s = ax.scatter(e["track"]["x"], e["track"]["y"], c=v, s=10, cmap=cmap)
            fig.colorbar(s, ax=ax, label=label)
            ax.set_aspect("equal"); ax.set_title(domain); ax.grid(alpha=.3)
            ax.set_xlabel("east (m)"); ax.set_ylabel("north (m)")
        fig.suptitle(f"{label} along the route")
        _save(fig, figs / f"{idx:02d}_route_{col}.png", produced)

    # 7. Acceleration map -----------------------------------------------------
    # Vehicle acceleration only — the head IMU never feeds this figure.
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    for ax, (domain, e) in zip(axes, data.items()):
        rows = e["track"]["rows"]
        v = e["vehicle"]
        pick, ok = _nearest_index(v["timestamp_ns"].values,
                                  rows["timestamp_ns"].values, 1.0)
        acc = np.full(len(rows), np.nan)
        acc[ok] = v["acceleration_mps2"].values[pick[ok]]
        lim = float(np.nanmax(np.abs(acc))) if np.isfinite(acc).any() else 1.0
        s = ax.scatter(e["track"]["x"], e["track"]["y"], c=acc, s=10,
                       cmap="coolwarm", vmin=-lim, vmax=lim)
        fig.colorbar(s, ax=ax, label="longitudinal acceleration (m/s^2)")
        ax.set_aspect("equal"); ax.grid(alpha=.3)
        ax.set_title(f"{domain} ({int(np.isfinite(acc).sum())} located samples)")
        ax.set_xlabel("east (m)"); ax.set_ylabel("north (m)")
    fig.suptitle("Vehicle acceleration and deceleration along the route\n"
                 "GPS-derived; the head IMU is not used as vehicle acceleration")
    _save(fig, figs / "07_route_acceleration.png", produced)

    # 5. Heart-rate map -------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    for ax, (domain, e) in zip(axes, data.items()):
        rows = e["track"]["rows"]
        beats = e["beats"]
        hr = np.full(len(rows), np.nan)
        if beats is not None:
            gb = beats[beats["valid"].astype(bool)]
            if len(gb):
                bt = gb["timestamp_ns"].values
                for i, t in enumerate(rows["timestamp_ns"].values):
                    k = int(np.argmin(np.abs(bt - t)))
                    if abs(bt[k] - t) < 5e9:
                        hr[i] = gb["heart_rate_bpm"].values[k]
        s = ax.scatter(e["track"]["x"], e["track"]["y"], c=hr, s=10, cmap="coolwarm")
        fig.colorbar(s, ax=ax, label="heart rate (bpm)")
        ax.set_aspect("equal"); ax.set_title(domain); ax.grid(alpha=.3)
    fig.suptitle("Heart rate along the route (physiological proxy, not medical)")
    _save(fig, figs / "05_route_heart_rate.png", produced)

    # 8. Solid-line candidates ------------------------------------------------
    cand_path = Path(args.reports) / "solid_line" / "solid_line_candidates.csv"
    fig, ax = plt.subplots(figsize=(9, 7))
    for domain, e in data.items():
        ax.plot(e["track"]["x"], e["track"]["y"], lw=1, alpha=.4,
                color=DOMAIN_COLOUR[domain], label=domain)
    if cand_path.exists():
        cand = pd.read_csv(cand_path)
        for domain, e in data.items():
            sub = cand[cand.domain == domain]
            rows = e["track"]["rows"]
            xs, ys = [], []
            for t in sub["start_ns"].values:
                k = int(np.argmin(np.abs(rows["timestamp_ns"].values - t)))
                xs.append(e["track"]["x"][k]); ys.append(e["track"]["y"][k])
            ax.scatter(xs, ys, s=40, marker="x",
                       color=DOMAIN_COLOUR[domain],
                       label=f"{domain} candidates ({len(sub)}, all not_evaluable)")
    ax.set_aspect("equal"); ax.grid(alpha=.3); ax.legend(fontsize=8)
    ax.set_title("Lateral-excursion candidates\nGPS cannot resolve a lane crossing "
                 "at this accuracy; every candidate is not_evaluable")
    _save(fig, figs / "08_solid_line_candidates.png", produced)

    # 9-12. Gaze figures ------------------------------------------------------
    sg = json.loads((Path(args.reports) / "semantic_gaze_summary.json").read_text())
    for domain, e in data.items():
        g = e["gaze"]
        if g is None or g.empty:
            continue
        v = g[g["semantic_valid"].astype(bool)]
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.hexbin(v["rect_u"], 1512 - v["rect_v"], gridsize=32, cmap="inferno",
                  extent=(0, 2016, 0, 1512))
        ax.set_xlim(0, 2016); ax.set_ylim(0, 1512); ax.set_aspect("equal")
        ax.set_title(f"{domain}: gaze density in the rectified image\n"
                     f"{len(v)} samples over {sg['domains'][domain]['blocks'][0]['block']['duration_s']:.0f} s")
        ax.set_xlabel("u (px)"); ax.set_ylabel("v (px, flipped)")
        _save(fig, figs / f"{9 if domain == 'car' else 10:02d}_gaze_heatmap_{domain}.png",
              produced)

        m = sg["domains"][domain]["blocks"][0]
        tm = np.array(m["transition_matrix"], float)
        names = m["transition_class_names"]
        keep = [i for i in range(len(names)) if tm[i].sum() + tm[:, i].sum() > 0]
        if keep:
            fig, ax = plt.subplots(figsize=(7, 6))
            im = ax.imshow(tm[np.ix_(keep, keep)], cmap="Blues")
            ax.set_xticks(range(len(keep)), [names[i] for i in keep], rotation=90,
                          fontsize=7)
            ax.set_yticks(range(len(keep)), [names[i] for i in keep], fontsize=7)
            fig.colorbar(im, ax=ax, label="transitions")
            ax.set_title(f"{domain}: semantic gaze transitions")
            _save(fig, figs / f"11_gaze_transitions_{domain}.png", produced)

    fig, ax = plt.subplots(figsize=(10, 5))
    width = 0.38
    classes: List[str] = []
    for domain in data:
        if domain in sg["domains"]:
            classes = list(sg["domains"][domain]["blocks"][0]["foveal_probability_mass"])
    classes = [c for c in classes if any(
        (sg["domains"][d]["blocks"][0]["foveal_probability_mass"].get(c) or 0) > 0.01
        for d in sg["domains"])]
    xpos = np.arange(len(classes))
    for i, (domain, e) in enumerate(data.items()):
        if domain not in sg["domains"]:
            continue
        mass = sg["domains"][domain]["blocks"][0]["foveal_probability_mass"]
        ax.bar(xpos + (i - .5) * width, [100 * (mass.get(c) or 0) for c in classes],
               width, label=domain, color=DOMAIN_COLOUR[domain])
    ax.set_xticks(xpos, classes, rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("foveal probability mass (%)"); ax.legend(); ax.grid(alpha=.3, axis="y")
    ax.set_title("Where gaze fell, by foveal probability mass\n"
                 "(geometry-robust counterpart to the top-1 share)")
    _save(fig, figs / "12_gaze_class_distribution.png", produced)

    # 13-14. Event-related curves --------------------------------------------
    ev = pd.read_parquet(out_root / "event_response_metrics.parquet")
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
    kinds = ["roundabout_traverse", "junction_crossing", "curve", "straight"]
    for ax, domain in zip(axes, data):
        s = ev[ev.domain == domain]
        vals, labels, ns = [], [], []
        for k in kinds:
            q = s[s.kind == k]["delta_phys_immediate_ppg_heart_rate_bpm"].dropna()
            if len(q):
                vals.append(q.values); labels.append(f"{k}\nn={len(q)}"); ns.append(len(q))
        if vals:
            ax.boxplot(vals, tick_labels=labels)
        ax.axhline(0, color="#888", lw=.8)
        ax.set_title(domain); ax.grid(alpha=.3, axis="y")
        ax.tick_params(labelsize=7)
    axes[0].set_ylabel("HR change vs local baseline (bpm)")
    fig.suptitle("Event-associated heart-rate change (proxy; not a stress measure)")
    _save(fig, figs / "13_event_heart_rate.png", produced)

    # 14. Event-related gaze. The frozen semantic block covers 30 s of each
    # recording, so almost no mapped event has gaze under it. The figure reports
    # that coverage honestly instead of implying a curve that does not exist.
    phases = [("phys_anticipation", "anticipation\n-10..0 s"),
              ("phys_immediate", "immediate\n0..+10 s")]
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
    for ax, domain in zip(axes, data):
        s = ev[ev.domain == domain]
        vals, labels = [], []
        for prefix, label in phases:
            q = s[f"{prefix}_gaze_foveal_entropy"].dropna()
            if len(q):
                vals.append(q.values); labels.append(f"{label}\nn={len(q)}")
        if vals:
            ax.boxplot(vals, tick_labels=labels)
        else:
            ax.text(.5, .5, "no mapped event overlaps\nthe frozen semantic block",
                    ha="center", va="center", fontsize=9, color="#a33",
                    transform=ax.transAxes)
            ax.set_xticks([])
        ax.set_title(f"{domain} ({len(s)} mapped events)")
        ax.grid(alpha=.3, axis="y"); ax.tick_params(labelsize=7)
    axes[0].set_ylabel("foveal gaze entropy (bits)")
    fig.suptitle("Event-related semantic gaze\n"
                 "coverage-limited: no baseline window has gaze, so no delta is "
                 "computable", y=1.06)
    _save(fig, figs / "14_event_gaze.png", produced)

    # 23. Head motion around the same events (supplementary).
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
    for ax, domain in zip(axes, data):
        s = ev[ev.domain == domain]
        vals, labels = [], []
        for k in kinds:
            q = s[s.kind == k]["phys_immediate_head_abs_yaw_rate_rad_s"].dropna()
            if len(q):
                vals.append(q.values); labels.append(f"{k}\nn={len(q)}")
        if vals:
            ax.boxplot(vals, tick_labels=labels)
        ax.set_title(domain); ax.grid(alpha=.3, axis="y"); ax.tick_params(labelsize=7)
    axes[0].set_ylabel("|head yaw rate| (rad/s)")
    fig.suptitle("Head motion around mapped events")
    _save(fig, figs / "23_event_head_motion.png", produced)

    # 16. Gaze entropy vs road complexity ------------------------------------
    # The paired table carries gaze for the car only, so this is built per domain
    # over each block's own bins, with ONE normalisation shared by both domains
    # so that the x axis means the same thing in each.
    bins_meta = pd.read_csv(Path(args.reports) / "route" / "shared_route_bins.csv")
    bins_meta = bins_meta[bins_meta.bin_size_m == bin_size]
    frames = []
    for domain, e in data.items():
        g, fb = e["gaze"], e["bins"]
        if g is None or fb is None or g.empty:
            continue
        v = g[g["semantic_valid"].astype(bool)]
        j = v.merge(fb[["frame_index", "bin_key"]], on="frame_index", how="left")
        j = j.dropna(subset=["bin_key"])
        if j.empty:
            continue
        agg = (j.groupby("bin_key")["foveal_entropy"]
                .agg(mean_entropy="mean", n_samples="size").reset_index())
        agg = agg.merge(
            bins_meta[bins_meta.domain == domain][
                ["bin_key", "curvature_1_per_m", "distance_to_junction_m",
                 "road_lanes"]], on="bin_key", how="left")
        agg["domain"] = domain
        frames.append(agg)
    fig, ax = plt.subplots(figsize=(7, 5))
    if frames:
        allb = pd.concat(frames, ignore_index=True)
        rci = road_complexity_index(
            curvature=allb["curvature_1_per_m"].values,
            junction_density=-allb["distance_to_junction_m"].values,
            lanes=allb["road_lanes"].values)
        allb["road_complexity_index"] = rci.values
        for domain in data:
            sub = allb[allb.domain == domain]
            if sub.empty:
                continue
            ax.scatter(sub["road_complexity_index"], sub["mean_entropy"],
                       s=26 + 2 * sub["n_samples"], alpha=.75,
                       color=DOMAIN_COLOUR[domain],
                       label=f"{domain} (n={len(sub)} bins)")
        ax.legend()
        note = f"components used: {', '.join(rci.components_used)}"
    else:
        ax.text(.5, .5, "no bin carries both gaze and map complexity",
                ha="center", va="center", transform=ax.transAxes, color="#a33")
        note = "no data"
    ax.set_xlabel("road complexity index (normalised within this figure's bins)")
    ax.set_ylabel("mean foveal gaze entropy (bits)")
    ax.grid(alpha=.3)
    ax.set_title(f"Gaze entropy vs road complexity, per {bin_size:.0f} m bin\n"
                 f"frozen semantic blocks only - {note}")
    _save(fig, figs / "16_scatter_gaze_entropy_complexity.png", produced)

    # 15, 24. Scatter against road complexity over the paired bins ------------
    paired = pd.read_csv(Path(args.reports) / "paired" / "paired_route_comparison.csv")
    for n, (ycol, ylabel) in [(15, ("heart_rate_bpm", "heart rate (bpm)")),
                              (24, ("head_angular_speed_rad_s",
                                    "head angular speed (rad/s)"))]:
        fig, ax = plt.subplots(figsize=(7, 5))
        for domain, suffix in (("car", "_car"), ("motorcycle", "_moto")):
            x = paired.get(f"road_complexity_index{suffix}")
            y = paired.get(f"{ycol}{suffix}")
            if x is None or y is None:
                continue
            ax.scatter(x, y, s=24, alpha=.75, color=DOMAIN_COLOUR[domain],
                       label=f"{domain} (n={int(np.isfinite(y).sum())} bins)")
        ax.set_xlabel("road complexity index (map components only)")
        ax.set_ylabel(ylabel); ax.grid(alpha=.3); ax.legend()
        ax.set_title(f"{ylabel} vs road complexity, per {bin_size:.0f} m shared bin")
        _save(fig, figs / f"{n:02d}_scatter_{ycol}_complexity.png", produced)

    # 17-19. Timelines --------------------------------------------------------
    fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=False)
    for ax, (domain, e) in zip(axes, data.items()):
        q = e["quality"]
        if q is None:
            continue
        t = (q["start_ns"].values - q["start_ns"].values[0]) / 1e9
        ax.plot(t, q["sqi"].values, color=DOMAIN_COLOUR[domain], lw=1)
        ax.fill_between(t, 0, 1, where=~q["usable"].astype(bool).values,
                        color="#d62728", alpha=.2, label="rejected window")
        ax.set_ylabel("SQI"); ax.set_title(f"{domain} PPG quality"); ax.grid(alpha=.3)
        ax.legend(fontsize=7)
    axes[-1].set_xlabel("time (s)")
    _save(fig, figs / "17_ppg_quality_timeline.png", produced)

    fig, axes = plt.subplots(2, 1, figsize=(12, 6))
    for ax, (domain, e) in zip(axes, data.items()):
        v = e["vehicle"]
        t = (v["timestamp_ns"].values - v["timestamp_ns"].values[0]) / 1e9
        ax.plot(t, v["speed_mps"], color=DOMAIN_COLOUR[domain], lw=1, label="speed")
        ax2 = ax.twinx()
        ax2.plot(t, v["acceleration_mps2"], color="#888", lw=.8, alpha=.7,
                 label="acceleration")
        ax.set_ylabel("m/s"); ax2.set_ylabel("m/s^2")
        ax.set_title(f"{domain} speed and acceleration (GPS-derived)")
        ax.grid(alpha=.3)
    axes[-1].set_xlabel("time (s)")
    _save(fig, figs / "18_speed_acceleration_timeline.png", produced)

    fig, axes = plt.subplots(2, 1, figsize=(12, 6))
    for ax, (domain, e) in zip(axes, data.items()):
        h = e["head"]
        t = (h["frame_timestamp_ns"].values - h["frame_timestamp_ns"].values[0]) / 1e9
        ax.plot(t, h["angular_speed_rad_s"], lw=.6, color=DOMAIN_COLOUR[domain],
                label="head angular speed")
        ax.plot(t, h["vibration_rms_msec2"], lw=.8, color="#333", alpha=.6,
                label="vibration RMS")
        ax.set_title(f"{domain} head motion (head IMU, not vehicle acceleration)")
        ax.grid(alpha=.3); ax.legend(fontsize=7)
    axes[-1].set_xlabel("time (s)")
    _save(fig, figs / "19_head_motion_timeline.png", produced)

    # 20. Paired comparison ---------------------------------------------------
    stats = json.loads((Path(args.reports) / "paired" /
                        "paired_comparison_summary.json").read_text())
    comps = [c for c in stats["comparisons"]
             if not c.get("skipped") and not c.get("is_place_property")]
    fig, ax = plt.subplots(figsize=(9, 6))
    names = [c["metric"] for c in comps]
    est = [c["cliffs_delta"] for c in comps]
    sig = [c["significant_after_fdr"] for c in comps]
    y = np.arange(len(names))
    ax.barh(y, est, color=["#2ca02c" if s else "#bbbbbb" for s in sig])
    ax.set_yticks(y, names, fontsize=8)
    ax.axvline(0, color="#333", lw=.8)
    ax.set_xlabel("Cliff's delta (motorcycle vs car), green = survives FDR")
    ax.set_title(f"Paired comparison on {stats['paired_bins_with_metrics']} shared "
                 f"{bin_size:.0f} m bins\n"
                 f"effective sample size ~{stats['effective_independent_blocks']} "
                 "independent blocks")
    ax.grid(alpha=.3, axis="x")
    _save(fig, figs / "20_paired_comparison.png", produced)

    # 21. Sensor quality map --------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    for ax, (domain, e) in zip(axes, data.items()):
        rows = e["track"]["rows"]
        s = ax.scatter(e["track"]["x"], e["track"]["y"],
                       c=rows["match_quality"].values, s=10, cmap="RdYlGn",
                       vmin=0, vmax=1)
        fig.colorbar(s, ax=ax, label="map-match quality")
        ax.set_aspect("equal"); ax.set_title(domain); ax.grid(alpha=.3)
    fig.suptitle("Map-matching quality along the route")
    _save(fig, figs / "21_sensor_quality_map.png", produced)

    log.info("wrote %d figures", len(produced))
    absent = missing_required(produced)
    if absent:
        log.error("required figures not produced: %s", ", ".join(absent))
        return 1

    dash = build_dashboard(figs, data, sg, stats, bin_size, args)
    atomic_write_json(figs / "figure_manifest.json", {
        "schema": "article1_behavior_figures_v1",
        "result_status": EXPLORATORY_MARKER,
        "figures": produced, "dashboard": str(dash),
        "privacy": {d: e["track"]["redaction"] for d, e in data.items()},
    })
    print(json.dumps({"figures": len(produced), "dashboard": str(dash)}, indent=2))
    return 0


def build_dashboard(figs: Path, data, sg, stats, bin_size, args) -> Path:
    """A self-contained local HTML page over the figures and headline numbers."""
    dest = Path("output/article1/behavior_analysis/dashboard")
    dest.mkdir(parents=True, exist_ok=True)
    images = sorted(figs.glob("*.png"))
    rel = Path("..") / "figures"

    # How much of each frozen semantic block actually landed on the shared route.
    paired = pd.read_csv(Path(args.reports) / "paired" / "paired_route_comparison.csv")
    counts = {d: int(paired[f"gaze_foveal_entropy{s}"].notna().sum())
              if f"gaze_foveal_entropy{s}" in paired else 0
              for d, s in (("car", "_car"), ("motorcycle", "_moto"))}
    gaze_on_shared = "; ".join(
        f"the {d} block covers {n} of the {len(paired)} paired bins"
        for d, n in counts.items()) + ","

    rows = []
    for c in stats["comparisons"]:
        if c.get("skipped"):
            rows.append(f"<tr class='skip'><td>{c['metric']}</td><td colspan='5'>"
                        f"skipped - {c['reason']}</td></tr>")
            continue
        cls = "sig" if c["significant_after_fdr"] else ""
        if c.get("is_place_property"):
            cls = "skip"
        ci = ("n/a" if c["ci_low"] is None
              else f"[{c['ci_low']:.4g}, {c['ci_high']:.4g}]")
        p = ("n/a" if c["permutation_p_fdr"] is None
             else f"{c['permutation_p_fdr']:.4f}")
        note = (" (place property: zero by construction)"
                if c.get("is_place_property") else "")
        rows.append(
            f"<tr class='{cls}'>"
            f"<td>{c['metric']}{note}</td>"
            f"<td>{c['n_paired_bins']}</td>"
            f"<td>{c['paired_median_difference_moto_minus_car']:+.4g}</td>"
            f"<td>{ci}</td>"
            f"<td>{c['cliffs_delta']:+.2f}</td>"
            f"<td>{p}</td></tr>")

    gallery = "\n".join(
        f"<figure data-name='{p.stem}'><img src='{rel / p.name}' loading='lazy'>"
        f"<figcaption>{p.stem.replace('_', ' ')}</figcaption></figure>"
        for p in images)

    html = f"""<!doctype html>
<meta charset="utf-8">
<title>Article 1 behaviour analysis - exploratory pilot</title>
<style>
 body {{ font-family: system-ui, sans-serif; margin: 0 auto; max-width: 1200px;
        padding: 1.5rem; color: #222; }}
 .banner {{ background:#fff4e5; border-left:4px solid #e8912d; padding:.8rem 1rem;
            margin-bottom:1.2rem; }}
 table {{ border-collapse: collapse; width: 100%; font-size: .85rem; }}
 th, td {{ border-bottom: 1px solid #ddd; padding: .35rem .5rem; text-align: left; }}
 tr.sig {{ background: #eaf7ea; }}
 tr.skip {{ color: #888; font-style: italic; }}
 #gallery {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(380px, 1fr));
             gap: 1rem; }}
 figure {{ margin: 0; border: 1px solid #e2e2e2; border-radius: 6px; padding: .5rem; }}
 figure img {{ width: 100%; height: auto; }}
 figcaption {{ font-size: .78rem; color: #555; margin-top: .3rem; }}
 input, select {{ padding: .35rem; margin-right: .5rem; }}
 h2 {{ margin-top: 2rem; }}
</style>
<h1>Article 1 - multimodal driver behaviour</h1>
<div class="banner">
 <strong>exploratory_pilot.</strong> One participant, one session per vehicle;
 the car was recorded at ~10 fps and the motorcycle at ~15 fps. Nothing here
 generalises to a population, and the PPG is a physiological proxy with no
 medical meaning.
</div>

<h2>Headline</h2>
<ul>
 <li>Shared route: <b>{stats['paired_bins_with_metrics']}</b> paired
     {bin_size:.0f} m bins, about <b>{stats['effective_independent_blocks']}</b>
     independent 30 s blocks.</li>
 <li>Semantic gaze coverage:
     {" ".join(f"{d} {100 * sg['domains'][d]['semantic_coverage']['coverage_fraction']:.1f}%"
               for d in sg['domains'])} of each recording.</li>
 <li>Semantic gaze is <b>absent from the paired comparison</b>: {gaze_on_shared}
     so no bin carries gaze for both vehicles.</li>
</ul>

<h2>Paired comparison (motorcycle minus car)</h2>
<label>filter metric <input id="mf" placeholder="type to filter"></label>
<table id="stats">
 <tr><th>metric</th><th>bins</th><th>median diff</th><th>95% CI</th>
     <th>Cliff's d</th><th>p (FDR)</th></tr>
 {"".join(rows)}
</table>

<h2>Figures</h2>
<label>filter <input id="ff" placeholder="e.g. gaze, route, ppg"></label>
<div id="gallery">{gallery}</div>

<script>
 const wire = (input, sel, attr) => document.getElementById(input)
   .addEventListener('input', e => {{
     const q = e.target.value.toLowerCase();
     document.querySelectorAll(sel).forEach(n => {{
       const hay = (attr ? n.getAttribute(attr) : n.textContent).toLowerCase();
       n.style.display = hay.includes(q) ? '' : 'none';
     }});
   }});
 wire('ff', '#gallery figure', 'data-name');
 wire('mf', '#stats tr:not(:first-child)', null);
</script>
"""
    path = dest / "index.html"
    atomic_write_text(path, html)
    return path


if __name__ == "__main__":
    raise SystemExit(main())
