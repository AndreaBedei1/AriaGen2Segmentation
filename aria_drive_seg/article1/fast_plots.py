"""Reduced publication figure set with locked car/motorcycle comparison scales."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, Sequence

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from ..behavior.privacy import redact_track
from ..config import Config
from ..io_utils import atomic_write_json
from ..taxonomy import Taxonomy
from .fast_analysis import RECORDING_IDS, ROAD_CLASSES

DOMAIN_COLORS = {"car": "#0072B2", "motorcycle": "#D55E00"}
DOMAIN_LABELS = {"car": "Auto", "motorcycle": "Moto"}
PHASE_ORDER = ["pre", "event", "post"]
FOOTER = "Exploratory pilot — one participant, one session per vehicle"


def shared_limits(values: Iterable[Sequence[float]], include_zero: bool = True,
                  pad_fraction: float = .05) -> tuple[float, float]:
    finite = np.concatenate([np.asarray(v, float).ravel() for v in values])
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return (0.0, 1.0)
    lo, hi = float(finite.min()), float(finite.max())
    if include_zero:
        lo, hi = min(0.0, lo), max(0.0, hi)
    span = hi - lo
    pad = span * pad_fraction if span > 0 else max(abs(hi), 1.0) * pad_fraction
    return lo - pad, hi + pad


def apply_shared_limits(axes, values: Iterable[Sequence[float]], axis: str = "y",
                        include_zero: bool = True) -> tuple[float, float]:
    limits = shared_limits(values, include_zero=include_zero)
    for ax in np.asarray(axes, dtype=object).ravel():
        (ax.set_ylim if axis == "y" else ax.set_xlim)(limits)
    return limits


def _style() -> None:
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 8,
        "axes.labelsize": 9, "axes.titlesize": 10,
        "xtick.labelsize": 7, "ytick.labelsize": 7,
        "legend.fontsize": 7, "axes.spines.top": False,
        "axes.spines.right": False, "pdf.fonttype": 42,
    })


def _save(fig, root: Path, stem: str, manifest: list[dict]) -> None:
    root.mkdir(parents=True, exist_ok=True)
    # Reserve a dedicated footer strip.  With ``bbox_inches='tight'`` a long
    # x-axis label can otherwise reach the pilot-status note on wide figures.
    fig.subplots_adjust(bottom=max(float(fig.subplotpars.bottom), .16))
    fig.text(.99, .006, FOOTER, ha="right", va="bottom", fontsize=6, color="#666")
    png = root / f"{stem}.png"; pdf = root / f"{stem}.pdf"
    fig.savefig(png, dpi=300, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    manifest.append({"stem": stem, "png": str(png), "pdf": str(pdf),
                     "same_scale_car_motorcycle": True})


def _semantic_summaries(run_dirs):
    return {d: json.loads((Path(root) / "gaze" /
                           "semantic_gaze_summary.json").read_text())
            for d, root in run_dirs.items()}


def plot_all(run_dirs: Dict[str, str | Path], analysis_dir: str | Path,
             figures_dir: str | Path, cfg: Config,
             behavior_root: str | Path = "output/article1/behavior_analysis",
             behavior_reports: str | Path = "reports/article1_behavior_analysis"):
    import pandas as pd
    _style(); figures = Path(figures_dir); analysis = Path(analysis_dir)
    behavior_root = Path(behavior_root); behavior_reports = Path(behavior_reports)
    taxonomy = Taxonomy.load(cfg.resolve(
        cfg.get("semantic_gaze_fast_external.classes")))
    classes = [c.name for c in taxonomy.classes
               if c.name not in {"unknown", "mirror"}]
    labels = [name.replace("_", " ") for name in classes]
    manifest: list[dict] = []

    # 1. One shared coordinate system, redacted endpoints.
    tracks = {}
    for domain, rec_id in RECORDING_IDS.items():
        data = pd.read_parquet(behavior_root / rec_id / "map_matched.parquet")
        data = data[data.matched.astype(bool)].reset_index(drop=True)
        red = redact_track(data.matched_x_m.values, data.matched_y_m.values, 250.0)
        tracks[domain] = data[red.keep].copy()
    reference = tracks["motorcycle"]
    ox, oy = float(reference.matched_x_m.iloc[0]), float(reference.matched_y_m.iloc[0])
    fig, ax = plt.subplots(figsize=(7.2, 5.2))
    for domain in ("motorcycle", "car"):
        track = tracks[domain]
        ax.plot(track.matched_x_m - ox, track.matched_y_m - oy,
                color=DOMAIN_COLORS[domain], lw=(2.1 if domain == "car" else 1.3),
                linestyle=("--" if domain == "car" else "-"),
                label=DOMAIN_LABELS[domain])
    ax.set_aspect("equal"); ax.grid(alpha=.18)
    ax.set(xlabel="Est (m, coordinate relative)",
           ylabel="Nord (m, coordinate relative)",
           title="Percorsi registrati e tratto condiviso")
    ax.legend(frameon=False)
    _save(fig, figures, "01_route_map_shared", manifest)

    # 2. Full-recording foveal mass, one common percentage axis.
    summaries = _semantic_summaries(run_dirs)
    y = np.arange(len(classes)); h = .37
    fig, ax = plt.subplots(figsize=(7.2, 6.2))
    for offset, domain in ((-h / 2, "car"), (h / 2, "motorcycle")):
        mass = summaries[domain]["metrics"]["foveal_probability_mass"]
        values = [100 * float(mass.get(name) or 0) for name in classes]
        ax.barh(y + offset, values, h, color=DOMAIN_COLORS[domain],
                label=DOMAIN_LABELS[domain])
    ax.set(yticks=y, yticklabels=labels, xlabel="Massa foveale media (%)",
           title="Distribuzione del gaze semantico sull’intera registrazione")
    ax.invert_yaxis(); ax.grid(axis="x", alpha=.18); ax.legend(frameon=False)
    _save(fig, figures, "02_semantic_gaze_distribution", manifest)

    # 3. Seconds and fixation rate; each metric shares one axis across domains.
    metrics = {d: pd.read_csv(Path(root) / "gaze" / "class_metrics.csv").set_index(
        "class_name") for d, root in run_dirs.items()}
    fig, axes = plt.subplots(1, 2, figsize=(12.0, 6.2), sharey=True)
    for ax, column, xlabel in ((axes[0], "dwell_time_s", "Dwell (s)"),
                               (axes[1], "fixations_per_minute", "Fixations/min")):
        for offset, domain in ((-h / 2, "car"), (h / 2, "motorcycle")):
            values = [float(metrics[domain].loc[name, column] or 0) for name in classes]
            ax.barh(y + offset, values, h, color=DOMAIN_COLORS[domain],
                    label=DOMAIN_LABELS[domain])
        ax.set_xlabel(xlabel); ax.grid(axis="x", alpha=.18)
    axes[0].set(yticks=y, yticklabels=labels); axes[0].invert_yaxis()
    axes[0].legend(frameon=False); fig.suptitle("Dwell e fixation per classe semantica")
    _save(fig, figures, "03_dwell_fixation_by_class", manifest)

    # 4. Shared-route bin profile, identical 0–100% y scale.
    paired_all = pd.read_parquet(analysis / "paired_route_bins.parquet")
    paired = paired_all[paired_all.class_name.isin(ROAD_CLASSES)]
    road = paired.groupby(["domain", "bin_order"], as_index=False).foveal_mass_percent.sum()
    fig, ax = plt.subplots(figsize=(8.2, 3.4))
    for domain in ("car", "motorcycle"):
        data = road[road.domain == domain]
        ax.plot(data.bin_order, data.foveal_mass_percent, lw=1.6,
                color=DOMAIN_COLORS[domain], label=DOMAIN_LABELS[domain])
    ax.set_ylim(0, 100); ax.set(xlabel="Bin spaziale condiviso (50 m, ordine di percorso)",
                               ylabel="Gaze road-relevant (%)",
                               title="Gaze semantico lungo il tratto condiviso")
    ax.grid(alpha=.18); ax.legend(frameon=False)
    _save(fig, figures, "04_semantic_gaze_shared_route", manifest)

    # 5. Event units, not frames; all facets share 0–100%.
    events = pd.read_parquet(analysis / "event_metrics.parquet")
    event_types = [e for e in ("roundabout", "junction", "curve")
                   if e in set(events.event_type)]
    fig, axes = plt.subplots(1, len(event_types), figsize=(4 * len(event_types), 3.5),
                             sharey=True, squeeze=False)
    for ax, event_type in zip(axes[0], event_types):
        data = events[events.event_type == event_type]
        for domain in ("car", "motorcycle"):
            values = [data[(data.domain == domain) & (data.phase == phase)]
                      .road_relevant_mass_percent.mean() for phase in PHASE_ORDER]
            ax.plot(PHASE_ORDER, values, marker="o", color=DOMAIN_COLORS[domain],
                    label=DOMAIN_LABELS[domain])
        ax.set_title(event_type); ax.grid(alpha=.18); ax.set_ylim(0, 100)
    axes[0, 0].set_ylabel("Gaze road-relevant (%)"); axes[0, 0].legend(frameon=False)
    fig.suptitle("Risposta gaze per eventi comparabili (media per evento)")
    _save(fig, figures, "05_event_related_gaze", manifest)

    # 6. Ten-second block summaries avoid treating native rows as independent.
    block_rows = []
    for domain, rec_id in RECORDING_IDS.items():
        vehicle = pd.read_parquet(behavior_root / rec_id / "vehicle_dynamics.parquet")
        head = pd.read_parquet(behavior_root / rec_id / "head_dynamics_per_frame.parquet")
        for table, ts_col, value_col, metric in (
                (vehicle, "timestamp_ns", "speed_mps", "Velocità (m/s)"),
                (head, "timestamp_ns", "angular_speed_rad_s",
                 "Head motion (rad/s)")):
            block = ((table[ts_col] - table[ts_col].min()) // int(10e9)).astype(int)
            for value in table.groupby(block)[value_col].mean().dropna():
                block_rows.append({"domain": domain, "metric": metric, "value": value})
    blocks = pd.DataFrame(block_rows)
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.4))
    for ax, metric in zip(axes, blocks.metric.unique()):
        values = [blocks[(blocks.domain == d) & (blocks.metric == metric)].value.values
                  for d in ("car", "motorcycle")]
        boxes = ax.boxplot(values, labels=["Auto", "Moto"], patch_artist=True,
                           showfliers=False)
        for patch, domain in zip(boxes["boxes"], ("car", "motorcycle")):
            patch.set_facecolor(DOMAIN_COLORS[domain]); patch.set_alpha(.7)
        ax.set_ylabel(metric); ax.grid(axis="y", alpha=.18)
    fig.suptitle("Velocità e movimento della testa (finestre reali di 10 s)")
    _save(fig, figures, "06_speed_head_motion", manifest)

    # 7. Event-level PPG: immediate minus baseline, one common delta scale.
    wide = pd.read_parquet(behavior_root / "event_response_metrics.parquet")
    wide["event_type"] = wide.kind.map(
        {"roundabout_traverse": "roundabout", "junction_crossing": "junction",
         "curve": "curve"})
    wide = wide[wide.event_type.notna()].copy()
    wide["hr_delta_bpm"] = (wide.phys_immediate_ppg_heart_rate_bpm -
                            wide.phys_baseline_ppg_heart_rate_bpm)
    means = wide.groupby(["event_type", "domain"]).hr_delta_bpm.mean().unstack()
    x = np.arange(len(means)); width = .36
    fig, ax = plt.subplots(figsize=(6.2, 3.4))
    for off, domain in ((-width / 2, "car"), (width / 2, "motorcycle")):
        ax.bar(x + off, means.get(domain, np.nan), width,
               color=DOMAIN_COLORS[domain], label=DOMAIN_LABELS[domain])
    finite_hr = np.abs(means.values[np.isfinite(means.values)])
    lim = max(5.0, float(finite_hr.max()) * 1.15 if finite_hr.size else 5.0)
    ax.set_ylim(-lim, lim); ax.axhline(0, color="#333", lw=.8)
    ax.set(xticks=x, xticklabels=means.index,
           ylabel="Δ HR immediate − baseline (bpm)",
           title="PPG/HR event-related (finestre con quality gate)")
    ax.legend(frameon=False); ax.grid(axis="y", alpha=.18)
    _save(fig, figures, "07_ppg_hr_event_related", manifest)

    # 8. Candidate states only; no automatic verdict.
    candidates = pd.read_csv(behavior_reports / "solid_line" /
                             "solid_line_candidates.csv")
    states = ["candidate_compliant", "candidate_noncompliant", "uncertain",
              "not_evaluable"]
    counts = candidates.groupby(["domain", "state"]).size()
    fig, ax = plt.subplots(figsize=(6.2, 3.4)); x = np.arange(len(states))
    for off, domain in ((-width / 2, "car"), (width / 2, "motorcycle")):
        ax.bar(x + off, [counts.get((domain, s), 0) for s in states], width,
               color=DOMAIN_COLORS[domain], label=DOMAIN_LABELS[domain])
    ax.set(xticks=x, xticklabels=[s.replace("candidate_", "").replace("_", "\n")
                                 for s in states], ylabel="Candidati (n)",
           title="Attraversamento linea continua: stati da revisionare")
    ax.legend(frameon=False); ax.grid(axis="y", alpha=.18)
    _save(fig, figures, "08_solid_line_candidates", manifest)

    # 9. Explicit paired difference, separate from common-scale absolutes.
    total = paired_all.groupby(
        ["domain", "class_name"]).foveal_mass_percent.mean().unstack(0)
    total["difference"] = total.get("motorcycle", 0) - total.get("car", 0)
    total = total.reindex([c for c in classes if c in total.index])
    fig, ax = plt.subplots(figsize=(7.2, 5.2))
    colors = ["#D55E00" if value >= 0 else "#0072B2" for value in total.difference]
    ax.barh(np.arange(len(total)), total.difference, color=colors)
    ax.axvline(0, color="#333", lw=.8)
    ax.set(yticks=np.arange(len(total)),
           yticklabels=[c.replace("_", " ") for c in total.index],
           xlabel="Differenza Moto − Auto (punti percentuali)",
           title="Sintesi paired sul tratto condiviso")
    ax.invert_yaxis(); ax.grid(axis="x", alpha=.18)
    _save(fig, figures, "09_paired_summary_difference", manifest)

    atomic_write_json(figures / "figure_manifest.json", {
        "schema": "article1_fast_semantic_gaze_figures_v1",
        "result_status": "exploratory_pilot", "figures": manifest,
        "class_order": classes, "domain_colors": DOMAIN_COLORS,
        "comparisons_use_common_scales": True,
        "technical_diagnostics_are_supplementary": True})
    return manifest
