"""The three figures the final behaviour statistics add, and nothing more.

The publication set is the nine figures `fast_plots.plot_all` already produces.
This module regenerates those into the final report directory unchanged — same
class order, same palette, same locked car/motorcycle scales — and appends
exactly three:

10. blink and pupil comparison;
11. event-related eye response;
12. paired gaze-physiology summary.

Every panel keeps the house rules: one question per figure, car and motorcycle on
one shared scale for any quantity compared between them, the same colour for the
same vehicle everywhere, and the pilot status in the footer.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from ..config import Config
from ..io_utils import atomic_write_json
from .fast_plots import (DOMAIN_COLORS, DOMAIN_LABELS, _save, _style,
                         apply_shared_limits, plot_all)

#: Window order used by every event-related panel.
EVENT_WINDOW_ORDER = ["phys_baseline", "phys_anticipation", "phys_immediate",
                      "phys_delayed"]
EVENT_WINDOW_LABELS = ["baseline", "anticipazione", "immediata", "ritardata"]

#: Colour-blind-safe accent for a difference that survived FDR.
SURVIVES_COLOR = "#009E73"
NOT_SURVIVES_COLOR = "#999999"


def _boxes(ax, values: Sequence[Sequence[float]], ylabel: str) -> None:
    """Two boxes, car then motorcycle, in the house colours."""
    boxes = ax.boxplot([np.asarray(v, float)[np.isfinite(v)] for v in values],
                       tick_labels=[DOMAIN_LABELS["car"], DOMAIN_LABELS["motorcycle"]],
                       patch_artist=True, showfliers=False)
    for patch, domain in zip(boxes["boxes"], ("car", "motorcycle")):
        patch.set_facecolor(DOMAIN_COLORS[domain])
        patch.set_alpha(.7)
    for element in ("medians", "whiskers", "caps"):
        for line in boxes[element]:
            line.set_color("#333")
    ax.set_ylabel(ylabel)
    ax.grid(axis="y", alpha=.18)


def plot_blink_pupil(analysis: Path, recordings: Dict[str, str], figures: Path,
                     manifest: List[dict]) -> None:
    """Figure 10 — do the two vehicles differ in blink and pupil behaviour?"""
    import pandas as pd

    windows = {d: pd.read_parquet(analysis / rec / "blink_rate_windows.parquet")
               for d, rec in recordings.items()}
    events = {d: pd.read_parquet(analysis / rec / "blink_events.parquet")
              for d, rec in recordings.items()}
    pupil = {d: pd.read_parquet(analysis / rec / "pupil_samples.parquet")
             for d, rec in recordings.items()}
    summaries = {d: json.loads((analysis / rec / "eye_state_summary.json").read_text())
                 for d, rec in recordings.items()}

    fig, axes = plt.subplots(1, 4, figsize=(12.4, 4.0))
    fig.subplots_adjust(wspace=.42, top=.80, bottom=.20)

    rates = [windows[d]["blinks_per_minute"].to_numpy(float)
             for d in ("car", "motorcycle")]
    _boxes(axes[0], rates, "Blink/min (finestre reali di 60 s)")
    axes[0].set_title("Frequenza di blink")
    _percentile_limits(axes[0], rates)

    durations = [events[d][events[d]["is_identified_blink"].astype(bool)][
        "duration_s"].to_numpy(float) * 1000.0 for d in ("car", "motorcycle")]
    _boxes(axes[1], durations, "Durata del blink (ms)")
    axes[1].set_title("Durata del blink")
    _percentile_limits(axes[1], durations)

    closure = [100 * float(summaries[d]["blink"]["eye_closure"]["both_eyes"][
        "eye_closure_fraction"] or 0) for d in ("car", "motorcycle")]
    axes[2].bar([0, 1], closure, .55,
                color=[DOMAIN_COLORS["car"], DOMAIN_COLORS["motorcycle"]])
    axes[2].set(xticks=[0, 1],
                xticklabels=[DOMAIN_LABELS["car"], DOMAIN_LABELS["motorcycle"]],
                ylabel="Occhi chiusi (% dei campioni validi)",
                title="Frazione a occhi chiusi")
    axes[2].set_ylim(0, max(12.0, max(closure) * 1.25))
    axes[2].grid(axis="y", alpha=.18)

    residual = [pupil[d]["pupil_light_adjusted_residual_m"].to_numpy(float) * 1000.0
                for d in ("car", "motorcycle")]
    _boxes(axes[3], residual, "Residuo pupillare (mm)")
    axes[3].set_title("Pupilla corretta per log(lux)")
    # Percentile limits, not min/max: a handful of tracking outliers reach several
    # millimetres and would compress the box to a line.
    _percentile_limits(axes[3], residual, low=1.0, high=99.0)
    axes[3].axhline(0, color="#333", lw=.8)

    fig.suptitle("Blink e pupilla: auto contro moto sull’intera registrazione",
                 y=.97)
    fig.text(.01, .015, "La frazione a occhi chiusi è una misura oculare, non un "
                        "indice di sonnolenza. Il diametro pupillare non ha "
                        "interpretazione clinica.",
             fontsize=6, color="#555", ha="left", va="bottom")
    _save(fig, figures, "10_blink_pupil_comparison", manifest)


def _percentile_limits(ax, values: Sequence[Sequence[float]], low: float = 0.0,
                       high: float = 100.0, pad: float = .08) -> None:
    """One shared y range for both vehicles, robust to tracking outliers."""
    pooled = np.concatenate([np.asarray(v, float).ravel() for v in values])
    pooled = pooled[np.isfinite(pooled)]
    if pooled.size == 0:
        return
    lo, hi = np.percentile(pooled, [low, high])
    span = float(hi - lo) or max(abs(float(hi)), 1.0)
    ax.set_ylim(float(lo) - span * pad, float(hi) + span * pad)


def plot_event_eye_response(analysis: Path, figures: Path,
                            manifest: List[dict]) -> None:
    """Figure 11 — how do the eyes respond around comparable road events?"""
    import pandas as pd

    events = pd.read_parquet(analysis / "event_response.parquet")
    events = events[events["window"].isin(EVENT_WINDOW_ORDER)]
    kinds = [k for k in ("roundabout", "junction", "curve")
             if k in set(events["event_type"])]
    metrics = (("blink_rate_per_minute", "Blink/min"),
               ("eye_closure_fraction", "Occhi chiusi (frazione)"),
               ("pupil_light_adjusted_residual_mm", "Residuo pupillare (mm)"))

    fig, axes = plt.subplots(len(metrics), len(kinds),
                             figsize=(3.6 * len(kinds), 2.5 * len(metrics)),
                             squeeze=False, sharex=True)
    for row, (metric, ylabel) in enumerate(metrics):
        series = []
        for column, kind in enumerate(kinds):
            ax = axes[row][column]
            subset = events[events["event_type"] == kind]
            for domain in ("car", "motorcycle"):
                values = [subset[(subset.domain == domain) &
                                 (subset.window == window)][metric].median()
                          for window in EVENT_WINDOW_ORDER]
                series.append(values)
                ax.plot(EVENT_WINDOW_LABELS, values, marker="o", lw=1.5,
                        color=DOMAIN_COLORS[domain], label=DOMAIN_LABELS[domain])
            ax.grid(alpha=.18)
            if row == 0:
                ax.set_title(kind)
            if column == 0:
                ax.set_ylabel(ylabel)
        # One shared scale per metric, so a row is comparable across event types
        # and across vehicles.
        apply_shared_limits(axes[row], series, include_zero=False)
    axes[0][0].legend(frameon=False)
    for ax in axes[-1]:
        ax.tick_params(axis="x", rotation=20)
    fig.suptitle("Risposta oculare per evento (mediana per evento, unità = evento)")
    _save(fig, figures, "11_event_related_eye_response", manifest)


def plot_paired_summary(analysis: Path, figures: Path,
                        manifest: List[dict]) -> None:
    """Figure 12 — which paired differences survive on the shared route?"""
    import pandas as pd

    stats = pd.read_parquet(analysis / "paired_statistics.parquet")
    stats = stats[(stats["family"] == "bin") & ~stats["skipped"].astype(bool)]
    if stats.empty:
        return
    # A difference is only readable against its own spread, so every metric is
    # shown as a standardised effect: Cliff's delta, which is already bounded and
    # unit-free, with the raw difference and its interval in the label.
    stats = stats.sort_values("cliffs_delta")
    y = np.arange(len(stats))
    colors = [SURVIVES_COLOR if survives else NOT_SURVIVES_COLOR
              for survives in stats["significant_after_fdr"]]

    fig, axes = plt.subplots(1, 2, figsize=(12.6, 6.6),
                             gridspec_kw={"width_ratios": [1.0, 1.05]})
    fig.subplots_adjust(left=.24, right=.98, top=.88, bottom=.14, wspace=.06)
    axes[0].barh(y, stats["cliffs_delta"], color=colors)
    axes[0].axvline(0, color="#333", lw=.8)
    axes[0].set(yticks=y,
                yticklabels=[m.replace("_", " ") for m in stats["metric"]],
                xlabel="Cliff's δ (moto − auto)", xlim=(-1.05, 1.05),
                title="Dimensione dell’effetto")
    axes[0].grid(axis="x", alpha=.18)
    axes[0].set_ylim(-.7, len(stats) - .3)

    # The right panel is the numbers themselves. A second bar chart cannot work
    # here: these metrics live on scales from 0.06 (a fraction) to 750 (ms), and
    # normalising by the car median explodes wherever that median is near zero,
    # which is exactly where the interesting classes are.
    axes[1].axis("off")
    axes[1].set_xlim(0, 1)
    axes[1].set_ylim(-.7, len(stats) - .3)
    columns = [(0.02, "auto"), (0.20, "moto"), (0.40, "moto − auto"),
               (0.62, "IC 95%"), (0.98, "p FDR")]
    for x, header in columns:
        axes[1].text(x, len(stats) - .45, header, fontsize=8, weight="bold",
                     color="#333", va="center",
                     ha="right" if x > .9 else "left")
    for i, (_, row) in enumerate(stats.iterrows()):
        colour = SURVIVES_COLOR if row["significant_after_fdr"] else "#444"
        weight = "bold" if row["significant_after_fdr"] else "normal"
        cells = [(0.02, _fmt(row["car_median"])),
                 (0.20, _fmt(row["motorcycle_median"])),
                 (0.40, _fmt(row["difference_moto_minus_car"])),
                 (0.62, f"[{_fmt(row['ci_low'])}, {_fmt(row['ci_high'])}]"),
                 (0.98, _fmt_p(row["permutation_p_fdr"]))]
        for x, text in cells:
            axes[1].text(x, i, text, fontsize=7, color=colour, weight=weight,
                         va="center", ha="right" if x > .9 else "left")
    axes[1].set_title("Mediane, differenza e intervallo (unità di ciascuna metrica)")

    handles = [plt.Line2D([], [], marker="s", ls="", color=SURVIVES_COLOR,
                          label="sopravvive a FDR 0.05"),
               plt.Line2D([], [], marker="s", ls="", color=NOT_SURVIVES_COLOR,
                          label="non sopravvive")]
    axes[0].legend(handles=handles, frameon=False, loc="lower right")
    fig.suptitle("Confronto paired auto–moto sui 42 bin condivisi da 50 m "
                 "(unità = bin, ≈6 blocchi indipendenti)", y=.965)
    fig.text(.01, .015, "Intervalli da block bootstrap, p da permutazione a "
                        "blocchi, FDR Benjamini–Hochberg. Unità di misura per "
                        "metrica in paired_statistics.csv.",
             fontsize=6, color="#555", ha="left", va="bottom")
    _save(fig, figures, "12_paired_gaze_physiology_summary", manifest)


def _fmt(value: Optional[float]) -> str:
    """Compact fixed-point, with enough digits for the smallest metric here."""
    if value is None or not np.isfinite(value):
        return "—"
    magnitude = abs(float(value))
    if magnitude >= 100:
        return f"{value:.0f}"
    if magnitude >= 1:
        return f"{value:.2f}"
    return f"{value:.3f}"


def _fmt_p(value: Optional[float]) -> str:
    if value is None or not np.isfinite(value):
        return "—"
    return f"{value:.3f}" if value >= 0.001 else "<0.001"


def plot_final(run_dirs: Dict[str, str | Path], analysis_dir: str | Path,
               final_analysis_dir: str | Path, figures_dir: str | Path,
               cfg: Config, recordings: Dict[str, str],
               behavior_root: str | Path = "output/article1/behavior_analysis",
               behavior_reports: str | Path = "reports/article1_behavior_analysis"
               ) -> List[dict]:
    """The nine publication figures, unchanged, plus the three additions."""
    figures = Path(figures_dir)
    manifest = plot_all(run_dirs, analysis_dir, figures, cfg,
                        behavior_root=behavior_root,
                        behavior_reports=behavior_reports)
    _style()
    final = Path(final_analysis_dir)
    plot_blink_pupil(final, recordings, figures, manifest)
    plot_event_eye_response(final, figures, manifest)
    plot_paired_summary(final, figures, manifest)

    atomic_write_json(figures / "figure_manifest.json", {
        "schema": "article1_final_behavior_figures_v1",
        "result_status": "exploratory_pilot",
        "figures": manifest,
        "figure_count": len(manifest),
        "domain_colors": DOMAIN_COLORS,
        "comparisons_use_common_scales": True,
        "one_question_per_figure": True,
        "additions": ["10_blink_pupil_comparison",
                      "11_event_related_eye_response",
                      "12_paired_gaze_physiology_summary"],
        "eye_closure_is_not_a_drowsiness_measure": True,
        "pupil_has_no_clinical_interpretation": True,
    })
    return manifest
