"""Timestamp-paired comparison of the frozen 5 Hz and native fast runs.

All comparisons use the same real gaze timestamps, seconds, percentages,
fixations, events, or spatial bins.  Frame counts are retained only as pipeline
provenance and are never treated as independent behavioural samples.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

import numpy as np

from ..behavior.gaze_semantics import pixels_per_degree, scanpath_statistics
from ..config import Config
from ..io_utils import atomic_write_json
from ..taxonomy import Taxonomy
from .fast_analysis import ROAD_CLASSES
from .fast_plots import DOMAIN_LABELS, FOOTER, _style

RUN_LABELS = {"5hz": "5 Hz", "native": "Native"}
DOMAINS = ("car", "motorcycle")


def _atomic_parquet(path: Path, frame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp.parquet")
    frame.to_parquet(tmp, index=False); tmp.replace(path)


def _difference(value_5hz: Optional[float], value_native: Optional[float]
                ) -> tuple[Optional[float], Optional[float]]:
    if value_5hz is None or value_native is None:
        return None, None
    if not (np.isfinite(value_5hz) and np.isfinite(value_native)):
        return None, None
    absolute = float(value_native - value_5hz)
    percent = (float(absolute / abs(value_5hz) * 100.0)
               if abs(value_5hz) > 1e-12 else None)
    return absolute, percent


def _comparison_row(domain: str, scope: str, metric: str, unit: str,
                    value_5hz: Optional[float], value_native: Optional[float],
                    stratum: Optional[str] = None,
                    normalization: str = "timestamp_normalized") -> Dict[str, Any]:
    absolute, percent = _difference(value_5hz, value_native)
    return {
        "domain": domain, "scope": scope, "stratum": stratum,
        "metric": metric, "unit": unit,
        "value_5hz": value_5hz, "value_native": value_native,
        "difference_native_minus_5hz": absolute,
        "difference_percent_of_5hz": percent,
        "ci95_low": None, "ci95_high": None,
        "normalization": normalization,
    }


def _load_domain(root: Path, domain: str):
    import pandas as pd
    run = root / domain
    gaze = pd.read_parquet(run / "gaze" / "semantic_gaze.parquet").sort_values(
        "timestamp_ns").reset_index(drop=True)
    fixes = pd.read_parquet(run / "gaze" / "fixations.parquet")
    summary = json.loads((run / "gaze" / "semantic_gaze_summary.json").read_text())
    calibration = json.loads((run / "frames" / "calibration.json").read_text())
    return gaze, fixes, summary, calibration


def _run_metrics(root: Path, domain: str, class_names: Sequence[str]
                 ) -> Dict[str, Any]:
    gaze, fixes, summary, calibration = _load_domain(root, domain)
    valid = gaze[gaze.semantic_valid.astype(bool)]
    dt_s = (float(np.median(np.diff(gaze.timestamp_ns.to_numpy(np.int64))) / 1e9)
            if len(gaze) > 1 else 0.0)
    abs_dt = valid.segmentation_dt_ms.abs().to_numpy(float)
    stored = calibration["stored_resolution"]
    ppd = pixels_per_degree(
        float(calibration["pinhole_focal_at_stored_resolution"]))
    scanpath = scanpath_statistics(
        gaze.rect_u.to_numpy(float), gaze.rect_v.to_numpy(float),
        gaze.semantic_valid.to_numpy(bool), ppd,
        (int(stored["height"]), int(stored["width"])))
    valid_duration = (float((int(valid.timestamp_ns.iloc[-1]) -
                             int(valid.timestamp_ns.iloc[0])) / 1e9)
                      if len(valid) > 1 else 0.0)
    if scanpath.get("scanpath_length_deg") is not None:
        scanpath["scanpath_length_deg_per_s"] = (
            float(scanpath["scanpath_length_deg"] / valid_duration)
            if valid_duration > 0 else None)
    ids = valid.top1_class_id.to_numpy(np.int64)
    switches = int(np.count_nonzero(ids[1:] != ids[:-1])) if ids.size > 1 else 0
    total_valid_s = float(len(valid) * dt_s)
    road = valid.top1_class.isin(ROAD_CLASSES)
    fix_durations = fixes.duration_s.dropna().to_numpy(float)
    metrics = {
        "coverage_percent": float(gaze.semantic_valid.mean() * 100.0),
        "associated_gaze_samples": float(len(valid)),
        "excluded_gaze_samples": float(len(gaze) - len(valid)),
        "gaze_mask_dt_median_ms": float(np.median(abs_dt)) if abs_dt.size else None,
        "gaze_mask_dt_p95_ms": float(np.percentile(abs_dt, 95)) if abs_dt.size else None,
        "gaze_mask_dt_max_ms": float(abs_dt.max()) if abs_dt.size else None,
        "fixation_count": float(len(fixes)),
        "fixation_mean_duration_s": (float(fix_durations.mean())
                                       if fix_durations.size else None),
        "fixation_median_duration_s": (float(np.median(fix_durations))
                                         if fix_durations.size else None),
        "semantic_transitions_per_s": (float(switches / total_valid_s)
                                         if total_valid_s > 0 else None),
        "semantic_distribution_entropy": float(
            summary["metrics"]["semantic_distribution_entropy"]),
        "mean_foveal_entropy": (float(valid.foveal_entropy.mean())
                                  if len(valid) else None),
        "scanpath_length_deg_per_s": scanpath.get("scanpath_length_deg_per_s"),
        "off_road_time_percent": float((~road).mean() * 100.0) if len(valid) else None,
        "off_road_glances_per_minute": summary["metrics"]["off_road_glances"].get(
            "per_minute"),
        "off_road_max_duration_s": summary["metrics"]["off_road_glances"].get(
            "max_duration_s"),
        "possible_mirror_gaze_candidates": float(
            gaze.possible_mirror_gaze_candidate.astype(bool).sum()),
    }
    per_class: Dict[str, Dict[str, float]] = {}
    for name in class_names:
        top = valid.top1_class == name
        per_class[name] = {
            "foveal_mass_percent": (float(valid[f"p_{name}"].mean() * 100.0)
                                     if len(valid) else 0.0),
            "top1_share_percent": float(top.mean() * 100.0) if len(valid) else 0.0,
            "dwell_time_s": float(top.sum() * dt_s),
            "fixations_per_minute": float(
                summary["metrics"]["per_class"][name].get("fixations_per_minute") or 0.0),
        }
    return {"global": metrics, "per_class": per_class, "gaze": gaze,
            "sample_interval_s": dt_s, "ppd": ppd}


def _block_rows(domain: str, run_name: str, data: Dict[str, Any],
                class_names: Sequence[str], block_duration_s: float):
    rows = []
    gaze = data["gaze"].copy()
    start = int(gaze.timestamp_ns.min())
    gaze["block_id"] = ((gaze.timestamp_ns - start) //
                        int(block_duration_s * 1e9)).astype(int)
    for block_id, block in gaze.groupby("block_id", sort=True):
        valid = block[block.semantic_valid.astype(bool)]
        block_s = (float((int(block.timestamp_ns.max()) -
                          int(block.timestamp_ns.min())) / 1e9)
                   if len(block) > 1 else block_duration_s)
        if block_s <= 0:
            continue
        base = {"domain": domain, "run": run_name, "block_id": int(block_id),
                "block_duration_s": block_s}
        rows.append({**base, "scope": "global", "stratum": None,
                     "metric": "coverage_percent",
                     "value": float(block.semantic_valid.mean() * 100.0)})
        if valid.empty:
            continue
        ids = valid.top1_class_id.to_numpy(np.int64)
        transitions = int(np.count_nonzero(ids[1:] != ids[:-1]))
        global_values = {
            "gaze_mask_dt_median_ms": float(valid.segmentation_dt_ms.abs().median()),
            "mean_foveal_entropy": float(valid.foveal_entropy.mean()),
            "semantic_transitions_per_s": float(transitions / block_s),
            "off_road_time_percent": float(
                (~valid.top1_class.isin(ROAD_CLASSES)).mean() * 100.0),
        }
        points = valid[["rect_u", "rect_v"]].to_numpy(float)
        if len(points) > 1:
            global_values["scanpath_length_deg_per_s"] = float(
                np.hypot(np.diff(points[:, 0]), np.diff(points[:, 1])).sum() /
                data["ppd"] / block_s)
        for metric, value in global_values.items():
            rows.append({**base, "scope": "global", "stratum": None,
                         "metric": metric, "value": value})
        for name in class_names:
            top = valid.top1_class == name
            values = {
                "foveal_mass_percent": float(valid[f"p_{name}"].mean() * 100.0),
                "top1_share_percent": float(top.mean() * 100.0),
                "dwell_time_s": float(top.sum() * data["sample_interval_s"]),
            }
            for metric, value in values.items():
                rows.append({**base, "scope": "class", "stratum": name,
                             "metric": metric, "value": value})
    return rows


def _bootstrap_ci(values: np.ndarray, repeats: int, rng: np.random.Generator
                  ) -> tuple[Optional[float], Optional[float]]:
    values = np.asarray(values, float)
    values = values[np.isfinite(values)]
    if values.size < 2:
        return None, None
    draws = rng.choice(values, size=(repeats, values.size), replace=True).mean(axis=1)
    return float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))


def _event_comparison(baseline_root: Path, native_root: Path, repeats: int,
                      rng: np.random.Generator):
    import pandas as pd
    keys = ["event_id", "domain", "event_type", "phase"]
    metrics = ["observed_time_s", "road_relevant_mass_percent",
               "mean_foveal_entropy", "fixation_time_percent"]
    five = pd.read_parquet(baseline_root / "event_metrics.parquet")
    native = pd.read_parquet(native_root / "event_metrics.parquet")
    paired = five.merge(native, on=keys, how="inner", suffixes=("_5hz", "_native"))
    rows = []
    for (domain, event_type, phase), group in paired.groupby(
            ["domain", "event_type", "phase"], sort=True):
        for metric in metrics:
            a = group[f"{metric}_5hz"].to_numpy(float)
            b = group[f"{metric}_native"].to_numpy(float)
            row = _comparison_row(
                domain, "event", metric,
                "seconds" if metric == "observed_time_s" else (
                    "percent" if metric.endswith("percent") else "normalized_entropy"),
                float(np.mean(a)), float(np.mean(b)),
                stratum=f"{event_type}:{phase}", normalization="event_mean")
            lo, hi = _bootstrap_ci(b - a, repeats, rng)
            row.update({"ci95_low": lo, "ci95_high": hi,
                        "paired_units": int(len(group))})
            rows.append(row)
    return paired, pd.DataFrame(rows)


def _route_comparison(baseline_root: Path, native_root: Path):
    import pandas as pd
    keys = ["domain", "bin_key", "bin_order", "bin_size_m", "class_name"]
    five = pd.read_parquet(baseline_root / "paired_route_bins.parquet")
    native = pd.read_parquet(native_root / "paired_route_bins.parquet")
    columns = keys + ["gaze_time_s", "foveal_mass_percent", "top1_time_percent"]
    paired = five[columns].merge(native[columns], on=keys, how="inner",
                                 suffixes=("_5hz", "_native"))
    for metric in ("gaze_time_s", "foveal_mass_percent", "top1_time_percent"):
        paired[f"{metric}_difference"] = (paired[f"{metric}_native"] -
                                           paired[f"{metric}_5hz"])
        denom = paired[f"{metric}_5hz"].abs()
        paired[f"{metric}_difference_percent"] = np.where(
            denom > 1e-12, paired[f"{metric}_difference"] / denom * 100.0, np.nan)
    return paired


def _semantic_runs(frame, class_name: str, max_gap_s: float = .075):
    selected = frame[frame.semantic_valid.astype(bool) &
                     (frame.top1_class == class_name)].sort_values("timestamp_ns")
    if selected.empty:
        return []
    rows = list(selected.itertuples(index=False)); runs = []; start = previous = rows[0]
    for row in rows[1:]:
        if (int(row.timestamp_ns) - int(previous.timestamp_ns)) / 1e9 > max_gap_s:
            runs.append((start, previous)); start = row
        previous = row
    runs.append((start, previous))
    return runs


def _short_events_recovered(five, native, class_names: Sequence[str],
                            max_duration_s: float = .6):
    import pandas as pd
    paired = native.merge(
        five[["timestamp_ns", "semantic_valid", "top1_class", "segmentation_frame_index"]],
        on="timestamp_ns", how="inner", suffixes=("_native", "_5hz"))
    rows = []
    for name in class_names:
        for start, end in _semantic_runs(native, name):
            duration = float((int(end.timestamp_ns) - int(start.timestamp_ns)) / 1e9)
            if duration > max_duration_s:
                continue
            sample = paired[(paired.timestamp_ns >= int(start.timestamp_ns)) &
                            (paired.timestamp_ns <= int(end.timestamp_ns))]
            if sample.empty or (sample.top1_class_5hz == name).any():
                continue
            rows.append({
                "class_name": name, "start_ns": int(start.timestamp_ns),
                "end_ns": int(end.timestamp_ns), "duration_s": duration,
                "native_samples": int(len(sample)),
                "native_frame_start": int(start.segmentation_frame_index),
                "native_frame_end": int(end.segmentation_frame_index),
                "five_hz_dominant_class": (str(sample.top1_class_5hz.mode().iloc[0])
                                             if sample.top1_class_5hz.notna().any()
                                             else None),
            })
    return pd.DataFrame(rows)


def _short_semantic_runs_per_minute(gaze, maximum_s: float = .2) -> float:
    valid = gaze[gaze.semantic_valid.astype(bool)].sort_values("timestamp_ns")
    if len(valid) < 2:
        return 0.0
    ts = valid.timestamp_ns.to_numpy(np.int64)
    labels = valid.top1_class.to_numpy(object)
    starts = np.r_[0, np.flatnonzero(labels[1:] != labels[:-1]) + 1]
    ends = np.r_[starts[1:], len(valid)]
    durations = np.array([
        (int(ts[end - 1]) - int(ts[start])) / 1e9
        for start, end in zip(starts, ends)], float)
    total_s = float((int(ts[-1]) - int(ts[0])) / 1e9)
    return float(np.count_nonzero(durations <= maximum_s) * 60.0 / total_s)


def _mask_flicker(root: Path, domain: str) -> Dict[str, float]:
    import cv2
    import pandas as pd
    run = root / domain
    index = pd.read_parquet(run / "segmentation" /
                            "segmentation_index.parquet").sort_values(
                                "capture_timestamp_ns")
    changes = []; previous = None
    for row in index.itertuples(index=False):
        mask = cv2.imread(str(run / str(row.mask_path)), cv2.IMREAD_UNCHANGED)
        if mask is None:
            raise FileNotFoundError(run / str(row.mask_path))
        if previous is not None:
            changes.append(float(np.mean(mask != previous)))
        previous = mask
    values = np.asarray(changes, float)
    elapsed_s = float((int(index.capture_timestamp_ns.iloc[-1]) -
                       int(index.capture_timestamp_ns.iloc[0])) / 1e9)
    return {
        "frames": float(len(index)),
        "mean_consecutive_pixel_change_fraction": float(values.mean()),
        "median_consecutive_pixel_change_fraction": float(np.median(values)),
        "p95_consecutive_pixel_change_fraction": float(np.percentile(values, 95)),
        "summed_pixel_change_fraction_per_s": float(values.sum() / elapsed_s),
    }


def _matched_mask_agreement(baseline_root: Path, native_root: Path,
                            domain: str) -> float:
    import cv2
    import pandas as pd
    five_run = baseline_root / domain; native_run = native_root / domain
    five = pd.read_parquet(five_run / "segmentation" /
                           "segmentation_index.parquet")
    native = pd.read_parquet(native_run / "segmentation" /
                             "segmentation_index.parquet").set_index("frame_index")
    equal = total = 0
    for row in five.itertuples(index=False):
        a = cv2.imread(str(five_run / str(row.mask_path)), cv2.IMREAD_UNCHANGED)
        b_row = native.loc[int(row.frame_index)]
        b = cv2.imread(str(native_run / str(b_row.mask_path)), cv2.IMREAD_UNCHANGED)
        if a is None or b is None or a.shape != b.shape:
            raise RuntimeError(f"unreadable matched masks for {domain} frame {row.frame_index}")
        equal += int(np.count_nonzero(a == b)); total += int(a.size)
    return float(equal / total) if total else float("nan")


def build_flicker_comparison(baseline_root: Path, native_root: Path,
                             metrics_by_run: Dict[str, Dict[str, Any]]):
    import pandas as pd
    rows = []
    for domain in DOMAINS:
        agreement = _matched_mask_agreement(baseline_root, native_root, domain)
        for run_name, root in (("5hz", baseline_root), ("native", native_root)):
            values = _mask_flicker(root, domain)
            gaze = metrics_by_run[run_name][domain]["gaze"]
            values.update({
                "domain": domain, "run": run_name,
                "semantic_transitions_per_s": metrics_by_run[run_name][domain][
                    "global"]["semantic_transitions_per_s"],
                "short_semantic_runs_per_minute":
                    _short_semantic_runs_per_minute(gaze),
                "matched_real_frame_agreement_5hz_native": agreement,
                "interpretation": (
                    "raw frame-to-frame change includes scene/camera motion; semantic "
                    "gaze transitions and same-frame agreement are the primary flicker checks"),
            })
            rows.append(values)
    return pd.DataFrame(rows)


def _attach_block_confidence(comparison, blocks, repeats: int,
                             rng: np.random.Generator):
    import pandas as pd
    keys = ["domain", "block_id", "scope", "stratum", "metric"]
    five = blocks[blocks.run == "5hz"]
    native = blocks[blocks.run == "native"]
    paired = five[keys + ["value"]].merge(
        native[keys + ["value"]], on=keys, suffixes=("_5hz", "_native"))
    paired["difference"] = paired.value_native - paired.value_5hz
    for key, group in paired.groupby(["domain", "scope", "stratum", "metric"],
                                     dropna=False):
        domain, scope, stratum, metric = key
        match = ((comparison.domain == domain) & (comparison.scope == scope) &
                 (comparison.metric == metric))
        if pd.isna(stratum):
            match &= comparison.stratum.isna()
        else:
            match &= comparison.stratum == stratum
        lo, hi = _bootstrap_ci(group.difference.to_numpy(float), repeats, rng)
        comparison.loc[match, ["ci95_low", "ci95_high"]] = [lo, hi]
    return paired, comparison


def plot_native_vs_5hz(comparison, output_dir: str | Path,
                       class_names: Sequence[str]) -> Dict[str, str]:
    import matplotlib.pyplot as plt
    _style(); output = Path(output_dir); output.mkdir(parents=True, exist_ok=True)
    y = np.arange(len(class_names)); h = .34
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 9.0), sharey=True)
    value_axes, delta_axes = axes[:, 0], axes[:, 1]
    all_values, all_delta = [], []
    for row, domain in enumerate(DOMAINS):
        data = comparison[(comparison.domain == domain) &
                          (comparison.scope == "class") &
                          (comparison.metric == "foveal_mass_percent")].set_index(
                              "stratum").reindex(class_names)
        five = data.value_5hz.to_numpy(float); native = data.value_native.to_numpy(float)
        delta = data.difference_native_minus_5hz.to_numpy(float)
        all_values.extend([five, native]); all_delta.append(delta)
        value_axes[row].barh(y - h / 2, five, h, color="#777777", label="5 Hz")
        value_axes[row].barh(y + h / 2, native, h, color="#009E73", label="Native")
        colors = ["#D55E00" if value >= 0 else "#0072B2" for value in delta]
        delta_axes[row].barh(y, delta, color=colors)
        delta_axes[row].axvline(0, color="#333333", lw=.8)
        value_axes[row].set_title(DOMAIN_LABELS[domain])
        delta_axes[row].set_title(f"{DOMAIN_LABELS[domain]}: native − 5 Hz")
        for ax in axes[row]:
            ax.grid(axis="x", alpha=.18)
    max_value = max(float(np.nanmax(v)) for v in all_values if np.isfinite(v).any())
    max_delta = max(float(np.nanmax(np.abs(v))) for v in all_delta if np.isfinite(v).any())
    for ax in value_axes:
        ax.set_xlim(0, max_value * 1.08 if max_value else 1)
    for ax in delta_axes:
        ax.set_xlim(-max_delta * 1.12 if max_delta else -1,
                    max_delta * 1.12 if max_delta else 1)
    labels = [name.replace("_", " ") for name in class_names]
    for ax in axes[:, 0]:
        ax.set(yticks=y, yticklabels=labels); ax.invert_yaxis()
        ax.set_xlabel("Massa foveale media (%)")
    for ax in axes[:, 1]:
        ax.set_xlabel("Differenza (punti percentuali)")
    value_axes[0].legend(frameon=False)
    fig.suptitle("Semantic gaze: frequenza RGB nativa vs 5 Hz")
    fig.subplots_adjust(left=.25, bottom=.09, top=.93, hspace=.28, wspace=.18)
    fig.text(.99, .008, FOOTER, ha="right", fontsize=6, color="#666666")
    png = output / "native_vs_5hz_semantic_gaze.png"
    pdf = output / "native_vs_5hz_semantic_gaze.pdf"
    fig.savefig(png, dpi=300, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight"); plt.close(fig)
    return {"png": str(png), "pdf": str(pdf),
            "common_value_scale": True, "common_difference_scale": True}


def build_native_comparison(baseline_root: str | Path,
                            native_root: str | Path,
                            output_dir: str | Path, figures_dir: str | Path,
                            cfg: Config) -> Dict[str, Any]:
    """Build exact timestamp-paired 5 Hz/native tables and comparison figure."""
    import pandas as pd
    baseline_root = Path(baseline_root); native_root = Path(native_root)
    output = Path(output_dir); output.mkdir(parents=True, exist_ok=True)
    compare_cfg = cfg.get("native_comparison", {})
    block_s = float(compare_cfg.get("block_duration_s", 10.0))
    repeats = int(compare_cfg.get("bootstrap_repeats", 2000))
    rng = np.random.default_rng(int(compare_cfg.get("random_seed", 20260803)))
    taxonomy = Taxonomy.load(cfg.resolve(
        cfg.get("semantic_gaze_fast_external.classes")))
    class_names = taxonomy.names()
    unit_map = {
        "coverage_percent": "percent", "associated_gaze_samples": "samples",
        "excluded_gaze_samples": "samples", "gaze_mask_dt_median_ms": "ms",
        "gaze_mask_dt_p95_ms": "ms", "gaze_mask_dt_max_ms": "ms",
        "fixation_count": "fixations", "fixation_mean_duration_s": "seconds",
        "fixation_median_duration_s": "seconds",
        "semantic_transitions_per_s": "transitions_per_second",
        "semantic_distribution_entropy": "normalized_entropy",
        "mean_foveal_entropy": "normalized_entropy",
        "scanpath_length_deg_per_s": "degrees_per_second",
        "off_road_time_percent": "percent",
        "off_road_glances_per_minute": "events_per_minute",
        "off_road_max_duration_s": "seconds",
        "possible_mirror_gaze_candidates": "review_candidates",
    }
    comparison_rows = []; block_rows = []; recovered = []
    metrics_by_run: Dict[str, Dict[str, Any]] = {"5hz": {}, "native": {}}
    for domain in DOMAINS:
        metrics_by_run["5hz"][domain] = _run_metrics(
            baseline_root, domain, class_names)
        metrics_by_run["native"][domain] = _run_metrics(
            native_root, domain, class_names)
        five_gaze = metrics_by_run["5hz"][domain]["gaze"]
        native_gaze = metrics_by_run["native"][domain]["gaze"]
        if not np.array_equal(five_gaze.timestamp_ns.to_numpy(np.int64),
                              native_gaze.timestamp_ns.to_numpy(np.int64)):
            raise ValueError(f"{domain}: gaze timestamps differ between runs")
        for metric in unit_map:
            comparison_rows.append(_comparison_row(
                domain, "global", metric, unit_map[metric],
                metrics_by_run["5hz"][domain]["global"].get(metric),
                metrics_by_run["native"][domain]["global"].get(metric)))
        for name in class_names:
            for metric, unit in (("foveal_mass_percent", "percent"),
                                 ("top1_share_percent", "percent"),
                                 ("dwell_time_s", "seconds"),
                                 ("fixations_per_minute", "fixations_per_minute")):
                comparison_rows.append(_comparison_row(
                    domain, "class", metric, unit,
                    metrics_by_run["5hz"][domain]["per_class"][name][metric],
                    metrics_by_run["native"][domain]["per_class"][name][metric],
                    stratum=name,
                    normalization=("seconds" if metric == "dwell_time_s"
                                   else "valid_gaze_time")))
        for run_name in ("5hz", "native"):
            block_rows.extend(_block_rows(
                domain, run_name, metrics_by_run[run_name][domain],
                class_names, block_s))
        short = _short_events_recovered(five_gaze, native_gaze, class_names)
        if len(short):
            short.insert(0, "domain", domain); recovered.append(short)

    comparison = pd.DataFrame(comparison_rows)
    blocks = pd.DataFrame(block_rows)
    paired_blocks, comparison = _attach_block_confidence(
        comparison, blocks, repeats, rng)
    event_pairs, event_summary = _event_comparison(
        baseline_root, native_root, repeats, rng)
    route = _route_comparison(baseline_root, native_root)
    recovered_frame = (pd.concat(recovered, ignore_index=True)
                       if recovered else pd.DataFrame(columns=[
                           "domain", "class_name", "start_ns", "end_ns",
                           "duration_s", "native_samples"]))

    _atomic_parquet(output / "native_vs_5hz_metrics.parquet", comparison)
    comparison.to_csv(output / "native_vs_5hz_metrics.csv", index=False)
    _atomic_parquet(output / "native_vs_5hz_block_metrics.parquet", blocks)
    _atomic_parquet(output / "native_vs_5hz_paired_block_differences.parquet",
                    paired_blocks)
    _atomic_parquet(output / "native_vs_5hz_event_pairs.parquet", event_pairs)
    _atomic_parquet(output / "native_vs_5hz_event_summary.parquet", event_summary)
    event_summary.to_csv(output / "native_vs_5hz_event_summary.csv", index=False)
    _atomic_parquet(output / "native_vs_5hz_shared_route_bins.parquet", route)
    route.to_csv(output / "native_vs_5hz_shared_route_bins.csv", index=False)
    _atomic_parquet(output / "native_recovered_short_events.parquet", recovered_frame)
    recovered_frame.to_csv(output / "native_recovered_short_events.csv", index=False)
    flicker = build_flicker_comparison(
        baseline_root, native_root, metrics_by_run)
    _atomic_parquet(output / "native_vs_5hz_flicker.parquet", flicker)
    flicker.to_csv(output / "native_vs_5hz_flicker.csv", index=False)
    figure = plot_native_vs_5hz(comparison, figures_dir, class_names)

    summary = {
        "schema": "article1_fast_semantic_gaze_native_comparison_v1",
        "result_status": "exploratory_pilot",
        "baseline_root": str(baseline_root), "native_root": str(native_root),
        "gaze_timestamps_exactly_paired": True,
        "block_duration_s": block_s, "bootstrap_repeats": repeats,
        "comparison_rows": int(len(comparison)),
        "paired_block_rows": int(len(paired_blocks)),
        "paired_event_rows": int(len(event_pairs)),
        "shared_route_rows": int(len(route)),
        "shared_route_bins": int(route.bin_key.nunique()),
        "recovered_short_events": int(len(recovered_frame)),
        "flicker_rows": int(len(flicker)),
        "frame_counts_used_as_behavioral_samples": False,
        "units": ["seconds", "percent", "events", "spatial_bins", "fixations"],
        "figure": figure,
    }
    atomic_write_json(output / "native_vs_5hz_summary.json", summary)
    return summary
