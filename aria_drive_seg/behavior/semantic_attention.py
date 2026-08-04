"""Semantic attention metrics over the full-recording semantic gaze.

The unit here is a **set of gaze samples** — a whole recording, a 50 m route bin,
or an event window — and every metric is computed the same way whichever it is,
so a bin-level number and a recording-level number mean the same thing.

**Foveal probability mass is the primary metric, and top-1 share is secondary.**
That is a measurement decision, not a preference. A driver looking far down the
road puts the fovea on the vanishing point, where the road surface has narrowed
to less than the ~3° foveal window; the surrounding scene then wins the argmax
and "percent of gaze on road_surface" collapses for a reason that has nothing to
do with attention. The probability mass a class holds inside the foveal window
does not have that failure mode — a road filling a third of the fovea contributes
a third whether or not it is the argmax. Both are computed; the mass figures are
the ones any comparison rests on.

Counts are always accompanied by a rate, because the two recordings differ in
length and in gaze cadence and a raw count between them would be a unit error.
"""
from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

NS_PER_S = 1_000_000_000

#: Classes whose share carries a primary claim.
ROAD_RELEVANT = ("road_surface", "lane_marking", "regulatory_road_marking",
                 "vehicle", "two_wheeler", "pedestrian", "traffic_sign",
                 "traffic_light", "road_boundary_or_sidewalk")

#: The rider's own vehicle. In the fast taxonomy this is a segmentation class, but
#: it aggregates a windscreen pillar, a tank and a dashboard, so it is a coarse
#: "eyes inside the vehicle" indicator and not a mirror or instrument metric.
INTERIOR_COCKPIT = ("interior_cockpit",)

#: Road users with no protective structure of their own.
VULNERABLE_ROAD_USER = ("pedestrian", "two_wheeler")

#: Static regulatory information the driver has to read.
SIGN_AND_SIGNAL = ("traffic_sign", "traffic_light", "regulatory_road_marking")

CLASS_GROUPS: Dict[str, Sequence[str]] = {
    "road_relevant": ROAD_RELEVANT,
    "interior_cockpit": INTERIOR_COCKPIT,
    "vulnerable_road_user": VULNERABLE_ROAD_USER,
    "sign_and_signal": SIGN_AND_SIGNAL,
}

#: Classes that stay exploratory proxies and may not carry a primary claim.
PROXY_ONLY = ("mirror",)

PRIMARY_METRIC_NOTE = (
    "foveal probability mass is the primary metric. Top-1 share is reported "
    "alongside it but is biased by viewing geometry: gaze at the vanishing point "
    "lands where the road subtends less than the foveal window, so the argmax "
    "goes to the surrounding scene.")


def _entropy(fractions: np.ndarray, n_classes: int) -> Optional[float]:
    """Shannon entropy normalised to [0, 1] by log(n_classes)."""
    p = np.asarray(fractions, float)
    p = p[np.isfinite(p) & (p > 0)]
    if p.size == 0 or n_classes < 2:
        return None
    return float(-(p * np.log(p)).sum() / math.log(n_classes))


def _describe(values: Sequence[float]) -> Dict[str, Optional[float]]:
    v = np.asarray(values, float)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return {"count": 0, "mean_s": None, "median_s": None, "p95_s": None}
    return {"count": int(v.size), "mean_s": float(v.mean()),
            "median_s": float(np.median(v)),
            "p95_s": float(np.percentile(v, 95))}


def attention_metrics(samples, class_names: Sequence[str],
                      sample_interval_s: float,
                      fixations=None,
                      px_per_deg: Optional[float] = None,
                      image_shape: Optional[Sequence[int]] = None,
                      ) -> Dict[str, Any]:
    """Every Phase-4 metric for one set of semantic-gaze samples.

    `samples` must already be restricted to the unit of interest and to the rows
    whose foveal reading is valid; this function does not filter, so a caller can
    never be surprised about which rows a number came from.
    """
    import pandas as pd

    df = samples if isinstance(samples, pd.DataFrame) else pd.DataFrame(samples)
    n = int(len(df))
    n_classes = len(class_names)
    observed_s = float(n * sample_interval_s)
    if n == 0:
        return {"samples": 0, "observed_time_s": 0.0,
                "sample_interval_s": sample_interval_s,
                "per_class": {}, "groups": {}, "usable": False,
                "reason": "no valid semantic-gaze sample in this unit"}

    ts = df["timestamp_ns"].to_numpy(np.int64)
    top1 = df["top1_class"].to_numpy(object)
    unit_start_ns = int(ts.min())

    fix_lookup: Dict[int, float] = {}
    if fixations is not None:
        fix_frame = (fixations if isinstance(fixations, pd.DataFrame)
                     else pd.DataFrame([f.to_dict() for f in fixations]))
        if len(fix_frame):
            fix_lookup = dict(zip(fix_frame["fixation_id"].astype(int),
                                  fix_frame["duration_s"].astype(float)))
    has_fixation_id = "fixation_id" in df.columns

    per_class: Dict[str, Any] = {}
    for name in class_names:
        mass_col = f"p_{name}"
        selected = top1 == name
        count = int(selected.sum())
        mass = (float(df[mass_col].mean()) if mass_col in df.columns else None)

        fixation_ids: List[int] = []
        if has_fixation_id and count:
            fixation_ids = sorted(
                set(int(v) for v in df.loc[selected, "fixation_id"]) - {-1})
        durations = [fix_lookup[i] for i in fixation_ids if i in fix_lookup]

        # A revisit is a return to the class after having left it: the number of
        # separate runs minus the first one.
        runs = _count_runs(selected)
        first_index = int(np.argmax(selected)) if count else None

        per_class[name] = {
            "foveal_mass_fraction": mass,
            "foveal_mass_percent": None if mass is None else mass * 100.0,
            "top1_samples": count,
            "top1_share_percent": float(count / n * 100.0),
            "dwell_time_s": float(count * sample_interval_s),
            "fixation_count": len(fixation_ids),
            "fixations_per_minute": (float(len(fixation_ids) * 60.0 / observed_s)
                                     if observed_s > 0 else None),
            "fixation_duration": _describe(durations),
            "visit_runs": runs,
            "revisits": max(0, runs - 1),
            "revisit_rate_per_minute": (float(max(0, runs - 1) * 60.0 / observed_s)
                                        if observed_s > 0 else None),
            "time_to_first_fixation_s": (
                float((int(ts[first_index]) - unit_start_ns) / NS_PER_S)
                if first_index is not None else None),
            "is_proxy_only": name in PROXY_ONLY,
            "metric_status": ("exploratory_proxy_needs_manual_review"
                              if name in PROXY_ONLY else "primary"),
        }

    groups: Dict[str, Any] = {}
    for group_name, members in CLASS_GROUPS.items():
        present = [c for c in members if c in per_class]
        mass = [per_class[c]["foveal_mass_fraction"] for c in present
                if per_class[c]["foveal_mass_fraction"] is not None]
        groups[group_name] = {
            "classes": present,
            "foveal_mass_percent": float(sum(mass) * 100.0) if mass else None,
            "top1_share_percent": float(sum(
                per_class[c]["top1_share_percent"] for c in present)),
            "dwell_time_s": float(sum(per_class[c]["dwell_time_s"] for c in present)),
        }

    switches = _count_transitions(top1)
    shares = np.array([per_class[c]["top1_share_percent"] / 100.0
                       for c in class_names])
    mass_vector = np.array([
        per_class[c]["foveal_mass_fraction"] or 0.0 for c in class_names])

    out: Dict[str, Any] = {
        "usable": True,
        "samples": n,
        "observed_time_s": observed_s,
        "sample_interval_s": sample_interval_s,
        "per_class": per_class,
        "groups": groups,
        "semantic_transitions": switches,
        "semantic_transition_rate_per_s": (float(switches / observed_s)
                                           if observed_s > 0 else None),
        "gaze_entropy": {
            "foveal_entropy_mean": (float(df["foveal_entropy"].mean())
                                    if "foveal_entropy" in df else None),
            "foveal_entropy_median": (float(df["foveal_entropy"].median())
                                      if "foveal_entropy" in df else None),
            "top1_distribution_entropy": _entropy(shares, n_classes),
            "foveal_mass_distribution_entropy": _entropy(mass_vector, n_classes),
            "normalisation": f"log({n_classes})",
        },
        "fixation_time_percent": (float(df["is_fixation"].mean() * 100.0)
                                  if "is_fixation" in df else None),
        "primary_metric": "foveal_mass_percent",
        "primary_metric_note": PRIMARY_METRIC_NOTE,
    }
    if px_per_deg and image_shape is not None and {"rect_u", "rect_v"} <= set(df.columns):
        out["scanpath"] = _scanpath(df, px_per_deg, image_shape, ts)
    return out


def _count_runs(flag: np.ndarray) -> int:
    """Number of maximal True runs."""
    f = np.asarray(flag, bool)
    if f.size == 0:
        return 0
    return int(f[0]) + int(np.sum(np.diff(f.astype(np.int8)) == 1))


def _count_transitions(labels: np.ndarray) -> int:
    """Class-to-class changes, ignoring repeats of the same class."""
    values = [v for v in labels if v is not None]
    return int(sum(1 for a, b in zip(values[:-1], values[1:]) if a != b))


def _scanpath(df, px_per_deg: float, image_shape: Sequence[int],
              ts: np.ndarray) -> Dict[str, Any]:
    """Scanpath length, dispersion and eccentricity, all in degrees."""
    u = df["rect_u"].to_numpy(float)
    v = df["rect_v"].to_numpy(float)
    ok = np.isfinite(u) & np.isfinite(v)
    u, v = u[ok], v[ok]
    if u.size < 2:
        return {"samples": int(u.size), "length_deg": None,
                "length_deg_per_s": None, "dispersion_deg": None,
                "mean_eccentricity_deg": None}
    steps = np.hypot(np.diff(u), np.diff(v)) / px_per_deg
    height, width = int(image_shape[0]), int(image_shape[1])
    eccentricity = np.hypot(u - width / 2.0, v - height / 2.0) / px_per_deg
    duration_s = float((ts[ok][-1] - ts[ok][0]) / NS_PER_S) if ok.sum() > 1 else 0.0
    return {
        "samples": int(u.size),
        "length_deg": float(steps.sum()),
        "length_deg_per_s": (float(steps.sum() / duration_s)
                             if duration_s > 0 else None),
        "dispersion_deg": float(np.hypot(np.std(u), np.std(v)) / px_per_deg),
        "mean_eccentricity_deg": float(eccentricity.mean()),
        "p95_eccentricity_deg": float(np.percentile(eccentricity, 95)),
        "median_step_deg": float(np.median(steps)),
        "p95_step_deg": float(np.percentile(steps, 95)),
    }


# --------------------------------------------------------------------------- #
# Flattening, for the per-bin and per-event tables
# --------------------------------------------------------------------------- #
#: Scalar metrics carried onto every bin and event row, in a stable order.
UNIT_METRICS = (
    "road_relevant_mass_percent",
    "interior_cockpit_mass_percent",
    "vulnerable_road_user_mass_percent",
    "sign_and_signal_mass_percent",
    "road_relevant_top1_percent",
    "foveal_entropy_mean",
    "top1_distribution_entropy",
    "semantic_transition_rate_per_s",
    "fixation_time_percent",
    "scanpath_length_deg_per_s",
)


def flatten_unit(metrics: Dict[str, Any]) -> Dict[str, Any]:
    """The scalar summary of one unit, ready to be a table row."""
    if not metrics.get("usable"):
        return {name: None for name in UNIT_METRICS} | {
            "samples": int(metrics.get("samples", 0)),
            "observed_time_s": float(metrics.get("observed_time_s", 0.0))}
    groups = metrics["groups"]
    entropy = metrics["gaze_entropy"]
    scanpath = metrics.get("scanpath") or {}
    return {
        "samples": metrics["samples"],
        "observed_time_s": metrics["observed_time_s"],
        "road_relevant_mass_percent": groups["road_relevant"]["foveal_mass_percent"],
        "interior_cockpit_mass_percent":
            groups["interior_cockpit"]["foveal_mass_percent"],
        "vulnerable_road_user_mass_percent":
            groups["vulnerable_road_user"]["foveal_mass_percent"],
        "sign_and_signal_mass_percent":
            groups["sign_and_signal"]["foveal_mass_percent"],
        "road_relevant_top1_percent": groups["road_relevant"]["top1_share_percent"],
        "foveal_entropy_mean": entropy["foveal_entropy_mean"],
        "top1_distribution_entropy": entropy["top1_distribution_entropy"],
        "semantic_transition_rate_per_s": metrics["semantic_transition_rate_per_s"],
        "fixation_time_percent": metrics["fixation_time_percent"],
        "scanpath_length_deg_per_s": scanpath.get("length_deg_per_s"),
    }


def per_class_rows(metrics: Dict[str, Any], class_names: Sequence[str],
                   **base: Any) -> List[Dict[str, Any]]:
    """One row per class, for the long-format per-bin table."""
    rows: List[Dict[str, Any]] = []
    for name in class_names:
        entry = metrics.get("per_class", {}).get(name)
        if entry is None:
            continue
        duration = entry["fixation_duration"]
        rows.append({
            **base, "class_name": name,
            "foveal_mass_percent": entry["foveal_mass_percent"],
            "top1_time_percent": entry["top1_share_percent"],
            "dwell_time_s": entry["dwell_time_s"],
            "fixation_count": entry["fixation_count"],
            "fixations_per_minute": entry["fixations_per_minute"],
            "mean_fixation_duration_s": duration["mean_s"],
            "median_fixation_duration_s": duration["median_s"],
            "p95_fixation_duration_s": duration["p95_s"],
            "revisit_rate_per_minute": entry["revisit_rate_per_minute"],
            "time_to_first_fixation_s": entry["time_to_first_fixation_s"],
            "metric_status": entry["metric_status"],
        })
    return rows
