"""Spatial route binning and car/motorcycle pairing.

The two recordings are aligned **in space, never in time**. They were driven
weeks apart, in different traffic, at different speeds; the only thing they share
is the road.

A bin is defined by the *map*, not by either trajectory: `(osm_way_id,
floor(offset_along_way / bin_size), travel_direction)`. Both vehicles fall into
the same bin when they were at the same place on the same road going the same
way, which is exactly the condition a paired comparison needs. Defining bins from
one vehicle's track instead would make the pairing depend on which vehicle was
chosen as the reference.

Frames are assigned to bins through their nearest **real** GPS fix. At 1 Hz that
fix can be up to half a second — 5 to 8 metres — from the frame, so the positional
resolution of this assignment is reported next to every bin size and a bin
smaller than that resolution is flagged rather than silently trusted.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .osm import angular_difference_deg


def bin_key(osm_way_id: np.ndarray, offset_m: np.ndarray, reversed_travel: np.ndarray,
            bin_size_m: float) -> np.ndarray:
    """Stable string key for a (way, along-way bin, direction) cell."""
    idx = np.floor(np.asarray(offset_m, float) / float(bin_size_m)).astype(np.int64)
    direction = np.where(np.asarray(reversed_travel, bool), "B", "A")
    return np.array([f"w{int(w)}_b{int(i)}_{d}"
                     for w, i, d in zip(np.asarray(osm_way_id, np.int64), idx, direction)])


def positional_resolution_m(gps_rate_hz: float, typical_speed_mps: float) -> float:
    """Half the distance covered between two GPS fixes.

    This is the finest spatial detail a nearest-real-fix assignment can support.
    A bin narrower than this does not resolve anything a coarser bin would not.
    """
    if gps_rate_hz <= 0:
        return float("inf")
    return 0.5 * float(typical_speed_mps) / float(gps_rate_hz)


@dataclass
class RouteBins:
    """Per-domain occupancy of the map-defined spatial bins."""

    bin_size_m: float
    domain: str
    recording_id: str
    keys: np.ndarray
    osm_way_id: np.ndarray
    bin_index: np.ndarray
    direction: np.ndarray
    n_samples: np.ndarray
    mean_x_m: np.ndarray
    mean_y_m: np.ndarray
    mean_heading_deg: np.ndarray
    mean_match_quality: np.ndarray
    mean_snap_distance_m: np.ndarray
    first_timestamp_ns: np.ndarray
    last_timestamp_ns: np.ndarray
    attributes: Dict[str, np.ndarray]

    def to_frame(self):
        import pandas as pd
        data = {
            "bin_key": self.keys, "bin_size_m": self.bin_size_m,
            "domain": self.domain, "recording_id": self.recording_id,
            "osm_way_id": self.osm_way_id, "bin_index": self.bin_index,
            "direction": self.direction, "n_samples": self.n_samples,
            "mean_x_m": self.mean_x_m, "mean_y_m": self.mean_y_m,
            "mean_heading_deg": self.mean_heading_deg,
            "mean_match_quality": self.mean_match_quality,
            "mean_snap_distance_m": self.mean_snap_distance_m,
            "first_timestamp_ns": self.first_timestamp_ns,
            "last_timestamp_ns": self.last_timestamp_ns,
        }
        data.update(self.attributes)
        return pd.DataFrame(data)


def _circular_mean_deg(values: np.ndarray) -> float:
    """Mean of headings. A plain arithmetic mean of 350 and 10 gives 180."""
    v = np.asarray(values, float)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return float("nan")
    r = np.radians(v)
    return float(np.degrees(np.arctan2(np.mean(np.sin(r)), np.mean(np.cos(r)))) % 360.0)


#: Bin attributes carried from the map. Numeric ones are averaged over the samples
#: in the bin; categorical ones take the modal value.
NUMERIC_ATTRIBUTES = ("road_lanes", "road_maxspeed_kph", "curvature_1_per_m",
                      "distance_to_junction_m", "distance_to_roundabout_m",
                      "distance_to_traffic_signals_m", "distance_to_crossing_m",
                      "distance_to_stop_m", "distance_to_give_way_m")
CATEGORICAL_ATTRIBUTES = ("road_class", "road_oneway", "road_surface")
BOOLEAN_ATTRIBUTES = ("road_is_roundabout", "road_bridge", "road_tunnel",
                      "road_is_service", "against_oneway")


def build_route_bins(matched, domain: str, recording_id: str,
                     bin_size_m: float,
                     min_match_quality: float = 0.0) -> RouteBins:
    """Aggregate a domain's matched GPS samples into map-defined spatial bins.

    `matched` is the `map_matched.parquet` frame. Only matched samples take part:
    an unmatched sample has no place on the road and cannot occupy a bin.
    """
    import pandas as pd

    df = matched[matched["matched"].astype(bool)]
    if min_match_quality > 0:
        df = df[df["match_quality"] >= float(min_match_quality)]
    if df.empty:
        empty_f, empty_i = np.zeros(0), np.zeros(0, np.int64)
        return RouteBins(bin_size_m, domain, recording_id, np.zeros(0, dtype=object),
                         empty_i, empty_i, np.zeros(0, dtype=object), empty_i,
                         empty_f, empty_f, empty_f, empty_f, empty_f,
                         empty_i, empty_i, {})

    keys = bin_key(df["osm_way_id"].values, df["offset_along_way_m"].values,
                   df["travel_reversed"].values, bin_size_m)
    work = df.copy()
    work["_key"] = keys
    work["_bin_index"] = np.floor(
        work["offset_along_way_m"].values / float(bin_size_m)).astype(np.int64)
    work["_direction"] = np.where(work["travel_reversed"].values, "B", "A")

    groups = work.groupby("_key", sort=True)
    order = list(groups.groups)

    attributes: Dict[str, np.ndarray] = {}
    for col in NUMERIC_ATTRIBUTES:
        if col in work:
            attributes[col] = groups[col].mean().reindex(order).values
    for col in CATEGORICAL_ATTRIBUTES:
        if col in work:
            attributes[col] = groups[col].agg(
                lambda s: s.mode().iloc[0] if not s.mode().empty else None
            ).reindex(order).values
    for col in BOOLEAN_ATTRIBUTES:
        if col in work:
            attributes[col] = groups[col].agg(
                lambda s: bool(s.astype(bool).mean() >= 0.5)).reindex(order).values

    heading = groups["road_heading_deg"].agg(
        lambda s: _circular_mean_deg(s.values)).reindex(order).values

    return RouteBins(
        bin_size_m=float(bin_size_m), domain=domain, recording_id=recording_id,
        keys=np.asarray(order, dtype=object),
        osm_way_id=groups["osm_way_id"].first().reindex(order).values.astype(np.int64),
        bin_index=groups["_bin_index"].first().reindex(order).values.astype(np.int64),
        direction=groups["_direction"].first().reindex(order).values,
        n_samples=groups.size().reindex(order).values.astype(np.int64),
        mean_x_m=groups["matched_x_m"].mean().reindex(order).values,
        mean_y_m=groups["matched_y_m"].mean().reindex(order).values,
        mean_heading_deg=heading,
        mean_match_quality=groups["match_quality"].mean().reindex(order).values,
        mean_snap_distance_m=groups["snap_distance_m"].mean().reindex(order).values,
        first_timestamp_ns=groups["timestamp_ns"].min().reindex(order).values,
        last_timestamp_ns=groups["timestamp_ns"].max().reindex(order).values,
        attributes=attributes,
    )


@dataclass
class PairingCriteria:
    max_lateral_distance_m: float = 25.0
    max_heading_difference_deg: float = 45.0
    min_samples_per_bin: int = 2
    require_same_direction: bool = True
    require_map_match: bool = True
    min_match_quality: float = 0.0


def pair_route_bins(bins_a: RouteBins, bins_b: RouteBins,
                    criteria: PairingCriteria):
    """Pair bins occupied by both domains, keeping every rejection and its reason.

    Rejections are returned, not dropped: "how much of the shared route survived
    pairing, and why the rest did not" is a result, not bookkeeping.
    """
    import pandas as pd

    fa = bins_a.to_frame().set_index("bin_key")
    fb = bins_b.to_frame().set_index("bin_key")
    shared = sorted(set(fa.index) & set(fb.index))

    rows: List[Dict[str, Any]] = []
    for key in shared:
        a, b = fa.loc[key], fb.loc[key]
        dist = float(np.hypot(a.mean_x_m - b.mean_x_m, a.mean_y_m - b.mean_y_m))
        dh = float(angular_difference_deg(np.array(a.mean_heading_deg),
                                          np.array(b.mean_heading_deg)))
        reasons: List[str] = []
        if criteria.require_same_direction and a.direction != b.direction:
            reasons.append("opposite travel direction")
        if dist > criteria.max_lateral_distance_m:
            reasons.append(f"bin centroids {dist:.1f} m apart")
        if np.isfinite(dh) and dh > criteria.max_heading_difference_deg:
            reasons.append(f"heading differs by {dh:.0f} deg")
        if min(a.n_samples, b.n_samples) < criteria.min_samples_per_bin:
            reasons.append(f"only {int(min(a.n_samples, b.n_samples))} samples "
                           f"in the sparser domain")
        if min(a.mean_match_quality, b.mean_match_quality) < criteria.min_match_quality:
            reasons.append("map-match quality below threshold")

        row: Dict[str, Any] = {
            "bin_key": key, "bin_size_m": bins_a.bin_size_m,
            "osm_way_id": int(a.osm_way_id), "bin_index": int(a.bin_index),
            "paired": not reasons,
            "rejection_reason": "; ".join(reasons) or None,
            "centroid_distance_m": dist,
            "heading_difference_deg": dh,
            "direction_a": a.direction, "direction_b": b.direction,
            "n_samples_a": int(a.n_samples), "n_samples_b": int(b.n_samples),
            "match_quality_a": float(a.mean_match_quality),
            "match_quality_b": float(b.mean_match_quality),
            "pair_quality": float(np.sqrt(max(a.mean_match_quality, 0.0)
                                          * max(b.mean_match_quality, 0.0))),
            "first_timestamp_ns_a": int(a.first_timestamp_ns),
            "first_timestamp_ns_b": int(b.first_timestamp_ns),
            "last_timestamp_ns_a": int(a.last_timestamp_ns),
            "last_timestamp_ns_b": int(b.last_timestamp_ns),
        }
        # Road context is a property of the place, so it is taken from the domain
        # whose map match is better rather than averaged across the two.
        better = a if a.mean_match_quality >= b.mean_match_quality else b
        for col in (*NUMERIC_ATTRIBUTES, *CATEGORICAL_ATTRIBUTES, *BOOLEAN_ATTRIBUTES):
            if col in fa.columns:
                row[col] = better.get(col)
        rows.append(row)

    return pd.DataFrame(rows)


def assign_frames_to_bins(frame_timestamp_ns: np.ndarray,
                          matched,
                          bin_size_m: float,
                          max_temporal_distance_s: float = 0.75,
                          speed_mps: Optional[np.ndarray] = None
                          ) -> Dict[str, np.ndarray]:
    """Attach every RGB frame to the spatial bin of its nearest real GPS fix.

    No position is invented between fixes. The returned
    `position_uncertainty_m` is the honest consequence of that choice: the
    distance the vehicle could have travelled in the temporal gap to the fix that
    was used.
    """
    ts = np.asarray(frame_timestamp_ns, dtype=np.int64)
    ok = matched[matched["matched"].astype(bool)].sort_values("timestamp_ns")
    n = ts.size
    out = {
        "bin_key": np.array([None] * n, dtype=object),
        "gps_dt_s": np.full(n, np.nan),
        "position_uncertainty_m": np.full(n, np.nan),
        "route_progress_m": np.full(n, np.nan),
        "bin_valid": np.zeros(n, bool),
        "bin_reason": np.array(["no matched GPS fix"] * n, dtype=object),
    }
    if ok.empty:
        return out

    gt = ok["timestamp_ns"].values.astype(np.int64)
    keys = bin_key(ok["osm_way_id"].values, ok["offset_along_way_m"].values,
                   ok["travel_reversed"].values, bin_size_m)
    progress = ok["route_progress_m"].values

    pos = np.clip(np.searchsorted(gt, ts), 0, gt.size - 1)
    left = np.clip(pos - 1, 0, gt.size - 1)
    take_left = np.abs(ts - gt[left]) <= np.abs(gt[pos] - ts)
    idx = np.where(take_left, left, pos)
    dt_s = np.abs(ts - gt[idx]) / 1e9

    speed = (np.asarray(speed_mps, float) if speed_mps is not None
             else np.full(n, np.nan))
    if "gps_speed_mps" in ok:
        per_fix = np.asarray(ok["gps_speed_mps"].values, float)[idx]
        fallback = float(np.nanmedian(ok["gps_speed_mps"].values))
    else:
        per_fix = np.full(n, np.nan)
        fallback = np.nan
    eff_speed = np.where(np.isfinite(speed), speed,
                         np.where(np.isfinite(per_fix), per_fix, fallback))

    valid = dt_s <= float(max_temporal_distance_s)
    out["gps_dt_s"] = dt_s
    out["position_uncertainty_m"] = dt_s * eff_speed
    out["route_progress_m"] = np.where(valid, progress[idx], np.nan)
    out["bin_key"] = np.where(valid, keys[idx], None)
    out["bin_valid"] = valid
    out["bin_reason"] = np.where(
        valid, "ok",
        np.array([f"nearest matched GPS fix {d:.2f} s away" for d in dt_s],
                 dtype=object))
    return out


def expected_fixes_per_bin(bin_size_m: float, gps_rate_hz: float,
                           speed_mps: float) -> float:
    """How many real GPS fixes a bin of this size can be expected to contain.

    A 1 Hz fix at 12 m/s lands roughly every 12 m, so a 10 m bin usually holds one
    fix and sometimes none. A pairing rule that demands two fixes per bin is then
    not a quality filter but an impossibility, and the bin size — not the rule —
    is what has to change.
    """
    if not np.isfinite(speed_mps) or speed_mps <= 0 or gps_rate_hz <= 0:
        return float("nan")
    return float(bin_size_m) / (float(speed_mps) / float(gps_rate_hz))


def shared_route_summary(pairs, bin_size_m: float, resolution_m: float,
                         expected_fixes: Optional[float] = None,
                         min_samples_per_bin: int = 2) -> Dict[str, Any]:
    """Headline numbers for one bin size, including the resolution caveats."""
    total = int(len(pairs))
    paired = int(pairs["paired"].sum()) if total else 0
    reasons: Dict[str, int] = {}
    if total:
        for r in pairs.loc[~pairs["paired"], "rejection_reason"]:
            head = str(r).split(";")[0].strip()
            reasons[head] = reasons.get(head, 0) + 1

    undersampled = (expected_fixes is not None and np.isfinite(expected_fixes)
                    and expected_fixes < min_samples_per_bin)
    return {
        "bin_size_m": float(bin_size_m),
        "shared_bins_considered": total,
        "paired_bins": paired,
        "paired_length_m": paired * float(bin_size_m),
        "rejection_reasons": reasons,
        "positional_resolution_m": float(resolution_m),
        "expected_gps_fixes_per_bin": (None if expected_fixes is None
                                       else float(expected_fixes)),
        "bin_below_positional_resolution": bool(bin_size_m < resolution_m),
        "gps_cadence_undersamples_this_bin": bool(undersampled),
        "resolution_caveat": (
            f"a {bin_size_m:.0f} m bin holds about {expected_fixes:.1f} GPS fixes "
            f"at this speed and cadence, below the {min_samples_per_bin} required "
            "for pairing; the low paired count is a sampling limit, not a finding "
            "about the routes" if undersampled else
            (f"a {bin_size_m:.0f} m bin is finer than the {resolution_m:.1f} m "
             "positional resolution a 1 Hz GPS supports"
             if bin_size_m < resolution_m else None)),
    }
