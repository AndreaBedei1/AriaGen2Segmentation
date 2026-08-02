"""HMM map matching of a GPS track onto an OSM road network.

A per-point nearest-road snap is not good enough here: the car's GPS has a median
reported accuracy of ~28 m, which is wider than the gap between parallel roads, so
independent snapping flips between carriageways. This module uses the standard
hidden-Markov formulation (Newson & Krumm): candidate road positions are states,
the perpendicular distance is the emission and the agreement between the distance
the vehicle actually moved and the distance along the road is the transition.

Everything it emits carries its own quality: snap distance, heading error, whether
the candidate set was empty, and whether the traversal ran against a one-way tag.
A point that cannot be matched is returned **unmatched with a reason**, never
attached to the nearest road regardless of plausibility.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .osm import (RoadNetwork, angular_difference_deg, menger_curvature,
                  point_segment_projection)

#: Emission scale floor. A reported accuracy below this is not believed: consumer
#: GPS underestimates its own error, especially in an urban canyon.
MIN_EMISSION_SIGMA_M = 8.0

#: Transition scale: how much slack, in metres, between the straight-line step and
#: the along-road step before a transition is heavily penalised.
TRANSITION_BETA_M = 15.0

#: Cost added when two candidates are on ways that do not touch. Finite on
#: purpose: the network extract can be missing a connecting way.
DISCONNECTED_PENALTY_M = 60.0


@dataclass
class MatchResult:
    """Per-GPS-sample map-matching output. Arrays are parallel to the input track."""

    timestamp_ns: np.ndarray
    latitude: np.ndarray
    longitude: np.ndarray
    matched: np.ndarray                 # bool
    reason: List[str]
    matched_x: np.ndarray
    matched_y: np.ndarray
    matched_lat: np.ndarray
    matched_lon: np.ndarray
    snap_distance_m: np.ndarray
    way_index: np.ndarray               # -1 when unmatched
    osm_way_id: np.ndarray
    offset_along_way_m: np.ndarray
    route_progress_m: np.ndarray
    gps_heading_deg: np.ndarray
    road_heading_deg: np.ndarray
    heading_error_deg: np.ndarray
    travel_reversed: np.ndarray         # travelling against the way's digitisation
    against_oneway: np.ndarray
    curvature_1_per_m: np.ndarray
    match_quality: np.ndarray           # 0..1
    warnings: List[List[str]] = field(default_factory=list)
    road_attributes: Dict[str, np.ndarray] = field(default_factory=dict)
    distances: Dict[str, np.ndarray] = field(default_factory=dict)

    def __len__(self) -> int:
        return int(self.timestamp_ns.size)

    def to_frame(self):
        import pandas as pd
        data: Dict[str, Any] = {
            "timestamp_ns": self.timestamp_ns,
            "latitude": self.latitude, "longitude": self.longitude,
            "matched": self.matched, "match_reason": self.reason,
            "matched_lat": self.matched_lat, "matched_lon": self.matched_lon,
            "matched_x_m": self.matched_x, "matched_y_m": self.matched_y,
            "snap_distance_m": self.snap_distance_m,
            "osm_way_id": self.osm_way_id,
            "offset_along_way_m": self.offset_along_way_m,
            "route_progress_m": self.route_progress_m,
            "gps_heading_deg": self.gps_heading_deg,
            "road_heading_deg": self.road_heading_deg,
            "heading_error_deg": self.heading_error_deg,
            "travel_reversed": self.travel_reversed,
            "against_oneway": self.against_oneway,
            "curvature_1_per_m": self.curvature_1_per_m,
            "match_quality": self.match_quality,
            "warnings": ["|".join(w) for w in self.warnings],
        }
        data.update({f"road_{k}": v for k, v in self.road_attributes.items()})
        data.update({f"distance_to_{k}_m": v for k, v in self.distances.items()})
        return pd.DataFrame(data)

    def summary(self) -> Dict[str, Any]:
        m = self.matched
        snap = self.snap_distance_m[m]
        herr = self.heading_error_deg[m]
        herr = herr[np.isfinite(herr)]
        reasons: Dict[str, int] = {}
        for ok, r in zip(self.matched, self.reason):
            if not ok:
                reasons[r] = reasons.get(r, 0) + 1
        return {
            "samples": len(self),
            "matched": int(m.sum()),
            "matched_fraction": float(m.mean()) if len(self) else 0.0,
            "unmatched_reasons": reasons,
            "snap_distance_m": _stats(snap),
            "heading_error_deg": _stats(herr),
            "median_match_quality": (float(np.median(self.match_quality[m]))
                                     if m.any() else None),
            "against_oneway_samples": int(np.sum(self.against_oneway & m)),
            "route_length_m": (float(np.nanmax(self.route_progress_m))
                               if m.any() else 0.0),
        }


def _stats(a: np.ndarray) -> Dict[str, Optional[float]]:
    a = np.asarray(a, float)
    a = a[np.isfinite(a)]
    if a.size == 0:
        return {"count": 0, "median": None, "p90": None, "max": None}
    return {"count": int(a.size), "median": float(np.median(a)),
            "p90": float(np.percentile(a, 90)), "max": float(np.max(a))}


def track_heading_deg(x: np.ndarray, y: np.ndarray,
                      min_step_m: float = 1.0) -> np.ndarray:
    """Heading of travel from consecutive positions, NaN where the step is tiny.

    A heading computed from a sub-metre step of a 3-30 m accurate fix is noise, so
    it is returned as NaN rather than as a number that looks usable.
    """
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    n = x.size
    head = np.full(n, np.nan)
    if n < 2:
        return head
    dx = np.diff(x)
    dy = np.diff(y)
    step = np.hypot(dx, dy)
    seg = np.where(step >= min_step_m,
                   np.degrees(np.arctan2(dx, dy)) % 360.0, np.nan)
    head[:-1] = seg
    head[-1] = seg[-1] if seg.size else np.nan
    # Fill an interior NaN from the previous usable heading; leave leading NaNs.
    for i in range(1, n):
        if np.isnan(head[i]):
            head[i] = head[i - 1]
    return head


def _candidates(net: RoadNetwork, px: float, py: float, radius: float,
                max_candidates: int = 24) -> List[Tuple[int, float, float, float]]:
    """(segment index, distance, t, along-way offset) for segments near a point."""
    if net.seg_way.size == 0:
        return []
    # Cheap bbox pre-filter before the exact perpendicular distance.
    near = ((np.minimum(net.seg_a[:, 0], net.seg_b[:, 0]) - radius <= px) &
            (np.maximum(net.seg_a[:, 0], net.seg_b[:, 0]) + radius >= px) &
            (np.minimum(net.seg_a[:, 1], net.seg_b[:, 1]) - radius <= py) &
            (np.maximum(net.seg_a[:, 1], net.seg_b[:, 1]) + radius >= py))
    idx = np.flatnonzero(near)
    if idx.size == 0:
        return []
    d, t, _ = point_segment_projection(
        np.full(idx.size, px), np.full(idx.size, py),
        net.seg_a[idx, 0], net.seg_a[idx, 1],
        net.seg_b[idx, 0], net.seg_b[idx, 1])
    keep = d <= radius
    idx, d, t = idx[keep], d[keep], t[keep]
    if idx.size == 0:
        return []
    # One candidate per way: the closest point on that way. Otherwise a long road
    # contributes dozens of near-identical states and crowds out the alternatives.
    best: Dict[int, Tuple[int, float, float]] = {}
    for k in range(idx.size):
        w = int(net.seg_way[idx[k]])
        if w not in best or d[k] < best[w][1]:
            best[w] = (int(idx[k]), float(d[k]), float(t[k]))
    out = [(s, dist, tt, float(net.seg_offset_m[s] + tt * net.seg_length_m[s]))
           for s, dist, tt in best.values()]
    out.sort(key=lambda c: c[1])
    return out[:max_candidates]


def _route_distance(net: RoadNetwork, way_nodes: List[set],
                    c_from: Tuple[int, float, float, float],
                    c_to: Tuple[int, float, float, float]) -> float:
    """Approximate along-road distance between two candidate positions."""
    w_from = int(net.seg_way[c_from[0]])
    w_to = int(net.seg_way[c_to[0]])
    if w_from == w_to:
        return abs(c_to[3] - c_from[3])
    fx = net.seg_a[c_from[0]] + c_from[2] * (net.seg_b[c_from[0]] - net.seg_a[c_from[0]])
    tx = net.seg_a[c_to[0]] + c_to[2] * (net.seg_b[c_to[0]] - net.seg_a[c_to[0]])
    straight = float(np.hypot(*(tx - fx)))
    if way_nodes[w_from] & way_nodes[w_to]:
        # Ways touch: the true path runs through a shared node, which is at most a
        # short detour. Charge the straight line plus a small connection cost.
        return straight + 5.0
    return straight + DISCONNECTED_PENALTY_M


def match_track(net: RoadNetwork,
                timestamp_ns: Sequence[int],
                latitude: Sequence[float],
                longitude: Sequence[float],
                accuracy_m: Optional[Sequence[float]] = None,
                max_snap_distance_m: float = 30.0,
                max_heading_error_deg: float = 60.0,
                heading_valid_min_speed_mps: float = 2.0,
                speed_mps: Optional[Sequence[float]] = None,
                ) -> MatchResult:
    """Map-match a GPS track with a Viterbi decode over candidate road positions."""
    ts = np.asarray(timestamp_ns, dtype=np.int64)
    lat = np.asarray(latitude, dtype=float)
    lon = np.asarray(longitude, dtype=float)
    n = ts.size
    acc = (np.asarray(accuracy_m, float) if accuracy_m is not None
           else np.full(n, np.nan))
    spd = (np.asarray(speed_mps, float) if speed_mps is not None
           else np.full(n, np.nan))

    x, y = net.project(lat, lon)
    gps_heading = track_heading_deg(x, y)

    way_nodes = [set(w.node_ids) for w in net.ways]

    # --- candidate generation ------------------------------------------------
    cand: List[List[Tuple[int, float, float, float]]] = []
    for i in range(n):
        sigma = max(MIN_EMISSION_SIGMA_M,
                    float(acc[i]) if np.isfinite(acc[i]) else MIN_EMISSION_SIGMA_M)
        # Search wider than the acceptance radius when the fix says it is poor,
        # so a genuinely uncertain point can still reach the right road.
        radius = max(float(max_snap_distance_m), 2.0 * sigma)
        cand.append(_candidates(net, float(x[i]), float(y[i]), radius))

    # --- Viterbi -------------------------------------------------------------
    log_prob: List[np.ndarray] = []
    back: List[np.ndarray] = []
    for i in range(n):
        cs = cand[i]
        if not cs:
            log_prob.append(np.zeros(0))
            back.append(np.zeros(0, np.int64))
            continue
        sigma = max(MIN_EMISSION_SIGMA_M,
                    float(acc[i]) if np.isfinite(acc[i]) else MIN_EMISSION_SIGMA_M)
        emission = np.array([-0.5 * (c[1] / sigma) ** 2 for c in cs])

        prev_i = _previous_with_candidates(cand, i)
        if prev_i is None or log_prob[prev_i].size == 0:
            log_prob.append(emission)
            back.append(np.full(len(cs), -1, np.int64))
            continue

        gps_step = float(np.hypot(x[i] - x[prev_i], y[i] - y[prev_i]))
        prev_cs = cand[prev_i]
        scores = np.empty((len(prev_cs), len(cs)))
        for a, ca in enumerate(prev_cs):
            for b, cb in enumerate(cs):
                rd = _route_distance(net, way_nodes, ca, cb)
                scores[a, b] = -abs(rd - gps_step) / TRANSITION_BETA_M
        total = log_prob[prev_i][:, None] + scores
        best_prev = np.argmax(total, axis=0)
        log_prob.append(total[best_prev, np.arange(len(cs))] + emission)
        back.append(best_prev.astype(np.int64))

    # --- backtrace -----------------------------------------------------------
    chosen = np.full(n, -1, np.int64)
    last = _last_with_candidates(log_prob)
    if last is not None:
        chosen[last] = int(np.argmax(log_prob[last]))
        cur, k = last, int(chosen[last])
        while True:
            prev_i = _previous_with_candidates(cand, cur)
            if prev_i is None or back[cur].size == 0 or back[cur][k] < 0:
                break
            k = int(back[cur][k])
            chosen[prev_i] = k
            cur = prev_i

    # --- assemble ------------------------------------------------------------
    out = _empty_result(n, ts, lat, lon)
    out.gps_heading_deg = gps_heading
    progress = 0.0
    prev_choice: Optional[Tuple[int, float, float, float]] = None

    for i in range(n):
        warns: List[str] = []
        if not cand[i]:
            out.reason[i] = "no road within the search radius"
            out.warnings.append(warns)
            continue
        k = int(chosen[i])
        if k < 0:
            out.reason[i] = "no viterbi path reached this sample"
            out.warnings.append(warns)
            continue
        seg, dist, t, offset = cand[i][k]
        wi = int(net.seg_way[seg])
        way = net.ways[wi]

        road_head = float(net.seg_heading_deg[seg])
        gh = gps_heading[i]
        reversed_travel = False
        herr = np.nan
        if np.isfinite(gh):
            fwd = angular_difference_deg(np.array(gh), np.array(road_head))
            rev = angular_difference_deg(np.array(gh), np.array((road_head + 180) % 360))
            reversed_travel = bool(rev < fwd)
            herr = float(min(float(fwd), float(rev)))

        # Heading gating is suspended at low speed, where GPS heading is noise.
        heading_trustworthy = (np.isfinite(gh) and
                               (not np.isfinite(spd[i]) or
                                spd[i] >= heading_valid_min_speed_mps))
        if dist > max_snap_distance_m:
            out.reason[i] = (f"snap distance {dist:.1f} m exceeds "
                             f"{max_snap_distance_m:.0f} m")
            out.warnings.append(warns)
            continue
        if heading_trustworthy and np.isfinite(herr) and herr > max_heading_error_deg:
            out.reason[i] = (f"heading error {herr:.0f} deg exceeds "
                             f"{max_heading_error_deg:.0f} deg")
            out.warnings.append(warns)
            continue

        if not heading_trustworthy:
            warns.append("heading_not_trustworthy_low_speed")
        if not np.isfinite(acc[i]):
            warns.append("gps_accuracy_unreported")
        elif acc[i] > 25.0:
            warns.append(f"gps_accuracy_{acc[i]:.0f}m")
        if dist > 0.5 * max_snap_distance_m:
            warns.append("large_snap_distance")

        against_oneway = bool(way.oneway and reversed_travel
                              and not way.reversed_oneway)
        if against_oneway:
            warns.append("travel_against_oneway_tag")

        cx = net.seg_a[seg] + t * (net.seg_b[seg] - net.seg_a[seg])
        mlat, mlon = net.unproject(cx[0], cx[1])

        if prev_choice is not None:
            progress += _route_distance(net, way_nodes, prev_choice, cand[i][k])
        prev_choice = cand[i][k]

        sigma = max(MIN_EMISSION_SIGMA_M,
                    float(acc[i]) if np.isfinite(acc[i]) else MIN_EMISSION_SIGMA_M)
        q_dist = float(np.clip(1.0 - dist / max_snap_distance_m, 0.0, 1.0))
        q_head = (float(np.clip(1.0 - herr / max_heading_error_deg, 0.0, 1.0))
                  if (heading_trustworthy and np.isfinite(herr)) else 0.5)
        q_acc = float(np.clip(MIN_EMISSION_SIGMA_M / sigma, 0.0, 1.0))

        out.matched[i] = True
        out.reason[i] = "ok"
        out.matched_x[i], out.matched_y[i] = float(cx[0]), float(cx[1])
        out.matched_lat[i], out.matched_lon[i] = float(mlat), float(mlon)
        out.snap_distance_m[i] = dist
        out.way_index[i] = wi
        out.osm_way_id[i] = way.osm_id
        out.offset_along_way_m[i] = offset
        out.route_progress_m[i] = progress
        out.road_heading_deg[i] = ((road_head + 180.0) % 360.0 if reversed_travel
                                   else road_head)
        out.heading_error_deg[i] = herr
        out.travel_reversed[i] = reversed_travel
        out.against_oneway[i] = against_oneway
        out.match_quality[i] = float(np.cbrt(q_dist * q_head * q_acc))
        out.warnings.append(warns)

    _attach_road_attributes(net, out)
    _attach_feature_distances(net, out)
    _attach_curvature(net, out)
    return out


def _previous_with_candidates(cand: List[List], i: int) -> Optional[int]:
    for j in range(i - 1, -1, -1):
        if cand[j]:
            return j
    return None


def _last_with_candidates(log_prob: List[np.ndarray]) -> Optional[int]:
    for j in range(len(log_prob) - 1, -1, -1):
        if log_prob[j].size:
            return j
    return None


def _empty_result(n: int, ts, lat, lon) -> MatchResult:
    return MatchResult(
        timestamp_ns=ts, latitude=lat, longitude=lon,
        matched=np.zeros(n, bool), reason=["unprocessed"] * n,
        matched_x=np.full(n, np.nan), matched_y=np.full(n, np.nan),
        matched_lat=np.full(n, np.nan), matched_lon=np.full(n, np.nan),
        snap_distance_m=np.full(n, np.nan),
        way_index=np.full(n, -1, np.int64), osm_way_id=np.full(n, -1, np.int64),
        offset_along_way_m=np.full(n, np.nan),
        route_progress_m=np.full(n, np.nan),
        gps_heading_deg=np.full(n, np.nan), road_heading_deg=np.full(n, np.nan),
        heading_error_deg=np.full(n, np.nan),
        travel_reversed=np.zeros(n, bool), against_oneway=np.zeros(n, bool),
        curvature_1_per_m=np.full(n, np.nan), match_quality=np.zeros(n),
        warnings=[],
    )


def _attach_road_attributes(net: RoadNetwork, out: MatchResult) -> None:
    """Copy the matched way's tags onto each sample, keeping absent tags absent."""
    n = len(out)
    attrs: Dict[str, np.ndarray] = {
        "class": np.array([None] * n, dtype=object),
        "is_roundabout": np.zeros(n, bool),
        "oneway": np.array([None] * n, dtype=object),
        "lanes": np.full(n, np.nan),
        "maxspeed_kph": np.full(n, np.nan),
        "bridge": np.zeros(n, bool),
        "tunnel": np.zeros(n, bool),
        "is_service": np.zeros(n, bool),
        "surface": np.array([None] * n, dtype=object),
    }
    for i in range(n):
        wi = int(out.way_index[i])
        if wi < 0:
            continue
        d = net.ways[wi].to_dict()
        attrs["class"][i] = d["highway"]
        attrs["is_roundabout"][i] = d["is_roundabout"]
        attrs["oneway"][i] = d["oneway"]
        attrs["lanes"][i] = np.nan if d["lanes"] is None else float(d["lanes"])
        attrs["maxspeed_kph"][i] = (np.nan if d["maxspeed_kph"] is None
                                    else float(d["maxspeed_kph"]))
        attrs["bridge"][i] = d["bridge"]
        attrs["tunnel"][i] = d["tunnel"]
        attrs["is_service"][i] = d["is_service"]
        attrs["surface"][i] = d["surface"]
    out.road_attributes = attrs


def _attach_feature_distances(net: RoadNetwork, out: MatchResult) -> None:
    """Distance from the matched position to junctions, roundabouts and controls."""
    x = np.where(out.matched, out.matched_x, np.nan)
    y = np.where(out.matched, out.matched_y, np.nan)
    from .osm import distance_to_features
    dists: Dict[str, np.ndarray] = {}
    for feature in ("junction", "roundabout", "traffic_signals", "crossing",
                    "stop", "give_way", "level_crossing", "mini_roundabout"):
        d = np.full(len(out), np.nan)
        ok = np.isfinite(x)
        if ok.any():
            d[ok] = distance_to_features(net, x[ok], y[ok], feature)
        dists[feature] = d
    out.distances = dists


def _attach_curvature(net: RoadNetwork, out: MatchResult) -> None:
    """Curvature of the matched road, from the way geometry around the match.

    Computed from the *road* geometry rather than from the GPS track, because at
    28 m accuracy the track's own curvature is dominated by fix noise.
    """
    n = len(out)
    k = np.full(n, np.nan)
    per_way: Dict[int, Tuple[np.ndarray, np.ndarray]] = {}
    for i in range(n):
        wi = int(out.way_index[i])
        if wi < 0:
            continue
        if wi not in per_way:
            pts = [net.node_xy[nd] for nd in net.ways[wi].node_ids
                   if nd in net.node_xy]
            if len(pts) < 3:
                per_way[wi] = (np.zeros(0), np.zeros(0))
            else:
                arr = np.asarray(pts, float)
                seg = np.sqrt(((arr[1:] - arr[:-1]) ** 2).sum(axis=1))
                off = np.concatenate([[0.0], np.cumsum(seg)])
                per_way[wi] = (off, menger_curvature(arr[:, 0], arr[:, 1]))
        off, curv = per_way[wi]
        if off.size == 0:
            continue
        j = int(np.argmin(np.abs(off - out.offset_along_way_m[i])))
        k[i] = curv[j]
    out.curvature_1_per_m = k
