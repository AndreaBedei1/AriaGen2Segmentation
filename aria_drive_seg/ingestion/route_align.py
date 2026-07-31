"""Preliminary spatial alignment of two recordings of the same route.

Alignment is **spatial**, never temporal: two drives of the same road happen at
different absolute times, at different speeds and with a different number of
frames, so neither wall-clock time nor frame index can pair them. The priority
order is GPS, then VIO/SLAM trajectory, then relative spatial progression.

Every pairing carries its own quality evidence and a warning list. Pairs that are
not trustworthy are emitted with `accepted = False` rather than silently dropped or
forced, because "these two points do not correspond" is itself a result.

The current car recording is a provisional 10 fps development baseline, so every
pairing produced here is exploratory and must be recomputed once both recordings
exist at the final protocol rate.
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

EARTH_RADIUS_M = 6_371_008.8


def haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance in metres."""
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = p2 - p1
    dl = math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * EARTH_RADIUS_M * math.asin(math.sqrt(a))


def bearing_deg(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Initial bearing from point 1 to point 2, degrees clockwise from north."""
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dl = math.radians(lon2 - lon1)
    y = math.sin(dl) * math.cos(p2)
    x = math.cos(p1) * math.sin(p2) - math.sin(p1) * math.cos(p2) * math.cos(dl)
    return (math.degrees(math.atan2(y, x)) + 360.0) % 360.0


def angular_difference_deg(a: float, b: float) -> float:
    """Smallest absolute difference between two bearings, in [0, 180]."""
    d = abs((a - b) % 360.0)
    return d if d <= 180.0 else 360.0 - d


@dataclass
class RouteTrack:
    """A recording's trajectory with cumulative distance and normalised progress."""

    recording_id: str
    domain: str
    source: str                      # gps | vio | none
    timestamp_ns: np.ndarray
    latitude: np.ndarray
    longitude: np.ndarray
    accuracy_m: np.ndarray
    speed_mps: np.ndarray
    cumulative_distance_m: np.ndarray
    progression: np.ndarray          # 0..1 along the travelled path
    heading_deg: np.ndarray
    stopped: np.ndarray              # bool
    warnings: List[str] = field(default_factory=list)

    @property
    def total_distance_m(self) -> float:
        return float(self.cumulative_distance_m[-1]) if self.cumulative_distance_m.size else 0.0

    def summary(self) -> Dict[str, Any]:
        return {
            "recording_id": self.recording_id,
            "domain": self.domain,
            "source": self.source,
            "sample_count": int(self.timestamp_ns.size),
            "total_distance_m": self.total_distance_m,
            "duration_s": (float((self.timestamp_ns[-1] - self.timestamp_ns[0]) / 1e9)
                           if self.timestamp_ns.size > 1 else 0.0),
            "median_accuracy_m": (float(np.median(self.accuracy_m))
                                  if self.accuracy_m.size else None),
            "stopped_sample_fraction": (float(np.mean(self.stopped))
                                        if self.stopped.size else 0.0),
            "warnings": self.warnings,
        }


def build_track(recording_id: str, domain: str,
                timestamps_ns: Sequence[int], latitudes: Sequence[float],
                longitudes: Sequence[float],
                accuracies: Optional[Sequence[Optional[float]]] = None,
                speeds: Optional[Sequence[Optional[float]]] = None,
                stop_speed_mps: float = 0.5,
                source: str = "gps") -> RouteTrack:
    """Build a route track with cumulative distance, heading and stop detection."""
    ts = np.asarray(timestamps_ns, dtype=np.int64)
    lat = np.asarray(latitudes, dtype=np.float64)
    lon = np.asarray(longitudes, dtype=np.float64)
    n = ts.size
    acc = np.asarray([np.nan if a is None else a for a in (accuracies or [])],
                     dtype=np.float64) if accuracies else np.full(n, np.nan)
    warnings: List[str] = []

    if n < 2:
        warnings.append("fewer than two positions: no usable trajectory")
        zeros = np.zeros(n)
        return RouteTrack(recording_id, domain, source, ts, lat, lon, acc, zeros,
                          zeros, zeros, zeros, np.zeros(n, dtype=bool), warnings)

    step = np.zeros(n)
    heading = np.zeros(n)
    for i in range(1, n):
        step[i] = haversine_m(lat[i - 1], lon[i - 1], lat[i], lon[i])
        heading[i] = bearing_deg(lat[i - 1], lon[i - 1], lat[i], lon[i])
    heading[0] = heading[1] if n > 1 else 0.0

    cumulative = np.cumsum(step)
    total = float(cumulative[-1])
    progression = cumulative / total if total > 0 else np.zeros(n)

    dt_s = np.diff(ts) / 1e9
    derived_speed = np.zeros(n)
    derived_speed[1:] = np.where(dt_s > 0, step[1:] / np.maximum(dt_s, 1e-9), 0.0)
    if speeds is not None:
        given = np.asarray([np.nan if s is None else s for s in speeds], dtype=np.float64)
        speed = np.where(np.isfinite(given), given, derived_speed)
    else:
        speed = derived_speed

    stopped = speed < stop_speed_mps
    if total <= 0:
        warnings.append("the trajectory has zero length: positions never change")
    if np.isfinite(acc).any() and float(np.nanmedian(acc)) > 20.0:
        warnings.append(
            f"median GPS accuracy {np.nanmedian(acc):.1f} m is degraded")
    jump = step > 200.0
    if np.any(jump):
        warnings.append(
            f"{int(np.sum(jump))} position jumps above 200 m: possible GPS outliers")

    return RouteTrack(recording_id, domain, source, ts, lat, lon, acc, speed,
                      cumulative, progression, heading, stopped, warnings)


@dataclass
class PairCandidate:
    pairing_id: str
    recording_id_car: str
    recording_id_motorcycle: str
    timestamp_ns_car: int
    timestamp_ns_motorcycle: int
    progression_car: float
    progression_motorcycle: float
    latitude_car: float
    longitude_car: float
    latitude_motorcycle: float
    longitude_motorcycle: float
    distance_m: float
    heading_difference_deg: float
    pairing_quality: float
    sensors_used: List[str]
    accepted: bool
    warnings: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def pair_tracks(car: RouteTrack, motorcycle: RouteTrack,
                max_distance_m: float = 40.0,
                max_heading_difference_deg: float = 45.0) -> Dict[str, Any]:
    """Pair the two trajectories by spatial proximity plus travel direction.

    Direction matters: passing the same junction in the opposite direction is a
    different visual experience, so a spatially close point with an incompatible
    heading is rejected rather than paired.
    """
    if car.timestamp_ns.size < 2 or motorcycle.timestamp_ns.size < 2:
        return {"pairs": [], "summary": {
            "paired": False,
            "reason": "at least one recording has no usable trajectory"}}

    pairs: List[PairCandidate] = []
    for j in range(motorcycle.timestamp_ns.size):
        d = np.array([haversine_m(motorcycle.latitude[j], motorcycle.longitude[j],
                                  car.latitude[i], car.longitude[i])
                      for i in range(car.timestamp_ns.size)])
        i = int(np.argmin(d))
        distance = float(d[i])
        heading_diff = angular_difference_deg(
            float(motorcycle.heading_deg[j]), float(car.heading_deg[i]))

        warnings: List[str] = []
        accepted = True
        if distance > max_distance_m:
            accepted = False
            warnings.append(
                f"nearest car position is {distance:.1f} m away, above the "
                f"{max_distance_m:.0f} m threshold: routes do not overlap here")
        if heading_diff > max_heading_difference_deg:
            accepted = False
            warnings.append(
                f"heading differs by {heading_diff:.0f}deg: the two vehicles are not "
                "travelling in a comparable direction at this point")
        if motorcycle.stopped[j] or car.stopped[i]:
            warnings.append("at least one vehicle is stopped at this point")
        for track, idx, label in ((motorcycle, j, "motorcycle"), (car, i, "car")):
            a = track.accuracy_m[idx] if idx < track.accuracy_m.size else np.nan
            if np.isfinite(a) and a > 20.0:
                warnings.append(f"{label} GPS accuracy {a:.0f} m is degraded here")

        # Quality decays with distance and heading mismatch; both are normalised
        # against their own acceptance threshold so the score stays interpretable.
        quality = float(max(0.0, (1.0 - distance / max_distance_m))
                        * max(0.0, (1.0 - heading_diff / max_heading_difference_deg)))
        pairs.append(PairCandidate(
            pairing_id=f"pair_{j:05d}",
            recording_id_car=car.recording_id,
            recording_id_motorcycle=motorcycle.recording_id,
            timestamp_ns_car=int(car.timestamp_ns[i]),
            timestamp_ns_motorcycle=int(motorcycle.timestamp_ns[j]),
            progression_car=float(car.progression[i]),
            progression_motorcycle=float(motorcycle.progression[j]),
            latitude_car=float(car.latitude[i]),
            longitude_car=float(car.longitude[i]),
            latitude_motorcycle=float(motorcycle.latitude[j]),
            longitude_motorcycle=float(motorcycle.longitude[j]),
            distance_m=distance,
            heading_difference_deg=heading_diff,
            pairing_quality=quality,
            sensors_used=[car.source, motorcycle.source],
            accepted=accepted,
            warnings=warnings,
        ))

    accepted = [p for p in pairs if p.accepted]
    summary: Dict[str, Any] = {
        "paired": bool(accepted),
        "status": "exploratory_preliminary",
        "candidate_count": len(pairs),
        "accepted_count": len(accepted),
        "accepted_fraction": len(accepted) / len(pairs) if pairs else 0.0,
        "max_distance_m": max_distance_m,
        "max_heading_difference_deg": max_heading_difference_deg,
        "car": car.summary(),
        "motorcycle": motorcycle.summary(),
    }
    if accepted:
        summary["median_pair_distance_m"] = float(np.median([p.distance_m for p in accepted]))
        summary["median_pair_quality"] = float(np.median([p.pairing_quality for p in accepted]))
        summary["common_route"] = _common_route(accepted, motorcycle, car)
    else:
        summary["common_route"] = None
        summary["reason"] = (
            "no motorcycle position is within the distance and heading thresholds "
            "of any car position: the two recordings do not share a route segment")
    return {"pairs": [p.to_dict() for p in pairs], "summary": summary}


def _common_route(accepted: List[PairCandidate], motorcycle: RouteTrack,
                  car: RouteTrack) -> Dict[str, Any]:
    mp = np.array([p.progression_motorcycle for p in accepted])
    cp = np.array([p.progression_car for p in accepted])
    ts = np.array([p.timestamp_ns_motorcycle for p in accepted], dtype=np.int64)
    return {
        "motorcycle_progression_range": [float(mp.min()), float(mp.max())],
        "car_progression_range": [float(cp.min()), float(cp.max())],
        "motorcycle_covered_fraction": float(mp.max() - mp.min()),
        "car_covered_fraction": float(cp.max() - cp.min()),
        "motorcycle_time_range_ns": [int(ts.min()), int(ts.max())],
        "approximate_common_distance_m": float(
            (mp.max() - mp.min()) * motorcycle.total_distance_m),
        "progression_monotonic": bool(np.all(np.diff(cp[np.argsort(mp)]) >= -0.05)),
    }


def detect_route_events(track: RouteTrack, stop_min_duration_s: float = 5.0,
                        reversal_heading_deg: float = 140.0) -> Dict[str, Any]:
    """Report stops, direction reversals and GPS dropouts along one trajectory."""
    ts, n = track.timestamp_ns, track.timestamp_ns.size
    stops: List[Dict[str, Any]] = []
    i = 0
    while i < n:
        if track.stopped[i]:
            j = i
            while j + 1 < n and track.stopped[j + 1]:
                j += 1
            duration = float((ts[j] - ts[i]) / 1e9)
            if duration >= stop_min_duration_s:
                stops.append({"start_timestamp_ns": int(ts[i]),
                              "end_timestamp_ns": int(ts[j]),
                              "duration_s": duration,
                              "progression": float(track.progression[i])})
            i = j + 1
        else:
            i += 1

    reversals = []
    for k in range(1, n):
        if track.stopped[k]:
            continue
        if angular_difference_deg(float(track.heading_deg[k]),
                                  float(track.heading_deg[k - 1])) > reversal_heading_deg:
            reversals.append({"timestamp_ns": int(ts[k]),
                              "progression": float(track.progression[k]),
                              "heading_before_deg": float(track.heading_deg[k - 1]),
                              "heading_after_deg": float(track.heading_deg[k])})

    dt = np.diff(ts) / 1e9 if n > 1 else np.array([])
    median_dt = float(np.median(dt)) if dt.size else 0.0
    dropouts = [
        {"start_timestamp_ns": int(ts[k]), "end_timestamp_ns": int(ts[k + 1]),
         "gap_s": float(dt[k])}
        for k in range(dt.size) if median_dt > 0 and dt[k] > 3 * median_dt
    ]
    return {"stops": stops, "reversals": reversals, "gps_dropouts": dropouts,
            "median_sample_interval_s": median_dt}


def progression_of_timestamp(track: RouteTrack, timestamp_ns: int
                             ) -> Tuple[Optional[float], Optional[float]]:
    """Route progression at a given device time, with the temporal distance used."""
    if track.timestamp_ns.size == 0:
        return None, None
    i = int(np.argmin(np.abs(track.timestamp_ns - int(timestamp_ns))))
    dt_s = float(abs(int(track.timestamp_ns[i]) - int(timestamp_ns)) / 1e9)
    return float(track.progression[i]), dt_s
