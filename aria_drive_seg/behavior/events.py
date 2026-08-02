"""Cartographic road events, and what every other signal did around them.

An event here is a **place on the map the vehicle reached**, not a pattern spotted
in a signal: a roundabout, a junction, a set of traffic signals, a curve, a stop.
Defining events from the map rather than from the data being analysed is what
keeps the event-related response from being circular — a braking event found in
the speed trace would guarantee a speed response.

Every window is a duration in seconds and is clipped to what actually exists:
a baseline window that runs off the start of the recording, or that overlaps the
previous event, is shortened and the shortening is reported. A window that ends
up too short for the statistic it feeds is refused rather than computed.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from ..ingestion.timeline import NS_PER_S

#: Event kinds this module can raise from the map.
EVENT_KINDS = ("roundabout_entry", "roundabout_traverse", "roundabout_exit",
               "junction_approach", "junction_crossing", "traffic_signals",
               "pedestrian_crossing", "curve", "straight", "stop", "restart",
               "road_class_change")

#: Kinds that describe parts of one manoeuvre. A roundabout's entry does not
#: contaminate its own traverse's baseline — it *is* the traverse — so members of
#: a family never trim each other's windows.
EVENT_FAMILIES = (
    {"roundabout_entry", "roundabout_traverse", "roundabout_exit"},
    {"junction_approach", "junction_crossing"},
    {"stop", "restart"},
)


def same_family(kind_a: str, kind_b: str) -> bool:
    """True when two kinds describe parts of the same manoeuvre."""
    if kind_a == kind_b:
        return True
    return any(kind_a in f and kind_b in f for f in EVENT_FAMILIES)


#: Only these kinds can spoil another event's baseline. `curve`, `straight` and
#: `road_class_change` partition the whole drive between them — every metre is in
#: one of them — so treating them as contaminants would leave no event anywhere
#: with a clean baseline. Worse, the straight before a curve is precisely the
#: baseline that curve wants.
CONTAMINATING_KINDS = frozenset({
    "roundabout_entry", "roundabout_traverse", "roundabout_exit",
    "junction_approach", "junction_crossing", "traffic_signals",
    "pedestrian_crossing", "stop", "restart",
})


@dataclass
class RoadEvent:
    event_id: str
    kind: str
    domain: str
    recording_id: str
    start_ns: int
    end_ns: int
    peak_ns: int
    route_progress_m: float
    osm_way_id: int
    road_class: Optional[str]
    detail: Dict[str, Any] = field(default_factory=dict)

    @property
    def duration_s(self) -> float:
        return float((self.end_ns - self.start_ns) / NS_PER_S)

    def to_dict(self) -> Dict[str, Any]:
        return {"event_id": self.event_id, "kind": self.kind, "domain": self.domain,
                "recording_id": self.recording_id, "start_ns": self.start_ns,
                "end_ns": self.end_ns, "peak_ns": self.peak_ns,
                "duration_s": self.duration_s,
                "route_progress_m": self.route_progress_m,
                "osm_way_id": self.osm_way_id, "road_class": self.road_class,
                **{f"detail_{k}": v for k, v in self.detail.items()}}


def _runs(flag: np.ndarray) -> List[Tuple[int, int]]:
    """Half-open index ranges where a boolean array is True."""
    f = np.asarray(flag, bool)
    if f.size == 0:
        return []
    edges = np.diff(f.astype(np.int8))
    starts = list(np.flatnonzero(edges == 1) + 1)
    ends = list(np.flatnonzero(edges == -1) + 1)
    if f[0]:
        starts.insert(0, 0)
    if f[-1]:
        ends.append(f.size)
    return list(zip(starts, ends))


def detect_events(matched, domain: str, recording_id: str,
                  speed_mps: Optional[np.ndarray] = None,
                  approach_radius_m: Optional[Dict[str, float]] = None,
                  curvature_threshold: float = 0.02,
                  stopped_speed_mps: float = 0.5,
                  min_curve_duration_s: float = 1.0) -> List[RoadEvent]:
    """Raise map-defined events from a map-matched track."""
    df = matched[matched["matched"].astype(bool)].sort_values("timestamp_ns")
    if df.empty:
        return []
    radius = {"roundabout": 40.0, "junction": 30.0, "traffic_signals": 40.0,
              "crossing": 25.0, **(approach_radius_m or {})}

    ts = df["timestamp_ns"].values.astype(np.int64)
    progress = df["route_progress_m"].values
    way = df["osm_way_id"].values.astype(np.int64)
    road_class = df["road_class"].values
    curv = np.abs(df["curvature_1_per_m"].values)
    events: List[RoadEvent] = []
    counter = {k: 0 for k in EVENT_KINDS}

    def add(kind: str, a: int, b: int, peak: int, **detail) -> None:
        counter[kind] += 1
        events.append(RoadEvent(
            event_id=f"{domain}_{kind}_{counter[kind]:03d}", kind=kind,
            domain=domain, recording_id=recording_id,
            start_ns=int(ts[a]), end_ns=int(ts[min(b, ts.size) - 1]),
            peak_ns=int(ts[peak]), route_progress_m=float(progress[peak]),
            osm_way_id=int(way[peak]),
            road_class=(None if road_class[peak] is None else str(road_class[peak])),
            detail=detail))

    # --- roundabouts: the map says which way is one -------------------------
    if "road_is_roundabout" in df:
        for a, b in _runs(df["road_is_roundabout"].values.astype(bool)):
            mid = a + (b - a) // 2
            add("roundabout_traverse", a, b, mid,
                length_m=float(progress[b - 1] - progress[a]))
            add("roundabout_entry", a, min(a + 1, b), a)
            add("roundabout_exit", max(b - 1, a), b, b - 1)

    # --- point features: the sample closest to each approach ----------------
    for feature, kind in (("junction", "junction_crossing"),
                          ("traffic_signals", "traffic_signals"),
                          ("crossing", "pedestrian_crossing")):
        col = f"distance_to_{feature}_m"
        if col not in df:
            continue
        d = df[col].values
        near = np.isfinite(d) & (d <= radius.get(feature, 30.0))
        for a, b in _runs(near):
            seg = d[a:b]
            peak = a + int(np.argmin(seg))
            add(kind, a, b, peak, closest_distance_m=float(np.min(seg)))
            if kind == "junction_crossing":
                add("junction_approach", a, max(peak, a + 1), a,
                    approach_distance_m=float(d[a]))

    # --- curves and straights ----------------------------------------------
    curved = np.isfinite(curv) & (curv >= curvature_threshold)
    for flag, kind in ((curved, "curve"), (~curved & np.isfinite(curv), "straight")):
        for a, b in _runs(flag):
            if float((ts[b - 1] - ts[a]) / NS_PER_S) < min_curve_duration_s:
                continue
            peak = a + int(np.argmax(curv[a:b])) if kind == "curve" else a + (b - a) // 2
            add(kind, a, b, peak,
                mean_curvature_1_per_m=float(np.nanmean(curv[a:b])),
                max_curvature_1_per_m=float(np.nanmax(curv[a:b])))

    # --- stops and restarts -------------------------------------------------
    if speed_mps is not None:
        v = np.asarray(speed_mps, float)
        if v.size == ts.size:
            for a, b in _runs(np.isfinite(v) & (v < stopped_speed_mps)):
                add("stop", a, b, a, duration_s=float((ts[b - 1] - ts[a]) / NS_PER_S))
                if b < ts.size:
                    add("restart", b - 1, min(b + 1, ts.size), min(b, ts.size - 1))

    # --- road class changes -------------------------------------------------
    for i in range(1, ts.size):
        if road_class[i] != road_class[i - 1] and road_class[i] is not None:
            add("road_class_change", i - 1, min(i + 1, ts.size), i,
                previous_class=str(road_class[i - 1]), new_class=str(road_class[i]))

    events.sort(key=lambda e: (e.peak_ns, e.kind))
    return events


# --------------------------------------------------------------------------- #
# Event-related windows
# --------------------------------------------------------------------------- #
@dataclass
class Window:
    name: str
    start_ns: int
    end_ns: int
    requested_start_ns: int
    requested_end_ns: int
    clipped: bool
    clip_reason: Optional[str]

    @property
    def duration_s(self) -> float:
        return float(max(0, self.end_ns - self.start_ns) / NS_PER_S)


def build_windows(event: RoadEvent, spec: Dict[str, Sequence[float]],
                  recording_start_ns: int, recording_end_ns: int,
                  other_events: Optional[Sequence[RoadEvent]] = None,
                  avoid_overlap: bool = True) -> Dict[str, Window]:
    """Windows in seconds around an event, clipped to reality and reported as such.

    Offsets are measured from the event's start, except the post-event family,
    which is measured from its end — a 40 s roundabout and a 2 s junction should
    both get "the ten seconds after it finished".
    """
    out: Dict[str, Window] = {}
    for name, (a_s, b_s) in spec.items():
        anchor = event.end_ns if a_s >= 0 and name not in ("event",) else event.start_ns
        req_a = int(anchor + a_s * NS_PER_S)
        req_b = int(anchor + b_s * NS_PER_S)
        a, b = req_a, req_b
        reasons: List[str] = []
        if a < recording_start_ns:
            a = recording_start_ns
            reasons.append("clipped to the start of the recording")
        if b > recording_end_ns:
            b = recording_end_ns
            reasons.append("clipped to the end of the recording")
        if avoid_overlap and other_events:
            for o in other_events:
                if o.event_id == event.event_id:
                    continue
                if o.kind not in CONTAMINATING_KINDS:
                    continue
                # A neighbour from the *same manoeuvre* is not a contaminant:
                # a roundabout's own entry and exit are part of it.
                if not same_family(o.kind, event.kind) and a < o.end_ns and o.start_ns < b:
                    if o.end_ns <= event.start_ns and o.end_ns > a:
                        a = o.end_ns
                        reasons.append(f"trimmed to avoid {o.kind}")
                    elif o.start_ns >= event.end_ns and o.start_ns < b:
                        b = o.start_ns
                        reasons.append(f"trimmed to avoid {o.kind}")
        out[name] = Window(name, a, max(a, b), req_a, req_b,
                           clipped=bool(reasons),
                           clip_reason="; ".join(reasons) or None)
    return out


def summarise_window(window: Window, timestamp_ns: np.ndarray,
                     signals: Dict[str, np.ndarray],
                     min_samples: int = 2) -> Dict[str, Any]:
    """Mean of each signal inside a window, or a refusal with a reason."""
    ts = np.asarray(timestamp_ns, np.int64)
    lo = int(np.searchsorted(ts, window.start_ns, "left"))
    hi = int(np.searchsorted(ts, window.end_ns, "right"))
    n = hi - lo
    out: Dict[str, Any] = {
        f"{window.name}_duration_s": window.duration_s,
        f"{window.name}_samples": n,
        f"{window.name}_clipped": window.clipped,
        f"{window.name}_clip_reason": window.clip_reason,
    }
    for key, values in signals.items():
        v = np.asarray(values, float)[lo:hi] if n > 0 else np.zeros(0)
        v = v[np.isfinite(v)]
        out[f"{window.name}_{key}"] = (float(np.mean(v))
                                       if v.size >= min_samples else None)
    return out


def event_response(events: Sequence[RoadEvent],
                   windows_spec: Dict[str, Sequence[float]],
                   sources: Dict[str, Tuple[np.ndarray, Dict[str, np.ndarray]]],
                   recording_start_ns: int, recording_end_ns: int,
                   baseline_window: Optional[str] = None,
                   avoid_overlap: bool = True) -> List[Dict[str, Any]]:
    """Per-event, per-window aggregates across every supplied signal family.

    `sources` maps a family name to (timestamps, {signal: values}). Families keep
    their own clocks — PPG at 256 Hz, GPS at 1 Hz, gaze at frame rate — and are
    each windowed on their own real samples.

    When `baseline_window` names one of the windows, every other window also gets
    a delta against it. The delta is the interesting quantity and computing it
    here keeps its definition in one place.
    """
    rows: List[Dict[str, Any]] = []
    for ev in events:
        wins = build_windows(ev, windows_spec, recording_start_ns,
                             recording_end_ns, events, avoid_overlap)
        row: Dict[str, Any] = ev.to_dict()
        for family, (ts, signals) in sources.items():
            named = {f"{family}_{k}": v for k, v in signals.items()}
            for w in wins.values():
                row.update(summarise_window(w, ts, named))
        if baseline_window and baseline_window in wins:
            for family, (_, signals) in sources.items():
                for k in signals:
                    base = row.get(f"{baseline_window}_{family}_{k}")
                    if base is None:
                        continue
                    for wname in wins:
                        if wname == baseline_window:
                            continue
                        cur = row.get(f"{wname}_{family}_{k}")
                        row[f"delta_{wname}_{family}_{k}"] = (
                            None if cur is None else float(cur - base))
        rows.append(row)
    return rows


def event_summary(events: Sequence[RoadEvent]) -> Dict[str, Any]:
    """Counts and durations per event kind, normalised per minute of driving."""
    if not events:
        return {"total": 0, "kinds": {}}
    span_s = float((max(e.end_ns for e in events)
                    - min(e.start_ns for e in events)) / NS_PER_S)
    kinds: Dict[str, Any] = {}
    for kind in sorted({e.kind for e in events}):
        sel = [e for e in events if e.kind == kind]
        durations = [e.duration_s for e in sel]
        kinds[kind] = {
            "count": len(sel),
            "per_minute": (float(len(sel) * 60.0 / span_s) if span_s > 0 else None),
            "median_duration_s": float(np.median(durations)),
            "total_duration_s": float(np.sum(durations)),
        }
    return {"total": len(events), "span_s": span_s, "kinds": kinds}
