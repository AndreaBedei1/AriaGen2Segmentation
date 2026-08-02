"""Candidate detection for crossing a solid line into the opposing carriageway.

Nothing in this module decides that a traffic law was broken. It produces
**candidates** with the evidence attached, in four states —
`candidate_compliant`, `candidate_noncompliant`, `uncertain`, `not_evaluable` —
and a human decides. That is not caution for its own sake: calling a manoeuvre
illegal requires knowing the line was continuous, that the vehicle really crossed
it, that it was not avoiding an obstacle, not turning at a junction, not being
directed around works, and none of that is in a GPS trace.

A candidate needs several independent kinds of evidence before it is raised. One
segmentation frame is never enough: a single dropped `lane_marking` mask is a
routine model failure, and building an accusation on one would be building it on
the model's worst moment.

The detector also reports its own resolution. Lateral displacement is measured
against the map centreline, and a fix with a 13 m error cannot resolve a 3 m lane
change. Where the geometry cannot support a decision, the answer is
`not_evaluable`, which is a result and not a gap.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from ..ingestion.timeline import NS_PER_S

#: The four states automatic detection is allowed to emit. A definitive label is
#: not one of them and can only come from review.
CANDIDATE_STATES = ("candidate_compliant", "candidate_noncompliant",
                    "uncertain", "not_evaluable")

#: Manoeuvre contexts that make a line crossing ordinary rather than suspect.
BENIGN_CONTEXTS = ("junction", "roundabout", "road_class_change")


def signed_lateral_offset(gps_x: np.ndarray, gps_y: np.ndarray,
                          matched_x: np.ndarray, matched_y: np.ndarray,
                          road_heading_deg: np.ndarray) -> np.ndarray:
    """Signed distance from the road centreline: positive to the right of travel.

    The sign is what matters. An unsigned snap distance cannot tell "drifted right
    onto the verge" from "crossed left into oncoming traffic", and those are very
    different events.
    """
    hx = np.sin(np.radians(np.asarray(road_heading_deg, float)))
    hy = np.cos(np.radians(np.asarray(road_heading_deg, float)))
    dx = np.asarray(gps_x, float) - np.asarray(matched_x, float)
    dy = np.asarray(gps_y, float) - np.asarray(matched_y, float)
    # z-component of heading x offset: positive when the offset is to the right.
    return hx * dy - hy * dx


@dataclass
class MarkingObservation:
    """What the frozen segmentation says about the ego-lane boundary markings."""

    frame_index: int
    timestamp_ns: int
    left_marking_present: bool
    right_marking_present: bool
    left_coverage: float
    right_coverage: float
    marking_pixels: int
    confidence: float


def observe_lane_markings(block, frame_indices: Sequence[int],
                          timestamps_ns: Sequence[int],
                          lane_marking_id: int,
                          horizon_fraction: float = 0.55,
                          ) -> List[MarkingObservation]:
    """Measure lane-marking presence either side of the ego lane, per frame.

    Only the lower part of the image is used: above the horizon a lane marking is
    a few pixels of a road hundreds of metres away, and its presence says nothing
    about the line the vehicle is next to now.
    """
    out: List[MarkingObservation] = []
    for fi, ts in zip(frame_indices, timestamps_ns):
        layers = block.load(int(fi))
        if layers is None:
            continue
        mask = layers["mask"]
        conf = layers["confidence"]
        h, w = mask.shape[:2]
        y0 = int(h * horizon_fraction)
        lower = mask[y0:, :]
        lower_conf = conf[y0:, :]
        is_marking = lower == lane_marking_id
        mid = w // 2
        left, right = is_marking[:, :mid], is_marking[:, mid:]
        out.append(MarkingObservation(
            frame_index=int(fi), timestamp_ns=int(ts),
            left_marking_present=bool(left.any()),
            right_marking_present=bool(right.any()),
            left_coverage=float(left.mean()),
            right_coverage=float(right.mean()),
            marking_pixels=int(is_marking.sum()),
            confidence=(float(lower_conf[is_marking].mean())
                        if is_marking.any() else 0.0),
        ))
    return out


def classify_continuity(observations: Sequence[MarkingObservation], side: str,
                        frame_interval_s: float,
                        min_persistence_s: float = 0.5) -> Dict[str, Any]:
    """Solid or broken, from how the marking behaves over TIME rather than space.

    A solid line is present in essentially every frame; a broken one alternates
    with a period set by the road standard and the vehicle's speed. The duty cycle
    over a run of frames separates them far more reliably than trying to measure
    dash length in a perspective image.

    A short observation cannot tell the two apart — at 30 km/h a single dash and
    its gap take over a second — so a run below `min_persistence_s` returns
    `unknown` rather than guessing.
    """
    key = f"{side}_marking_present"
    present = np.array([getattr(o, key) for o in observations], bool)
    n = present.size
    duration_s = n * frame_interval_s
    out: Dict[str, Any] = {
        "side": side, "frames": int(n), "duration_s": duration_s,
        "duty_cycle": float(present.mean()) if n else 0.0,
        "continuity": "unknown", "confidence": 0.0,
        "reason": None,
    }
    if duration_s < min_persistence_s:
        out["reason"] = (f"only {duration_s:.2f} s observed; below the "
                         f"{min_persistence_s:.2f} s needed to tell a solid line "
                         "from a dashed one")
        return out
    if not present.any():
        out["continuity"] = "absent"
        out["confidence"] = 1.0
        out["reason"] = "no lane marking segmented on this side"
        return out

    duty = float(present.mean())
    # Number of on/off transitions: a broken line switches, a solid one does not.
    switches = int(np.sum(np.diff(present.astype(np.int8)) != 0))
    if duty >= 0.9 and switches <= max(1, int(0.1 * n)):
        out["continuity"] = "solid"
        out["confidence"] = float(min(1.0, duty))
    elif 0.15 <= duty <= 0.8 and switches >= 2:
        out["continuity"] = "broken"
        out["confidence"] = float(min(1.0, switches / max(n * 0.1, 1.0)))
    else:
        out["continuity"] = "unknown"
        out["reason"] = (f"duty cycle {duty:.2f} with {switches} transitions "
                         "matches neither a solid nor a broken line cleanly")
    return out


def vision_lane_offset(block, frame_indices: Sequence[int],
                       timestamps_ns: Sequence[int], lane_marking_id: int,
                       assumed_lane_width_m: float = 3.5,
                       band_top_fraction: float = 0.72,
                       band_bottom_fraction: float = 0.95) -> List[Dict[str, Any]]:
    """Lateral position inside the lane, measured from the image, not from GPS.

    GPS cannot resolve a lane change here — the fix error is larger than a lane is
    wide. The image can: both lane boundaries are visible in the near field, and
    the distance between them in pixels *is* the lane width, which gives the scale
    needed to turn a pixel offset into metres without any calibration.

    The measurement band sits low in the image, a few metres ahead of the vehicle,
    where perspective compression is small and the markings are the ego lane's own
    rather than some other lane's.

    Returns one record per frame. `offset_m` is positive when the vehicle sits
    right of the lane centre. Frames where both boundaries are not visible get
    `offset_m = None`: with one boundary the scale is unknown, and guessing it
    would produce a confident number from an unmeasured quantity.
    """
    out: List[Dict[str, Any]] = []
    for fi, ts in zip(frame_indices, timestamps_ns):
        layers = block.load(int(fi))
        if layers is None:
            continue
        mask = layers["mask"]
        h, w = mask.shape[:2]
        y0 = int(h * band_top_fraction)
        y1 = int(h * band_bottom_fraction)
        band = mask[y0:y1, :] == lane_marking_id
        centre = w / 2.0

        rec: Dict[str, Any] = {
            "frame_index": int(fi), "timestamp_ns": int(ts),
            "left_boundary_px": None, "right_boundary_px": None,
            "lane_width_px": None, "offset_px": None, "offset_m": None,
            "metres_per_px": None, "reason": None,
        }
        if not band.any():
            rec["reason"] = "no lane marking in the measurement band"
            out.append(rec)
            continue

        cols = np.flatnonzero(band.any(axis=0))
        left_cols = cols[cols < centre]
        right_cols = cols[cols >= centre]
        if left_cols.size == 0 or right_cols.size == 0:
            rec["reason"] = ("only one lane boundary visible; the pixel-to-metre "
                             "scale is unmeasured, so no offset is reported")
            out.append(rec)
            continue

        # The boundaries closest to the vehicle's own path are the ego lane's.
        left = float(left_cols.max())
        right = float(right_cols.min())
        width_px = right - left
        if width_px < w * 0.05:
            rec["reason"] = (f"apparent lane width {width_px:.0f} px is implausibly "
                             "narrow; the two detections are probably one marking")
            out.append(rec)
            continue

        mpp = float(assumed_lane_width_m) / width_px
        lane_centre = 0.5 * (left + right)
        rec.update({
            "left_boundary_px": left, "right_boundary_px": right,
            "lane_width_px": width_px, "offset_px": float(centre - lane_centre),
            "metres_per_px": mpp,
            "offset_m": float((centre - lane_centre) * mpp),
        })
        out.append(rec)
    return out


def summarise_vision_offset(records: Sequence[Dict[str, Any]],
                            assumed_lane_width_m: float = 3.5) -> Dict[str, Any]:
    """Usability of the vision-based lateral offset over one block."""
    total = len(records)
    vals = np.array([r["offset_m"] for r in records if r["offset_m"] is not None],
                    dtype=float)
    reasons: Dict[str, int] = {}
    for r in records:
        if r["offset_m"] is None and r["reason"]:
            head = str(r["reason"]).split(";")[0]
            reasons[head] = reasons.get(head, 0) + 1
    return {
        "frames": total,
        "frames_with_offset": int(vals.size),
        "coverage_fraction": float(vals.size / total) if total else 0.0,
        "offset_m": {
            "median": float(np.median(vals)) if vals.size else None,
            "p05": float(np.percentile(vals, 5)) if vals.size else None,
            "p95": float(np.percentile(vals, 95)) if vals.size else None,
            "range_m": (float(np.percentile(vals, 95) - np.percentile(vals, 5))
                        if vals.size else None),
        },
        "unavailable_reasons": reasons,
        "assumed_lane_width_m": assumed_lane_width_m,
        "caveat": ("the pixel-to-metre scale assumes a "
                   f"{assumed_lane_width_m} m lane. The measured offset scales "
                   "linearly with that assumption, so its absolute value is only "
                   "as good as the lane width; changes over time within one block "
                   "do not depend on it."),
    }


@dataclass
class Candidate:
    candidate_id: str
    domain: str
    recording_id: str
    start_ns: int
    end_ns: int
    state: str
    manoeuvre: str
    evidence: Dict[str, Any] = field(default_factory=dict)
    evidence_sources: List[str] = field(default_factory=list)
    review_status: str = "pending_human_review"

    @property
    def duration_s(self) -> float:
        return float((self.end_ns - self.start_ns) / NS_PER_S)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "candidate_id": self.candidate_id, "domain": self.domain,
            "recording_id": self.recording_id,
            "start_ns": self.start_ns, "end_ns": self.end_ns,
            "duration_s": self.duration_s,
            "state": self.state, "manoeuvre": self.manoeuvre,
            "evidence_source_count": len(self.evidence_sources),
            "evidence_sources": "|".join(self.evidence_sources),
            "review_status": self.review_status,
            "automatic_label_is_not_final": True,
            **{f"evidence_{k}": v for k, v in self.evidence.items()},
        }


def lateral_resolution_m(accuracy_m: np.ndarray) -> float:
    """The smallest lateral displacement the positioning can actually resolve."""
    a = np.asarray(accuracy_m, float)
    a = a[np.isfinite(a)]
    return float(np.median(a)) if a.size else float("inf")


def detect_candidates(matched, domain: str, recording_id: str,
                      marking_runs: Optional[Dict[str, Any]] = None,
                      min_evidence_sources: int = 3,
                      min_lateral_displacement_m: float = 1.2,
                      min_time_beyond_line_s: float = 0.5,
                      junction_exclusion_radius_m: float = 25.0,
                      ) -> Tuple[List[Candidate], Dict[str, Any]]:
    """Raise candidates where several independent signals agree on a crossing."""
    df = matched[matched["matched"].astype(bool)].sort_values(
        "timestamp_ns").reset_index(drop=True)
    diagnostics: Dict[str, Any] = {
        "domain": domain, "samples_considered": int(len(df)),
        "candidates": 0, "states": {}, "not_evaluable_reasons": {},
    }
    if df.empty:
        diagnostics["not_evaluable_reasons"]["no matched GPS"] = 1
        return [], diagnostics

    # The offset of the RAW fix from the centreline foot the matcher chose,
    # signed by which side of the direction of travel it fell on.
    lateral = _signed_from_snap(df)

    resolution = lateral_resolution_m(df["gps_accuracy_m"].values
                                      if "gps_accuracy_m" in df else np.array([]))
    diagnostics["lateral_resolution_m"] = resolution
    diagnostics["min_lateral_displacement_m"] = float(min_lateral_displacement_m)
    resolvable = resolution <= min_lateral_displacement_m
    diagnostics["displacement_is_resolvable"] = bool(resolvable)
    if not resolvable:
        diagnostics["resolution_caveat"] = (
            f"the median positioning error is {resolution:.1f} m, larger than the "
            f"{min_lateral_displacement_m:.1f} m displacement a lane crossing "
            "produces. A lateral crossing cannot be resolved from this track, so "
            "every candidate here is at best 'uncertain' and most are "
            "'not_evaluable'.")

    ts = df["timestamp_ns"].values.astype(np.int64)
    candidates: List[Candidate] = []
    n = 0
    beyond = np.abs(lateral) >= min_lateral_displacement_m
    for a, b in _runs(beyond):
        duration = float((ts[b - 1] - ts[a]) / NS_PER_S)
        if duration < min_time_beyond_line_s:
            continue
        n += 1
        seg = df.iloc[a:b]
        sources: List[str] = ["signed_lateral_offset"]
        evidence: Dict[str, Any] = {
            "peak_lateral_offset_m": float(np.max(np.abs(lateral[a:b]))),
            "mean_lateral_offset_m": float(np.mean(lateral[a:b])),
            "crossed_to_left": bool(np.mean(lateral[a:b]) < 0),
            "duration_beyond_s": duration,
            "distance_beyond_m": float(
                seg["route_progress_m"].iloc[-1] - seg["route_progress_m"].iloc[0]),
            "mean_speed_mps": (float(np.nanmean(seg["gps_speed_mps"]))
                               if "gps_speed_mps" in seg else None),
            "mean_heading_error_deg": float(np.nanmean(seg["heading_error_deg"])),
            "median_gps_accuracy_m": float(np.nanmedian(seg["gps_accuracy_m"]))
            if "gps_accuracy_m" in seg else None,
            "road_class": (None if seg["road_class"].iloc[0] is None
                           else str(seg["road_class"].iloc[0])),
            "road_oneway": str(seg["road_oneway"].iloc[0]),
            "against_oneway": bool(seg["against_oneway"].any()),
            "min_distance_to_junction_m": float(
                np.nanmin(seg["distance_to_junction_m"])),
            "min_distance_to_roundabout_m": float(
                np.nanmin(seg["distance_to_roundabout_m"])),
        }
        if evidence["against_oneway"]:
            sources.append("map_direction_conflict")
        if np.isfinite(evidence["mean_heading_error_deg"]):
            sources.append("heading_vs_road")

        # --- segmentation evidence, when a dense block covers this window ----
        marking = _marking_for_window(marking_runs, int(ts[a]), int(ts[b - 1]))
        if marking:
            evidence.update({f"marking_{k}": v for k, v in marking.items()})
            if marking.get("continuity") in ("solid", "broken"):
                sources.append("lane_marking_continuity")
            if marking.get("observed_frames", 0) > 0:
                sources.append("lane_marking_presence")
        else:
            evidence["marking_continuity"] = None
            evidence["marking_reason"] = (
                "no dense frozen semantic block covers this window")

        # --- manoeuvre context ----------------------------------------------
        manoeuvre = "lane_keeping_or_drift"
        if evidence["min_distance_to_roundabout_m"] <= junction_exclusion_radius_m:
            manoeuvre = "roundabout_manoeuvre"
        elif evidence["min_distance_to_junction_m"] <= junction_exclusion_radius_m:
            manoeuvre = "junction_manoeuvre"
        elif marking and marking.get("continuity") == "broken":
            manoeuvre = "lane_change_on_broken_line"
        elif marking and marking.get("continuity") == "solid":
            manoeuvre = "candidate_solid_line_crossing"

        # --- state ------------------------------------------------------------
        state = _decide_state(evidence, marking, manoeuvre, sources,
                              min_evidence_sources, resolvable,
                              junction_exclusion_radius_m)
        diagnostics["states"][state] = diagnostics["states"].get(state, 0) + 1
        candidates.append(Candidate(
            candidate_id=f"{domain}_slc_{n:03d}", domain=domain,
            recording_id=recording_id, start_ns=int(ts[a]), end_ns=int(ts[b - 1]),
            state=state, manoeuvre=manoeuvre, evidence=evidence,
            evidence_sources=sources))

    diagnostics["candidates"] = len(candidates)
    return candidates, diagnostics


def _decide_state(evidence, marking, manoeuvre, sources, min_sources,
                  resolvable, junction_radius) -> str:
    """Map the evidence onto one of the four permitted states."""
    if not resolvable:
        return "not_evaluable"
    if len(sources) < min_sources:
        return "not_evaluable"
    if manoeuvre in ("roundabout_manoeuvre", "junction_manoeuvre"):
        # Crossing a line while turning at a junction is ordinary driving.
        return "candidate_compliant"
    if marking is None or marking.get("continuity") in (None, "unknown", "absent"):
        return "uncertain"
    if marking.get("continuity") == "broken":
        return "candidate_compliant"
    if marking.get("continuity") == "solid":
        # Even here it is only a candidate: an obstacle, roadworks or a police
        # direction all make crossing a solid line lawful, and none is visible
        # in this evidence set.
        return "candidate_noncompliant"
    return "uncertain"


def _runs(flag: np.ndarray) -> List[Tuple[int, int]]:
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


def _signed_from_snap(df) -> np.ndarray:
    """Snap distance, signed by which side of the road the raw fix fell on."""
    import numpy as np
    net_x = df["matched_x_m"].values
    net_y = df["matched_y_m"].values
    # Raw fix, projected the same way the matcher projected it. The matched frame
    # stores lat/lon for both, so the offset is recovered in metres locally.
    dlat = df["latitude"].values - df["matched_lat"].values
    dlon = df["longitude"].values - df["matched_lon"].values
    lat0 = float(np.nanmean(df["matched_lat"].values))
    dx = np.radians(dlon) * 6_371_008.8 * np.cos(np.radians(lat0))
    dy = np.radians(dlat) * 6_371_008.8
    hx = np.sin(np.radians(df["road_heading_deg"].values))
    hy = np.cos(np.radians(df["road_heading_deg"].values))
    return hx * dy - hy * dx


def _marking_for_window(marking_runs, start_ns, end_ns) -> Optional[Dict[str, Any]]:
    """Continuity classification for the block covering a time window, if any."""
    if not marking_runs:
        return None
    for run in marking_runs.get("runs", []):
        if run["start_ns"] <= end_ns and start_ns <= run["end_ns"]:
            return run
    return None


def summarise(candidates: Sequence[Candidate],
              diagnostics: Dict[str, Any]) -> Dict[str, Any]:
    """Counts per state, with the reminder that none of them is a verdict."""
    states: Dict[str, int] = {}
    manoeuvres: Dict[str, int] = {}
    for c in candidates:
        states[c.state] = states.get(c.state, 0) + 1
        manoeuvres[c.manoeuvre] = manoeuvres.get(c.manoeuvre, 0) + 1
    return {
        "candidates": len(candidates),
        "states": states,
        "manoeuvres": manoeuvres,
        "permitted_states": list(CANDIDATE_STATES),
        "review_required": True,
        "note": ("every state here is a candidate awaiting human review. No row "
                 "asserts that a traffic rule was broken, and the pipeline has no "
                 "code path that can produce such an assertion."),
        **{k: v for k, v in diagnostics.items() if k != "states"},
    }
