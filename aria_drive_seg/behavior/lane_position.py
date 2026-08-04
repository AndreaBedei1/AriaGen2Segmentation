"""Visual lane-position proxy from lane markings and road geometry.

**This is a proxy and is named one everywhere.** It measures where the *camera*
sits between the two lane markings visible in the near field of the segmented
image. It is not the vehicle's metric position in the lane. The camera is on the
rider's head, so a head lean moves it without the vehicle moving; the
segmentation can miss or invent a marking; and the mapping from pixels to metres
rests on an assumed lane width. Every artefact this module writes carries
`visual_lane_position_proxy` and `is_vehicle_metric_position: false`.

The primary quantity is deliberately **normalised**: the offset is expressed as a
fraction of the lane's own half-width, measured in the same image. That makes it
independent of the assumed lane width, of the camera's focal length and of the
perspective scale at the measurement band — the three things a metric offset
would inherit and then hide. A metric value is also reported, under its
assumption, because a reader wants an order of magnitude.

Road geometry enters as a **rejection**: the strip between the two candidate
markings must actually be road surface. Two markings with a hedge between them
are two different lanes' markings, or a marking and a white van, and the pair is
discarded rather than turned into an implausibly wide lane.
"""
from __future__ import annotations

from typing import Any, Dict, List, Sequence, Tuple

import numpy as np

NS_PER_S = 1_000_000_000

PROXY_LABEL = "visual_lane_position_proxy"

PROXY_NOTE = (
    "visual lane-position proxy: the horizontal position of the head-mounted "
    "camera between the two lane markings visible in the near field, normalised "
    "by the lane's own half-width in the same image. It is not a metric "
    "measurement of where the vehicle sits in its lane, and it moves when the "
    "rider's head moves.")

#: |normalised offset| below this counts as "near the centre" — the central third
#: of the lane.
NEAR_CENTRE = 1.0 / 3.0

#: |normalised offset| above this counts as "near a boundary" — the outer third.
NEAR_BOUNDARY = 2.0 / 3.0


# --------------------------------------------------------------------------- #
# Measurement band
# --------------------------------------------------------------------------- #
def road_row_profile(masks, frame_indices: Sequence[int], road_surface_id: int):
    """Median fraction of each image row that is road surface, over a sample."""
    rows: List[np.ndarray] = []
    for frame_index in frame_indices:
        mask = masks(int(frame_index))
        if mask is not None:
            rows.append((mask == int(road_surface_id)).mean(axis=1))
    if not rows:
        return np.zeros(0)
    return np.median(np.stack(rows), axis=0)


def select_measurement_band(row_road_fraction: Sequence[float],
                            min_road_row_fraction: float = 0.15,
                            lower_portion: float = 0.60,
                            min_band_fraction: float = 0.05,
                            ) -> Dict[str, Any]:
    """Choose the near-field road band from the segmentation itself.

    A fixed band cannot serve both vehicles. On the motorcycle the road runs to
    the bottom of the frame; in the car the same rows are 85% dashboard, because
    a head-mounted camera behind a windscreen sees the bonnet where the
    motorcycle sees tarmac. Using the motorcycle's band on the car would measure
    the dashboard and report it as a lane position.

    So the band is measured: it is the **lowest contiguous run of rows that are
    predominantly road surface**, restricted to that run's lower portion, where
    perspective compression is smallest and the markings are the ego lane's own.
    """
    profile = np.asarray(row_road_fraction, float)
    height = profile.size
    out: Dict[str, Any] = {
        "selected": False, "top_fraction": None, "bottom_fraction": None,
        "min_road_row_fraction": float(min_road_row_fraction),
        "lower_portion": float(lower_portion), "reason": None,
    }
    if height == 0:
        out["reason"] = "no frames were readable, so no row profile exists"
        return out
    above = profile >= float(min_road_row_fraction)
    runs = _runs(above)
    if not runs:
        out["reason"] = (f"no image row is at least {min_road_row_fraction:.0%} "
                         "road surface; this recording has no usable near-field "
                         "road band")
        return out
    start, end = runs[-1]                     # lowest run in the image
    run_height = end - start
    band_height = max(int(round(run_height * float(lower_portion))),
                      int(round(height * float(min_band_fraction))))
    band_height = min(band_height, run_height)
    top = end - band_height
    out.update({
        "selected": True,
        "top_fraction": float(top / height),
        "bottom_fraction": float(end / height),
        "road_run_top_fraction": float(start / height),
        "road_run_bottom_fraction": float(end / height),
        "median_road_fraction_in_band": float(np.mean(profile[top:end])),
        "method": "lowest_contiguous_road_row_run_lower_portion",
        "note": ("the band is measured from this recording's own segmentation, "
                 "because a band that is near-field road on a motorcycle is "
                 "dashboard in a car"),
    })
    return out


def _marking_groups(columns: np.ndarray, max_gap_px: int) -> List[Tuple[float, float]]:
    """Contiguous column groups, as (first, last), merging gaps up to `max_gap_px`.

    One painted line is many adjacent columns and, through a dashed segment or a
    segmentation hole, sometimes two nearly-adjacent groups. Grouping is what lets
    a *pair* of lines be found; taking raw columns would make a single wide line
    look like two boundaries a few pixels apart.
    """
    if columns.size == 0:
        return []
    groups: List[Tuple[float, float]] = []
    start = previous = float(columns[0])
    for column in columns[1:]:
        if column - previous > max_gap_px:
            groups.append((start, previous))
            start = float(column)
        previous = float(column)
    groups.append((start, previous))
    return groups


def measure_frame(mask: np.ndarray, lane_marking_id: int, road_surface_id: int,
                  band_top_fraction: float = 0.72,
                  band_bottom_fraction: float = 0.95,
                  min_lane_width_fraction: float = 0.05,
                  max_lane_width_fraction: float = 0.90,
                  min_road_fraction_between: float = 0.50,
                  marking_merge_gap_fraction: float = 0.01,
                  ) -> Dict[str, Any]:
    """Normalised lateral offset from one segmentation mask, or a stated reason.

    Only a band low in the image is used. Above it, a lane marking is a few pixels
    of a road hundreds of metres away and says nothing about the line the vehicle
    is beside now.

    The two markings are chosen as the **adjacent pair whose lane centre is
    nearest the camera axis** among the pairs that pass the width and road-surface
    checks. Crucially the camera axis is *not* required to lie between them: a
    vehicle straddling a line has both markings on one side, and a rule that
    picked one marking from each side of the axis would bound the offset to ±1 by
    construction and could never report a crossing at all.
    """
    height, width = mask.shape[:2]
    y0 = int(height * float(band_top_fraction))
    y1 = max(y0 + 1, int(height * float(band_bottom_fraction)))
    band = mask[y0:y1, :]
    centre = width / 2.0

    record: Dict[str, Any] = {
        "left_boundary_px": None, "right_boundary_px": None,
        "lane_width_px": None, "lane_centre_px": None,
        "normalized_offset": None, "road_fraction_between": None,
        "candidate_pairs": 0, "measured": False, "reason": None,
    }
    is_marking = band == int(lane_marking_id)
    if not is_marking.any():
        record["reason"] = "no lane marking in the measurement band"
        return record

    columns = np.flatnonzero(is_marking.any(axis=0))
    groups = _marking_groups(
        columns, max_gap_px=max(1, int(round(width * marking_merge_gap_fraction))))
    if len(groups) < 2:
        record["reason"] = (
            "only one lane marking is visible; with one boundary the lane's "
            "half-width is unmeasured and no normalised offset exists")
        return record

    min_width = width * float(min_lane_width_fraction)
    max_width = width * float(max_lane_width_fraction)
    rejections: List[str] = []
    candidates: List[Dict[str, Any]] = []
    for (left_start, left_end), (right_start, right_end) in zip(groups[:-1],
                                                               groups[1:]):
        # Inner edges of the two painted lines bound the carriageway between them.
        left, right = float(left_end), float(right_start)
        lane_width_px = right - left
        if lane_width_px < min_width:
            rejections.append(
                f"apparent lane width {lane_width_px:.0f} px is implausibly narrow")
            continue
        if lane_width_px > max_width:
            rejections.append(
                f"apparent lane width {lane_width_px:.0f} px spans almost the "
                "whole image")
            continue
        strip = band[:, int(np.ceil(left)):int(np.floor(right)) + 1]
        road_fraction = (float(np.mean(strip == int(road_surface_id)))
                         if strip.size else 0.0)
        if road_fraction < float(min_road_fraction_between):
            rejections.append(
                f"only {road_fraction:.0%} of the strip between two markings is "
                "road surface")
            continue
        candidates.append({
            "left": left, "right": right, "lane_width_px": lane_width_px,
            "lane_centre_px": 0.5 * (left + right),
            "road_fraction_between": road_fraction,
        })

    record["candidate_pairs"] = len(candidates)
    if not candidates:
        record["reason"] = (rejections[0] if rejections
                            else "no adjacent pair of markings bounds a lane")
        return record

    # The ego lane is the candidate whose centre is nearest the camera axis.
    best = min(candidates, key=lambda c: abs(c["lane_centre_px"] - centre))
    record.update({
        "left_boundary_px": best["left"], "right_boundary_px": best["right"],
        "lane_width_px": best["lane_width_px"],
        "lane_centre_px": best["lane_centre_px"],
        "road_fraction_between": best["road_fraction_between"],
        # Positive when the camera sits right of the lane centre. Normalised by
        # the half-width, so ±1 is exactly a boundary and beyond ±1 is beyond it.
        "normalized_offset": float((centre - best["lane_centre_px"]) /
                                   (best["lane_width_px"] / 2.0)),
        "measured": True,
    })
    return record


def measure_run(masks, frame_indices: Sequence[int], timestamps_ns: Sequence[int],
                lane_marking_id: int, road_surface_id: int,
                assumed_lane_width_m: float = 3.5, **kwargs) -> List[Dict[str, Any]]:
    """One record per frame. `masks` is any callable frame_index -> mask array."""
    out: List[Dict[str, Any]] = []
    for frame_index, ts in zip(frame_indices, timestamps_ns):
        mask = masks(int(frame_index))
        if mask is None:
            out.append({"frame_index": int(frame_index), "timestamp_ns": int(ts),
                        "measured": False, "normalized_offset": None,
                        "offset_m": None,
                        "reason": "no segmentation mask for this frame"})
            continue
        record = measure_frame(mask, lane_marking_id, road_surface_id, **kwargs)
        offset = record["normalized_offset"]
        record.update({
            "frame_index": int(frame_index), "timestamp_ns": int(ts),
            # The metric value is the normalised one scaled by the assumed lane
            # half-width. It inherits that assumption; the normalised value does
            # not, which is why the normalised value is the primary one.
            "offset_m": (None if offset is None
                         else float(offset * assumed_lane_width_m / 2.0)),
        })
        out.append(record)
    return out


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


def crossing_candidates(timestamp_ns: Sequence[int], offset: Sequence[float],
                        min_duration_s: float = 0.5,
                        boundary: float = 1.0) -> List[Dict[str, Any]]:
    """Runs where the proxy sits beyond a lane boundary for long enough.

    A **candidate**, never a verdict. The proxy can exceed a boundary because the
    rider leaned their head, because the segmentation dropped a marking, or
    because the vehicle really was over the line, and this module cannot tell
    those apart. Whether a crossing was lawful is a further question again, and is
    not asked here at all.
    """
    ts = np.asarray(timestamp_ns, np.int64)
    values = np.asarray(offset, float)
    beyond = np.isfinite(values) & (np.abs(values) > float(boundary))
    out: List[Dict[str, Any]] = []
    for a, b in _runs(beyond):
        duration_s = float((ts[b - 1] - ts[a]) / NS_PER_S)
        if duration_s < float(min_duration_s):
            continue
        segment = values[a:b]
        out.append({
            "start_ns": int(ts[a]), "end_ns": int(ts[b - 1]),
            "duration_beyond_boundary_s": duration_s,
            "side": "right" if float(np.mean(segment)) > 0 else "left",
            "peak_normalized_offset": float(segment[int(np.argmax(np.abs(segment)))]),
            "samples": int(b - a),
            "state": "candidate_requires_review",
            "is_traffic_violation_claim": False,
        })
    return out


def summarise(records: Sequence[Dict[str, Any]], sample_interval_s: float,
              assumed_lane_width_m: float = 3.5,
              min_crossing_duration_s: float = 0.5) -> Dict[str, Any]:
    """Coverage, band, time-in-zone and crossing candidates for one recording."""
    total = len(records)
    measured = [r for r in records if r.get("measured")]
    values = np.asarray([r["normalized_offset"] for r in measured], float)
    ts = np.asarray([r["timestamp_ns"] for r in measured], np.int64)

    reasons: Dict[str, int] = {}
    for record in records:
        if not record.get("measured") and record.get("reason"):
            head = str(record["reason"]).split(";")[0]
            reasons[head] = reasons.get(head, 0) + 1

    if values.size == 0:
        return {
            "metric": PROXY_LABEL, "is_vehicle_metric_position": False,
            "frames": total, "frames_measured": 0, "coverage_fraction": 0.0,
            "unavailable_reasons": reasons, "note": PROXY_NOTE,
            "reason": "no frame produced a usable pair of lane boundaries",
        }

    p05, p95 = np.percentile(values, [5, 95])
    near_centre = np.abs(values) <= NEAR_CENTRE
    near_left = values <= -NEAR_BOUNDARY
    near_right = values >= NEAR_BOUNDARY
    candidates = crossing_candidates(ts, values,
                                     min_duration_s=min_crossing_duration_s)
    observed_s = float(values.size * sample_interval_s)

    return {
        "metric": PROXY_LABEL,
        "is_vehicle_metric_position": False,
        "frames": total,
        "frames_measured": int(values.size),
        "coverage_fraction": float(values.size / total) if total else 0.0,
        "observed_time_s": observed_s,
        "sample_interval_s": float(sample_interval_s),
        "normalized_offset": {
            "median": float(np.median(values)),
            "mean": float(values.mean()),
            "sd": float(values.std(ddof=1)) if values.size > 1 else None,
            "iqr": float(np.percentile(values, 75) - np.percentile(values, 25)),
            "p05": float(p05), "p95": float(p95),
            "p05_p95_band_width": float(p95 - p05),
            "min": float(values.min()), "max": float(values.max()),
            "units": "fraction of the lane half-width; +-1 is a lane boundary",
        },
        "offset_m_under_assumed_lane_width": {
            "assumed_lane_width_m": float(assumed_lane_width_m),
            "median": float(np.median(values) * assumed_lane_width_m / 2.0),
            "p05_p95_band_width": float((p95 - p05) * assumed_lane_width_m / 2.0),
            "caveat": ("scales linearly with the assumed lane width; the "
                       "normalised figures above do not depend on it"),
        },
        "time_in_zone": {
            "near_centre_fraction": float(near_centre.mean()),
            "near_left_boundary_fraction": float(near_left.mean()),
            "near_right_boundary_fraction": float(near_right.mean()),
            "near_centre_s": float(near_centre.sum() * sample_interval_s),
            "near_left_boundary_s": float(near_left.sum() * sample_interval_s),
            "near_right_boundary_s": float(near_right.sum() * sample_interval_s),
            "zone_definition": {
                "near_centre": f"|offset| <= {NEAR_CENTRE:.3f}",
                "near_boundary": f"|offset| >= {NEAR_BOUNDARY:.3f}",
            },
        },
        "crossing_candidates": {
            "count": len(candidates),
            "per_minute": (float(len(candidates) * 60.0 / observed_s)
                           if observed_s > 0 else None),
            "total_duration_beyond_boundary_s": float(sum(
                c["duration_beyond_boundary_s"] for c in candidates)),
            "median_duration_beyond_boundary_s": (
                float(np.median([c["duration_beyond_boundary_s"] for c in candidates]))
                if candidates else None),
            "max_duration_beyond_boundary_s": (
                float(max(c["duration_beyond_boundary_s"] for c in candidates))
                if candidates else None),
            "left": sum(1 for c in candidates if c["side"] == "left"),
            "right": sum(1 for c in candidates if c["side"] == "right"),
            "min_duration_s": float(min_crossing_duration_s),
            "state": "candidates_require_review",
            "is_traffic_violation_claim": False,
        },
        "unavailable_reasons": reasons,
        "note": PROXY_NOTE,
    }
