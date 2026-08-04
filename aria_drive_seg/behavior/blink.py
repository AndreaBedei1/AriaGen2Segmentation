"""Blink events and blink metrics from the per-sample eye state.

Three decisions in this module are worth reading before any number it produces.

**The blink flag's polarity is measured, not assumed.** In both pilot recordings
the record's `blink` field is `True` for ~89–93% of samples, in runs whose median
length is over a second, while `False` occurs in runs whose median length is
~130 ms and whose maximum is ~430 ms. A person does not spend 90% of a drive with
their eyes shut, and 130 ms *is* a blink; the short-run polarity is the closed
one. `determine_blink_polarity` establishes that from the run-length distribution
of each recording and records the evidence. If the two polarities are not clearly
separated it returns `undetermined` and refuses to emit events, rather than
picking one.

**Nothing is interpolated.** A blink is a run of consecutive *real* samples. If
the stream skips, the run ends: `max_gap_s` splits an event rather than bridging
it, because bridging would invent eye state that was never recorded.

**Eye-closure fraction is a measurement, not a diagnosis.** It is the fraction of
valid samples in the closed state. It is not drowsiness, not PERCLOS validated
against a reference, and not a fitness-to-drive statement. Nothing downstream may
label it as one.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

NS_PER_S = 1_000_000_000

#: The five states an event may take, exactly as the analysis brief defines them.
EVENT_CATEGORIES = ("bilateral_blink", "left_only", "right_only", "uncertain",
                    "invalid_tracking_gap")

#: Categories counted as an identified blink when a rate is computed.
BLINK_CATEGORIES = ("bilateral_blink", "left_only", "right_only")

#: A closure longer than this is not a blink. It is kept as an event and reported,
#: but under `uncertain`, because calling a two-second closure a blink would put a
#: physiologically different state into a blink-rate denominator.
MAX_BLINK_DURATION_S = 1.0

#: Consecutive samples further apart than this do not belong to the same event.
#: At 30 Hz the nominal spacing is 33.3 ms, so this tolerates one dropped sample
#: and no more.
DEFAULT_MAX_GAP_S = 0.075

CLOSURE_INTERPRETATION_NOTE = (
    "eye-closure fraction is the measured fraction of valid samples in the closed "
    "state. It is an ocular measurement, not a drowsiness score, not a validated "
    "PERCLOS, and not a clinical or fitness-to-drive statement.")


# --------------------------------------------------------------------------- #
# Polarity
# --------------------------------------------------------------------------- #
def _runs(flag: np.ndarray) -> List[Tuple[int, int]]:
    """Half-open index ranges of maximal True runs."""
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


def _run_durations_s(flag: np.ndarray, timestamp_ns: np.ndarray) -> np.ndarray:
    ts = np.asarray(timestamp_ns, np.int64)
    out = []
    for a, b in _runs(flag):
        # A single-sample run still occupies one sampling interval; using the
        # difference alone would report it as zero-length.
        if b - a >= 2:
            out.append(float((ts[b - 1] - ts[a]) / NS_PER_S))
        elif ts.size > 1:
            out.append(float(np.median(np.diff(ts)) / NS_PER_S))
    return np.asarray(out, float)


def determine_blink_polarity(blink: Sequence[bool], valid: Sequence[bool],
                             timestamp_ns: Sequence[int],
                             max_closed_fraction: float = 0.35,
                             ) -> Dict[str, Any]:
    """Decide which value of the raw `blink` flag means *eye closed*, from the data.

    A blink is short and rare; an open eye is long and common. The polarity whose
    runs are the shorter **and** whose share of samples is the smaller is the
    closed state. Both conditions must agree: either alone could be satisfied by a
    signal that is not a blink flag at all.
    """
    flag = np.asarray(blink, bool)
    ok = np.asarray(valid, bool)
    ts = np.asarray(timestamp_ns, np.int64)
    evidence: Dict[str, Any] = {"valid_samples": int(ok.sum())}
    if ok.sum() < 2:
        return {**evidence, "closed_value": None, "decided": False,
                "reason": "fewer than two valid blink samples"}

    stats = {}
    for value in (True, False):
        state = (flag == value) & ok
        durations = _run_durations_s(state, ts)
        stats[value] = {
            "sample_fraction": float(state.sum() / ok.sum()),
            "runs": int(durations.size),
            "median_run_s": float(np.median(durations)) if durations.size else None,
            "p95_run_s": (float(np.percentile(durations, 95))
                          if durations.size else None),
            "max_run_s": float(durations.max()) if durations.size else None,
        }
    evidence["run_statistics"] = {str(k): v for k, v in stats.items()}

    candidates = [v for v in (True, False)
                  if stats[v]["median_run_s"] is not None
                  and stats[not v]["median_run_s"] is not None
                  and stats[v]["median_run_s"] < stats[not v]["median_run_s"]
                  and stats[v]["sample_fraction"] < stats[not v]["sample_fraction"]
                  and stats[v]["sample_fraction"] <= max_closed_fraction]
    if len(candidates) != 1:
        return {**evidence, "closed_value": None, "decided": False,
                "reason": ("neither value of the blink flag is both rarer and "
                           "shorter-running than the other, so which one means "
                           "'eye closed' cannot be established from this "
                           "recording")}
    closed = candidates[0]
    return {
        **evidence,
        "closed_value": bool(closed),
        "decided": True,
        "method": "shorter_and_rarer_run_polarity",
        "max_closed_fraction": max_closed_fraction,
        "field_name_matches_polarity": bool(closed),
        "note": (
            "the raw field is named `blink`, but in this recording the value that "
            f"behaves like an eye closure is {closed}. The polarity is established "
            "from the run-length distribution rather than from the field name, and "
            "the raw values are stored unchanged."),
    }


def closed_series(frame, polarity: Dict[str, Any]) -> Dict[str, np.ndarray]:
    """Per-eye closed/known boolean series under a decided polarity."""
    if not polarity.get("decided"):
        raise ValueError("blink polarity is undetermined; refusing to derive "
                         "closure without it")
    closed_value = bool(polarity["closed_value"])
    out: Dict[str, np.ndarray] = {}
    for side in ("left", "right"):
        known = frame[f"{side}_blink_valid"].to_numpy(bool)
        raw = frame[f"{side}_blink"].to_numpy(bool)
        out[f"{side}_known"] = known
        out[f"{side}_closed"] = known & (raw == closed_value)
    return out


# --------------------------------------------------------------------------- #
# Events
# --------------------------------------------------------------------------- #
@dataclass
class BlinkEvent:
    event_index: int
    start_ns: int
    end_ns: int
    duration_s: float
    samples: int
    category: str
    reason: Optional[str]
    left_closed_samples: int
    right_closed_samples: int
    both_closed_samples: int
    invalid_samples: int
    gaze_invalid_samples: int

    def to_dict(self) -> Dict[str, Any]:
        return {
            "event_index": self.event_index, "start_ns": self.start_ns,
            "end_ns": self.end_ns, "duration_s": self.duration_s,
            "samples": self.samples, "category": self.category,
            "reason": self.reason,
            "left_closed_samples": self.left_closed_samples,
            "right_closed_samples": self.right_closed_samples,
            "both_closed_samples": self.both_closed_samples,
            "invalid_samples": self.invalid_samples,
            "gaze_invalid_samples": self.gaze_invalid_samples,
            "is_identified_blink": self.category in BLINK_CATEGORIES,
        }


def build_blink_events(frame, polarity: Dict[str, Any],
                       max_gap_s: float = DEFAULT_MAX_GAP_S,
                       max_blink_duration_s: float = MAX_BLINK_DURATION_S,
                       ) -> List[BlinkEvent]:
    """Group consecutive non-open samples into events and classify each one.

    An event is a maximal run of consecutive real samples that are not in the
    open state — either at least one eye closed, or both eyes' blink flags
    invalid. A sampling gap larger than `max_gap_s` ends the run, because the
    state during the gap was not recorded and this module never invents it.
    """
    series = closed_series(frame, polarity)
    left_closed, right_closed = series["left_closed"], series["right_closed"]
    left_known, right_known = series["left_known"], series["right_known"]
    ts = frame["timestamp_ns"].to_numpy(np.int64)
    gaze_valid = (frame["combined_gaze_direction_valid"].to_numpy(bool)
                  if "combined_gaze_direction_valid" in frame
                  else np.ones(ts.size, bool))

    unknown = (~left_known) & (~right_known)
    active = left_closed | right_closed | unknown
    interval_s = float(np.median(np.diff(ts)) / NS_PER_S) if ts.size > 1 else 0.0

    events: List[BlinkEvent] = []
    for a, b in _runs(active):
        # Split on real sampling gaps rather than bridging them.
        segment_start = a
        for i in range(a + 1, b + 1):
            gap_break = (i < b and
                         float((ts[i] - ts[i - 1]) / NS_PER_S) > max_gap_s)
            if i == b or gap_break:
                events.append(_classify(
                    len(events), segment_start, i, ts, interval_s,
                    left_closed, right_closed, unknown, gaze_valid,
                    max_blink_duration_s))
                segment_start = i
    return events


def _classify(index: int, a: int, b: int, ts: np.ndarray, interval_s: float,
              left_closed: np.ndarray, right_closed: np.ndarray,
              unknown: np.ndarray, gaze_valid: np.ndarray,
              max_blink_duration_s: float) -> BlinkEvent:
    n = b - a
    duration_s = (float((ts[b - 1] - ts[a]) / NS_PER_S) + interval_s
                  if n >= 1 else 0.0)
    left = int(left_closed[a:b].sum())
    right = int(right_closed[a:b].sum())
    both = int((left_closed[a:b] & right_closed[a:b]).sum())
    invalid = int(unknown[a:b].sum())
    reason: Optional[str] = None

    if invalid:
        category = "invalid_tracking_gap"
        reason = (f"{invalid} of {n} samples have both blink flags marked "
                  "invalid, so eye state is unknown across this run")
    elif both:
        category = "bilateral_blink"
    elif left and not right:
        category = "left_only"
    elif right and not left:
        category = "right_only"
    else:
        category = "uncertain"
        reason = ("both eyes close within the run but never on the same sample; "
                  "this is not a clean bilateral blink and not a clean "
                  "monocular one")

    if category in BLINK_CATEGORIES and duration_s > max_blink_duration_s:
        reason = (f"closure lasts {duration_s:.2f} s, longer than the "
                  f"{max_blink_duration_s:.2f} s a blink occupies; reported as a "
                  "prolonged closure rather than counted as a blink")
        category = "uncertain"

    return BlinkEvent(
        event_index=index, start_ns=int(ts[a]), end_ns=int(ts[b - 1]),
        duration_s=duration_s, samples=n, category=category, reason=reason,
        left_closed_samples=left, right_closed_samples=right,
        both_closed_samples=both, invalid_samples=invalid,
        gaze_invalid_samples=int((~gaze_valid[a:b]).sum()))


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #
def _describe_durations(values: Sequence[float]) -> Dict[str, Optional[float]]:
    v = np.asarray(values, float)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return {"count": 0, "mean_s": None, "median_s": None, "p95_s": None,
                "min_s": None, "max_s": None}
    return {"count": int(v.size), "mean_s": float(v.mean()),
            "median_s": float(np.median(v)),
            "p95_s": float(np.percentile(v, 95)),
            "min_s": float(v.min()), "max_s": float(v.max())}


def windowed_blink_rate(events: Sequence[BlinkEvent], start_ns: int, end_ns: int,
                        window_s: float = 60.0, step_s: float = 10.0
                        ) -> List[Dict[str, Any]]:
    """Blink rate in sliding windows defined in **seconds**, not in samples.

    A window is only reported when it fits entirely inside the recording, so a
    partial window at either end never produces a rate computed over less time
    than it claims.
    """
    if window_s <= 0 or step_s <= 0:
        raise ValueError("window_s and step_s must be positive")
    starts_ns = np.asarray([e.start_ns for e in events
                            if e.category in BLINK_CATEGORIES], np.int64)
    window_ns = int(round(window_s * NS_PER_S))
    step_ns = int(round(step_s * NS_PER_S))
    out: List[Dict[str, Any]] = []
    t = int(start_ns)
    while t + window_ns <= int(end_ns):
        count = int(np.sum((starts_ns >= t) & (starts_ns < t + window_ns)))
        out.append({
            "window_start_ns": t, "window_end_ns": t + window_ns,
            "window_s": float(window_s), "step_s": float(step_s),
            "rel_start_s": float((t - int(start_ns)) / NS_PER_S),
            "blinks": count,
            "blinks_per_minute": float(count * 60.0 / window_s),
        })
        t += step_ns
    return out


def blink_metrics(frame, events: Sequence[BlinkEvent], polarity: Dict[str, Any],
                  window_s: float = 60.0, step_s: float = 10.0) -> Dict[str, Any]:
    """Rate, duration, inter-blink interval, closure fraction and gaze coupling."""
    ts = frame["timestamp_ns"].to_numpy(np.int64)
    duration_s = float((ts[-1] - ts[0]) / NS_PER_S) if ts.size > 1 else 0.0
    interval_s = float(np.median(np.diff(ts)) / NS_PER_S) if ts.size > 1 else 0.0
    series = closed_series(frame, polarity)

    identified = [e for e in events if e.category in BLINK_CATEGORIES]
    by_category = {c: sum(1 for e in events if e.category == c)
                   for c in EVENT_CATEGORIES}
    starts = np.asarray([e.start_ns for e in identified], np.int64)
    ends = np.asarray([e.end_ns for e in identified], np.int64)
    # Inter-blink interval: onset to onset, the convention that does not shrink
    # when a blink happens to be long.
    ibi = (np.diff(starts) / NS_PER_S) if starts.size > 1 else np.zeros(0)
    # Eye closure is counted per eye on valid samples only; an unknown sample is
    # not evidence of an open eye.
    closure: Dict[str, Any] = {}
    for side in ("left", "right"):
        known = series[f"{side}_known"]
        closed = series[f"{side}_closed"]
        closure[side] = {
            "valid_samples": int(known.sum()),
            "closed_samples": int(closed.sum()),
            "eye_closure_fraction": (float(closed.sum() / known.sum())
                                     if known.sum() else None),
            "closed_time_s": float(closed.sum() * interval_s),
        }
    both_known = series["left_known"] & series["right_known"]
    both_closed = series["left_closed"] & series["right_closed"]

    gaze_valid = (frame["combined_gaze_direction_valid"].to_numpy(bool)
                  if "combined_gaze_direction_valid" in frame
                  else np.ones(ts.size, bool))
    any_closed = series["left_closed"] | series["right_closed"]
    invalid_gaze = ~gaze_valid

    return {
        "recording_duration_s": duration_s,
        "samples": int(ts.size),
        "sample_interval_s": interval_s,
        "polarity": polarity,
        "events_total": len(events),
        "events_by_category": by_category,
        "identified_blinks": len(identified),
        "blinks_per_minute_whole_recording": (
            float(len(identified) * 60.0 / duration_s) if duration_s > 0 else None),
        "blinks_per_minute_including_uncertain": (
            float((len(identified) + by_category["uncertain"]) * 60.0 / duration_s)
            if duration_s > 0 else None),
        "duration": _describe_durations([e.duration_s for e in identified]),
        "duration_by_category": {
            c: _describe_durations([e.duration_s for e in events if e.category == c])
            for c in EVENT_CATEGORIES},
        "inter_blink_interval_s": _describe_durations(ibi),
        "bilateral_share": (float(by_category["bilateral_blink"] / len(identified))
                            if identified else None),
        "monocular_share": (
            float((by_category["left_only"] + by_category["right_only"])
                  / len(identified)) if identified else None),
        "left_right_asymmetry": {
            "left_only_events": by_category["left_only"],
            "right_only_events": by_category["right_only"],
            "note": ("a monocular event is as likely to be a per-eye tracking "
                     "failure as a genuine one-eyed blink; it is reported, not "
                     "interpreted"),
        },
        "eye_closure": {
            **closure,
            "both_eyes": {
                "valid_samples": int(both_known.sum()),
                "closed_samples": int(both_closed.sum()),
                "eye_closure_fraction": (float(both_closed.sum() / both_known.sum())
                                         if both_known.sum() else None),
                "closed_time_s": float(both_closed.sum() * interval_s),
            },
            "interpretation": CLOSURE_INTERPRETATION_NOTE,
            "is_drowsiness_measure": False,
        },
        "gaze_invalidation": {
            "gaze_invalid_fraction": float(invalid_gaze.mean()) if ts.size else None,
            "gaze_invalid_fraction_during_closure": (
                float(invalid_gaze[any_closed].mean()) if any_closed.any() else None),
            "gaze_invalid_fraction_while_open": (
                float(invalid_gaze[~any_closed].mean()) if (~any_closed).any() else None),
            "invalid_gaze_samples_explained_by_closure": (
                float((invalid_gaze & any_closed).sum() / invalid_gaze.sum())
                if invalid_gaze.sum() else None),
            "note": ("gaze cannot be estimated through a closed eyelid, so gaze "
                     "invalidation during a blink is expected instrumentation "
                     "behaviour and not a separate finding"),
        },
        "windows": windowed_blink_rate(
            events, int(ts[0]), int(ts[-1]), window_s=window_s, step_s=step_s)
        if ts.size > 1 else [],
        "window_definition": {"window_s": window_s, "step_s": step_s,
                              "units": "seconds", "partial_windows_reported": False},
        "interpolated": False,
        "blink_interpolated": False,
        "medical_interpretation": "not attempted and not supportable",
    }


def blink_rate_in_window(onset_ns: Sequence[int], duration_s: Sequence[float],
                         start_ns: int, end_ns: int) -> Dict[str, Any]:
    """Blink statistics inside one arbitrary time window, for event analysis.

    Takes the onsets and durations of the **identified** blinks as arrays rather
    than the event objects, so the same function serves a caller holding a list of
    `BlinkEvent` and one holding the events table read back from parquet.
    """
    onsets = np.asarray(onset_ns, np.int64)
    durations = np.asarray(duration_s, float)
    window_s = float((int(end_ns) - int(start_ns)) / NS_PER_S)
    inside = (onsets >= int(start_ns)) & (onsets < int(end_ns))
    selected = durations[inside]
    return {
        "window_s": window_s,
        "blinks": int(inside.sum()),
        "blinks_per_minute": (float(int(inside.sum()) * 60.0 / window_s)
                              if window_s > 0 else None),
        "mean_blink_duration_s": (float(selected.mean()) if selected.size else None),
        "median_blink_duration_s": (float(np.median(selected))
                                    if selected.size else None),
    }


def identified_blink_arrays(events: Sequence[BlinkEvent]):
    """Onsets and durations of the identified blinks, ready for the window helper."""
    selected = [e for e in events if e.category in BLINK_CATEGORIES]
    return (np.asarray([e.start_ns for e in selected], np.int64),
            np.asarray([e.duration_s for e in selected], float))


def closure_fraction_in_window(frame, polarity: Dict[str, Any], start_ns: int,
                               end_ns: int) -> Optional[float]:
    """Fraction of valid samples with both eyes closed inside a time window."""
    ts = frame["timestamp_ns"].to_numpy(np.int64)
    sel = (ts >= int(start_ns)) & (ts < int(end_ns))
    if not sel.any():
        return None
    series = closed_series(frame, polarity)
    known = (series["left_known"] & series["right_known"])[sel]
    closed = (series["left_closed"] & series["right_closed"])[sel]
    return float(closed.sum() / known.sum()) if known.sum() else None
