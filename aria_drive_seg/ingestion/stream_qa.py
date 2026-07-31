"""Per-recording stream quality assurance.

Produces, for every stream of a VRS: measured rate, jitter percentiles, gaps,
non-monotonic timestamps, continuous segments, drift, and synchronisation against
the RGB reference. The RGB stream additionally gets resolution, codec, duplicate
timestamp/image detection and image-quality statistics.

The nominal rate declared by the device is reported *next to* the measured rate and
never substituted for it.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np

from ..logging_utils import get_logger
from .rgb_scan import RgbScan, duplicate_report
from .timeline import (NS_PER_S, continuous_segments, find_gaps, measure_rate,
                       synchronisation_report)

log = get_logger("stream_qa")

# Streams we explicitly look for, so that an absent modality is reported as
# "checked and missing" rather than silently omitted.
EXPECTED_STREAMS = [
    "camera-rgb", "eyegaze", "camera-et-left", "camera-et-right", "handtracking",
    "slam-front-left", "slam-front-right", "slam-side-left", "slam-side-right",
    "imu-left", "imu-right", "mag0", "baro0", "gps-app", "ppg", "temperature",
    "als",
]


@dataclass
class StreamQa:
    stream_id: str
    label: str
    present: bool
    sample_count: int
    nominal_rate_hz: Optional[float]
    rate: Optional[Dict[str, Any]]
    gaps: List[Dict[str, Any]]
    gap_summary: Dict[str, Any]
    segments: List[Dict[str, Any]]
    drift: Optional[Dict[str, Any]]
    synchronisation: Optional[Dict[str, Any]]
    notes: List[str]

    # A bursty stream produces thousands of one-sample "segments" and thousands of
    # inter-burst intervals. The summary statistics stay exact; only the enumerations
    # are capped, and the full gap list is always written to the CSV.
    MAX_LISTED = 200

    def to_dict(self) -> Dict[str, Any]:
        segments = sorted(self.segments, key=lambda s: -s["duration_s"])[:self.MAX_LISTED]
        return {
            "stream_id": self.stream_id, "label": self.label, "present": self.present,
            "sample_count": self.sample_count,
            "nominal_rate_hz": self.nominal_rate_hz,
            "rate": self.rate, "gap_summary": self.gap_summary,
            "gaps": self.gaps[:self.MAX_LISTED],
            "gaps_total": len(self.gaps),
            "gaps_truncated": len(self.gaps) > self.MAX_LISTED,
            "continuous_segment_count": len(self.segments),
            "continuous_segments": segments,
            "continuous_segments_truncated": len(self.segments) > self.MAX_LISTED,
            "continuous_segments_note": ("listed longest first and capped; "
                                         "continuous_segment_count is exact"),
            "longest_segment_s": max((s["duration_s"] for s in self.segments),
                                     default=0.0),
            "drift": self.drift,
            "synchronisation_vs_rgb": self.synchronisation,
            "notes": self.notes,
        }


def _drift(timestamps_ns: np.ndarray) -> Optional[Dict[str, Any]]:
    """Compare the ideal constant-rate clock with the observed timestamps.

    A large residual means the stream did not keep a constant cadence over the
    recording (thermal throttling, pauses); a small one means jitter only.
    """
    ts = np.asarray(timestamps_ns, dtype=np.int64)
    if ts.size < 3:
        return None
    n = np.arange(ts.size, dtype=np.float64)
    t = (ts - ts[0]).astype(np.float64)
    slope, intercept = np.polyfit(n, t, 1)
    residual = t - (slope * n + intercept)
    return {
        "fitted_period_ms": float(slope / 1e6),
        "fitted_rate_hz": float(NS_PER_S / slope) if slope > 0 else None,
        "max_abs_residual_ms": float(np.max(np.abs(residual)) / 1e6),
        "final_residual_ms": float(residual[-1] / 1e6),
        "residual_std_ms": float(np.std(residual) / 1e6),
    }


def _gap_summary(gaps: List, median_dt_ns: Optional[float]) -> Dict[str, Any]:
    over15 = sum(1 for g in gaps if g.over_1p5_periods)
    over2 = sum(1 for g in gaps if g.over_2_periods)
    over250 = sum(1 for g in gaps if g.over_250ms)
    nonmono = sum(1 for g in gaps if g.non_monotonic)
    missing = sum(g.missing_estimate for g in gaps)
    longest = max((g.dt_ms for g in gaps), default=0.0)
    return {
        "total_flagged_intervals": len(gaps),
        "over_1p5_periods": over15,
        "over_2_periods": over2,
        "over_250ms": over250,
        "non_monotonic": nonmono,
        "estimated_missing_samples": missing,
        "longest_interval_ms": longest,
        "median_period_ms": (median_dt_ns / 1e6) if median_dt_ns else None,
    }


def qa_stream(label: str, stream_id: str, timestamps_ns: np.ndarray,
              nominal_rate_hz: Optional[float],
              rgb_timestamps_ns: Optional[np.ndarray],
              gap_absolute_ms: float = 250.0) -> StreamQa:
    """QA a single stream from its timestamps."""
    rate = measure_rate(timestamps_ns)
    med_ns = (rate.median_dt_ms * 1e6) if rate.median_dt_ms else None
    gaps = find_gaps(timestamps_ns, med_ns, gap_absolute_ms)
    segments = continuous_segments(timestamps_ns, 2.0, gap_absolute_ms, med_ns)

    notes: List[str] = []
    # A stream that delivers several samples in a burst and then waits has a median
    # period far shorter than its average one. Reporting the median-based rate alone
    # would claim a kilohertz sensor where the device actually samples once a second.
    bursty = False
    if rate.effective_fps and rate.effective_fps_span and rate.effective_fps_span > 0:
        ratio = rate.effective_fps / rate.effective_fps_span
        if ratio > 2.0:
            bursty = True
            notes.append(
                f"bursty stream: the median-interval rate ({rate.effective_fps:.1f} Hz) "
                f"is {ratio:.0f}x the span rate ({rate.effective_fps_span:.2f} Hz); "
                "samples arrive in bursts, so the span rate describes its real "
                "throughput and the gaps below are the inter-burst waits, not losses")

    if nominal_rate_hz and rate.effective_fps and not bursty:
        rel = abs(rate.effective_fps - nominal_rate_hz) / nominal_rate_hz
        if rel > 0.05:
            notes.append(
                f"measured rate {rate.effective_fps:.3f} Hz differs from the declared "
                f"nominal {nominal_rate_hz:.3f} Hz by {rel:.1%}; the measured value "
                "is authoritative")
    if rate.non_monotonic_count:
        notes.append(f"{rate.non_monotonic_count} non-monotonic intervals")
    if len(segments) > 1 and not bursty:
        notes.append(f"stream splits into {len(segments)} continuous segments")

    rate_doc = rate.to_dict()
    rate_doc["bursty"] = bursty
    rate_doc["representative_rate_hz"] = (rate.effective_fps_span if bursty
                                          else rate.effective_fps)
    return StreamQa(
        stream_id=stream_id, label=label, present=True,
        sample_count=int(rate.count), nominal_rate_hz=nominal_rate_hz,
        rate=rate_doc, gaps=[g.to_dict() for g in gaps],
        gap_summary=_gap_summary(gaps, med_ns),
        segments=[s.to_dict() for s in segments],
        drift=_drift(timestamps_ns),
        synchronisation=(synchronisation_report(rgb_timestamps_ns, timestamps_ns)
                         if rgb_timestamps_ns is not None else None),
        notes=notes,
    )


def _nominal_rate(provider, sid, label: str) -> Optional[float]:
    """Best-effort read of the rate the device declared for a stream."""
    dp = provider._dp
    for getter in ("get_image_configuration", "get_imu_configuration",
                   "get_gps_configuration", "get_hand_pose_configuration",
                   "get_eye_gaze_configuration", "get_magnetometer_configuration",
                   "get_barometer_configuration", "get_ppg_configuration",
                   "get_als_configuration", "get_temperature_configuration"):
        fn = getattr(dp, getter, None)
        if fn is None:
            continue
        try:
            cfg = fn(sid)
        except Exception:
            continue
        for attr in ("nominal_rate_hz", "nominalRateHz"):
            v = getattr(cfg, attr, None)
            if v:
                return float(v)
    return None


def run_stream_qa(provider, recording_id: str, domain: str,
                  source_path: str, source_sha256: str,
                  rgb_scan: Optional[RgbScan] = None,
                  rgb_label: str = "camera-rgb") -> Dict[str, Any]:
    """Full QA of every stream in a recording."""
    rgb_ts = provider.rgb_timestamps_ns(rgb_label) if provider.has_label(rgb_label) else None

    streams: Dict[str, Any] = {}
    seen_labels = set()
    for info in provider.list_streams():
        label = str(info.label)
        seen_labels.add(label)
        try:
            ts = provider.timestamps_ns(label)
        except Exception as exc:  # a stream can exist but be unreadable
            streams[label] = {"stream_id": info.stream_id, "label": label,
                              "present": True, "readable": False, "error": str(exc),
                              "sample_count": int(info.num_data)}
            continue
        sid = provider.stream_id(label)
        qa = qa_stream(label, info.stream_id, ts, _nominal_rate(provider, sid, label),
                       rgb_ts if label != rgb_label else None)
        streams[label] = qa.to_dict()

    for label in EXPECTED_STREAMS:
        if label not in seen_labels:
            streams[label] = {"label": label, "present": False,
                              "note": "checked and not present in this recording"}

    doc: Dict[str, Any] = {
        "recording_id": recording_id,
        "domain": domain,
        "source_file": source_path,
        "source_file_sha256": source_sha256,
        "metadata": _metadata(provider),
        "streams": streams,
    }

    if rgb_ts is not None:
        rgb_rate = measure_rate(rgb_ts)
        cfg = provider.rgb_config(rgb_label)
        rgb_doc: Dict[str, Any] = {
            "label": rgb_label,
            "width": cfg["width"], "height": cfg["height"],
            "pixel_format": cfg["pixel_format"],
            "codec": _codec(provider, rgb_label),
            "frame_count": int(rgb_rate.count),
            "rate": rgb_rate.to_dict(),
        }
        if rgb_scan is not None:
            rgb_doc["duplicates"] = duplicate_report(rgb_scan)
            rgb_doc["image_quality"] = _image_quality(rgb_scan)
        doc["rgb"] = rgb_doc

    doc["summary"] = _summary(doc)
    return doc


def _metadata(provider) -> Dict[str, Any]:
    try:
        md = provider._dp.get_metadata()
    except Exception:
        return {}
    out = {}
    for attr in ("device_serial", "device_id", "recording_profile",
                 "shared_session_id", "start_time_epoch_sec",
                 "end_time_epoch_sec", "duration_sec", "filename"):
        v = getattr(md, attr, None)
        if v is not None:
            out[attr] = v if isinstance(v, (int, float)) else str(v)
    out["time_sync_mode"] = str(getattr(md, "time_sync_mode", ""))
    return out


def _codec(provider, label: str) -> Dict[str, Any]:
    """Describe the image encoding.

    The provider does not expose the video codec on the image configuration, so the
    sensor and pixel format are reported and the codec is explicitly marked as not
    exposed rather than guessed.
    """
    out: Dict[str, Any] = {"codec": None,
                           "codec_source": "not exposed by the provider API"}
    try:
        cfg = provider._dp.get_image_configuration(provider.stream_id(label))
    except Exception:
        return out
    for attr in ("codec_name", "codecName"):
        v = getattr(cfg, attr, None)
        if v:
            out["codec"] = str(v)
            out["codec_source"] = "image configuration"
    for attr in ("sensor_model", "sensor_serial", "image_stride", "pixel_format",
                 "gamma_factor", "exposure_duration_min", "exposure_duration_max"):
        v = getattr(cfg, attr, None)
        if v is None:
            continue
        if isinstance(v, float) and not np.isfinite(v):
            # keep the JSON strictly parseable: NaN is not valid JSON
            out[attr] = None
        else:
            out[attr] = v if isinstance(v, (int, float, str)) else str(v)
    return out


def _image_quality(scan: RgbScan) -> Dict[str, Any]:
    def stats(a: np.ndarray) -> Dict[str, float]:
        return {"mean": float(np.mean(a)), "median": float(np.median(a)),
                "p05": float(np.percentile(a, 5)), "p95": float(np.percentile(a, 95)),
                "min": float(np.min(a)), "max": float(np.max(a))}

    return {
        "mean_luminance": stats(scan.mean_luminance),
        "blur_variance": stats(scan.blur_variance),
        "frame_difference": stats(scan.frame_difference[1:]) if scan.frame_difference.size > 1 else {},
        "very_dark_frames": int(np.sum(scan.mean_luminance < 0.08)),
        "very_bright_frames": int(np.sum(scan.mean_luminance > 0.92)),
        "clipped_highlight_frames": int(np.sum(scan.bright_fraction > 0.25)),
        "low_sharpness_frames": int(np.sum(
            scan.blur_variance < 0.25 * float(np.median(scan.blur_variance)))),
    }


def _summary(doc: Dict[str, Any]) -> Dict[str, Any]:
    rgb = doc.get("rgb")
    streams = doc.get("streams", {})
    present = sorted(k for k, v in streams.items() if v.get("present"))
    missing = sorted(k for k, v in streams.items() if not v.get("present"))

    out: Dict[str, Any] = {
        "streams_present": present,
        "streams_checked_and_missing": missing,
        "stream_count_present": len(present),
    }
    if rgb:
        rate = rgb["rate"]
        rgb_qa = streams.get(rgb["label"], {})
        segs = rgb_qa.get("continuous_segments", [])
        out.update({
            "rgb_frame_count": rgb["frame_count"],
            "rgb_effective_fps": rate["effective_fps"],
            "rgb_effective_fps_span": rate["effective_fps_span"],
            "rgb_duration_s": rate["duration_s"],
            "rgb_resolution": [rgb["width"], rgb["height"]],
            "rgb_monotonic": rate["non_monotonic_count"] == 0,
            "rgb_gap_summary": rgb_qa.get("gap_summary", {}),
            "rgb_continuous_segment_count": rgb_qa.get(
                "continuous_segment_count", len(segs)),
            "rgb_longest_segment_s": rgb_qa.get(
                "longest_segment_s",
                max((s["duration_s"] for s in segs), default=0.0)),
        })
    return out


def gaps_to_rows(doc: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Flatten every stream's gap list into CSV rows."""
    rows: List[Dict[str, Any]] = []
    for label, s in doc.get("streams", {}).items():
        for g in s.get("gaps", []):
            rows.append({
                "recording_id": doc["recording_id"],
                "domain": doc["domain"],
                "stream_label": label,
                "stream_id": s.get("stream_id"),
                **g,
            })
    return rows
