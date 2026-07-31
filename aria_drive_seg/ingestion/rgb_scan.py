"""One sequential decode pass over an RGB stream producing per-frame QA features.

Decoding a full Aria RGB stream is the expensive part of ingestion, so it is done
exactly once and every downstream stage (duplicate detection, image-quality QA,
segment ranking, ego-structure evidence, hand-region motion) reads the cached
features instead of decoding again.

Nothing here resamples, interpolates or reorders: features are stored against the
original `frame_index` and `timestamp_ns` of the source stream.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from ..logging_utils import get_logger

log = get_logger("rgb_scan")

THUMB_W, THUMB_H = 128, 96


@dataclass
class RgbScan:
    """Per-frame features of an RGB stream, indexed by position in the stream."""

    recording_id: str
    source_file_sha256: str
    source_stream_id: str
    frame_index: np.ndarray        # int64, original index in the VRS stream
    timestamp_ns: np.ndarray       # int64, original capture timestamp
    width: int
    height: int
    mean_luminance: np.ndarray     # float32, 0-1
    std_luminance: np.ndarray      # float32, 0-1
    blur_variance: np.ndarray      # float32, variance of Laplacian (higher = sharper)
    dark_fraction: np.ndarray      # float32, pixels below 16/255
    bright_fraction: np.ndarray    # float32, pixels above 240/255
    dhash: np.ndarray              # uint64, perceptual hash for duplicate detection
    frame_difference: np.ndarray   # float32, mean abs diff vs previous frame, 0-1
    thumbnails: np.ndarray         # uint8 (n, THUMB_H, THUMB_W) grayscale

    def to_meta(self) -> Dict[str, Any]:
        return {
            "recording_id": self.recording_id,
            "source_file_sha256": self.source_file_sha256,
            "source_stream_id": self.source_stream_id,
            "count": int(self.frame_index.size),
            "width": self.width,
            "height": self.height,
            "thumbnail_size": [THUMB_W, THUMB_H],
        }

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp.npz")
        np.savez_compressed(
            tmp,
            recording_id=np.array(self.recording_id),
            source_file_sha256=np.array(self.source_file_sha256),
            source_stream_id=np.array(self.source_stream_id),
            frame_index=self.frame_index,
            timestamp_ns=self.timestamp_ns,
            size=np.array([self.width, self.height], dtype=np.int64),
            mean_luminance=self.mean_luminance,
            std_luminance=self.std_luminance,
            blur_variance=self.blur_variance,
            dark_fraction=self.dark_fraction,
            bright_fraction=self.bright_fraction,
            dhash=self.dhash,
            frame_difference=self.frame_difference,
            thumbnails=self.thumbnails,
        )
        tmp.replace(path)

    @classmethod
    def load(cls, path: str | Path) -> "RgbScan":
        d = np.load(path, allow_pickle=False)
        size = d["size"]
        return cls(
            recording_id=str(d["recording_id"]),
            source_file_sha256=str(d["source_file_sha256"]),
            source_stream_id=str(d["source_stream_id"]),
            frame_index=d["frame_index"],
            timestamp_ns=d["timestamp_ns"],
            width=int(size[0]), height=int(size[1]),
            mean_luminance=d["mean_luminance"],
            std_luminance=d["std_luminance"],
            blur_variance=d["blur_variance"],
            dark_fraction=d["dark_fraction"],
            bright_fraction=d["bright_fraction"],
            dhash=d["dhash"],
            frame_difference=d["frame_difference"],
            thumbnails=d["thumbnails"],
        )


def _dhash(gray_small: np.ndarray) -> np.uint64:
    """64-bit difference hash of a 9x8 downscale (classic dHash)."""
    import cv2
    g = cv2.resize(gray_small, (9, 8), interpolation=cv2.INTER_AREA)
    bits = (g[:, 1:] > g[:, :-1]).ravel()
    out = np.uint64(0)
    for b in bits:
        out = np.uint64(out << np.uint64(1)) | np.uint64(1 if b else 0)
    return out


def hamming(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Bitwise Hamming distance between two arrays of uint64 hashes."""
    x = np.bitwise_xor(a.astype(np.uint64), b.astype(np.uint64))
    counts = np.zeros(x.shape, dtype=np.int32)
    for _ in range(64):
        counts += (x & np.uint64(1)).astype(np.int32)
        x = x >> np.uint64(1)
    return counts


def scan_rgb_stream(provider, recording_id: str, source_sha256: str,
                    rgb_label: str = "camera-rgb",
                    progress_every: int = 500) -> RgbScan:
    """Decode every RGB frame once and compute the per-frame feature set."""
    import cv2

    sid = provider.rgb_stream(rgb_label)
    timestamps = provider.rgb_timestamps_ns(rgb_label)
    n = int(timestamps.size)
    cfg = provider.rgb_config(rgb_label)

    mean_l = np.zeros(n, dtype=np.float32)
    std_l = np.zeros(n, dtype=np.float32)
    blur = np.zeros(n, dtype=np.float32)
    dark = np.zeros(n, dtype=np.float32)
    bright = np.zeros(n, dtype=np.float32)
    dh = np.zeros(n, dtype=np.uint64)
    diff = np.zeros(n, dtype=np.float32)
    thumbs = np.zeros((n, THUMB_H, THUMB_W), dtype=np.uint8)
    actual_ts = np.zeros(n, dtype=np.int64)

    previous: Optional[np.ndarray] = None
    log.info("scanning %d RGB frames of %s", n, recording_id)
    for i in range(n):
        raw, ts = provider.rgb_by_index(i, rgb_label)
        actual_ts[i] = ts
        gray = cv2.cvtColor(raw, cv2.COLOR_RGB2GRAY)
        small = cv2.resize(gray, (THUMB_W, THUMB_H), interpolation=cv2.INTER_AREA)
        thumbs[i] = small
        f = small.astype(np.float32)
        mean_l[i] = f.mean() / 255.0
        std_l[i] = f.std() / 255.0
        blur[i] = float(cv2.Laplacian(small, cv2.CV_32F).var())
        dark[i] = float((gray < 16).mean())
        bright[i] = float((gray > 240).mean())
        dh[i] = _dhash(small)
        if previous is not None:
            diff[i] = float(np.abs(f - previous).mean() / 255.0)
        previous = f
        if progress_every and (i + 1) % progress_every == 0:
            log.info("  %d/%d frames", i + 1, n)

    return RgbScan(
        recording_id=recording_id,
        source_file_sha256=source_sha256,
        source_stream_id=str(sid),
        frame_index=np.arange(n, dtype=np.int64),
        timestamp_ns=actual_ts,
        width=int(cfg["width"]), height=int(cfg["height"]),
        mean_luminance=mean_l, std_luminance=std_l, blur_variance=blur,
        dark_fraction=dark, bright_fraction=bright, dhash=dh,
        frame_difference=diff, thumbnails=thumbs,
    )


def duplicate_report(scan: RgbScan, near_identical_hamming: int = 2,
                     duplicate_pixel_difference: float = 0.002) -> Dict[str, Any]:
    """Distinguish genuinely repeated frames from merely similar consecutive ones.

    A perceptual-hash distance of 0-2 between consecutive frames is completely
    normal in a video: at 15 fps a slow scene barely changes in 67 ms. Calling those
    "duplicates" would report roughly half of any driving recording as broken.

    A **duplicate** therefore requires both an identical hash *and* a near-zero mean
    pixel difference, which is what a genuinely repeated or frozen frame looks like.
    The looser hash-only count is still reported, as a low-scene-change statistic
    rather than a defect.
    """
    ts = scan.timestamp_ns
    dup_ts = int(np.sum(np.diff(ts) == 0)) if ts.size > 1 else 0

    if scan.dhash.size < 2:
        return {"duplicate_timestamps": dup_ts, "duplicate_images": 0,
                "duplicate_image_pairs": []}

    dist = hamming(scan.dhash[1:], scan.dhash[:-1])
    diff = scan.frame_difference[1:]
    duplicate = (dist == 0) & (diff <= duplicate_pixel_difference)
    idx = np.flatnonzero(duplicate)
    pairs = [
        {
            "first_frame_index": int(scan.frame_index[i]),
            "second_frame_index": int(scan.frame_index[i + 1]),
            "first_timestamp_ns": int(ts[i]),
            "second_timestamp_ns": int(ts[i + 1]),
            "hamming_distance": int(dist[i]),
            "frame_difference": float(diff[i]),
        }
        for i in idx[:200]
    ]
    return {
        "duplicate_timestamps": dup_ts,
        "duplicate_images": int(idx.size),
        "duplicate_definition": (
            f"identical perceptual hash AND mean pixel difference <= "
            f"{duplicate_pixel_difference}"),
        "duplicate_image_pairs": pairs,
        "duplicate_image_pairs_truncated": bool(idx.size > len(pairs)),
        "near_identical_consecutive_pairs": int(np.sum(dist <= near_identical_hamming)),
        "near_identical_hamming_threshold": near_identical_hamming,
        "near_identical_note": (
            "consecutive frames of a video are expected to be perceptually similar; "
            "this is a scene-change statistic, not a duplication defect"),
    }


def sample_frames_for_domain(provider, count: int = 12,
                             rgb_label: str = "camera-rgb") -> List[np.ndarray]:
    """Sample frames spread across the whole recording by timestamp fraction.

    Sampling is done on the timestamp span rather than on an index step so that the
    spread is identical for recordings with different sampling rates.
    """
    ts = provider.rgb_timestamps_ns(rgb_label)
    if ts.size == 0:
        return []
    targets = np.linspace(float(ts[0]), float(ts[-1]), num=count)
    indices = sorted({int(np.argmin(np.abs(ts - t))) for t in targets})
    return [provider.rgb_by_index(i, rgb_label)[0] for i in indices]
