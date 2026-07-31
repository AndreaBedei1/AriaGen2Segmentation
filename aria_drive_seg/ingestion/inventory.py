"""Acquisition inventory: discover candidate data files and resolve their domain.

Two hard rules are encoded here.

1. **Nothing is identified by file name.** The motorcycle recording is found by
   scanning the project tree, reading the VRS metadata and stream structure and
   looking at the actual pixels. The name is recorded as an attribute, never used
   as evidence.
2. **The domain is never inferred from the frame rate.** `classify_domain` receives
   a `DomainEvidence` object that structurally cannot carry a sampling rate. The
   car recording currently runs at 10 fps and the motorcycle at 15 fps, but that
   difference is a temporary acquisition artefact and must never become a feature.
"""
from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from ..hashing import sha256_file

# Extensions worth inventorying for this project.
DATA_SUFFIXES = {
    ".vrs": "vrs_recording",
    ".mp4": "video",
    ".mov": "video",
    ".csv": "tabular",
    ".json": "structured",
    ".jsonl": "structured",
    ".parquet": "tabular",
    ".npz": "array_bundle",
    ".gpx": "trajectory",
}

# Directories that hold derived pipeline products, not acquisitions.
DERIVED_DIR_NAMES = {
    "output", "outputs", "runs", "run200", "val10", "val300", "weights",
    "__pycache__", ".git", ".pytest_cache", "node_modules", ".venv",
}


# --------------------------------------------------------------------------- #
# Hash cache (large VRS files must not be re-hashed on every run)
# --------------------------------------------------------------------------- #
class HashCache:
    """Cache SHA-256 by (absolute path, size, mtime_ns).

    The cache is an optimisation only: any change to size or mtime forces a
    recompute, so a modified file can never keep a stale hash.
    """

    def __init__(self, path: Optional[str | Path] = None):
        self.path = Path(path) if path else None
        self._data: Dict[str, Dict[str, Any]] = {}
        if self.path and self.path.exists():
            try:
                self._data = json.loads(self.path.read_text())
            except Exception:
                self._data = {}

    def get(self, file_path: Path) -> str:
        st = file_path.stat()
        key = str(file_path.resolve())
        entry = self._data.get(key)
        if entry and entry.get("size") == st.st_size and entry.get("mtime_ns") == st.st_mtime_ns:
            return str(entry["sha256"])
        digest = sha256_file(file_path)
        self._data[key] = {"size": st.st_size, "mtime_ns": st.st_mtime_ns,
                           "sha256": digest}
        self.save()
        return digest

    def save(self) -> None:
        if not self.path:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self._data, indent=2))
        tmp.replace(self.path)


# --------------------------------------------------------------------------- #
# Domain evidence and classification
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class DomainEvidence:
    """Frame-rate-free visual evidence used to decide car vs motorcycle.

    Every field is a spatial or photometric property of the ego-structure. There is
    deliberately no sampling-rate, frame-count or duration field: the classifier
    cannot depend on the temporary 10 fps / 15 fps difference even by accident.
    """

    # fraction of pixels that barely change across the sampled frames
    static_fraction: float
    # same, restricted to the top / bottom quarter of the image
    top_band_static_fraction: float
    bottom_band_static_fraction: float
    # mean luminance (0-1) of the top quarter: sky is bright, a car roof is dark
    top_band_luminance: float
    # fraction of the image border occupied by static structure (cabin enclosure)
    border_static_fraction: float
    # how many sampled frames the evidence was pooled over (a sample-size
    # qualifier, not a rate: it does not identify the vehicle)
    sampled_frames: int

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class DomainDecision:
    domain: str                  # car | motorcycle | unknown
    confidence: float
    rationale: List[str]
    evidence: Optional[Dict[str, Any]] = None
    method: str = "static_ego_structure_v1"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# Decision thresholds. A car cabin encloses the camera: the roof makes the top band
# dark and static and the pillars make the border static. A motorcycle leaves the
# top band as open sky and keeps only a small static handlebar/mirror region low in
# the frame.
CAR_TOP_STATIC_MIN = 0.55
CAR_TOP_LUMINANCE_MAX = 0.25
MOTO_TOP_STATIC_MAX = 0.35
MOTO_TOP_LUMINANCE_MIN = 0.30


def classify_domain(evidence: DomainEvidence) -> DomainDecision:
    """Decide car vs motorcycle from ego-structure evidence alone."""
    reasons: List[str] = []
    car_score = 0.0
    moto_score = 0.0

    if evidence.top_band_static_fraction >= CAR_TOP_STATIC_MIN:
        car_score += 1.0
        reasons.append(
            f"top band is {evidence.top_band_static_fraction:.2%} static: "
            "consistent with a roof above the camera")
    elif evidence.top_band_static_fraction <= MOTO_TOP_STATIC_MAX:
        moto_score += 1.0
        reasons.append(
            f"top band is only {evidence.top_band_static_fraction:.2%} static: "
            "consistent with open sky above the rider")

    if evidence.top_band_luminance <= CAR_TOP_LUMINANCE_MAX:
        car_score += 0.7
        reasons.append(
            f"top band luminance {evidence.top_band_luminance:.2f} is dark: "
            "consistent with an interior headliner")
    elif evidence.top_band_luminance >= MOTO_TOP_LUMINANCE_MIN:
        moto_score += 0.7
        reasons.append(
            f"top band luminance {evidence.top_band_luminance:.2f} is bright: "
            "consistent with sky")

    if evidence.border_static_fraction >= 0.55:
        car_score += 0.8
        reasons.append(
            f"{evidence.border_static_fraction:.2%} of the image border is static: "
            "consistent with an enclosing cabin (pillars, doors, roof)")
    elif evidence.border_static_fraction <= 0.40:
        moto_score += 0.8
        reasons.append(
            f"only {evidence.border_static_fraction:.2%} of the image border is "
            "static: the camera is not enclosed by a cabin")

    if evidence.bottom_band_static_fraction >= 0.35:
        reasons.append(
            f"bottom band is {evidence.bottom_band_static_fraction:.2%} static: "
            "ego control structure is present in the lower frame (both domains)")

    total = car_score + moto_score
    if total <= 0 or evidence.sampled_frames < 2:
        return DomainDecision("unknown", 0.0,
                              reasons + ["insufficient evidence to separate the domains"],
                              evidence.to_dict())
    if car_score > moto_score:
        return DomainDecision("car", car_score / total, reasons, evidence.to_dict())
    if moto_score > car_score:
        return DomainDecision("motorcycle", moto_score / total, reasons, evidence.to_dict())
    return DomainDecision("unknown", 0.5,
                          reasons + ["car and motorcycle evidence are balanced"],
                          evidence.to_dict())


# The Aria RGB camera is a fisheye: the image corners are an optically black
# vignette that is static and dark in every recording. Including it would make both
# domains look equally "enclosed", so every statistic below is computed over the
# valid (non-vignette) image region only.
VIGNETTE_LUMINANCE = 0.06
# Auto-exposure drifts over a multi-minute drive, so genuinely rigid structure still
# varies somewhat between frames sampled minutes apart. The threshold is a tolerance
# on that drift, not a noise floor.
STATIC_STD_THRESHOLD = 20.0


def collect_domain_evidence(frames: Sequence[np.ndarray],
                            static_std_threshold: float = STATIC_STD_THRESHOLD
                            ) -> DomainEvidence:
    """Pool ego-structure evidence from a set of sampled RGB frames.

    Frames are expected to be sampled far apart in time so that scene content
    decorrelates and only the ego structure stays constant.
    """
    if len(frames) < 2:
        raise ValueError("at least two sampled frames are required")

    import cv2

    small = []
    for f in frames:
        g = cv2.cvtColor(f, cv2.COLOR_RGB2GRAY) if f.ndim == 3 else f
        small.append(cv2.resize(g, (192, 144), interpolation=cv2.INTER_AREA)
                     .astype(np.float32))
    stack = np.stack(small, axis=0)

    std = stack.std(axis=0)
    mean_lum = stack.mean(axis=0) / 255.0
    valid = mean_lum > VIGNETTE_LUMINANCE
    static = (std < static_std_threshold) & valid

    h, w = static.shape
    top = slice(0, h // 4)
    bottom = slice(3 * h // 4, h)
    # central columns only: the top corners are vignette in both domains, so the
    # discriminative luminance is the one straight above the driver/rider
    centre = slice(w // 4, 3 * w // 4)

    border = np.zeros(static.shape, dtype=bool)
    bw = max(1, w // 12)
    bh = max(1, h // 12)
    border[:bh, centre] = True
    border[-bh:, centre] = True
    border[h // 4:3 * h // 4, :bw] = True
    border[h // 4:3 * h // 4, -bw:] = True

    def frac(mask_static: np.ndarray, region: np.ndarray) -> float:
        denom = int(np.count_nonzero(region))
        return float(np.count_nonzero(mask_static & region) / denom) if denom else 0.0

    top_region = np.zeros(static.shape, dtype=bool)
    top_region[top] = True
    bottom_region = np.zeros(static.shape, dtype=bool)
    bottom_region[bottom] = True

    top_valid = valid & top_region
    return DomainEvidence(
        static_fraction=frac(static, valid),
        top_band_static_fraction=frac(static, top_valid),
        bottom_band_static_fraction=frac(static, valid & bottom_region),
        top_band_luminance=float(mean_lum[top, centre][valid[top, centre]].mean())
        if np.any(valid[top, centre]) else 0.0,
        border_static_fraction=frac(static, valid & border),
        sampled_frames=len(frames),
    )


# --------------------------------------------------------------------------- #
# File records
# --------------------------------------------------------------------------- #
@dataclass
class FileRecord:
    relative_path: str
    absolute_path: str
    name: str
    extension: str
    size_bytes: int
    sha256: Optional[str]
    modified_iso: str
    modified_epoch: float
    kind: str
    estimated_domain: str = "unknown"
    domain_confidence: float = 0.0
    domain_rationale: List[str] = field(default_factory=list)
    domain_evidence: Optional[Dict[str, Any]] = None
    recording_id: Optional[str] = None
    duration_s: Optional[float] = None
    rgb_width: Optional[int] = None
    rgb_height: Optional[int] = None
    rgb_frame_count: Optional[int] = None
    rgb_effective_fps: Optional[float] = None
    num_streams: Optional[int] = None
    stream_labels: List[str] = field(default_factory=list)
    device_serial: Optional[str] = None
    recording_profile: Optional[str] = None
    start_time_epoch_sec: Optional[int] = None
    complete: Optional[bool] = None
    notes: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def scan_files(root: str | Path,
               hash_cache: Optional[HashCache] = None,
               hash_limit_bytes: Optional[int] = None,
               skip_dirs: Optional[set] = None) -> List[FileRecord]:
    """Walk the project tree and record every acquisition-like file."""
    import datetime as _dt

    root = Path(root).resolve()
    skip = set(DERIVED_DIR_NAMES if skip_dirs is None else skip_dirs)
    records: List[FileRecord] = []

    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in skip and not d.startswith(".")]
        for fn in sorted(filenames):
            p = Path(dirpath) / fn
            suffix = p.suffix.lower()
            if suffix not in DATA_SUFFIXES:
                continue
            try:
                st = p.stat()
            except OSError:
                continue
            digest = None
            if hash_limit_bytes is None or st.st_size <= hash_limit_bytes:
                digest = hash_cache.get(p) if hash_cache else sha256_file(p)
            records.append(FileRecord(
                relative_path=str(p.relative_to(root)),
                absolute_path=str(p),
                name=p.name,
                extension=suffix,
                size_bytes=int(st.st_size),
                sha256=digest,
                modified_iso=_dt.datetime.fromtimestamp(
                    st.st_mtime, _dt.timezone.utc).isoformat(),
                modified_epoch=float(st.st_mtime),
                kind=DATA_SUFFIXES[suffix],
            ))
    return records


def recording_id_for(record: FileRecord) -> str:
    """A stable, content-derived recording identifier.

    Derived from the SHA-256 so that the same acquisition keeps its identity even if
    the file is renamed or moved, and two different files can never collide.
    """
    if record.sha256:
        return f"{record.estimated_domain}_{record.sha256[:12]}"
    return f"{record.estimated_domain}_{record.name}"
