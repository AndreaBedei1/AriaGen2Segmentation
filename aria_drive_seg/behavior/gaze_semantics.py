"""Semantic gaze: what the rider was looking at, as a distribution.

A gaze point is not a pixel. The fovea subtends a couple of degrees, the eye
tracker has its own error, and the segmentation has a boundary of its own, so
reading the class of one pixel would turn all three uncertainties into a
confident wrong answer at every class boundary. Every sample here is read through
a Gaussian foveal window and comes out as a **distribution over classes**, with
the runner-up and the entropy kept next to the winner.

The frozen baseline stores a class label, a confidence and an entropy per pixel,
but dense per-class probabilities for only a handful of diagnostic frames. The
foveal distribution is therefore built from the labels, weighted by the Gaussian
and by each pixel's own confidence — a spatial class distribution over the
foveal area. It is not the model's probability simplex, and it is named
`foveal_*` throughout so the two are never confused.

Gaze never flows back into segmentation. This module only reads the frozen
baseline's outputs.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from ..ingestion.timeline import NS_PER_S

#: Classes whose gaze share is a primary metric.
ROAD_RELEVANT = ("road_surface", "lane_marking", "regulatory_road_marking",
                 "vehicle", "pedestrian", "traffic_sign", "traffic_light",
                 "road_boundary_or_obstacle")

#: Classes that stay exploratory proxies and must not carry a primary claim.
PROXY_ONLY = ("mirror", "instrument_display", "control_and_ego_vehicle")


def pixels_per_degree(focal_px: float) -> float:
    """Pinhole scale near the optical axis."""
    return float(focal_px) * math.tan(math.radians(1.0))


# --------------------------------------------------------------------------- #
# Gaze kinematics and I-VT fixation identification
# --------------------------------------------------------------------------- #
@dataclass
class GazeKinematics:
    timestamp_ns: np.ndarray
    yaw_rad: np.ndarray
    pitch_rad: np.ndarray
    angular_velocity_deg_s: np.ndarray
    is_fixation: np.ndarray
    fixation_id: np.ndarray          # -1 outside a fixation
    valid: np.ndarray

    def to_frame(self):
        import pandas as pd
        return pd.DataFrame({
            "timestamp_ns": self.timestamp_ns, "yaw_rad": self.yaw_rad,
            "pitch_rad": self.pitch_rad,
            "angular_velocity_deg_s": self.angular_velocity_deg_s,
            "is_fixation": self.is_fixation, "fixation_id": self.fixation_id,
            "valid": self.valid,
        })


@dataclass
class Fixation:
    fixation_id: int
    start_ns: int
    end_ns: int
    duration_s: float
    mean_yaw_rad: float
    mean_pitch_rad: float
    dispersion_deg: float
    sample_count: int

    def to_dict(self) -> Dict[str, Any]:
        return {"fixation_id": self.fixation_id, "start_ns": self.start_ns,
                "end_ns": self.end_ns, "duration_s": self.duration_s,
                "mean_yaw_rad": self.mean_yaw_rad,
                "mean_pitch_rad": self.mean_pitch_rad,
                "dispersion_deg": self.dispersion_deg,
                "sample_count": self.sample_count}


def angular_velocity_deg_s(timestamp_ns: np.ndarray, yaw_rad: np.ndarray,
                           pitch_rad: np.ndarray) -> np.ndarray:
    """Great-circle angular speed of the gaze direction.

    Computed as the angle between successive unit gaze vectors rather than as the
    Euclidean distance in (yaw, pitch): the latter overstates motion away from
    the centre, where a degree of yaw spans less than a degree of visual angle.
    """
    ts = np.asarray(timestamp_ns, np.int64)
    yaw = np.asarray(yaw_rad, float)
    pitch = np.asarray(pitch_rad, float)
    n = ts.size
    out = np.full(n, np.nan)
    if n < 2:
        return out
    # Unit direction from (yaw, pitch) in the eye frame.
    x = np.cos(pitch) * np.sin(yaw)
    y = np.sin(pitch)
    z = np.cos(pitch) * np.cos(yaw)
    v = np.stack([x, y, z], axis=1)
    dot = np.clip(np.einsum("ij,ij->i", v[:-1], v[1:]), -1.0, 1.0)
    dtheta = np.degrees(np.arccos(dot))
    dt = np.diff(ts) / NS_PER_S
    with np.errstate(divide="ignore", invalid="ignore"):
        speed = np.where(dt > 0, dtheta / dt, np.nan)
    out[:-1] = speed
    out[-1] = speed[-1]
    return out


def identify_fixations(timestamp_ns: Sequence[int], yaw_rad: Sequence[float],
                       pitch_rad: Sequence[float],
                       valid: Optional[Sequence[bool]] = None,
                       velocity_threshold_deg_s: float = 30.0,
                       min_duration_s: float = 0.100,
                       max_gap_s: float = 0.075
                       ) -> Tuple[GazeKinematics, List[Fixation]]:
    """I-VT: samples below a velocity threshold, grouped into fixations.

    The threshold is in degrees per second and the minimum duration in seconds,
    so the same configuration means the same thing at any sampling rate. A short
    gap inside an otherwise stable run does not split the fixation; a longer one
    does, and the run has to re-earn the minimum duration.
    """
    ts = np.asarray(timestamp_ns, np.int64)
    yaw = np.asarray(yaw_rad, float)
    pitch = np.asarray(pitch_rad, float)
    ok = (np.asarray(valid, bool) if valid is not None
          else np.isfinite(yaw) & np.isfinite(pitch))
    ok = ok & np.isfinite(yaw) & np.isfinite(pitch)

    vel = angular_velocity_deg_s(ts, yaw, pitch)
    slow = ok & np.isfinite(vel) & (vel < float(velocity_threshold_deg_s))

    fixation_id = np.full(ts.size, -1, np.int64)
    fixations: List[Fixation] = []
    i = 0
    next_id = 0
    while i < ts.size:
        if not slow[i]:
            i += 1
            continue
        j = i
        last_good = i
        while j + 1 < ts.size:
            gap_s = float((ts[j + 1] - ts[last_good]) / NS_PER_S)
            if slow[j + 1]:
                j += 1
                last_good = j
            elif gap_s <= max_gap_s:
                j += 1                       # tolerate a brief excursion
            else:
                break
        duration = float((ts[last_good] - ts[i]) / NS_PER_S)
        if duration >= min_duration_s and last_good > i:
            sel = slice(i, last_good + 1)
            m = slow[sel]
            fy = float(np.mean(yaw[sel][m])) if m.any() else float("nan")
            fp = float(np.mean(pitch[sel][m])) if m.any() else float("nan")
            disp = _dispersion_deg(yaw[sel][m], pitch[sel][m]) if m.any() else float("nan")
            fixation_id[sel] = next_id
            fixations.append(Fixation(next_id, int(ts[i]), int(ts[last_good]),
                                      duration, fy, fp, disp, int(m.sum())))
            next_id += 1
        i = last_good + 1

    return (GazeKinematics(ts, yaw, pitch, vel, fixation_id >= 0, fixation_id, ok),
            fixations)


def _dispersion_deg(yaw: np.ndarray, pitch: np.ndarray) -> float:
    """Angular spread of a fixation: max angle from its own mean direction."""
    if yaw.size == 0:
        return float("nan")
    x = np.cos(pitch) * np.sin(yaw)
    y = np.sin(pitch)
    z = np.cos(pitch) * np.cos(yaw)
    v = np.stack([x, y, z], axis=1)
    mean = v.mean(axis=0)
    norm = np.linalg.norm(mean)
    if norm < 1e-9:
        return float("nan")
    mean = mean / norm
    return float(np.degrees(np.arccos(np.clip(v @ mean, -1.0, 1.0))).max())


# --------------------------------------------------------------------------- #
# Semantic block reader
# --------------------------------------------------------------------------- #
class SemanticBlock:
    """Read-only access to one dense frozen-baseline semantic block.

    The block is a contiguous run of real frames processed by the frozen
    pipeline. Nothing here modifies it; this class only locates and decodes what
    the baseline already wrote.
    """

    def __init__(self, root: str | Path, entropy_root: Optional[str | Path] = None):
        self.root = Path(root)
        self.final = self.root / "semantic_camera_final_pass"
        self.entropy_dir = (Path(entropy_root) if entropy_root is not None
                            else self.root / "semantic_camera" / "final_entropy")
        if not (self.final / "final_masks").is_dir():
            raise FileNotFoundError(f"no frozen semantic block at {self.final}")
        self.frames = sorted(int(p.stem.split("_")[1])
                             for p in (self.final / "final_masks").glob("frame_*.png"))
        self._cache: Dict[int, Dict[str, np.ndarray]] = {}

    def __contains__(self, frame_index: int) -> bool:
        return int(frame_index) in set(self.frames)

    @property
    def frame_range(self) -> Tuple[int, int]:
        return (self.frames[0], self.frames[-1]) if self.frames else (-1, -1)

    def load(self, frame_index: int) -> Optional[Dict[str, np.ndarray]]:
        """Mask, confidence, entropy and provenance for one frame."""
        import cv2
        fi = int(frame_index)
        if fi in self._cache:
            return self._cache[fi]
        stem = f"frame_{fi:06d}"
        mask_path = self.final / "final_masks" / f"{stem}.png"
        if not mask_path.exists():
            return None
        mask = cv2.imread(str(mask_path), cv2.IMREAD_UNCHANGED)
        conf = cv2.imread(str(self.final / "final_confidence" / f"{stem}.png"),
                          cv2.IMREAD_UNCHANGED)
        prov = cv2.imread(str(self.final / "final_provenance" / f"{stem}.png"),
                          cv2.IMREAD_UNCHANGED)
        ent_path = self.entropy_dir / f"{stem}.png"
        ent = (cv2.imread(str(ent_path), cv2.IMREAD_UNCHANGED)
               if ent_path.exists() else None)
        out = {
            "mask": mask.astype(np.uint16),
            # Confidence and entropy are stored as u8 fractions of 255.
            "confidence": (conf.astype(np.float32) / 255.0 if conf is not None
                           else np.ones(mask.shape, np.float32)),
            "entropy": (ent.astype(np.float32) / 255.0 if ent is not None else None),
            "provenance": (prov if prov is not None
                           else np.zeros(mask.shape, np.uint8)),
        }
        if len(self._cache) > 24:                 # bounded, frames are ~12 MB each
            self._cache.pop(next(iter(self._cache)))
        self._cache[fi] = out
        return out


# --------------------------------------------------------------------------- #
# Foveal sampling
# --------------------------------------------------------------------------- #
def foveal_weights(radius_px: int, sigma_px: float) -> np.ndarray:
    """Normalised Gaussian kernel over a square patch of half-width `radius_px`."""
    ax = np.arange(-radius_px, radius_px + 1, dtype=np.float64)
    xx, yy = np.meshgrid(ax, ax)
    w = np.exp(-(xx ** 2 + yy ** 2) / (2.0 * float(sigma_px) ** 2))
    w[(xx ** 2 + yy ** 2) > radius_px ** 2] = 0.0
    total = w.sum()
    return w / total if total > 0 else w


def sample_foveal_semantics(layers: Dict[str, np.ndarray], u: float, v: float,
                            weights: np.ndarray, n_classes: int
                            ) -> Optional[Dict[str, Any]]:
    """Class distribution, confidence and entropy in the foveal window.

    Returns None when the gaze falls outside the image: an off-image gaze is a
    real state that must stay distinguishable from "looking at nothing useful".
    """
    mask = layers["mask"]
    h, w = mask.shape[:2]
    ui, vi = int(round(u)), int(round(v))
    if not (0 <= ui < w and 0 <= vi < h):
        return None

    r = (weights.shape[0] - 1) // 2
    y0, y1 = max(0, vi - r), min(h, vi + r + 1)
    x0, x1 = max(0, ui - r), min(w, ui + r + 1)
    ky0, ky1 = y0 - (vi - r), weights.shape[0] - ((vi + r + 1) - y1)
    kx0, kx1 = x0 - (ui - r), weights.shape[1] - ((ui + r + 1) - x1)

    wpatch = weights[ky0:ky1, kx0:kx1]
    mpatch = mask[y0:y1, x0:x1]
    cpatch = layers["confidence"][y0:y1, x0:x1]
    if wpatch.size == 0 or mpatch.size == 0:
        return None

    # Weight by the Gaussian AND by each pixel's own confidence, so a boundary
    # pixel the model is unsure about does not vote as loudly as a confident one.
    ww = wpatch * np.maximum(cpatch, 1e-6)
    total = float(ww.sum())
    if total <= 0:
        return None
    probs = np.bincount(mpatch.ravel(), weights=ww.ravel(),
                        minlength=n_classes).astype(np.float64)
    probs = probs[:n_classes] / total

    order = np.argsort(probs)[::-1]
    top1, top2 = int(order[0]), int(order[1]) if probs.size > 1 else -1
    nz = probs[probs > 0]
    foveal_entropy = float(-(nz * np.log(nz)).sum() / math.log(n_classes)) if nz.size else 0.0

    ent_layer = layers.get("entropy")
    model_entropy = (float((ent_layer[y0:y1, x0:x1] * wpatch).sum() /
                           max(float(wpatch.sum()), 1e-9))
                     if ent_layer is not None else None)
    prov_patch = layers["provenance"][y0:y1, x0:x1]
    prov_counts = np.bincount(prov_patch.ravel(), weights=wpatch.ravel(), minlength=8)

    return {
        "foveal_probabilities": probs,
        "top1_class_id": top1,
        "top2_class_id": top2,
        "top1_probability": float(probs[top1]),
        "top2_probability": float(probs[top2]) if top2 >= 0 else 0.0,
        "foveal_entropy": foveal_entropy,
        "model_entropy": model_entropy,
        "confidence": float((cpatch * wpatch).sum() / max(float(wpatch.sum()), 1e-9)),
        "provenance_code": int(np.argmax(prov_counts)),
        "distance_from_centre_px": float(math.hypot(ui - w / 2.0, vi - h / 2.0)),
    }


# --------------------------------------------------------------------------- #
# Aggregate metrics
# --------------------------------------------------------------------------- #
def transition_matrix(class_ids: Sequence[int], n_classes: int) -> np.ndarray:
    """Counts of class-to-class changes, ignoring repeats of the same class."""
    ids = [int(c) for c in class_ids if c is not None and c >= 0]
    m = np.zeros((n_classes, n_classes), dtype=np.int64)
    for a, b in zip(ids[:-1], ids[1:]):
        if a != b:
            m[a, b] += 1
    return m


def semantic_gaze_metrics(samples, class_names: Sequence[str],
                          fixations: Sequence[Fixation],
                          sample_interval_s: float,
                          off_road_classes: Optional[Sequence[str]] = None
                          ) -> Dict[str, Any]:
    """Dwell, transitions, entropy and off-road statistics.

    Everything that could be a count is reported **per second** as well, because
    the two recordings run at different rates and a raw count is not comparable
    between them.
    """
    import pandas as pd

    df = samples if isinstance(samples, pd.DataFrame) else pd.DataFrame(samples)
    n_classes = len(class_names)
    valid = df[df["semantic_valid"].astype(bool)] if "semantic_valid" in df else df
    total_s = float(len(valid) * sample_interval_s)

    road_set = set(off_road_classes if off_road_classes is not None else ROAD_RELEVANT)
    name_of = {i: n for i, n in enumerate(class_names)}

    per_class: Dict[str, Any] = {}
    for cid, cname in name_of.items():
        sel = valid[valid["top1_class_id"] == cid]
        fix_ids = sorted(set(sel["fixation_id"]) - {-1}) if "fixation_id" in sel else []
        durations = [f.duration_s for f in fixations if f.fixation_id in set(fix_ids)]
        first = (float(sel["rel_time_s"].min()) if len(sel) and "rel_time_s" in sel
                 else None)
        per_class[cname] = {
            "samples": int(len(sel)),
            "dwell_time_s": float(len(sel) * sample_interval_s),
            "time_fraction": (float(len(sel) / len(valid)) if len(valid) else 0.0),
            "fixation_count": len(fix_ids),
            "fixations_per_minute": (float(len(fix_ids) * 60.0 / total_s)
                                     if total_s > 0 else None),
            "mean_fixation_duration_s": (float(np.mean(durations))
                                         if durations else None),
            "median_fixation_duration_s": (float(np.median(durations))
                                           if durations else None),
            "p95_fixation_duration_s": (float(np.percentile(durations, 95))
                                        if durations else None),
            "time_to_first_sample_s": first,
            "mean_top1_probability": (float(sel["top1_probability"].mean())
                                      if len(sel) else None),
            "mean_confidence": (float(sel["confidence"].mean())
                                if len(sel) else None),
            "is_proxy_only": cname in PROXY_ONLY,
            "metric_status": ("exploratory_proxy_needs_manual_review"
                              if cname in PROXY_ONLY else "primary"),
        }
        if len(sel) and total_s > 0:
            gaps = np.diff(np.sort(sel["rel_time_s"].values))
            per_class[cname]["return_rate_per_minute"] = float(
                np.sum(gaps > sample_interval_s * 2) * 60.0 / total_s)
        else:
            per_class[cname]["return_rate_per_minute"] = None

    ids = valid["top1_class_id"].tolist() if len(valid) else []
    tm = transition_matrix(ids, n_classes)
    switches = int(tm.sum())

    # Off-road glances: runs where the top-1 class is not road-relevant. Cockpit
    # proxies and out-of-image gaze are separated, because they are different
    # states and only one of them is "eyes off the scene".
    off_runs = _off_road_runs(df, road_set, name_of, sample_interval_s)

    dist = np.array([per_class[n]["time_fraction"] for n in class_names])
    nz = dist[dist > 0]
    dist_entropy = (float(-(nz * np.log(nz)).sum() / math.log(n_classes))
                    if nz.size else 0.0)

    # Top-1 share is biased by viewing geometry. A driver looking far down the
    # road puts the fovea on the vanishing point, where the road surface has
    # narrowed to less than the ~3 degree foveal window and the surrounding scene
    # wins the argmax — so "percent of gaze on road_surface" understates looking
    # at the road, for a reason that has nothing to do with attention.
    #
    # The probability mass a class holds inside the foveal window does not have
    # that failure mode: a road that fills a third of the fovea contributes a
    # third, whether or not it is the argmax. Both are reported.
    mass: Dict[str, Any] = {}
    for cname in class_names:
        col = f"p_{cname}"
        mass[cname] = (float(valid[col].mean()) if col in valid and len(valid)
                       else None)
    road_mass = float(sum(v for c, v in mass.items()
                          if c in road_set and v is not None))

    fix_dur = [f.duration_s for f in fixations]
    return {
        "valid_samples": int(len(valid)),
        "total_valid_gaze_time_s": total_s,
        "sample_interval_s": sample_interval_s,
        "per_class": per_class,
        "transition_matrix": tm.tolist(),
        "transition_class_names": list(class_names),
        "class_switches": switches,
        "gaze_switching_rate_per_s": (float(switches / total_s)
                                      if total_s > 0 else None),
        "semantic_distribution_entropy": dist_entropy,
        "fixations": {
            "count": len(fixations),
            "per_minute": (float(len(fixations) * 60.0 / total_s)
                           if total_s > 0 else None),
            "mean_duration_s": float(np.mean(fix_dur)) if fix_dur else None,
            "median_duration_s": float(np.median(fix_dur)) if fix_dur else None,
            "p95_duration_s": (float(np.percentile(fix_dur, 95))
                               if fix_dur else None),
            "mean_dispersion_deg": (
                float(np.nanmean([f.dispersion_deg for f in fixations]))
                if fixations else None),
        },
        "off_road_glances": off_runs,
        "road_relevant_time_fraction": float(sum(
            per_class[c]["time_fraction"] for c in road_set if c in per_class)),
        "proxy_class_time_fraction": float(sum(
            per_class[c]["time_fraction"] for c in PROXY_ONLY if c in per_class)),
        "foveal_probability_mass": mass,
        "road_relevant_probability_mass": road_mass,
        "top1_vs_mass_note": (
            "top-1 shares are biased by viewing geometry: gaze at the vanishing "
            "point lands where the road subtends less than the foveal window, so "
            "the surrounding scene wins the argmax. The probability-mass figures "
            "are the geometry-robust counterpart and should be read alongside."),
        "proxy_caveat": ("mirror, instrument_display and control_and_ego_vehicle "
                         "are exploratory cockpit proxies; their gaze shares are "
                         "reported for review, never as a primary result"),
    }


def _off_road_runs(df, road_set, name_of, sample_interval_s) -> Dict[str, Any]:
    """Contiguous runs of gaze away from the road-relevant classes."""
    if len(df) == 0:
        return {"count": 0, "total_s": 0.0, "max_duration_s": None,
                "mean_duration_s": None, "per_minute": None,
                "out_of_image_runs": 0, "cockpit_proxy_runs": 0}
    valid = df["semantic_valid"].astype(bool).values if "semantic_valid" in df else \
        np.ones(len(df), bool)
    cls = df["top1_class_id"].values
    is_off = np.array([
        (not v) or (name_of.get(int(c)) not in road_set)
        for v, c in zip(valid, cls)])

    runs, start = [], None
    for i, flag in enumerate(is_off):
        if flag and start is None:
            start = i
        elif not flag and start is not None:
            runs.append((start, i))
            start = None
    if start is not None:
        runs.append((start, len(is_off)))

    durations = [(b - a) * sample_interval_s for a, b in runs]
    total_s = float(len(df) * sample_interval_s)
    out_of_image = sum(1 for a, b in runs if not valid[a:b].any())
    cockpit = sum(1 for a, b in runs
                  if any(name_of.get(int(c)) in PROXY_ONLY for c in cls[a:b]))
    return {
        "count": len(runs),
        "total_s": float(sum(durations)),
        "time_fraction": (float(sum(durations) / total_s) if total_s > 0 else None),
        "max_duration_s": float(max(durations)) if durations else None,
        "mean_duration_s": float(np.mean(durations)) if durations else None,
        "per_minute": (float(len(runs) * 60.0 / total_s) if total_s > 0 else None),
        "out_of_image_runs": out_of_image,
        "cockpit_proxy_runs": cockpit,
        "note": ("a run is 'off road' when the foveal top-1 class is not one of "
                 "the road-relevant classes; out-of-image gaze and cockpit-proxy "
                 "gaze are counted separately because they are different states"),
    }


def scanpath_statistics(u: np.ndarray, v: np.ndarray, valid: np.ndarray,
                        px_per_deg: float, image_shape: Tuple[int, int]
                        ) -> Dict[str, Any]:
    """Scanpath length, dispersion and eccentricity, all in degrees."""
    ok = np.asarray(valid, bool) & np.isfinite(u) & np.isfinite(v)
    uu, vv = np.asarray(u, float)[ok], np.asarray(v, float)[ok]
    if uu.size < 2:
        return {"samples": int(uu.size), "scanpath_length_deg": None,
                "dispersion_deg": None, "mean_eccentricity_deg": None,
                "saccade_amplitude_deg": None}
    steps = np.hypot(np.diff(uu), np.diff(vv)) / px_per_deg
    h, w = image_shape
    ecc = np.hypot(uu - w / 2.0, vv - h / 2.0) / px_per_deg
    return {
        "samples": int(uu.size),
        "scanpath_length_deg": float(steps.sum()),
        "scanpath_length_deg_per_s": None,     # filled by the caller, needs duration
        "dispersion_deg": float(np.hypot(np.std(uu), np.std(vv)) / px_per_deg),
        "mean_eccentricity_deg": float(np.mean(ecc)),
        "p95_eccentricity_deg": float(np.percentile(ecc, 95)),
        "saccade_amplitude_deg": {
            "median": float(np.median(steps)),
            "p95": float(np.percentile(steps, 95)),
            "max": float(np.max(steps)),
        },
    }
