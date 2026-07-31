"""Exploratory car/motorcycle comparison of semantic-camera outputs.

Everything produced here is labelled `exploratory_preliminary`. It exists to expose
failure modes and domain shift before annotation, not to support a scientific claim
about how the vehicle changes visual attention.

Two rules shape every metric:

* the two recordings are currently sampled at different rates, so **every temporal
  quantity is normalised per second** and raw per-frame counts are never compared;
* the sampling-rate difference is a temporary acquisition artefact, so it is
  reported as a caveat on each affected metric and never used as a discriminating
  feature.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from ..io_utils import read_mask_u16
from ..taxonomy import Taxonomy
from .timeline import NS_PER_S, measure_rate, per_second

# Metrics whose value is affected by how densely the scene was sampled. They are
# still reported (hiding them would be worse) but always with this caveat.
RATE_SENSITIVE = (
    "class_switch_rate_per_second",
    "mean_frame_difference",
    "component_churn_per_second",
)

CAVEAT = ("the two recordings are currently sampled at different rates; this metric "
          "is rate sensitive and must be recomputed once both are at the final "
          "protocol rate")


@dataclass
class RecordingMetrics:
    recording_id: str
    domain: str
    frame_count: int
    duration_s: float
    effective_fps: float
    class_pixel_fraction: Dict[str, float]
    class_presence_fraction: Dict[str, float]
    mean_confidence: float
    confidence_percentiles: Dict[str, float]
    mean_entropy: float
    entropy_percentiles: Dict[str, float]
    provenance_fraction: Dict[str, float]
    fallback_fraction: float
    dense_coverage: float
    invalid_id_count: int
    class_switch_rate_per_second: Optional[float]
    component_count_per_class_frame: Dict[str, float]
    mean_component_area_px: Dict[str, float]
    lane_continuity: Optional[float]
    boundary_continuity: Optional[float]
    cockpit_fraction: float
    gaze: Dict[str, Any]
    hands: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        from dataclasses import asdict
        return asdict(self)


def _percentiles(values: np.ndarray) -> Dict[str, float]:
    if values.size == 0:
        return {}
    return {f"p{q}": float(np.percentile(values, q)) for q in (5, 25, 50, 75, 95)}


def _components(mask: np.ndarray, class_id: int) -> tuple:
    import cv2
    binary = (mask == class_id).astype(np.uint8)
    if not binary.any():
        return 0, 0.0
    n, _, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    areas = stats[1:, cv2.CC_STAT_AREA] if n > 1 else np.array([])
    return int(max(0, n - 1)), float(areas.mean()) if areas.size else 0.0


def collect_metrics(run_dir: str | Path, recording_id: str, domain: str,
                    taxonomy: Taxonomy,
                    semantic_subdir: str = "semantic_camera",
                    mask_subdir: str = "final_masks",
                    stride: int = 1,
                    gaze_summary: Optional[Dict[str, Any]] = None,
                    hand_summary: Optional[Dict[str, Any]] = None
                    ) -> RecordingMetrics:
    """Measure one recording's semantic-camera output."""
    root = Path(run_dir) / semantic_subdir
    meta_dir = root / "metadata"
    mask_dir = root / mask_subdir
    if not mask_dir.exists():
        raise FileNotFoundError(f"no masks at {mask_dir}")

    metas = sorted(meta_dir.glob("*.json"))[::stride]
    if not metas:
        raise FileNotFoundError(f"no metadata at {meta_dir}")

    names = taxonomy.names()
    pixel_fraction = {n: 0.0 for n in names}
    presence = {n: 0 for n in names}
    comp_count = {n: 0.0 for n in names}
    comp_area = {n: [] for n in names}
    prov_fraction: Dict[str, float] = {}
    confidences: List[float] = []
    entropies: List[float] = []
    coverages: List[float] = []
    fallback_flags: List[bool] = []
    timestamps: List[int] = []
    invalid_ids = 0
    switches: List[float] = []
    previous_mask: Optional[np.ndarray] = None

    prov_codes: Dict[str, str] = {}
    for path in metas:
        meta = json.loads(path.read_text())
        timestamps.append(int(meta["capture_timestamp_ns"]))
        coverages.append(float(meta.get("dense_coverage", 1.0)))
        fallback_flags.append(bool(meta.get("internal_is_fallback", False)))
        prov_codes = meta.get("provenance_codes", prov_codes)

        mask = read_mask_u16(mask_dir / f"{path.stem}.png")
        total = mask.size
        invalid_ids += int(np.count_nonzero(
            (mask < 1) | (mask > taxonomy.max_id)))
        for cid, name in enumerate(names):
            hits = int(np.count_nonzero(mask == cid))
            if hits:
                pixel_fraction[name] += hits / total
                presence[name] += 1
                n_comp, mean_area = _components(mask, cid)
                comp_count[name] += n_comp
                if mean_area:
                    comp_area[name].append(mean_area)

        for key, arr, dst in (("final_confidence", None, confidences),
                              ("final_entropy", None, entropies)):
            f = root / key / f"{path.stem}.png"
            if f.exists():
                import cv2
                v = cv2.imread(str(f), cv2.IMREAD_UNCHANGED)
                if v is not None:
                    dst.append(float(np.mean(v.astype(np.float32) / 255.0)))
        if not confidences and "mean_final_confidence" in meta:
            confidences.append(float(meta["mean_final_confidence"]))
        if not entropies and "mean_final_entropy" in meta:
            entropies.append(float(meta["mean_final_entropy"]))

        prov_path = root / "provenance" / f"{path.stem}.png"
        if prov_path.exists():
            import cv2
            prov = cv2.imread(str(prov_path), cv2.IMREAD_UNCHANGED)
            if prov is not None:
                for code in np.unique(prov):
                    label = prov_codes.get(str(int(code)), f"code_{int(code)}")
                    prov_fraction[label] = prov_fraction.get(label, 0.0) + \
                        float(np.count_nonzero(prov == code) / prov.size)

        if previous_mask is not None and previous_mask.shape == mask.shape:
            switches.append(float(np.mean(previous_mask != mask)))
        previous_mask = mask

    n = len(metas)
    rate = measure_rate(np.array(timestamps, dtype=np.int64))
    duration = rate.duration_s
    fps = rate.effective_fps or 0.0

    # switches are per adjacent SAMPLED pair; converting to a per-second rate is
    # what makes the two recordings comparable at all
    switch_per_second = None
    if switches and duration > 0:
        switch_per_second = per_second(float(np.sum(switches)), duration)

    return RecordingMetrics(
        recording_id=recording_id, domain=domain, frame_count=n,
        duration_s=duration, effective_fps=fps,
        class_pixel_fraction={k: v / n for k, v in pixel_fraction.items()},
        class_presence_fraction={k: v / n for k, v in presence.items()},
        mean_confidence=float(np.mean(confidences)) if confidences else float("nan"),
        confidence_percentiles=_percentiles(np.array(confidences)),
        mean_entropy=float(np.mean(entropies)) if entropies else float("nan"),
        entropy_percentiles=_percentiles(np.array(entropies)),
        provenance_fraction={k: v / n for k, v in prov_fraction.items()},
        fallback_fraction=float(np.mean(fallback_flags)) if fallback_flags else 0.0,
        dense_coverage=float(np.mean(coverages)) if coverages else 0.0,
        invalid_id_count=invalid_ids,
        class_switch_rate_per_second=switch_per_second,
        component_count_per_class_frame={
            k: (v / max(1, presence[k])) for k, v in comp_count.items()},
        mean_component_area_px={
            k: (float(np.mean(v)) if v else 0.0) for k, v in comp_area.items()},
        lane_continuity=_continuity(presence, n, ("lane_marking",
                                                  "regulatory_road_marking")),
        boundary_continuity=_continuity(presence, n, ("road_boundary_or_obstacle",)),
        cockpit_fraction=sum(pixel_fraction[c] / n for c in
                             ("mirror", "instrument_display",
                              "control_and_ego_vehicle")),
        gaze=gaze_summary or {"available": False},
        hands=hand_summary or {"available": False},
    )


def _continuity(presence: Dict[str, int], n: int,
                classes: Sequence[str]) -> Optional[float]:
    """Fraction of frames in which at least one of the classes is present."""
    if n == 0:
        return None
    return float(max(presence.get(c, 0) for c in classes) / n)


def compare(car: RecordingMetrics, motorcycle: RecordingMetrics,
            taxonomy: Taxonomy) -> Dict[str, Any]:
    """Build the exploratory comparison document."""
    per_class = {}
    for name in taxonomy.names():
        c = car.class_pixel_fraction.get(name, 0.0)
        m = motorcycle.class_pixel_fraction.get(name, 0.0)
        per_class[name] = {
            "car_pixel_fraction": c,
            "motorcycle_pixel_fraction": m,
            "difference": m - c,
            "car_presence_fraction": car.class_presence_fraction.get(name, 0.0),
            "motorcycle_presence_fraction":
                motorcycle.class_presence_fraction.get(name, 0.0),
            "car_components_per_class_frame":
                car.component_count_per_class_frame.get(name, 0.0),
            "motorcycle_components_per_class_frame":
                motorcycle.component_count_per_class_frame.get(name, 0.0),
        }

    findings = _attribute_findings(car, motorcycle, per_class)
    return {
        "status": "exploratory_preliminary",
        "is_scientific_conclusion": False,
        "caveats": [
            "The car recording is a provisional 10 fps development baseline; the "
            "final protocol is 15 fps for both vehicles.",
            "No reviewed ground truth exists, so no accuracy, IoU, precision or "
            "recall is reported: these are coverage, stability and agreement "
            "diagnostics only.",
            "The sampling-rate difference is never used as a feature and no "
            "vehicle classifier is trained here.",
            "The definitive comparison must be repeated when both recordings are "
            "acquired at the final protocol rate.",
        ],
        "rate_sensitive_metrics": {k: CAVEAT for k in RATE_SENSITIVE},
        "car": car.to_dict(),
        "motorcycle": motorcycle.to_dict(),
        "per_class": per_class,
        "temporal_per_second": {
            "car_class_switch_rate_per_second": car.class_switch_rate_per_second,
            "motorcycle_class_switch_rate_per_second":
                motorcycle.class_switch_rate_per_second,
            "note": ("switch rates are normalised per second; raw per-frame counts "
                     "of recordings sampled at different rates are never compared"),
            "caveat": CAVEAT,
        },
        "findings": findings,
    }


def _attribute_findings(car: RecordingMetrics, motorcycle: RecordingMetrics,
                        per_class: Dict[str, Any]) -> Dict[str, List[str]]:
    """Sort observed differences by their most likely cause."""
    common: List[str] = []
    moto_only: List[str] = []
    car_only: List[str] = []
    external: List[str] = []
    cockpit_proxy: List[str] = []
    fusion: List[str] = []
    temporal: List[str] = []
    rate: List[str] = []
    domain_shift: List[str] = []

    if car.fallback_fraction > 0 and motorcycle.fallback_fraction > 0:
        common.append(
            "both recordings use the unreviewed Grounding DINO + SAM2.1 cockpit "
            "fallback: no trained cockpit model exists yet")
    if car.dense_coverage >= 0.999 and motorcycle.dense_coverage >= 0.999:
        common.append("dense coverage is complete in both recordings")
    if car.invalid_id_count == 0 and motorcycle.invalid_id_count == 0:
        common.append("no invalid class id in either recording")

    cockpit_gap = motorcycle.cockpit_fraction - car.cockpit_fraction
    if abs(cockpit_gap) > 0.02:
        target = moto_only if cockpit_gap > 0 else car_only
        target.append(
            f"cockpit classes cover {abs(cockpit_gap):.1%} "
            f"{'more' if cockpit_gap > 0 else 'less'} of the frame than in the other "
            "domain; the motorcycle exposes far less ego structure to the camera")
        cockpit_proxy.append(
            "the cockpit stream is a geometric + open-vocabulary proxy tuned on car "
            "interiors; its bottom-fraction geometric prior does not describe a "
            "handlebar")

    for name, values in per_class.items():
        d = values["difference"]
        if abs(d) < 0.01:
            continue
        line = (f"{name}: {values['motorcycle_pixel_fraction']:.2%} on the "
                f"motorcycle vs {values['car_pixel_fraction']:.2%} on the car")
        if name in ("mirror", "instrument_display", "control_and_ego_vehicle"):
            cockpit_proxy.append(line)
        elif name in ("road_surface", "lane_marking", "regulatory_road_marking",
                      "road_boundary_or_obstacle", "other_environment"):
            external.append(line)
            domain_shift.append(
                f"{name} differs markedly: the rider's viewpoint is higher, less "
                "occluded and more roll-dynamic than the driver's")
        else:
            external.append(line)

    if (car.class_switch_rate_per_second is not None
            and motorcycle.class_switch_rate_per_second is not None):
        temporal.append(
            f"per-second class switching is "
            f"{motorcycle.class_switch_rate_per_second:.3f} on the motorcycle vs "
            f"{car.class_switch_rate_per_second:.3f} on the car")
        rate.append(
            "per-second switching still depends on how densely the motion was "
            "sampled; part of this difference is the 10 vs 15 fps artefact rather "
            "than a property of the vehicle")

    if abs(motorcycle.mean_entropy - car.mean_entropy) > 0.02:
        fusion.append(
            f"mean final entropy differs ({motorcycle.mean_entropy:.3f} vs "
            f"{car.mean_entropy:.3f}): the fusion is less certain in one domain")

    domain_shift.append(
        "vibration, roll and the absence of a windscreen change the motorcycle's "
        "image statistics independently of any model quality difference")

    return {
        "common_problems": common,
        "motorcycle_specific": moto_only,
        "car_specific": car_only,
        "external_model": external,
        "cockpit_proxy": cockpit_proxy,
        "fusion": fusion,
        "temporal": temporal,
        "probably_due_to_sampling_rate": rate,
        "probably_due_to_domain_shift": sorted(set(domain_shift)),
    }
