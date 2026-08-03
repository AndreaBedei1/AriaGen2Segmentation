"""Clean full-frame semantic-camera rendering for the fast pipeline."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

import numpy as np

from ..behavior.video import VideoWriter
from ..config import Config
from ..io_utils import atomic_write_json
from ..taxonomy import Taxonomy


def _overlay(image_bgr: np.ndarray, mask: np.ndarray, taxonomy: Taxonomy,
             alpha: float) -> np.ndarray:
    import cv2
    palette_bgr = taxonomy.palette()[:, ::-1]
    semantic = palette_bgr[np.clip(mask.astype(np.int64), 0, taxonomy.max_id)]
    return cv2.addWeighted(image_bgr, 1.0 - alpha, semantic, alpha, 0)


def _draw_gaze(image: np.ndarray, u: float, v: float, radius: int) -> None:
    import cv2
    if not (np.isfinite(u) and np.isfinite(v)):
        return
    point = (int(round(u)), int(round(v)))
    cv2.circle(image, point, radius + 2, (20, 20, 20), 2, cv2.LINE_AA)
    cv2.circle(image, point, radius, (255, 255, 255), 2, cv2.LINE_AA)


def _read_layers(root: Path, row, taxonomy: Taxonomy, alpha: float,
                 gaze_by_frame: Dict[int, tuple[float, float]], gaze_radius: int):
    import cv2
    image = cv2.imread(str(root / row.rectified_path), cv2.IMREAD_COLOR)
    mask = cv2.imread(str(root / row.mask_path), cv2.IMREAD_UNCHANGED)
    if image is None or mask is None:
        raise FileNotFoundError(f"missing RGB or mask for frame {int(row.frame_index)}")
    if mask.shape != image.shape[:2]:
        mask = cv2.resize(mask, (image.shape[1], image.shape[0]),
                          interpolation=cv2.INTER_NEAREST)
    frame = _overlay(image, mask, taxonomy, alpha)
    gaze = gaze_by_frame.get(int(row.frame_index))
    if gaze is not None:
        _draw_gaze(frame, gaze[0], gaze[1], gaze_radius)
    return frame


def _gaze_lookup(root: Path) -> Dict[int, tuple[float, float]]:
    import pandas as pd
    path = root / "gaze" / "semantic_gaze.parquet"
    if not path.exists():
        return {}
    gaze = pd.read_parquet(path)
    gaze = gaze[gaze.semantic_valid.astype(bool)]
    if gaze.empty:
        return {}
    # One unobtrusive point per segmented frame: median of the real gaze samples
    # associated to it.  The samples themselves remain intact in parquet.
    grouped = gaze.groupby("segmentation_frame_index")[["rect_u", "rect_v"]].median()
    return {int(i): (float(row.rect_u), float(row.rect_v))
            for i, row in grouped.iterrows()}


def choose_preview_rows(index, duration_s: float):
    """Deterministically choose the cleanest contiguous real-time preview."""
    import pandas as pd
    if index.empty:
        return index
    ts = index.capture_timestamp_ns.values.astype(np.int64)
    scores = (index.mean_confidence.values - index.mean_normalized_entropy.values -
              .5 * index.get("fraction_other_environment",
                             pd.Series(np.zeros(len(index)))).values)
    best = (0, min(len(index), 1), -np.inf)
    j = 0
    for i in range(len(index)):
        while j < len(index) and ts[j] - ts[i] <= duration_s * 1e9:
            j += 1
        if j <= i:
            continue
        score = float(np.nanmean(scores[i:j]))
        # Prefer a complete window; partial tail windows are eligible only when
        # the recording itself is shorter than requested.
        span = (ts[j - 1] - ts[i]) / 1e9
        if span >= min(duration_s * .9, (ts[-1] - ts[0]) / 1e9) and score > best[2]:
            best = (i, j, score)
    return index.iloc[best[0]:best[1]].copy()


def measured_fps_from_timestamps(index) -> float:
    """Measured presentation rate for observed, strictly ordered RGB records."""
    ts = index.capture_timestamp_ns.to_numpy(np.int64)
    if ts.size < 2 or np.any(np.diff(ts) <= 0):
        raise ValueError("at least two strictly increasing timestamps are required")
    duration_s = float((int(ts[-1]) - int(ts[0])) / 1e9)
    if duration_s <= 0:
        raise ValueError("timestamp duration must be positive")
    return float((ts.size - 1) / duration_s)


def _resolve_fps(value: Any, index) -> tuple[float, str]:
    if isinstance(value, str):
        if value not in {"measured", "measured_from_timestamps"}:
            raise ValueError(f"unsupported video fps mode: {value}")
        return measured_fps_from_timestamps(index), "measured_from_real_timestamps"
    fps = float(value)
    if not np.isfinite(fps) or fps <= 0:
        raise ValueError("video fps must be positive")
    return fps, "configured_constant"


def render_domain(input_dir: str | Path, output_path: str | Path, cfg: Config,
                  preview_path: Optional[str | Path] = None) -> Dict[str, Any]:
    import pandas as pd
    root = Path(input_dir); output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fast = cfg.get("semantic_gaze_fast_external", {})
    render = fast.get("render", {})
    taxonomy = Taxonomy.load(cfg.resolve(fast["classes"]))
    frames = pd.read_parquet(root / "frames" / "frames.parquet")
    seg = pd.read_parquet(root / "segmentation" / "segmentation_index.parquet")
    index = frames.merge(seg, on=["frame_index", "capture_timestamp_ns"], how="inner")
    index = index.sort_values("capture_timestamp_ns").reset_index(drop=True)
    if index.empty:
        raise ValueError("no RGB/segmentation rows to render")
    fps, fps_source = _resolve_fps(
        render.get("video_fps", fast.get("segmentation_frequency_hz", 5.0)), index)
    alpha = float(render.get("overlay_alpha", .62))
    radius = int(render.get("gaze_radius_px", 7))
    gaze = _gaze_lookup(root)
    size = (int(index.iloc[0].width), int(index.iloc[0].height))

    with VideoWriter(output_path, fps, size) as writer:
        for row in index.itertuples(index=False):
            writer.write(_read_layers(root, row, taxonomy, alpha, gaze, radius))

    preview_info = None
    if preview_path is not None:
        preview_path = Path(preview_path); preview_path.parent.mkdir(parents=True, exist_ok=True)
        preview = choose_preview_rows(
            index, float(render.get("preview_duration_s", 30.0)))
        with VideoWriter(preview_path, fps, size) as writer:
            for row in preview.itertuples(index=False):
                writer.write(_read_layers(root, row, taxonomy, alpha, gaze, radius))
        preview_info = {
            "path": str(preview_path), "frames": int(len(preview)),
            "start_timestamp_ns": int(preview.capture_timestamp_ns.iloc[0]),
            "end_timestamp_ns": int(preview.capture_timestamp_ns.iloc[-1])}
    info = {"path": str(output_path), "frames": int(len(index)), "fps": fps,
            "duration_s": float(len(index) / fps), "size": list(size),
            "fps_source": fps_source,
            "source_timestamp_duration_s": float(
                (int(index.capture_timestamp_ns.iloc[-1]) -
                 int(index.capture_timestamp_ns.iloc[0])) / 1e9),
            "layout": "single_full_frame_clean_semantic_overlay",
            "diagnostic_panels": False, "preview": preview_info}
    atomic_write_json(output_path.with_suffix(".manifest.json"), info)
    return info


def _label(image: np.ndarray, text: str) -> None:
    import cv2
    cv2.putText(image, text, (18, 34), cv2.FONT_HERSHEY_SIMPLEX, .75,
                (0, 0, 0), 5, cv2.LINE_AA)
    cv2.putText(image, text, (18, 34), cv2.FONT_HERSHEY_SIMPLEX, .75,
                (255, 255, 255), 2, cv2.LINE_AA)


def render_shared_route(car_dir: str | Path, motorcycle_dir: str | Path,
                        output_path: str | Path, cfg: Config,
                        behavior_root: str | Path = "output/article1/behavior_analysis",
                        route_report: str | Path =
                        "reports/article1_behavior_analysis/route") -> Dict[str, Any]:
    """Position-bin matched, clean car/motorcycle semantic comparison."""
    import cv2
    import pandas as pd
    roots = {"car": Path(car_dir), "motorcycle": Path(motorcycle_dir)}
    fast = cfg.get("semantic_gaze_fast_external", {})
    taxonomy = Taxonomy.load(cfg.resolve(fast["classes"]))
    alpha = float(fast.get("render", {}).get("overlay_alpha", .62))
    radius = int(fast.get("render", {}).get("gaze_radius_px", 7))
    render = fast.get("render", {})
    route_report = Path(route_report)
    summary = json.loads((route_report / "route_alignment_summary.json").read_text())
    bin_size = float(summary["analysis_bin_size_m"])
    pairs = pd.read_csv(route_report / "paired_route_segments.csv")
    keys = list(pairs[(pairs.bin_size_m == bin_size) &
                      pairs.paired.astype(bool)].bin_key)
    data: Dict[str, Any] = {}
    gaze: Dict[str, Any] = {}
    domain_fps: Dict[str, float] = {}
    behavior_root = Path(behavior_root)
    recording_ids = {"car": "car_2e84f0c3e245",
                     "motorcycle": "motorcycle_5ab8604a14df"}
    for domain, root in roots.items():
        frames = pd.read_parquet(root / "frames" / "frames.parquet")
        domain_fps[domain] = measured_fps_from_timestamps(
            frames.sort_values("capture_timestamp_ns"))
        seg = pd.read_parquet(root / "segmentation" / "segmentation_index.parquet")
        bins = pd.read_parquet(
            behavior_root / recording_ids[domain] / "frame_route_bins.parquet")
        merged = frames.merge(seg, on=["frame_index", "capture_timestamp_ns"])
        bcols = bins[["frame_index", "bin_key", "bin_valid"]].drop_duplicates("frame_index")
        data[domain] = merged.merge(bcols, on="frame_index", how="left")
        data[domain] = data[domain][data[domain].bin_valid.fillna(False).astype(bool)]
        gaze[domain] = _gaze_lookup(root)

    shared_value = render.get("shared_video_fps", render.get("video_fps", 5.0))
    if isinstance(shared_value, str) and shared_value.startswith("car_"):
        if shared_value.removeprefix("car_") not in {
                "measured", "measured_from_timestamps"}:
            raise ValueError(f"unsupported shared video fps mode: {shared_value}")
        fps, fps_source = domain_fps["car"], "car_measured_from_real_timestamps"
    elif isinstance(shared_value, str) and shared_value.startswith("motorcycle_"):
        if shared_value.removeprefix("motorcycle_") not in {
                "measured", "measured_from_timestamps"}:
            raise ValueError(f"unsupported shared video fps mode: {shared_value}")
        fps, fps_source = (domain_fps["motorcycle"],
                           "motorcycle_measured_from_real_timestamps")
    else:
        fps, fps_source = _resolve_fps(shared_value, data["car"].sort_values(
            "capture_timestamp_ns").drop_duplicates("capture_timestamp_ns"))

    ordered: list[tuple[Any, Any]] = []
    used_keys = []
    for key in keys:
        selections = {d: data[d][data[d].bin_key == key].sort_values(
            "capture_timestamp_ns") for d in data}
        n = min(len(selections["car"]), len(selections["motorcycle"]))
        if n == 0:
            continue
        # Equal-count ordinal samples preserve traversal order without making up
        # frames or copying a mask from an unobserved time.
        for d in selections:
            take = np.linspace(0, len(selections[d]) - 1, n).round().astype(int)
            selections[d] = selections[d].iloc[np.unique(take)]
        n = min(len(selections["car"]), len(selections["motorcycle"]))
        ordered.extend(zip(selections["car"].iloc[:n].itertuples(index=False),
                           selections["motorcycle"].iloc[:n].itertuples(index=False)))
        used_keys.append(key)
    if not ordered:
        raise ValueError("no segmented frames occupy paired shared-route bins")

    output_path = Path(output_path); output_path.parent.mkdir(parents=True, exist_ok=True)
    panel_size = (960, 720); size = (panel_size[0] * 2, panel_size[1])
    with VideoWriter(output_path, fps, size) as writer:
        for car_row, moto_row in ordered:
            panels = []
            for domain, row in (("car", car_row), ("motorcycle", moto_row)):
                panel = _read_layers(roots[domain], row, taxonomy, alpha,
                                     gaze[domain], radius)
                panel = cv2.resize(panel, panel_size, interpolation=cv2.INTER_AREA)
                _label(panel, "AUTO" if domain == "car" else "MOTO")
                panels.append(panel)
            writer.write(np.hstack(panels))
    info = {"path": str(output_path), "frames": len(ordered), "fps": fps,
            "duration_s": len(ordered) / fps, "paired_bin_size_m": bin_size,
            "fps_source": fps_source,
            "domain_native_fps": domain_fps,
            "paired_bins_rendered": len(used_keys), "layout": "clean_side_by_side",
            "matching": "spatial route bin and within-bin traversal order",
            "synthetic_frames": 0, "scientific_masks_resampled": False,
            "presentation_note": (
                "one MP4 requires one timebase; this spatial QA video uses the "
                "configured domain timebase without changing native scientific outputs")}
    atomic_write_json(output_path.with_suffix(".manifest.json"), info)
    return info
