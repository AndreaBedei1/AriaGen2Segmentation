"""`analyze` (§10, §17.1): gaze labels per method, method agreement, no-GT metrics."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

from ..config import Config
from ..io_utils import atomic_write_json
from ..logging_utils import get_logger
from ..taxonomy import Taxonomy
from . import compare as C
from .gaze_labels import compute_gaze_labels

log = get_logger("analyze")

METHODS = ["grounded_sam2", "oneformer_mapillary"]


def _present_methods(input_dir: str) -> List[str]:
    out = Path(input_dir)
    return [m for m in METHODS if (out / m / "canonical_masks").exists()
            and any((out / m / "canonical_masks").glob("frame_*.png"))]


def run_analyze(input_dir: str, cfg: Config, method: Optional[str] = None) -> Dict[str, Any]:
    tax = Taxonomy.load(cfg.resolve(cfg.get("project.classes")))
    out = Path(input_dir)
    (out / "comparison" / "metrics").mkdir(parents=True, exist_ok=True)
    methods = [method] if method else _present_methods(input_dir)
    if not methods:
        log.warning("no segmentation outputs found in %s", input_dir)
        return {}

    report: Dict[str, Any] = {"methods": methods, "per_method": {}, "gaze_temporal": {}}
    eval_ids = set(tax.eval_ids())

    # 1) gaze labels + no-GT metrics per method
    has_gaze = (out / "gaze" / "aligned_gaze.parquet").exists()
    for m in methods:
        report["per_method"][m] = C.per_method_metrics(input_dir, m, tax)
        if has_gaze:
            compute_gaze_labels(input_dir, m, cfg, tax,
                                unsupported_ids=set(range(tax.max_id + 1)) - eval_ids)
            report["gaze_temporal"][m] = C.gaze_temporal_distribution(input_dir, m, tax)

    # 2) method agreement (needs both)
    if len(methods) >= 2 and "grounded_sam2" in methods and "oneformer_mapillary" in methods:
        report["method_agreement"] = C.method_agreement(input_dir, tax)
        if has_gaze:
            report["gaze_agreement"] = C.gaze_agreement(input_dir, tax)

    atomic_write_json(out / "comparison" / "metrics" / "metrics_no_gt.json", report)
    _write_markdown(out / "reports" / "analysis_summary.md", report, tax)
    log.info("analyze done -> %s", out / "comparison" / "metrics" / "metrics_no_gt.json")
    return report


def _write_markdown(path: Path, report: Dict[str, Any], tax: Taxonomy) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    L = ["# Analysis summary (no ground truth)\n",
         "> Between-method agreement/IoU is **consistency, not accuracy** (needs GT, §17.2).\n"]
    L.append("## Per-method metrics\n")
    L.append("| method | frames | coverage | unknown% | classes/frame | ms/frame | fps |")
    L.append("|---|--:|--:|--:|--:|--:|--:|")
    for m, r in report.get("per_method", {}).items():
        L.append(f"| {m} | {r.get('frames')} | {_p(r.get('coverage_mean'))} | "
                 f"{_p(r.get('unknown_frac_mean'))} | {_f(r.get('classes_per_frame_mean'))} | "
                 f"{_f(r.get('mean_inference_ms'))} | {_f(r.get('fps'))} |")
    ma = report.get("method_agreement")
    if ma:
        L.append("\n## Method agreement (common eval taxonomy)\n")
        L.append(f"- pixelwise agreement: **{_p(ma.get('pixelwise_agreement_common'))}**")
        L.append(f"- mean IoU between methods: **{_f(ma.get('mean_iou_between_methods'))}** "
                 f"(consistency, not accuracy)")
    ga = report.get("gaze_agreement")
    if ga:
        L.append("\n## Gaze-class agreement between methods\n")
        L.append(f"- frames both valid: {ga.get('both_valid')}")
        L.append(f"- gaze-class agreement: **{_p(ga.get('gaze_class_agreement_frac'))}**")
        L.append(f"- gaze-class disagreement: **{_p(ga.get('gaze_class_disagreement_frac'))}**")
    for m, gt in report.get("gaze_temporal", {}).items():
        L.append(f"\n## Gaze time-on-class — {m}\n")
        L.append(f"- valid frames: {_p(gt.get('valid_frame_frac'))}, "
                 f"gaze-on-unknown: {_p(gt.get('gaze_on_unknown_frac'))}, "
                 f"stability: {_p(gt.get('class_temporal_stability'))}")
        L.append("\n| class | seconds |\n|---|--:|")
        for cname, sec in list(gt.get("time_per_class_s", {}).items())[:15]:
            L.append(f"| {cname} | {sec:.1f} |")
    from ..io_utils import atomic_write_text
    atomic_write_text(path, "\n".join(L) + "\n")


def _p(x): return "-" if x is None else f"{100*x:.1f}%"
def _f(x): return "-" if x is None else f"{x:.2f}"
