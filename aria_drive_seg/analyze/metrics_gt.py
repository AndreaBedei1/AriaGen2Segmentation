"""Ground-truth evaluation engine (§17.2).

Computes the full metric suite when human annotations exist. If no GT is present it
does NOTHING and reports that infra is ready — it never invents accuracy numbers.

Inputs:
  gt_dir       directory of GT canonical uint16 id-masks named frame_XXXXXX.png
  method masks <input>/<method>/canonical_masks/frame_XXXXXX.png
Only the common eval taxonomy (+unknown) is scored, so both methods are comparable.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from ..logging_utils import get_logger
from ..taxonomy import Taxonomy

log = get_logger("metrics_gt")


def _confusion(gt: np.ndarray, pred: np.ndarray, ids: List[int]) -> np.ndarray:
    idmap = {c: i for i, c in enumerate(ids)}
    n = len(ids)
    cm = np.zeros((n, n), dtype=np.int64)
    g = np.vectorize(lambda x: idmap.get(int(x), -1))(gt)
    p = np.vectorize(lambda x: idmap.get(int(x), -1))(pred)
    valid = (g >= 0) & (p >= 0)
    np.add.at(cm, (g[valid], p[valid]), 1)
    return cm


def _boundary(mask: np.ndarray, cid: int, width: int = 3) -> np.ndarray:
    import cv2
    m = (mask == cid).astype(np.uint8)
    if m.sum() == 0:
        return np.zeros_like(m, bool)
    er = cv2.erode(m, np.ones((width, width), np.uint8))
    return (m - er).astype(bool)


def boundary_f1(gt: np.ndarray, pred: np.ndarray, ids: List[int], tol: int = 3) -> Dict[int, float]:
    import cv2
    out = {}
    for c in ids:
        gb = _boundary(gt, c, tol)
        pb = _boundary(pred, c, tol)
        if gb.sum() == 0 and pb.sum() == 0:
            out[c] = float("nan"); continue
        # dilate for tolerance matching
        k = np.ones((2 * tol + 1, 2 * tol + 1), np.uint8)
        gb_d = cv2.dilate(gb.astype(np.uint8), k).astype(bool)
        pb_d = cv2.dilate(pb.astype(np.uint8), k).astype(bool)
        prec = (pb & gb_d).sum() / pb.sum() if pb.sum() else 0.0
        rec = (gb & pb_d).sum() / gb.sum() if gb.sum() else 0.0
        out[c] = float(2 * prec * rec / (prec + rec)) if (prec + rec) else 0.0
    return out


def evaluate_method(input_dir: str, method: str, gt_dir: str, tax: Taxonomy,
                    cfg=None) -> Dict[str, Any]:
    from ..io_utils import read_mask_u16
    from ..segmentation.base import SegLayout

    layout = SegLayout(input_dir, method)
    gtp = Path(gt_dir)
    gts = sorted(gtp.glob("frame_*.png"))
    if not gts:
        return {"method": method, "gt_frames": 0, "status": "no ground truth present"}

    ids = [0] + tax.eval_ids()  # unknown + common eval classes
    n = len(ids)
    cm = np.zeros((n, n), dtype=np.int64)
    bf1_acc: Dict[int, List[float]] = {c: [] for c in ids}
    frames_used = 0
    for g in gts:
        pr = layout.canonical / g.name
        if not pr.exists():
            continue
        gt = read_mask_u16(g)
        pred = read_mask_u16(pr)
        if gt.shape != pred.shape:
            continue
        cm += _confusion(gt, pred, ids)
        for c, v in boundary_f1(gt, pred, ids).items():
            if not np.isnan(v):
                bf1_acc[c].append(v)
        frames_used += 1

    if frames_used == 0:
        return {"method": method, "gt_frames": 0, "status": "no matching predictions"}

    # IoU / precision / recall per class
    iou, prec, rec = {}, {}, {}
    for i, c in enumerate(ids):
        tp = cm[i, i]
        fn = cm[i, :].sum() - tp
        fp = cm[:, i].sum() - tp
        iou[c] = tp / (tp + fp + fn) if (tp + fp + fn) else float("nan")
        prec[c] = tp / (tp + fp) if (tp + fp) else float("nan")
        rec[c] = tp / (tp + fn) if (tp + fn) else float("nan")

    eval_only = tax.eval_ids()
    miou = np.nanmean([iou[c] for c in eval_only])
    pix_acc = np.trace(cm) / cm.sum() if cm.sum() else float("nan")
    mean_cls_acc = np.nanmean([rec[c] for c in eval_only])
    # accuracy conditioned on assigned pixels (exclude pred==unknown)
    unk_i = ids.index(0)
    assigned = cm.sum() - cm[:, unk_i].sum()
    correct_assigned = np.trace(cm) - cm[unk_i, unk_i]
    acc_assigned = correct_assigned / assigned if assigned else float("nan")
    # unknown-as-error accuracy already reflected in pix_acc (unknown is a row/col)
    named = lambda d: {tax.name_of(c): (None if np.isnan(v) else round(float(v), 4)) for c, v in d.items()}
    bf1 = {tax.name_of(c): (round(float(np.mean(v)), 4) if v else None) for c, v in bf1_acc.items()}

    return {
        "method": method, "gt_frames": frames_used,
        "mIoU_eval": round(float(miou), 4),
        "pixel_accuracy": round(float(pix_acc), 4),
        "mean_class_accuracy": round(float(mean_cls_acc), 4),
        "accuracy_on_assigned": round(float(acc_assigned), 4),
        "per_class_iou": named(iou),
        "per_class_precision": named(prec),
        "per_class_recall": named(rec),
        "boundary_f1": bf1,
        "note": "unknown is scored as its own class, so unknown predictions count as errors "
                "against the GT (§17.2). accuracy_on_assigned excludes pred==unknown.",
    }


def gaze_class_accuracy(input_dir: str, method: str, gt_dir: str, tax: Taxonomy) -> Dict[str, Any]:
    """Class-at-gaze accuracy: predicted vs GT at the gaze pixel (§17.2)."""
    import pandas as pd
    from ..io_utils import read_mask_u16
    gz_p = Path(input_dir) / "gaze" / "aligned_gaze.parquet"
    if not gz_p.exists():
        return {}
    gz = pd.read_parquet(gz_p).set_index("frame_index")
    gtp = Path(gt_dir)
    total = correct = 0
    for g in sorted(gtp.glob("frame_*.png")):
        fi = int(g.stem.split("_")[1])
        if fi not in gz.index:
            continue
        row = gz.loc[fi]
        if not bool(row["valid"]) or row.get("rect_u") is None:
            continue
        gt = read_mask_u16(g)
        h, w = gt.shape
        u, v = int(round(row["rect_u"])), int(round(row["rect_v"]))
        if not (0 <= u < w and 0 <= v < h):
            continue
        from ..segmentation.base import SegLayout
        pr = SegLayout(input_dir, method).canonical / g.name
        if not pr.exists():
            continue
        pred = read_mask_u16(pr)
        total += 1
        if int(pred[v, u]) == int(gt[v, u]):
            correct += 1
    return {"gaze_frames_scored": total,
            "gaze_class_accuracy": round(correct / total, 4) if total else None}


def gaussian_gaze_weighted_accuracy(input_dir: str, method: str, gt_dir: str,
                                    tax: Taxonomy, sigma_px: float = 60.0) -> Dict[str, Any]:
    """Per-pixel accuracy weighted by a 2D gaussian centred on the gaze (§17.2).

    Emphasises correctness where the driver is actually looking rather than treating
    every pixel equally."""
    import pandas as pd
    from ..io_utils import read_mask_u16
    from ..segmentation.base import SegLayout

    gz_p = Path(input_dir) / "gaze" / "aligned_gaze.parquet"
    if not gz_p.exists():
        return {}
    gz = pd.read_parquet(gz_p).set_index("frame_index")
    layout = SegLayout(input_dir, method)
    num = den = 0.0
    frames = 0
    for g in sorted(Path(gt_dir).glob("frame_*.png")):
        fi = int(g.stem.split("_")[1])
        if fi not in gz.index:
            continue
        row = gz.loc[fi]
        if not bool(row["valid"]) or row.get("rect_u") is None:
            continue
        pr = layout.canonical / g.name
        if not pr.exists():
            continue
        gt = read_mask_u16(g)
        pred = read_mask_u16(pr)
        if gt.shape != pred.shape:
            continue
        h, w = gt.shape
        u, v = float(row["rect_u"]), float(row["rect_v"])
        ys, xs = np.mgrid[0:h, 0:w]
        weight = np.exp(-(((xs - u) ** 2 + (ys - v) ** 2) / (2 * sigma_px * sigma_px)))
        correct = (pred == gt).astype(np.float64)
        num += float((weight * correct).sum())
        den += float(weight.sum())
        frames += 1
    return {"gaussian_gaze_weighted_accuracy": round(num / den, 4) if den else None,
            "gaussian_sigma_px": sigma_px, "frames_scored": frames}
