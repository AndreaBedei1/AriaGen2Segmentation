"""Method-vs-method agreement + no-ground-truth metrics (§17.1).

IMPORTANT: agreement/IoU BETWEEN the two methods is NOT accuracy — it only measures
consistency on the classes both can produce. Real accuracy needs ground truth (§17.2).
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from ..config import Config
from ..logging_utils import get_logger
from ..taxonomy import Taxonomy

log = get_logger("compare")


def per_method_metrics(input_dir: str, method: str, tax: Taxonomy) -> Dict[str, Any]:
    import pandas as pd
    from ..io_utils import read_mask_u16
    from ..segmentation.base import SegLayout

    layout = SegLayout(input_dir, method)
    masks = sorted(layout.canonical.glob("frame_*.png"))
    if not masks:
        return {"method": method, "frames": 0}
    cov, unk, ncls = [], [], []
    for mp in masks:
        m = read_mask_u16(mp)
        assigned = m > 0
        cov.append(float(assigned.mean()))
        unk.append(float((m == 0).mean()))
        ncls.append(int(len(np.unique(m)) - (1 if (m == 0).any() else 0)))
    res = {
        "method": method, "frames": len(masks),
        "coverage_mean": float(np.mean(cov)),
        "unknown_frac_mean": float(np.mean(unk)),
        "classes_per_frame_mean": float(np.mean(ncls)),
    }
    meta_pq = layout.root / "metadata.parquet"
    if meta_pq.exists():
        mdf = pd.read_parquet(meta_pq)
        if "total_ms" in mdf and mdf["total_ms"].notna().any():
            tms = float(mdf["total_ms"].dropna().mean())
            res["mean_inference_ms"] = tms
            res["fps"] = 1000.0 / tms if tms > 0 else None
    return res


def method_agreement(input_dir: str, tax: Taxonomy,
                     m1: str = "grounded_sam2", m2: str = "oneformer_mapillary") -> Dict[str, Any]:
    import pandas as pd
    from ..io_utils import read_mask_u16
    from ..segmentation.base import SegLayout

    L1, L2 = SegLayout(input_dir, m1), SegLayout(input_dir, m2)
    eval_ids = set(tax.eval_ids())
    frames = sorted(set(p.stem for p in L1.canonical.glob("frame_*.png"))
                    & set(p.stem for p in L2.canonical.glob("frame_*.png")))
    if not frames:
        return {"frames": 0}

    inter = {c: 0 for c in eval_ids}
    union = {c: 0 for c in eval_ids}
    agree_px = 0
    common_px = 0
    per_frame = []
    for stem in frames:
        a = read_mask_u16(L1.canonical / f"{stem}.png")
        b = read_mask_u16(L2.canonical / f"{stem}.png")
        if a.shape != b.shape:
            continue
        # restrict to pixels where BOTH assigned a common eval class
        a_eval = np.isin(a, list(eval_ids))
        b_eval = np.isin(b, list(eval_ids))
        both = a_eval & b_eval
        n = int(both.sum())
        agr = int((a[both] == b[both]).sum())
        common_px += n
        agree_px += agr
        per_frame.append({"frame": stem, "common_px": n,
                          "agreement": agr / n if n else None})
        for c in eval_ids:
            ac = a == c
            bc = b == c
            inter[c] += int(np.logical_and(ac, bc).sum())
            union[c] += int(np.logical_or(ac, bc).sum())

    iou = {tax.name_of(c): (inter[c] / union[c]) if union[c] else None for c in eval_ids}
    miou = np.mean([v for v in iou.values() if v is not None]) if iou else None
    return {
        "frames": len(frames),
        "pixelwise_agreement_common": agree_px / common_px if common_px else None,
        "mean_iou_between_methods": float(miou) if miou is not None else None,
        "per_class_iou_between_methods": iou,
        "note": "agreement/IoU between methods is consistency, NOT accuracy (needs GT, §17.2)",
    }


def gaze_agreement(input_dir: str, tax: Taxonomy,
                   m1: str = "grounded_sam2", m2: str = "oneformer_mapillary") -> Dict[str, Any]:
    import pandas as pd
    base = Path(input_dir) / "comparison" / "gaze"
    p1, p2 = base / f"gaze_labels_{m1}.parquet", base / f"gaze_labels_{m2}.parquet"
    if not (p1.exists() and p2.exists()):
        return {"frames": 0}
    d1 = pd.read_parquet(p1).set_index("frame_index")
    d2 = pd.read_parquet(p2).set_index("frame_index")
    common = d1.index.intersection(d2.index)
    eval_names = {tax.name_of(c) for c in tax.eval_ids()}

    rows = []
    agree = both_valid = 0
    for fi in common:
        r1, r2 = d1.loc[fi], d2.loc[fi]
        c1, c2 = r1["class_at_pixel"], r2["class_at_pixel"]
        bv = bool(r1["gaze_valid"]) and bool(r2["gaze_valid"])
        # agreement only meaningful on the common eval taxonomy
        comparable = bv and (c1 in eval_names or c2 in eval_names)
        a = comparable and (c1 == c2)
        if bv:
            both_valid += 1
        if a:
            agree += 1
        rows.append({"frame_index": int(fi), f"{m1}_class": c1, f"{m2}_class": c2,
                     "both_valid": bv, "agree": bool(a), "comparable": bool(comparable)})
    outp = base / "gaze_method_agreement.parquet"
    pd.DataFrame(rows).to_parquet(outp, index=False)
    return {
        "frames": int(len(common)),
        "both_valid": int(both_valid),
        "gaze_class_agreement_frac": (agree / both_valid) if both_valid else None,
        "gaze_class_disagreement_frac": (1 - agree / both_valid) if both_valid else None,
        "parquet": str(outp),
    }


def gaze_temporal_distribution(input_dir: str, method: str, tax: Taxonomy) -> Dict[str, Any]:
    """Total observed time per gaze class + valid/unknown fractions (§10)."""
    import pandas as pd
    p = Path(input_dir) / "comparison" / "gaze" / f"gaze_labels_{method}.parquet"
    if not p.exists():
        return {}
    df = pd.read_parquet(p).sort_values("frame_index")
    ts = df["capture_timestamp_ns"].to_numpy()
    dt = np.diff(ts, append=ts[-1] + int(np.median(np.diff(ts))) if len(ts) > 1 else ts) / 1e9
    time_per_class: Dict[str, float] = {}
    for cname, d in zip(df["class_at_pixel"].fillna("unknown"), dt):
        time_per_class[cname] = time_per_class.get(cname, 0.0) + float(d)
    valid_frac = float(df["gaze_valid"].mean())
    unknown_frac = float((df["class_at_pixel"].fillna("unknown") == "unknown").mean())
    # stability: fraction of consecutive valid frames with the same class
    seq = df["class_at_pixel"].fillna("unknown").to_numpy()
    stable = float(np.mean(seq[1:] == seq[:-1])) if len(seq) > 1 else None
    return {
        "method": method,
        "valid_frame_frac": valid_frac,
        "gaze_on_unknown_frac": unknown_frac,
        "time_per_class_s": dict(sorted(time_per_class.items(), key=lambda kv: -kv[1])),
        "class_temporal_stability": stable,
    }
