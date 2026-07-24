"""`render` (§11): build overlay + comparison + gaze videos and hi-res diagnostics.

Overlays are drawn on COPIES; the inference frame is never mutated. The shared
deterministic palette makes the two methods' colours directly comparable."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from ..config import Config
from ..logging_utils import get_logger
from ..taxonomy import Taxonomy
from . import overlays as O

log = get_logger("render")

METHODS = ["grounded_sam2", "oneformer_mapillary"]


class _Video:
    def __init__(self, path: Path, fps: int):
        import imageio
        path.parent.mkdir(parents=True, exist_ok=True)
        self.w = imageio.get_writer(str(path), fps=fps, codec="libx264",
                                    quality=7, macro_block_size=8)
        self.path = path

    def add(self, rgb: np.ndarray):
        self.w.append_data(rgb)

    def close(self):
        self.w.close()


def run_render(input_dir: str, cfg: Config, what: str = "all") -> Dict[str, Any]:
    import pandas as pd
    import cv2
    from ..io_utils import read_mask_u16
    from ..segmentation.base import SegLayout

    out = Path(input_dir)
    tax = Taxonomy.load(cfg.resolve(cfg.get("project.classes")))
    fps = int(cfg.get("render.fps", 10))
    alpha = float(cfg.get("render.overlay_alpha", 0.5))
    fdf = pd.read_parquet(out / "frames" / "frames.parquet").sort_values("frame_index")
    t0 = int(fdf["capture_timestamp_ns"].min())

    present = [m for m in METHODS if (out / m / "canonical_masks").exists()
               and any((out / m / "canonical_masks").glob("*.png"))]
    gz_path = out / "gaze" / "aligned_gaze.parquet"
    gz = pd.read_parquet(gz_path).set_index("frame_index") if gz_path.exists() else None
    layouts = {m: SegLayout(input_dir, m) for m in present}

    vdir = out / "videos"
    writers: Dict[str, _Video] = {}

    def W(name: str) -> _Video:
        if name not in writers:
            writers[name] = _Video(vdir / f"{name}.mp4", fps)
        return writers[name]

    diag_dir = out / "comparison" / "figures"
    diag_dir.mkdir(parents=True, exist_ok=True)
    diag_idxs = set(np.linspace(0, len(fdf) - 1, min(6, len(fdf)), dtype=int).tolist())

    n = 0
    for pos, (_, row) in enumerate(fdf.iterrows()):
        fi = int(row["frame_index"])
        rp = out / row["rectified_path"] if row.get("rectified_path") else out / row["original_path"]
        bgr = cv2.imread(str(rp), cv2.IMREAD_COLOR)
        if bgr is None:
            continue
        rect = np.ascontiguousarray(bgr[:, :, ::-1])
        tsec = (int(row["capture_timestamp_ns"]) - t0) / 1e9
        g = gz.loc[fi] if (gz is not None and fi in gz.index) else None
        gu = g.get("rect_u") if g is not None else None
        gv = g.get("rect_v") if g is not None else None
        gvalid = bool(g["valid"]) if g is not None else False
        gdt = g.get("dt_ms") if g is not None else None
        gdepth = g.get("depth") if g is not None else None
        gdsrc = g.get("depth_source") if g is not None else None

        W("rgb_rectified").add(rect)

        panels_for_diag: Dict[str, np.ndarray] = {"rectified": rect}
        method_panels: List[np.ndarray] = []
        gaze_classes: Dict[str, str] = {}
        for m in present:
            mask = read_mask_u16(layouts[m].canonical_path(fi)) if layouts[m].canonical_path(fi).exists() else None
            if mask is None:
                continue
            base = O.blend_mask(rect, mask, tax, alpha)
            present_ids = [int(x) for x in np.unique(mask) if x != 0]
            base = O.draw_legend(base, present_ids, tax)
            cid, cname = O.class_at_gaze(mask, gu, gv, tax, 0)
            gaze_classes[m] = cname
            plain = O.draw_hud(base.copy(), [f"{m}  f={fi}  t={tsec:.1f}s"], font_scale=0.7)
            W(f"seg_{m}").add(plain)
            # gaze variant
            gv_img = O.draw_gaze(base.copy(), gu, gv, gvalid)
            hud = [f"{m}  f={fi}  t={tsec:.1f}s",
                   f"gaze={cname} valid={gvalid} dt={_f(gdt)}ms",
                   f"depth={_f(gdepth)}m({gdsrc})"]
            gv_img = O.draw_hud(gv_img, hud, font_scale=0.7)
            W(f"seg_{m}_gaze").add(gv_img)
            method_panels.append(gv_img)
            panels_for_diag[m] = gv_img

        # side-by-side comparison
        if len(method_panels) >= 1:
            combo = O.hstack_pad([rect] + method_panels)
            W("comparison_sidebyside").add(combo)
        # both + agreement HUD
        if len(present) == 2 and all(m in gaze_classes for m in present):
            agree = gaze_classes[present[0]] == gaze_classes[present[1]]
            ag = O.hstack_pad(method_panels)
            tag = "AGREE" if agree else "DISAGREE"
            ag = O.draw_hud(ag, [f"gaze: {present[0]}={gaze_classes[present[0]]} | "
                                 f"{present[1]}={gaze_classes[present[1]]}  -> {tag}"],
                            org=(12, ag.shape[0] - 40), font_scale=0.8)
            W("comparison_gaze_agreement").add(ag)

        if pos in diag_idxs:
            combo = O.hstack_pad(list(panels_for_diag.values()))
            cv2.imwrite(str(diag_dir / f"diag_frame_{fi:06d}.png"), combo[:, :, ::-1])
        n += 1

    for v in writers.values():
        v.close()
    log.info("render done: %d frames -> %s (%d videos)", n, vdir, len(writers))
    return {"frames": n, "videos": [str(v.path) for v in writers.values()],
            "figures_dir": str(diag_dir)}


def _f(x):
    try:
        return f"{float(x):.1f}"
    except (TypeError, ValueError):
        return "-"
