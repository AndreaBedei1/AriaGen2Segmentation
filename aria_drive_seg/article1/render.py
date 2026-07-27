"""Diagnostic rendering for Article 1 external segmentation."""
from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from ..config import Config
from ..io_utils import read_mask_u16
from ..taxonomy import Taxonomy, auto_color


def _blend(rgb, mask, palette, alpha=.48):
    out = rgb.copy()
    assigned = mask >= 0
    color = palette[np.clip(mask.astype(int), 0, len(palette) - 1)]
    out[assigned] = (alpha * color[assigned] + (1 - alpha) * out[assigned]).astype(np.uint8)
    return out


def _hud(img, title, frame_index, timestamp_s, labels, mask, extra):
    h, w = img.shape[:2]
    panel = np.zeros((h, 410, 3), np.uint8)
    panel[:] = (20, 23, 29)
    canvas = np.hstack([img, panel])
    cv2.rectangle(canvas, (0, 0), (w, 82), (12, 15, 20), -1)
    cv2.putText(canvas, title, (18, 32), cv2.FONT_HERSHEY_SIMPLEX, .75,
                (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(canvas, f"frame {frame_index}  t={timestamp_s:.3f}s", (18, 65),
                cv2.FONT_HERSHEY_SIMPLEX, .55, (210, 230, 255), 1, cv2.LINE_AA)
    y = 36
    for line in extra:
        cv2.putText(canvas, line, (w + 18, y), cv2.FONT_HERSHEY_SIMPLEX, .48,
                    (170, 225, 255), 1, cv2.LINE_AA)
        y += 25
    y += 15
    cv2.putText(canvas, "CLASSES PRESENT", (w + 18, y), cv2.FONT_HERSHEY_SIMPLEX,
                .53, (255, 255, 255), 2, cv2.LINE_AA)
    y += 28
    for cid in np.unique(mask):
        name = labels.get(int(cid), f"id_{cid}")
        cv2.putText(canvas, name, (w + 42, y), cv2.FONT_HERSHEY_SIMPLEX, .45,
                    (255, 255, 255), 1, cv2.LINE_AA)
        y += 23
    return canvas


def run_render_external(input_dir: str, cfg: Config, output_dir: str | None = None,
                        fps: float | None = None) -> dict:
    import imageio.v2 as imageio

    root = Path(input_dir)
    out = Path(output_dir) if output_dir else root / "videos"
    out.mkdir(parents=True, exist_ok=True)
    frames = pd.read_parquet(root / "frames" / "frames.parquet").sort_values("frame_index")
    first_ns = int(frames.capture_timestamp_ns.iloc[0])
    article_tax = Taxonomy.load(cfg.resolve(cfg.get("article1.classes")))
    article_palette = article_tax.palette()
    article_labels = {c.id: c.name for c in article_tax.classes}
    native_labels = {}
    model_cfg = cfg.resolve(cfg.get("oneformer_mapillary.mask2former_id"))
    config_json = model_cfg / "config.json"
    if config_json.exists():
        native_labels = {int(k): v for k, v in
                         json.loads(config_json.read_text()).get("id2label", {}).items()}
    native_palette = np.array([auto_color(i) for i in range(max(65, len(native_labels)))],
                              dtype=np.uint8)
    fps = float(fps or cfg.get("article1.render_fps", 2))
    specs = [
        ("01_mapillary_native.mp4", "Mapillary native", "native_masks",
         native_palette, native_labels),
        ("02_article1_external.mp4", "Article 1 external", "masks",
         article_palette, article_labels),
    ]
    paths = []
    for filename, title, mask_dir, palette, labels in specs:
        path = out / filename
        writer = imageio.get_writer(path, fps=fps, codec="libx264", quality=8,
                                   macro_block_size=8)
        for _, row in frames.iterrows():
            fi = int(row.frame_index)
            bgr = cv2.imread(str(root / row.rectified_path))
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            mask = read_mask_u16(root / "article1_external" / mask_dir /
                                 f"frame_{fi:06d}.png")
            view = _blend(rgb, mask, palette)
            edges = cv2.Canny((mask % 256).astype(np.uint8), 0, 1)
            view[edges > 0] = 255
            meta = json.loads((root / "article1_external" / "metadata" /
                               f"frame_{fi:06d}.json").read_text())
            extra = [
                f"vehicle: {cfg.get('article1.vehicle_type')}",
                f"session: {cfg.get('article1.session_id')}",
                f"coverage: {meta['coverage']:.1%}",
                f"unknown: {meta['unknown_rate']:.1%}",
                f"timing: {meta['total_ms']/1000:.2f}s",
                f"thin lane px: {meta['thin_lane_pixels']}",
                f"thin regulatory px: {meta['thin_regulatory_pixels']}",
            ]
            writer.append_data(_hud(view, title, fi,
                                    (int(row.capture_timestamp_ns) - first_ns) / 1e9,
                                    labels, mask, extra))
        writer.close()
        paths.append(str(path))
    return {"frames": len(frames), "fps": fps, "videos": paths}
