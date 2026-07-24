"""Shared segmentation output schema + resumable frame iteration (§12, §14).

Both methods write the SAME on-disk layout so downstream analysis/render are
method-agnostic:

    <input>/<method>/
        canonical_masks/frame_XXXXXX.png   uint16 canonical id-mask (lossless)
        confidence/frame_XXXXXX.png        uint8 per-pixel confidence (0-255)
        native/frame_XXXXXX.png            (method 2) native id-mask, lossless
        metadata/frame_XXXXXX.json         detections/scores/timings
        instances/frame_XXXXXX.npz         (optional) per-instance masks
        manifest.json                      resumability + config fingerprint
        metadata.parquet                   aggregated per-frame metadata
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple

import numpy as np

from ..io_utils import (Manifest, atomic_write_bytes, atomic_write_json,
                        file_ok, write_mask_u16)


@dataclass
class Detection:
    canonical_id: int
    canonical_name: str
    box: Tuple[float, float, float, float]   # x0,y0,x1,y1 in rectified pixels
    phrase: str = ""                          # canonical name (kept for back-compat)
    gdino_score: float = 0.0
    sam_iou: float = 0.0
    combined_score: float = 0.0
    area_px: int = 0
    priority: int = 0
    native_phrase: str = ""                   # raw phrase returned by Grounding DINO
    map_confidence: float = 0.0               # phrase->class mapping confidence
    provenance: str = "full_frame"            # full_frame | windshield_crop | tile:<i> | roi:<name>
    layer: str = "canonical"                  # canonical | exterior | cockpit | transparent | mirror

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["box"] = [float(x) for x in self.box]
        return d


@dataclass
class FrameOutput:
    frame_index: int
    capture_timestamp_ns: int
    method: str
    image_size: Tuple[int, int]              # (h, w)
    canonical_mask: np.ndarray               # uint16 (h,w)
    confidence: Optional[np.ndarray] = None  # float32 (h,w) in [0,1]
    native_mask: Optional[np.ndarray] = None # uint16 (h,w) native ids (method 2)
    detections: List[Detection] = field(default_factory=list)
    timings_ms: Dict[str, float] = field(default_factory=dict)
    extra: Dict[str, Any] = field(default_factory=dict)


class SegLayout:
    def __init__(self, input_dir: str | Path, method: str):
        self.root = Path(input_dir) / method
        self.method = method
        self.canonical = self.root / "canonical_masks"
        self.confidence = self.root / "confidence"
        self.native = self.root / "native"
        self.metadata = self.root / "metadata"
        self.instances = self.root / "instances"

    def ensure(self, native: bool = False, instances: bool = False) -> None:
        for d in (self.canonical, self.confidence, self.metadata):
            d.mkdir(parents=True, exist_ok=True)
        if native:
            self.native.mkdir(parents=True, exist_ok=True)
        if instances:
            self.instances.mkdir(parents=True, exist_ok=True)

    def name(self, frame_index: int) -> str:
        return f"frame_{frame_index:06d}"

    def canonical_path(self, i: int) -> Path:
        return self.canonical / f"{self.name(i)}.png"

    def confidence_path(self, i: int) -> Path:
        return self.confidence / f"{self.name(i)}.png"

    def native_path(self, i: int) -> Path:
        return self.native / f"{self.name(i)}.png"

    def metadata_path(self, i: int) -> Path:
        return self.metadata / f"{self.name(i)}.json"

    def is_frame_done(self, i: int, need_native: bool = False) -> bool:
        ok = file_ok(self.canonical_path(i)) and file_ok(self.metadata_path(i))
        if need_native:
            ok = ok and file_ok(self.native_path(i))
        return ok


def write_frame_output(layout: SegLayout, out: FrameOutput,
                       write_confidence: bool = True,
                       write_instances: bool = False) -> Dict[str, Any]:
    """Persist one frame's canonical mask (+optional confidence/native/instances)
    and metadata, all atomically."""
    write_mask_u16(layout.canonical_path(out.frame_index), out.canonical_mask)
    if out.native_mask is not None:
        write_mask_u16(layout.native_path(out.frame_index), out.native_mask)
    if write_confidence and out.confidence is not None:
        import cv2
        conf_u8 = np.clip(out.confidence * 255.0, 0, 255).astype(np.uint8)
        ok, buf = cv2.imencode(".png", conf_u8)
        if ok:
            atomic_write_bytes(layout.confidence_path(out.frame_index), buf.tobytes())
    meta = {
        "frame_index": out.frame_index,
        "capture_timestamp_ns": out.capture_timestamp_ns,
        "method": out.method,
        "image_size": list(out.image_size),
        "timings_ms": out.timings_ms,
        "num_detections": len(out.detections),
        "detections": [d.to_dict() for d in out.detections],
        **out.extra,
    }
    atomic_write_json(layout.metadata_path(out.frame_index), meta)
    return meta


@dataclass
class FrameRef:
    frame_index: int
    capture_timestamp_ns: int
    rectified_path: Path
    original_path: Path


def iter_frames(input_dir: str | Path, use_rectified: bool = True) -> List[FrameRef]:
    """Read the extract stage's frame index (Parquet) and return frame refs."""
    import pandas as pd
    root = Path(input_dir)
    df = pd.read_parquet(root / "frames" / "frames.parquet")
    refs = []
    for _, r in df.iterrows():
        rp = root / r["rectified_path"] if r.get("rectified_path") else None
        op = root / r["original_path"]
        refs.append(FrameRef(int(r["frame_index"]), int(r["capture_timestamp_ns"]),
                             rp if rp else op, op))
    return refs


def aggregate_metadata_parquet(layout: SegLayout) -> Optional[Path]:
    """Collect per-frame metadata json into a single parquet for analysis (§12)."""
    import json
    import pandas as pd
    rows = []
    for jp in sorted(layout.metadata.glob("frame_*.json")):
        m = json.loads(jp.read_text())
        rows.append({
            "frame_index": m["frame_index"],
            "capture_timestamp_ns": m["capture_timestamp_ns"],
            "num_detections": m.get("num_detections", 0),
            "total_ms": m.get("timings_ms", {}).get("total"),
            "classes": ",".join(sorted({d["canonical_name"] for d in m.get("detections", [])})),
        })
    if not rows:
        return None
    df = pd.DataFrame(rows).sort_values("frame_index")
    outp = layout.root / "metadata.parquet"
    tmp = layout.root / ".tmp_metadata.parquet"
    df.to_parquet(tmp, index=False)
    tmp.replace(outp)
    return outp
