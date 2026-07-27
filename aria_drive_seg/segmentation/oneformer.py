"""Method 2 — dense semantic segmentation on Mapillary Vistas (§8).

Two interchangeable backends over the SAME Mapillary v1.2 65-class taxonomy:
  * hf_mask2former  facebook/mask2former-swin-large-mapillary-vistas-semantic
                    (native HF transformers, no custom ops)  -> env B
  * hf_oneformer    OneFormer DiNAT-L Mapillary (task default; needs natten) -> env C

Both are mask-classification models, so one code path computes the per-pixel
semantic map AND confidence via the class/mask query einsum. The NATIVE id-mask is
always kept; the canonical mask is an ADDITIONAL view (native labels never deleted).
"""
from __future__ import annotations

import time
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import yaml

from ..config import Config
from ..hashing import sha256_file
from ..io_utils import (Manifest, append_jsonl, atomic_write_json,
                        config_fingerprint)
from ..logging_utils import get_logger
from ..taxonomy import Taxonomy
from .base import (Detection, FrameOutput, SegLayout, aggregate_metadata_parquet,
                  iter_frames, write_frame_output)

log = get_logger("oneformer")

ALLOWED_MASK2FORMER_MISSING = {
    "model.pixel_level_module.encoder.swin.layernorm.weight",
    "model.pixel_level_module.encoder.swin.layernorm.bias",
}
ALLOWED_MASK2FORMER_UNEXPECTED_PATTERNS = (
    r"^model\.pixel_level_module\.encoder\.swin\.encoder\.layers\.\d+"
    r"\.blocks\.\d+\.attention\.self\.relative_position_index$",
)


def validate_mask2former_loading_info(info: dict) -> dict:
    """Validate loading diagnostics independently so every failure mode is tested."""
    missing = set(info.get("missing_keys", ()))
    unexpected = list(info.get("unexpected_keys", ()))
    mismatched = list(info.get("mismatched_keys", ()))
    errors = list(info.get("error_msgs", ()))
    if missing != ALLOWED_MASK2FORMER_MISSING:
        raise RuntimeError(f"unapproved Mask2Former missing keys: {sorted(missing)}")
    rejected = [
        key for key in unexpected
        if not any(re.fullmatch(pattern, key)
                   for pattern in ALLOWED_MASK2FORMER_UNEXPECTED_PATTERNS)
    ]
    if rejected:
        raise RuntimeError(
            f"unapproved Mask2Former unexpected keys: {sorted(rejected)}")
    if mismatched:
        raise RuntimeError(f"Mask2Former mismatched shapes: {mismatched}")
    if errors:
        raise RuntimeError(f"Mask2Former loader errors: {errors}")
    return {
        "missing_keys": sorted(missing),
        "unexpected_keys": sorted(unexpected),
        "allowed_unexpected_patterns":
            list(ALLOWED_MASK2FORMER_UNEXPECTED_PATTERNS),
        "unexpected_key_policy":
            "legacy non-trainable relative_position_index buffers only",
        "mismatched_keys": [],
        "error_msgs": [],
        "gate_passed": True,
    }


def load_verified_mask2former(model_path: str):
    """Load the converted Mapillary checkpoint without random backbone parameters.

    Transformers >=5 adds the final SwinModel LayerNorm, but SwinBackbone consumes
    pre-final ``reshaped_hidden_states`` and normalizes them with
    ``hidden_states_norms``. The converted checkpoint therefore has no final norm.
    We set it explicitly to identity and fail closed for every other missing key.
    """
    import torch
    from transformers import Mask2FormerForUniversalSegmentation

    import transformers

    model, info = Mask2FormerForUniversalSegmentation.from_pretrained(
        model_path, output_loading_info=True, local_files_only=True)
    report = validate_mask2former_loading_info(info)
    norm = model.model.pixel_level_module.encoder.swin.layernorm
    with torch.no_grad():
        norm.weight.fill_(1.0)
        norm.bias.zero_()
    checkpoint = Path(model_path)
    weight_files = [
        path for path in (checkpoint / "model.safetensors",
                          checkpoint / "pytorch_model.bin") if path.exists()]
    report.update({
        "checkpoint_files_sha256": {
            path.name: sha256_file(path) for path in weight_files},
        "transformers_version": transformers.__version__,
        "pytorch_version": torch.__version__,
        "final_swin_layernorm_policy":
            "explicit_identity_unused_by_swin_backbone_feature_maps",
    })
    model._aria_loading_info = report
    return model


def _norm(s: str) -> str:
    return " ".join("".join(c if c.isalnum() else " " for c in s.lower()).split())


class MapillaryMapper:
    """native id -> canonical id LUT, built from model id2label + the mapping yaml."""

    def __init__(self, id2label: Dict[int, str], mapping_yaml: str | Path, tax: Taxonomy):
        self.id2label = {int(k): v for k, v in id2label.items()}
        self.tax = tax
        doc = yaml.safe_load(Path(mapping_yaml).read_text())
        raw = doc["mapping"]
        norm_map = {_norm(k): v for k, v in raw.items()}
        n = max(self.id2label) + 1
        self.lut = np.zeros(n, dtype=np.uint16)         # native id -> canonical id
        self.support: Dict[int, str] = {}
        self.canonical_of: Dict[int, str] = {}
        self.unmapped: List[str] = []
        for nid, label in self.id2label.items():
            entry = norm_map.get(_norm(label))
            if entry is None:
                self.unmapped.append(label)
                self.lut[nid] = 0
                self.support[nid] = "unsupported"
                self.canonical_of[nid] = "unknown"
                continue
            cname = entry["canonical"]
            self.lut[nid] = tax.id_of(cname) if cname in tax.by_name else 0
            self.support[nid] = entry.get("support", "supported")
            self.canonical_of[nid] = cname

    def to_canonical(self, native_mask: np.ndarray) -> np.ndarray:
        return self.lut[native_mask.astype(np.int64)].astype(np.uint16)


class OneFormerMapillarySegmenter:
    def __init__(self, cfg: Config, tax: Taxonomy):
        self.cfg = cfg
        self.tax = tax
        self.source = cfg.get("oneformer_mapillary.source", "hf_mask2former")
        self.device = cfg.get("oneformer_mapillary.device", "cuda")
        self.task = cfg.get("oneformer_mapillary.task", "semantic")
        self._model = None
        self._proc = None
        self._mapper: Optional[MapillaryMapper] = None
        self._amp_dtype = None
        self.model_id = None

    def load(self) -> None:
        import torch
        amp = self.cfg.get("hardware.amp", "bf16")
        self._amp_dtype = {"bf16": torch.bfloat16, "fp16": torch.float16,
                           "fp32": torch.float32}.get(amp, torch.bfloat16)
        if self.cfg.get("hardware.tf32", True):
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

        if self.source == "hf_oneformer":
            from transformers import OneFormerForUniversalSegmentation, OneFormerProcessor
            mid = self.cfg.get("oneformer_mapillary.oneformer_id")
            mid = self._resolve_local(mid)
            log.info("loading OneFormer %s", mid)
            self._proc = OneFormerProcessor.from_pretrained(mid)
            self._model = OneFormerForUniversalSegmentation.from_pretrained(mid)
        else:
            from transformers import AutoImageProcessor, Mask2FormerForUniversalSegmentation
            mid = self.cfg.get("oneformer_mapillary.mask2former_id")
            mid = self._resolve_local(mid)
            log.info("loading Mask2Former %s", mid)
            self._proc = AutoImageProcessor.from_pretrained(mid)
            self._model = load_verified_mask2former(mid)
        self.model_id = mid
        self._model = self._model.to(self.device).eval()
        if self.cfg.get("hardware.channels_last", True):
            self._model = self._model.to(memory_format=torch.channels_last)
        id2label = self._model.config.id2label
        self._mapper = MapillaryMapper(
            id2label, self.cfg.resolve(self.cfg.get("project.mapillary_map")), self.tax)
        if self._mapper.unmapped:
            log.warning("unmapped native labels: %s", self._mapper.unmapped)

    def _resolve_local(self, mid: Optional[str]) -> str:
        if mid is None:
            raise ValueError(f"no model id configured for source={self.source}")
        p = self.cfg.resolve(mid)
        # prefer a local pre-downloaded dir for offline
        for cand in (p, self.cfg.resolve(Path("weights") / Path(mid).name)):
            if Path(cand).exists():
                return str(cand)
        return mid

    # ------------------------------------------------------------------ #
    def segment(self, image_rgb: np.ndarray, frame_index: int,
                capture_ts_ns: int) -> FrameOutput:
        import torch
        import torch.nn.functional as F
        from PIL import Image

        h, w = image_rgb.shape[:2]
        pil = Image.fromarray(image_rgb)
        t0 = time.time()
        if self.source == "hf_oneformer":
            inputs = self._proc(images=pil, task_inputs=[self.task], return_tensors="pt")
        else:
            inputs = self._proc(images=pil, return_tensors="pt")
        inputs = {k: (v.to(self.device) if hasattr(v, "to") else v) for k, v in inputs.items()}

        with torch.inference_mode(), torch.autocast(self.device, dtype=self._amp_dtype,
                                                    enabled=self.device == "cuda"):
            outputs = self._model(**inputs)
        # Post-process in float32 OUTSIDE autocast (numpy has no bfloat16).
        # mask-classification -> per-pixel semantic prob (works for M2F & OneFormer).
        class_q = outputs.class_queries_logits.float()          # (B,Q,C+1)
        masks_q = outputs.masks_queries_logits.float()          # (B,Q,h',w')
        masks_up = F.interpolate(masks_q, size=(h, w), mode="bilinear", align_corners=False)
        class_prob = class_q.softmax(dim=-1)[..., :-1]          # drop no-object
        mask_prob = masks_up.sigmoid()
        semseg = torch.einsum("bqc,bqhw->bchw", class_prob, mask_prob)[0]  # (C,h,w)
        conf, ids = semseg.max(dim=0)
        native = ids.to("cpu").numpy().astype(np.uint16)
        confidence = conf.float().to("cpu").numpy().astype(np.float32)
        # normalise confidence to [0,1] (semseg sums over queries, can exceed 1)
        confidence = np.clip(confidence, 0.0, 1.0)
        canonical = self._mapper.to_canonical(native)
        total = (time.time() - t0) * 1e3

        dets = self._class_summary(native, canonical, confidence)
        return FrameOutput(
            frame_index=frame_index, capture_timestamp_ns=capture_ts_ns,
            method="oneformer_mapillary", image_size=(h, w),
            canonical_mask=canonical, confidence=confidence, native_mask=native,
            detections=dets,
            timings_ms={"total": round(total, 1)},
            extra={"coverage": float((canonical > 0).mean()),
                   "model": str(self.model_id), "source": self.source,
                   "input_size": [int(inputs["pixel_values"].shape[-1]),
                                  int(inputs["pixel_values"].shape[-2])]},
        )

    def _class_summary(self, native, canonical, confidence) -> List[Detection]:
        dets: List[Detection] = []
        for nid in np.unique(native):
            m = native == nid
            area = int(m.sum())
            cname = self._mapper.canonical_of.get(int(nid), "unknown")
            cid = int(self._mapper.lut[int(nid)])
            ys, xs = np.where(m)
            box = (float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max()))
            dets.append(Detection(
                canonical_id=cid, canonical_name=cname, box=box,
                phrase=self._mapper.id2label.get(int(nid), str(nid)),
                combined_score=float(confidence[m].mean()), area_px=area))
        return dets


def run_oneformer(input_dir: str, cfg: Config, resume: bool = True,
                  force: bool = False) -> int:
    import cv2
    tax = Taxonomy.load(cfg.resolve(cfg.get("project.classes")))
    layout = SegLayout(input_dir, "oneformer_mapillary")
    layout.ensure(native=True)
    frames = iter_frames(input_dir)

    fp = config_fingerprint("oneformer_mapillary", cfg.get("oneformer_mapillary"),
                            Path(cfg.resolve(cfg.get("project.mapillary_map"))).read_text(),
                            cfg.get("rectify"))
    manifest = Manifest.load_or_new(layout.root / "manifest.json", "oneformer_mapillary", fp,
                                    meta={"num_frames": len(frames),
                                          "source": cfg.get("oneformer_mapillary.source")})
    if force:
        manifest.done.clear()

    seg = OneFormerMapillarySegmenter(cfg, tax)
    seg.load()
    manifest.meta["model"] = str(seg.model_id)

    n_done = n_err = 0
    for k, ref in enumerate(frames):
        i = ref.frame_index
        if resume and not force and manifest.is_done(i) and layout.is_frame_done(i, need_native=True):
            n_done += 1
            continue
        try:
            bgr = cv2.imread(str(ref.rectified_path), cv2.IMREAD_COLOR)
            if bgr is None:
                raise IOError(f"cannot read {ref.rectified_path}")
            out = _seg_oom(seg, np.ascontiguousarray(bgr[:, :, ::-1]), i, ref.capture_timestamp_ns)
            write_frame_output(layout, out, write_confidence=bool(cfg.get("segmentation.write_confidence", True)))
            manifest.mark(i, {"coverage": out.extra.get("coverage"),
                              "total_ms": out.timings_ms.get("total")})
            n_done += 1
            if (k + 1) % 10 == 0:
                manifest.save()
                log.info("  %d/%d frames (%.0fms cov=%.2f)", k + 1, len(frames),
                         out.timings_ms.get("total", 0), out.extra.get("coverage", 0))
        except Exception as e:
            n_err += 1
            append_jsonl(Path(input_dir) / "logs" / "oneformer_errors.jsonl",
                         {"frame_index": i, "error": repr(e)})
            log.warning("frame %d failed: %s", i, e)
    manifest.save()
    parq = aggregate_metadata_parquet(layout)
    peak_vram = None
    try:
        import torch
        if torch.cuda.is_available():
            peak_vram = round(torch.cuda.max_memory_allocated() / 1e6, 1)
    except Exception:
        pass
    atomic_write_json(layout.root / "summary.json",
                      {"method": "oneformer_mapillary", "model": str(seg.model_id),
                       "source": seg.source, "frames_done": n_done, "errors": n_err,
                       "peak_vram_mb": peak_vram,
                       "unmapped_native_labels": seg._mapper.unmapped if seg._mapper else [],
                       "metadata_parquet": str(parq) if parq else None})
    log.info("oneformer_mapillary done: %d frames, %d errors -> %s", n_done, n_err, layout.root)
    return 0


def _seg_oom(seg, img, i, ts):
    import torch
    try:
        return seg.segment(img, i, ts)
    except torch.cuda.OutOfMemoryError:
        log.warning("CUDA OOM on frame %d; retrying after empty_cache", i)
        torch.cuda.empty_cache()
        return seg.segment(img, i, ts)
