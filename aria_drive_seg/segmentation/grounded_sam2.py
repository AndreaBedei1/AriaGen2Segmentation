"""Method 1 — open-vocabulary prompted segmentation: Grounding DINO + SAM 2.1 (§7).

Grounding DINO (HF transformers) proposes boxes from configurable text prompts (now
including variants + synonyms, see prompt_engine.py); SAM 2.1 turns each surviving box
into a mask; masks are filtered (per-class thresholds, geometric filters, max_instances),
scored, and composited by priority (overlap.py). This is NOT an exhaustive semantic
partition — unassigned pixels remain `unknown`.

Phase-1 correctness fixes vs the first version:
  * variants/synonyms are actually sent to GDINO and deduplicated;
  * grouped calls use a permissive CANDIDATE threshold, then each detection is re-filtered
    by its OWN class box_threshold + mapping confidence;
  * max_instances and a full set of geometric filters are applied;
  * robust deterministic phrase->class mapping; the native GDINO phrase is stored.
"""
from __future__ import annotations

import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import yaml

from ..logging_utils import get_logger
from ..taxonomy import Taxonomy
from .base import Detection, FrameOutput
from .layers import LAYERS, layer_of_id
from .overlap import Instance, class_aware_nms, resolve_overlaps
from .prompt_engine import (PhraseMapper, PromptSpec, box_geom_ok, build_specs,
                            group_captions)

log = get_logger("grounded_sam2")


class GroundedSAM2Segmenter:
    def __init__(self, cfg, taxonomy: Taxonomy):
        self.cfg = cfg
        self.tax = taxonomy
        self.device = cfg.get("grounded_sam2.device", "cuda")
        self._load_prompts()
        self._gdino = None
        self._gproc = None
        self._sam = None
        self._amp_dtype = None
        # global accumulators (per class) for calibration/statistics
        self.stats_accepted: Counter = Counter()
        self.stats_rejected: Counter = Counter()          # keyed by reason
        self.stats_rejected_by_class: Counter = Counter()

    # ------------------------------------------------------------------ #
    def _load_prompts(self) -> None:
        path = self.cfg.resolve(self.cfg.get("project.grounded_prompts"))
        doc = yaml.safe_load(Path(path).read_text())
        self.prompt_strategy = doc.get("prompt_strategy", "grouped")
        self.specs: Dict[str, PromptSpec] = build_specs(doc, set(self.tax.by_name))
        negatives: List[str] = []
        for s in self.specs.values():
            negatives.extend(s.negative_contexts)
        self.mapper = PhraseMapper(self.specs, negatives=negatives)

    def _build_query_groups(self) -> List[List[str]]:
        if self.prompt_strategy == "individual":
            return [[n] for n in self.specs]
        if self.prompt_strategy == "concatenated":
            return [list(self.specs.keys())]
        groups: Dict[str, List[str]] = defaultdict(list)
        solos: List[List[str]] = []
        for n, s in self.specs.items():
            if s.solo:
                solos.append([n])
            else:
                groups[s.group_tag].append(n)
        return list(groups.values()) + solos

    # ------------------------------------------------------------------ #
    def load(self) -> None:
        import torch
        from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

        amp = self.cfg.get("hardware.amp", "bf16")
        self._amp_dtype = {"bf16": torch.bfloat16, "fp16": torch.float16,
                           "fp32": torch.float32}.get(amp, torch.bfloat16)
        if self.cfg.get("hardware.tf32", True):
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

        gid = self.cfg.get("grounded_sam2.grounding_dino_id")
        local = self.cfg.resolve(Path("weights") / "grounding-dino-base")
        if local.exists():
            gid = str(local)
        log.info("loading Grounding DINO %s", gid)
        self._gproc = AutoProcessor.from_pretrained(gid)
        self._gdino = AutoModelForZeroShotObjectDetection.from_pretrained(gid).to(self.device).eval()

        cfg_file = self.cfg.get("grounded_sam2.sam2_cfg")
        ckpt = self.cfg.resolve(self.cfg.get("grounded_sam2.sam2_ckpt"))
        log.info("loading SAM2.1 cfg=%s ckpt=%s", cfg_file, ckpt)
        from sam2.build_sam import build_sam2
        from sam2.sam2_image_predictor import SAM2ImagePredictor
        sam2_model = build_sam2(cfg_file, str(ckpt), device=self.device)
        self._sam = SAM2ImagePredictor(sam2_model)

    # ------------------------------------------------------------------ #
    def _run_gdino(self, pil_img, caption: str, box_thr: float, txt_thr: float,
                   h: int, w: int):
        import torch
        inputs = self._gproc(images=pil_img, text=caption, return_tensors="pt").to(self.device)
        with torch.inference_mode(), torch.autocast(self.device, dtype=self._amp_dtype,
                                                    enabled=self.device == "cuda"):
            outputs = self._gdino(**inputs)
        proc = self._gproc.post_process_grounded_object_detection
        try:
            res = proc(outputs, inputs.input_ids, box_threshold=box_thr,
                       text_threshold=txt_thr, target_sizes=[(h, w)])[0]
        except TypeError:
            res = proc(outputs, inputs.input_ids, threshold=box_thr,
                       text_threshold=txt_thr, target_sizes=[(h, w)])[0]
        n = len(res["boxes"])
        boxes = res["boxes"].detach().cpu().numpy() if n else np.zeros((0, 4))
        scores = res["scores"].detach().cpu().numpy() if n else np.zeros((0,))
        labels = res.get("text_labels") or res.get("labels") or []
        return boxes, scores, list(labels)

    # ------------------------------------------------------------------ #
    def segment(self, image_rgb: np.ndarray, frame_index: int,
                capture_ts_ns: int, roi: Optional[Dict[str, Any]] = None) -> FrameOutput:
        import cv2
        import torch
        from PIL import Image

        h, w = image_rgb.shape[:2]
        pil = Image.fromarray(image_rgb)
        rej: Counter = Counter()
        t0 = time.time()

        # 1) candidate detections across query groups (variants+synonyms in caption)
        cand: List[Dict[str, Any]] = []
        for group in self._build_query_groups():
            for caption, cover in group_captions(self.specs, group):
                if not caption:
                    continue
                cbox = min(self.specs[n].candidate_box_threshold for n in cover)
                ctxt = min(self.specs[n].candidate_text_threshold for n in cover)
                boxes, scores, labels = self._run_gdino(pil, caption, cbox, ctxt, h, w)
                for b, s, lab in zip(boxes, scores, labels):
                    cand.append({"box": b, "score": float(s), "phrase": str(lab),
                                 "cover": cover})
        t_gdino = (time.time() - t0) * 1e3

        # 2) phrase->class mapping + per-class thresholds + box-level geometry
        acc_boxes, acc_scores, acc_cids, acc_meta = [], [], [], []
        for d in cand:
            mr = self.mapper.map(d["phrase"], d["cover"])
            if mr.name is None:
                rej[mr.reason] += 1
                continue
            spec = self.specs[mr.name]
            if mr.confidence < spec.min_map_confidence:
                rej["low_map_confidence"] += 1
                self.stats_rejected_by_class[mr.name] += 1
                continue
            if d["score"] < spec.box_threshold:
                rej["below_box_threshold"] += 1
                self.stats_rejected_by_class[mr.name] += 1
                continue
            x0, y0, x1, y1 = d["box"]
            ok, reason = box_geom_ok(spec, (x0, y0, x1, y1), int((x1 - x0) * (y1 - y0)), h, w)
            # box-area proxy for early geom filters that don't need the mask
            if reason in ("below_min_box_width", "below_min_box_height",
                          "below_min_aspect_ratio", "above_max_aspect_ratio",
                          "spans_full_frame_edges", "outside_allowed_region",
                          "inside_forbidden_region") and not ok:
                rej[reason] += 1
                self.stats_rejected_by_class[mr.name] += 1
                continue
            acc_boxes.append(d["box"]); acc_scores.append(d["score"])
            acc_cids.append(self.tax.id_of(mr.name))
            acc_meta.append({"name": mr.name, "phrase": d["phrase"],
                             "map_conf": mr.confidence,
                             "provenance": (roi or {}).get("name", "full_frame")})

        dets: List[Detection] = []
        instances: List[Instance] = []
        t_sam = 0.0
        if acc_boxes:
            boxes = np.array(acc_boxes, dtype=np.float32)
            scores = np.array(acc_scores, dtype=np.float32)
            cids = np.array(acc_cids, dtype=np.int64)
            iou_by_class = {self.tax.id_of(n): s.nms_iou for n, s in self.specs.items()}
            keep = class_aware_nms(boxes, scores, cids,
                                   iou_thr=float(self.cfg.get("grounded_sam2.nms_iou", 0.7)),
                                   iou_by_class=iou_by_class)
            boxes, scores = boxes[keep], scores[keep]
            meta = [acc_meta[k] for k in keep]

            t1 = time.time()
            self._sam.set_image(image_rgb)
            with torch.inference_mode(), torch.autocast(self.device, dtype=self._amp_dtype,
                                                        enabled=self.device == "cuda"):
                masks, ious, _ = self._sam.predict(box=boxes, multimask_output=False)
            t_sam = (time.time() - t1) * 1e3
            masks = np.asarray(masks)
            if masks.ndim == 4:
                masks = masks[:, 0]
            ious = np.asarray(ious).reshape(-1)

            per_class: Dict[str, List[Detection]] = defaultdict(list)
            per_class_inst: Dict[str, List[Instance]] = defaultdict(list)
            for i, m in enumerate(meta):
                name = m["name"]; spec = self.specs[name]
                mask = self._postprocess_mask(masks[i] > 0.0, spec, cv2)
                area = int(mask.sum())
                ok, reason = box_geom_ok(spec, tuple(float(x) for x in boxes[i]), area, h, w)
                if not ok:
                    rej[reason] += 1
                    self.stats_rejected_by_class[name] += 1
                    continue
                sam_iou = float(ious[i]) if i < len(ious) else 0.0
                combined = self._combine(float(scores[i]), sam_iou)
                cid = self.tax.id_of(name)
                det = Detection(cid, name, tuple(float(x) for x in boxes[i]),
                                phrase=name, gdino_score=float(scores[i]), sam_iou=sam_iou,
                                combined_score=combined, area_px=area, priority=spec.priority,
                                native_phrase=m["phrase"], map_confidence=float(m["map_conf"]),
                                provenance=m["provenance"])
                per_class[name].append(det)
                per_class_inst[name].append(Instance(mask, cid, combined, spec.priority))

            # 3) max_instances per class (keep highest combined score)
            for name, ds in per_class.items():
                spec = self.specs[name]
                order = sorted(range(len(ds)), key=lambda k: -ds[k].combined_score)
                cap = spec.max_instances if spec.max_instances else len(ds)
                dropped = max(0, len(ds) - cap)
                if dropped:
                    rej["max_instances"] += dropped
                    self.stats_rejected_by_class[name] += dropped
                for k in order[:cap]:
                    dets.append(ds[k]); instances.append(per_class_inst[name][k])
                    self.stats_accepted[name] += 1

        id_map, score_map = resolve_overlaps(h, w, instances)
        # multi-layer masks (Phase 3): partition instances by functional layer
        layer_inst: Dict[str, List[Instance]] = {L: [] for L in LAYERS}
        for inst in instances:
            layer_inst[layer_of_id(self.tax, inst.canonical_id)].append(inst)
        layers = {L: resolve_overlaps(h, w, layer_inst[L])[0] for L in LAYERS}
        for reason, c in rej.items():
            self.stats_rejected[reason] += c
        total = (time.time() - t0) * 1e3
        return FrameOutput(
            frame_index=frame_index, capture_timestamp_ns=capture_ts_ns,
            method="grounded_sam2", image_size=(h, w),
            canonical_mask=id_map, confidence=score_map, detections=dets,
            timings_ms={"grounding_dino": round(t_gdino, 1), "sam2": round(t_sam, 1),
                        "total": round(total, 1)},
            layers=layers,
            extra={"coverage": float((id_map > 0).mean()),
                   "num_candidates": len(cand), "num_accepted": len(dets),
                   "layer_coverage": {L: float((layers[L] > 0).mean()) for L in LAYERS},
                   "rejections": dict(rej)},
        )

    def _postprocess_mask(self, m: np.ndarray, spec: PromptSpec, cv2) -> np.ndarray:
        if spec.morph_close and spec.morph_close > 1:
            k = np.ones((spec.morph_close, spec.morph_close), np.uint8)
            m = cv2.morphologyEx(m.astype(np.uint8), cv2.MORPH_CLOSE, k).astype(bool)
        return m

    def _combine(self, gdino: float, sam_iou: float) -> float:
        mode = self.cfg.get("grounded_sam2.score_combine", "geometric")
        if mode == "geometric":
            return float(np.sqrt(max(gdino, 0.0) * max(sam_iou, 0.0)))
        if mode == "mean":
            return float(0.5 * (gdino + sam_iou))
        return float(gdino)
