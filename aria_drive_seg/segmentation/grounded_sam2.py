"""Method 1 — open-vocabulary prompted segmentation: Grounding DINO + SAM 2.1 (§7).

Grounding DINO (HF transformers) proposes boxes from configurable text prompts;
SAM 2.1 turns each box into a mask; masks are filtered, scored, and composited by
priority (overlap.py). This is NOT an exhaustive semantic partition — unassigned
pixels remain `unknown`.
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import yaml

from ..logging_utils import get_logger
from ..taxonomy import Taxonomy
from .base import Detection, FrameOutput
from .overlap import Instance, class_aware_nms, resolve_overlaps

log = get_logger("grounded_sam2")


def _tokens(s: str) -> set:
    return {w for w in "".join(c if c.isalnum() else " " for c in s.lower()).split() if len(w) > 2}


class PromptSpec:
    def __init__(self, name: str, cfg: Dict[str, Any], defaults: Dict[str, Any]):
        self.name = name
        g = lambda k: cfg.get(k, defaults.get(k))
        self.prompt = cfg["prompt"].strip().lower()
        self.synonyms = [s.strip().lower() for s in cfg.get("synonyms", [])]
        self.box_threshold = float(g("box_threshold"))
        self.text_threshold = float(g("text_threshold"))
        self.priority = int(cfg.get("priority", 0))
        self.min_area = int(g("min_area"))
        self.max_area_frac = g("max_area_frac")
        self.morph_close = int(g("morph_close"))
        self.max_instances = g("max_instances")
        self.solo = bool(cfg.get("solo", defaults.get("solo", False)))
        self.group_tag = cfg.get("group_tag", "misc")
        # phrases used to map a returned GDINO phrase back to this class
        self.phrases = [p.strip() for p in self.prompt.replace(".", " . ").split(".") if p.strip()]
        self.phrases += self.synonyms
        self.token_sets = [_tokens(p) for p in self.phrases if p]


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

    # ------------------------------------------------------------------ #
    def _load_prompts(self) -> None:
        path = self.cfg.resolve(self.cfg.get("project.grounded_prompts"))
        doc = yaml.safe_load(Path(path).read_text())
        defaults = doc.get("defaults", {})
        self.prompt_strategy = doc.get("prompt_strategy", "grouped")
        self.specs: Dict[str, PromptSpec] = {}
        for name, c in doc["classes"].items():
            if name not in self.tax.by_name:
                log.warning("prompt class %r not in taxonomy; skipping", name)
                continue
            self.specs[name] = PromptSpec(name, c, defaults)

    def _build_query_groups(self) -> List[List[str]]:
        """Batch classes into GDINO calls. solo classes get their own call."""
        if self.prompt_strategy == "individual":
            return [[n] for n in self.specs]
        groups: Dict[str, List[str]] = {}
        solos: List[List[str]] = []
        for n, s in self.specs.items():
            if s.solo or self.prompt_strategy == "concatenated" and False:
                solos.append([n])
            else:
                groups.setdefault(s.group_tag, []).append(n)
        if self.prompt_strategy == "concatenated":
            return [list(self.specs.keys())]
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
            gid = str(local)  # prefer pre-downloaded local weights (offline)
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
    def _run_gdino(self, pil_img, class_names: List[str], h: int, w: int):
        import torch
        # caption = union of class prompts; use the lowest thresholds in the group
        caption = " ".join(self.specs[n].prompt if self.specs[n].prompt.endswith(".")
                           else self.specs[n].prompt + "." for n in class_names)
        box_thr = min(self.specs[n].box_threshold for n in class_names)
        txt_thr = min(self.specs[n].text_threshold for n in class_names)
        inputs = self._gproc(images=pil_img, text=caption, return_tensors="pt").to(self.device)
        with torch.inference_mode(), torch.autocast(self.device, dtype=self._amp_dtype,
                                                    enabled=self.device == "cuda"):
            outputs = self._gdino(**inputs)
        results = self._post_process(outputs, inputs, box_thr, txt_thr, (h, w))
        boxes = results["boxes"].detach().cpu().numpy() if len(results["boxes"]) else np.zeros((0, 4))
        scores = results["scores"].detach().cpu().numpy() if len(results["boxes"]) else np.zeros((0,))
        labels = results.get("text_labels") or results.get("labels") or []
        return boxes, scores, list(labels)

    def _post_process(self, outputs, inputs, box_thr, txt_thr, size):
        proc = self._gproc.post_process_grounded_object_detection
        kw = dict(target_sizes=[size])
        try:
            return proc(outputs, inputs.input_ids, box_threshold=box_thr,
                        text_threshold=txt_thr, **kw)[0]
        except TypeError:
            return proc(outputs, inputs.input_ids, threshold=box_thr,
                        text_threshold=txt_thr, **kw)[0]

    def _phrase_to_class(self, phrase: str, candidates: List[str]) -> Optional[str]:
        pt = _tokens(phrase)
        if not pt:
            return None
        best, best_score = None, 0.0
        for n in candidates:
            for ts in self.specs[n].token_sets:
                if not ts:
                    continue
                j = len(pt & ts) / len(pt | ts)
                # also reward substring containment
                if any(w in phrase for w in ts):
                    j = max(j, 0.5 + 0.5 * j)
                if j > best_score:
                    best, best_score = n, j
        return best if best_score >= 0.3 else None

    # ------------------------------------------------------------------ #
    def segment(self, image_rgb: np.ndarray, frame_index: int,
                capture_ts_ns: int) -> FrameOutput:
        import cv2
        import torch
        from PIL import Image

        h, w = image_rgb.shape[:2]
        pil = Image.fromarray(image_rgb)
        t0 = time.time()

        all_boxes, all_scores, all_cids, all_names = [], [], [], []
        for group in self._build_query_groups():
            boxes, scores, labels = self._run_gdino(pil, group, h, w)
            for b, s, lab in zip(boxes, scores, labels):
                cls = self._phrase_to_class(str(lab), group)
                if cls is None:
                    continue
                all_boxes.append(b); all_scores.append(float(s))
                all_cids.append(self.tax.id_of(cls)); all_names.append(cls)
        t_gdino = (time.time() - t0) * 1e3

        dets: List[Detection] = []
        instances: List[Instance] = []
        if all_boxes:
            boxes = np.array(all_boxes, dtype=np.float32)
            scores = np.array(all_scores, dtype=np.float32)
            cids = np.array(all_cids, dtype=np.int64)
            keep = class_aware_nms(boxes, scores, cids,
                                   iou_thr=float(self.cfg.get("grounded_sam2.nms_iou", 0.7)))
            boxes, scores = boxes[keep], scores[keep]
            names = [all_names[k] for k in keep]

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

            for i, name in enumerate(names):
                spec = self.specs[name]
                m = masks[i] > 0.0
                m = self._postprocess_mask(m, spec, h, w, cv2)
                area = int(m.sum())
                if area < spec.min_area:
                    continue
                if spec.max_area_frac and area > spec.max_area_frac * h * w:
                    continue
                sam_iou = float(ious[i]) if i < len(ious) else 0.0
                gdino = float(scores[i])
                combined = self._combine(gdino, sam_iou)
                cid = self.tax.id_of(name)
                dets.append(Detection(cid, name, tuple(float(x) for x in boxes[i]),
                                      phrase=name, gdino_score=gdino, sam_iou=sam_iou,
                                      combined_score=combined, area_px=area,
                                      priority=spec.priority))
                instances.append(Instance(m, cid, combined, spec.priority))
        else:
            t_sam = 0.0

        id_map, score_map = resolve_overlaps(h, w, instances)
        total = (time.time() - t0) * 1e3
        return FrameOutput(
            frame_index=frame_index, capture_timestamp_ns=capture_ts_ns,
            method="grounded_sam2", image_size=(h, w),
            canonical_mask=id_map, confidence=score_map, detections=dets,
            timings_ms={"grounding_dino": round(t_gdino, 1), "sam2": round(t_sam, 1),
                        "total": round(total, 1)},
            extra={"coverage": float((id_map > 0).mean())},
        )

    def _postprocess_mask(self, m: np.ndarray, spec: PromptSpec, h: int, w: int, cv2) -> np.ndarray:
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
