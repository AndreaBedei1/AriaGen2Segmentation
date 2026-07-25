"""Method 1 — open-vocabulary prompted segmentation: Grounding DINO + SAM 2.1 (§7).

Grounding DINO proposes boxes from configurable text prompts (variants + synonyms,
prompt_engine.py); SAM 2.1 turns each surviving box into a mask; masks are filtered
(per-class thresholds, geometric filters, max_instances), scored, composited by
priority (overlap.py) and split into functional layers (layers.py). NOT an exhaustive
partition — unassigned pixels stay `unknown`.

Passes (a single reusable `_finalize_pass` runs each): a full-frame pass; optional
hierarchical-ROI passes on derived windshield/window/mirror/interior crops (Phase 4);
optional multi-scale tiles of the windshield crop for tiny objects (Phase 5). All
detections carry provenance and are fused across passes (box + mask NMS, max_instances).
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
from .hierarchy import decide_subclass, load_hierarchy
from .layers import LAYERS, layer_of_id
from .overlap import Instance, class_aware_nms, mask_iou, resolve_overlaps
from .prompt_engine import (PhraseMapper, PromptSpec, box_geom_ok, build_specs,
                            group_captions)
from . import roi as roimod

log = get_logger("grounded_sam2")

_SHAPE_REJECTS = {"below_min_box_width", "below_min_box_height", "below_min_aspect_ratio",
                  "above_max_aspect_ratio", "spans_full_frame_edges",
                  "outside_allowed_region", "inside_forbidden_region"}


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
        self.stats_accepted: Counter = Counter()
        self.stats_rejected: Counter = Counter()
        self.stats_rejected_by_class: Counter = Counter()
        self._roi_state = roimod.ROIState() if cfg.get("grounded_sam2.roi_stable", True) else None

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
        self._available = set(self.specs.keys())
        self.hierarchy = load_hierarchy(doc)

    def _global_classes(self) -> List[str]:
        """Classes prompted on the full frame. In the standard config the fine
        hierarchy subclasses are excluded (classified on parent crops instead)."""
        names = list(self.specs.keys())
        if (self.cfg.get("grounded_sam2.hierarchical_subclasses", True)
                and not self.cfg.get("grounded_sam2.global_fine_subclasses", False)):
            names = [n for n in names if n not in self.hierarchy["subclasses"]]
        return names

    def _build_query_groups(self, names: Optional[List[str]] = None) -> List[List[str]]:
        specs = names or list(self.specs.keys())
        if self.prompt_strategy == "individual":
            return [[n] for n in specs]
        if self.prompt_strategy == "concatenated":
            return [list(specs)]
        groups: Dict[str, List[str]] = defaultdict(list)
        solos: List[List[str]] = []
        for n in specs:
            s = self.specs[n]
            (solos.append([n]) if s.solo else groups[s.group_tag].append(n))
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

    def _gdino_candidates(self, pil, groups: List[List[str]], h: int, w: int):
        cand: List[Dict[str, Any]] = []
        for group in groups:
            for caption, cover in group_captions(self.specs, group):
                if not caption:
                    continue
                cbox = min(self.specs[n].candidate_box_threshold for n in cover)
                ctxt = min(self.specs[n].candidate_text_threshold for n in cover)
                boxes, scores, labels = self._run_gdino(pil, caption, cbox, ctxt, h, w)
                for b, s, lab in zip(boxes, scores, labels):
                    cand.append({"box": b, "score": float(s), "phrase": str(lab), "cover": cover})
        return cand

    def _finalize_pass(self, sam_image: np.ndarray, cand: List[Dict[str, Any]],
                       h_full: int, w_full: int, offset: Tuple[int, int], scale: float,
                       provenance: str, rej: Counter):
        """Map phrases -> classes, filter, NMS, run SAM on `sam_image`, paste masks to
        full-frame coords. Returns (detections, instances) in full-frame geometry."""
        import cv2
        import torch
        acc = []
        for d in cand:
            mr = self.mapper.map(d["phrase"], d["cover"])
            if mr.name is None:
                rej[mr.reason] += 1
                continue
            spec = self.specs[mr.name]
            if mr.confidence < spec.min_map_confidence:
                rej["low_map_confidence"] += 1; continue
            if d["score"] < spec.box_threshold:
                rej["below_box_threshold"] += 1; continue
            fbox = roimod.remap_box(d["box"], offset, scale)
            ok, reason = box_geom_ok(spec, fbox, int((fbox[2] - fbox[0]) * (fbox[3] - fbox[1])),
                                     h_full, w_full)
            if not ok and reason in _SHAPE_REJECTS:
                rej[reason] += 1; continue
            acc.append({"cbox": d["box"], "score": d["score"], "name": mr.name, "spec": spec,
                        "fbox": fbox, "phrase": d["phrase"], "map_conf": mr.confidence})
        if not acc:
            return [], []
        boxes = np.array([a["cbox"] for a in acc], dtype=np.float32)
        scores = np.array([a["score"] for a in acc], dtype=np.float32)
        cids = np.array([self.tax.id_of(a["name"]) for a in acc], dtype=np.int64)
        iou_by_class = {self.tax.id_of(n): s.nms_iou for n, s in self.specs.items()}
        keep = class_aware_nms(boxes, scores, cids,
                               iou_thr=float(self.cfg.get("grounded_sam2.nms_iou", 0.7)),
                               iou_by_class=iou_by_class)
        acc = [acc[k] for k in keep]
        boxes = boxes[keep]

        self._sam.set_image(sam_image)
        with torch.inference_mode(), torch.autocast(self.device, dtype=self._amp_dtype,
                                                    enabled=self.device == "cuda"):
            masks, ious, _ = self._sam.predict(box=boxes, multimask_output=False)
        masks = np.asarray(masks)
        if masks.ndim == 4:
            masks = masks[:, 0]
        ious = np.asarray(ious).reshape(-1)

        dets, inst = [], []
        for i, a in enumerate(acc):
            spec = a["spec"]
            m_crop = self._postprocess_mask(masks[i] > 0.0, spec, cv2)
            m_full = roimod.paste_mask(m_crop, offset, h_full, w_full, scale)
            area = int(m_full.sum())
            ok, reason = box_geom_ok(spec, a["fbox"], area, h_full, w_full)
            if not ok:
                rej[reason] += 1; continue
            sam_iou = float(ious[i]) if i < len(ious) else 0.0
            combined = self._combine(float(a["score"]), sam_iou)
            cid = self.tax.id_of(a["name"])
            dets.append(Detection(cid, a["name"], tuple(float(x) for x in a["fbox"]),
                                  phrase=a["name"], gdino_score=float(a["score"]), sam_iou=sam_iou,
                                  combined_score=combined, area_px=area, priority=spec.priority,
                                  native_phrase=a["phrase"], map_confidence=float(a["map_conf"]),
                                  provenance=provenance))
            inst.append(Instance(m_full, cid, combined, spec.priority))
        return dets, inst

    def _crop_pass(self, image: np.ndarray, box, class_names: List[str], provenance: str,
                   scale: float, rej: Counter):
        import cv2
        from PIL import Image
        H, W = image.shape[:2]
        x0, y0, x1, y1 = [int(v) for v in box]
        if x1 - x0 < 8 or y1 - y0 < 8:
            return [], []
        crop = image[y0:y1, x0:x1]
        if scale != 1.0:
            crop = cv2.resize(crop, (int((x1 - x0) * scale), int((y1 - y0) * scale)),
                              interpolation=cv2.INTER_LINEAR)
        ch, cw = crop.shape[:2]
        cand = self._gdino_candidates(Image.fromarray(crop), [class_names], ch, cw)
        return self._finalize_pass(np.ascontiguousarray(crop), cand, H, W, (x0, y0), scale,
                                   provenance, rej)

    # ------------------------------------------------------------------ #
    def segment(self, image_rgb: np.ndarray, frame_index: int,
                capture_ts_ns: int) -> FrameOutput:
        from PIL import Image
        h, w = image_rgb.shape[:2]
        rej: Counter = Counter()
        t0 = time.time()

        # ---- full-frame pass (fine hierarchy subclasses excluded in standard config) ----
        cand = self._gdino_candidates(Image.fromarray(image_rgb),
                                      self._build_query_groups(self._global_classes()), h, w)
        t_gdino = (time.time() - t0) * 1e3
        t1 = time.time()
        all_dets, all_inst = self._finalize_pass(image_rgb, cand, h, w, (0, 0), 1.0,
                                                 "full_frame", rej)

        rois: Dict[str, Any] = {}
        # ---- Phase 4: hierarchical ROI passes ----
        if self.cfg.get("grounded_sam2.roi_enabled", False):
            rois = roimod.derive_rois(all_dets, h, w, float(self.cfg.get("grounded_sam2.roi_pad_frac", 0.05)))
            if self._roi_state is not None:
                rois = self._roi_state.update(rois, h, w)
            rscale = float(self.cfg.get("grounded_sam2.roi_scale", 1.5))
            for name, box in rois.items():
                subset = roimod.roi_class_subset(name, self._available)
                if not subset:
                    continue
                d, ins = self._crop_pass(image_rgb, box, subset, f"roi:{name}", rscale, rej)
                all_dets += d; all_inst += ins

        # ---- Phase 5: multi-scale windshield tiles ----
        if self.cfg.get("grounded_sam2.multiscale_enabled", False):
            wbox = rois.get("windshield_view") or roimod.derive_rois(all_dets, h, w).get("windshield_view")
            if wbox:
                subset = roimod.roi_class_subset("windshield_view", self._available)
                mscale = float(self.cfg.get("grounded_sam2.ms_scale", 1.5))
                tiles = roimod.tiles_of(wbox, int(self.cfg.get("grounded_sam2.ms_tiles", 3)),
                                        float(self.cfg.get("grounded_sam2.ms_tile_overlap", 0.2)))
                for ti, tile in enumerate(tiles):
                    d, ins = self._crop_pass(image_rgb, tile, subset, f"tile:{ti}", mscale, rej)
                    all_dets += d; all_inst += ins
        t_sam = (time.time() - t1) * 1e3

        # ---- fuse across passes (box + mask NMS, max_instances) ----
        all_dets, all_inst = self._fuse(all_dets, all_inst, rej)

        # ---- hierarchical parent->subclass classification (on parent crops) ----
        if (self.cfg.get("grounded_sam2.hierarchical_subclasses", True)
                and not self.cfg.get("grounded_sam2.global_fine_subclasses", False)):
            all_dets, all_inst = self._classify_subclasses(image_rgb, all_dets, all_inst, rej)

        id_map, score_map = resolve_overlaps(h, w, all_inst)
        layer_inst: Dict[str, List[Instance]] = {L: [] for L in LAYERS}
        for inst in all_inst:
            layer_inst[layer_of_id(self.tax, inst.canonical_id)].append(inst)
        layers = {L: resolve_overlaps(h, w, layer_inst[L])[0] for L in LAYERS}
        for reason, c in rej.items():
            self.stats_rejected[reason] += c
        total = (time.time() - t0) * 1e3
        return FrameOutput(
            frame_index=frame_index, capture_timestamp_ns=capture_ts_ns,
            method="grounded_sam2", image_size=(h, w),
            canonical_mask=id_map, confidence=score_map, detections=all_dets,
            timings_ms={"grounding_dino": round(t_gdino, 1), "sam2": round(t_sam, 1),
                        "total": round(total, 1)},
            layers=layers,
            extra={"coverage": float((id_map > 0).mean()),
                   "num_candidates": len(cand), "num_accepted": len(all_dets),
                   "rois": {k: [int(x) for x in v] for k, v in rois.items()},
                   "provenance_counts": dict(Counter(d.provenance for d in all_dets)),
                   "layer_coverage": {L: float((layers[L] > 0).mean()) for L in LAYERS},
                   "rejections": dict(rej)},
        )

    def _fuse(self, dets: List[Detection], inst: List[Instance], rej: Counter):
        """Cross-pass duplicate fusion: class-aware box NMS, then same-class mask-IoU
        dedup, then max_instances per class."""
        if not dets:
            return [], []
        boxes = np.array([d.box for d in dets], dtype=np.float32)
        scores = np.array([d.combined_score for d in dets], dtype=np.float32)
        cids = np.array([d.canonical_id for d in dets], dtype=np.int64)
        iou_by_class = {self.tax.id_of(n): s.nms_iou for n, s in self.specs.items()}
        keep = class_aware_nms(boxes, scores, cids,
                               iou_thr=float(self.cfg.get("grounded_sam2.nms_iou", 0.7)),
                               iou_by_class=iou_by_class)
        dets = [dets[k] for k in keep]; inst = [inst[k] for k in keep]
        # same-class mask-IoU dedup (fuse full-frame vs tile duplicates)
        thr = float(self.cfg.get("segmentation.overlap_min_iou_dedup", 0.85))
        order = sorted(range(len(dets)), key=lambda k: -dets[k].combined_score)
        drop = set()
        for ii in range(len(order)):
            a = order[ii]
            if a in drop:
                continue
            for jj in range(ii + 1, len(order)):
                b = order[jj]
                if b in drop or dets[a].canonical_id != dets[b].canonical_id:
                    continue
                if mask_iou(inst[a].mask, inst[b].mask) > thr:
                    drop.add(b)
        dets = [d for k, d in enumerate(dets) if k not in drop]
        inst = [x for k, x in enumerate(inst) if k not in drop]
        if drop:
            rej["mask_dedup"] += len(drop)
        # max_instances per class
        per: Dict[str, List[int]] = defaultdict(list)
        for k, d in enumerate(dets):
            per[d.canonical_name].append(k)
        keep2 = []
        for name, idxs in per.items():
            cap = self.specs[name].max_instances or len(idxs)
            idxs = sorted(idxs, key=lambda k: -dets[k].combined_score)
            if len(idxs) > cap:
                rej["max_instances"] += len(idxs) - cap
                self.stats_rejected_by_class[name] += len(idxs) - cap
            keep2 += idxs[:cap]
            self.stats_accepted[name] += min(len(idxs), cap)
        keep2.sort()
        return [dets[k] for k in keep2], [inst[k] for k in keep2]

    def _subclass_scores(self, image: np.ndarray, box, subs: List[str]) -> Dict[str, float]:
        import cv2
        from PIL import Image
        x0, y0, x1, y1 = [int(v) for v in box]
        if x1 - x0 < 12 or y1 - y0 < 12:
            return {}
        crop = image[y0:y1, x0:x1]
        ch, cw = crop.shape[:2]
        if max(ch, cw) < 200:  # upscale small parent crops for the subclass query
            s = 200.0 / max(ch, cw)
            crop = cv2.resize(crop, (int(cw * s), int(ch * s)))
        ch2, cw2 = crop.shape[:2]
        cand = self._gdino_candidates(Image.fromarray(np.ascontiguousarray(crop)), [subs], ch2, cw2)
        scores: Dict[str, float] = {}
        for d in cand:
            mr = self.mapper.map(d["phrase"], subs)
            if mr.name in subs:
                scores[mr.name] = max(scores.get(mr.name, 0.0),
                                      float(d["score"]) * float(mr.confidence))
        return scores

    def _classify_subclasses(self, image: np.ndarray, dets: List[Detection],
                             insts: List[Instance], rej: Counter):
        """Detect subtype on the parent crop; relabel only on a confident, clear win.
        Mask geometry stays from the parent."""
        H, W = image.shape[:2]
        parents = self.hierarchy["parents"]
        ms, mm = self.hierarchy["min_score"], self.hierarchy["min_margin"]
        cap = self.hierarchy["max_parents_per_frame"]
        min_area = self.hierarchy["min_parent_area_frac"] * H * W
        cands = [(i, d) for i, d in enumerate(dets)
                 if d.canonical_name in parents and d.area_px >= min_area]
        cands.sort(key=lambda t: -t[1].area_px)
        for i, d in cands[:cap]:
            subs = [s for s in parents[d.canonical_name] if s in self.specs]
            if not subs:
                continue
            scores = self._subclass_scores(image, d.box, subs)
            name, status, conf = decide_subclass(scores, ms, mm)
            d.parent_class = d.canonical_name
            d.parent_confidence = d.combined_score
            d.subclass_status = status
            d.subclass_confidence = conf
            if status == "accepted" and name in self.specs:
                newid = self.tax.id_of(name)
                d.canonical_id = newid; d.canonical_name = name
                d.priority = self.specs[name].priority
                insts[i] = Instance(insts[i].mask, newid, insts[i].score, self.specs[name].priority)
                self.stats_accepted[name] += 1
            else:
                rej[f"subclass_{status}"] += 1
        return dets, insts

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
