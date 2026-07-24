"""Prompt engine for Grounded-SAM2 (Phase 1).

Fixes the three real bugs in the first implementation:
  * `synonyms`/`prompt_variants` are now REALLY sent to Grounding DINO (the caption is
    built from every phrase, deduplicated), controlled by per-class `prompt_mode`;
  * grouped calls use a permissive CANDIDATE threshold, then each detection is re-filtered
    by its OWN class `box_threshold` / mapping-confidence after phrase->class mapping;
  * robust, deterministic phrase->class mapping (exact / alias / singular-plural / token
    overlap with ambiguity rejection), returning a confidence and a reason.

Everything here is pure (no torch), so it is unit-tested in isolation.
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

_WORD = re.compile(r"[a-z0-9]+")


def normalize(s: str) -> str:
    return " ".join(_WORD.findall(s.lower()))


def singularize(word: str) -> str:
    """Cheap English singulariser sufficient for class/synonym matching."""
    if len(word) > 4 and word.endswith("ies"):
        return word[:-3] + "y"
    if len(word) > 4 and word.endswith("ses"):
        return word[:-2]
    if len(word) > 3 and word.endswith("s") and not word.endswith("ss"):
        return word[:-1]
    return word


def norm_tokens(s: str) -> List[str]:
    return [singularize(w) for w in normalize(s).split()]


def _phrase_key(s: str) -> str:
    return " ".join(norm_tokens(s))


@dataclass
class PromptSpec:
    name: str
    prompt: str
    prompt_variants: List[str] = field(default_factory=list)
    synonyms: List[str] = field(default_factory=list)
    negative_contexts: List[str] = field(default_factory=list)
    prompt_mode: str = "combined"          # combined | variants | individual
    box_threshold: float = 0.35
    text_threshold: float = 0.25
    candidate_box_threshold: float = 0.15
    candidate_text_threshold: float = 0.12
    min_map_confidence: float = 0.30
    priority: int = 0
    group_tag: str = "misc"
    solo: bool = False
    # geometric filters
    min_area: int = 400
    max_area_frac: Optional[float] = None
    min_aspect_ratio: Optional[float] = None   # w/h
    max_aspect_ratio: Optional[float] = None
    min_box_width: Optional[float] = None
    min_box_height: Optional[float] = None
    allowed_regions: List[List[float]] = field(default_factory=list)   # [x0,y0,x1,y1] frac
    forbidden_regions: List[List[float]] = field(default_factory=list)
    edge_margin: Optional[int] = None
    max_instances: Optional[int] = None
    nms_iou: Optional[float] = None
    morph_close: int = 3
    # derived
    all_phrases: List[str] = field(default_factory=list)
    phrase_keys: List[str] = field(default_factory=list)
    token_sets: List[frozenset] = field(default_factory=list)

    def finalize(self) -> "PromptSpec":
        seen = set()
        phrases: List[str] = []
        for p in [self.prompt, *self.prompt_variants, *self.synonyms]:
            p = (p or "").strip()
            if not p:
                continue
            k = _phrase_key(p)
            if k and k not in seen:
                seen.add(k)
                phrases.append(p)
        self.all_phrases = phrases
        self.phrase_keys = [_phrase_key(p) for p in phrases]
        self.token_sets = [frozenset(norm_tokens(p)) for p in phrases]
        return self


def build_specs(doc: Dict[str, Any], taxonomy_names: set) -> Dict[str, PromptSpec]:
    defaults = doc.get("defaults", {})
    out: Dict[str, PromptSpec] = {}
    for name, c in doc["classes"].items():
        if name not in taxonomy_names:
            continue
        g = lambda k, d=None: c.get(k, defaults.get(k, d))
        spec = PromptSpec(
            name=name,
            prompt=str(c["prompt"]),
            prompt_variants=list(c.get("prompt_variants", []) or []),
            synonyms=list(c.get("synonyms", []) or []),
            negative_contexts=list(c.get("negative_contexts", []) or []),
            prompt_mode=str(g("prompt_mode", "combined")),
            box_threshold=float(g("box_threshold", 0.35)),
            text_threshold=float(g("text_threshold", 0.25)),
            candidate_box_threshold=float(g("candidate_box_threshold", 0.15)),
            candidate_text_threshold=float(g("candidate_text_threshold", 0.12)),
            min_map_confidence=float(g("min_map_confidence", 0.30)),
            priority=int(c.get("priority", 0)),
            group_tag=str(c.get("group_tag", "misc")),
            solo=bool(c.get("solo", defaults.get("solo", False))),
            min_area=int(g("min_area", 400)),
            max_area_frac=g("max_area_frac"),
            min_aspect_ratio=g("min_aspect_ratio"),
            max_aspect_ratio=g("max_aspect_ratio"),
            min_box_width=g("min_box_width"),
            min_box_height=g("min_box_height"),
            allowed_regions=list(c.get("allowed_regions", []) or []),
            forbidden_regions=list(c.get("forbidden_regions", []) or []),
            edge_margin=g("edge_margin"),
            max_instances=c.get("max_instances", defaults.get("max_instances")),
            nms_iou=g("nms_iou"),
            morph_close=int(g("morph_close", 3)),
        ).finalize()
        out[name] = spec
    return out


@dataclass
class MapResult:
    name: Optional[str]
    confidence: float
    reason: str


class PhraseMapper:
    """Deterministic phrase -> canonical class mapping with a confidence + reason."""

    def __init__(self, specs: Dict[str, PromptSpec], negatives: Optional[List[str]] = None):
        self.specs = specs
        self.exact: Dict[str, str] = {}
        for name, s in specs.items():
            for k in s.phrase_keys:
                # first writer wins -> deterministic; skip collisions across classes
                self.exact.setdefault(k, name)
        # IDF-style token weights: tokens shared by many classes (e.g. "car") carry
        # little discriminative weight; distinctive tokens (e.g. "dashboard") carry a lot.
        df: Dict[str, int] = {}
        n_classes = max(1, len(specs))
        for s in specs.values():
            for t in set().union(*s.token_sets) if s.token_sets else set():
                df[t] = df.get(t, 0) + 1
        self.n_classes = n_classes
        self.tok_weight: Dict[str, float] = {
            t: math.log((n_classes + 1) / (c + 0.5)) + 0.1 for t, c in df.items()
        }
        # negative-context phrases
        self.neg_keys = {_phrase_key(p) for p in (negatives or []) if _phrase_key(p)}
        self.neg_token_sets = [frozenset(k.split()) for k in self.neg_keys]

    def _w(self, t: str) -> float:
        return self.tok_weight.get(t, 1.0)

    def _weighted_overlap(self, ptoks: frozenset, ts: frozenset) -> float:
        """Fraction of the phrase's DISTINCTIVE weight explained by class token set ts."""
        den = sum(self._w(t) for t in ptoks)
        if den <= 0:
            return 0.0
        num = sum(self._w(t) for t in ptoks if t in ts)
        base = num / den
        if ptoks and (ptoks.issubset(ts) or ts.issubset(ptoks)):
            base = max(base, 0.9 * base + 0.1)
        return base

    def _best_negative(self, ptoks: frozenset) -> float:
        return max((self._weighted_overlap(ptoks, ts) for ts in self.neg_token_sets), default=0.0)

    def map(self, phrase: str, candidates: Optional[List[str]] = None,
            ambiguity_margin: float = 0.06) -> MapResult:
        cand = candidates or list(self.specs.keys())
        key = _phrase_key(phrase)
        if not key:
            return MapResult(None, 0.0, "empty_phrase")
        ptoks = frozenset(key.split())
        # 0) exact negative-context phrase -> reject (before any class match)
        if key in self.neg_keys:
            return MapResult(None, 1.0, "negative_context")
        # 1) exact normalized (singularized) phrase match
        if key in self.exact and self.exact[key] in cand:
            return MapResult(self.exact[key], 1.0, "exact")
        if not ptoks:
            return MapResult(None, 0.0, "no_tokens")
        neg = self._best_negative(ptoks)
        # 2) IDF-weighted token overlap over candidate classes
        scored: List[Tuple[str, float]] = []
        for n in cand:
            best = max((self._weighted_overlap(ptoks, ts) for ts in self.specs[n].token_sets
                       if ts), default=0.0)
            if best > 0:
                scored.append((n, best))
        if not scored:
            return MapResult(None, neg, "negative_context" if neg > 0 else "no_overlap")
        scored.sort(key=lambda kv: (-kv[1], kv[0]))
        top_name, top = scored[0]
        if neg >= top:
            return MapResult(None, top, "negative_context")
        second = scored[1][1] if len(scored) > 1 else 0.0
        # reject genuine near-ties (two candidate classes explain the phrase equally).
        # In the default per-class prompt mode `candidates` has one entry, so this never
        # triggers there; it only guards the optional combined-caption mode.
        if second > 0 and (top - second) < ambiguity_margin:
            return MapResult(None, top, f"ambiguous({top_name}~{scored[1][0]})")
        return MapResult(top_name, float(top), "token_overlap")


def group_captions(specs: Dict[str, PromptSpec], group: List[str]
                   ) -> List[Tuple[str, List[str]]]:
    """Return (caption, [class_names it covers]) queries for a group of classes.

    combined  -> one caption with ALL phrases of ALL classes (deduped);
    variants  -> one caption per class (all its phrases), separate calls;
    individual-> one caption per class with only its main prompt.
    """
    combined_classes = [n for n in group if specs[n].prompt_mode == "combined"]
    queries: List[Tuple[str, List[str]]] = []
    if combined_classes:
        phrases: List[str] = []
        seen = set()
        for n in combined_classes:
            for p in specs[n].all_phrases:
                k = _phrase_key(p)
                if k not in seen:
                    seen.add(k)
                    phrases.append(p)
        queries.append((_caption(phrases), combined_classes))
    for n in group:
        if specs[n].prompt_mode == "variants":
            queries.append((_caption(specs[n].all_phrases), [n]))
        elif specs[n].prompt_mode == "individual":
            queries.append((_caption([specs[n].prompt]), [n]))
    return queries


def _caption(phrases: List[str]) -> str:
    parts = []
    for p in phrases:
        p = p.strip().lower().rstrip(".")
        if p:
            parts.append(p + ".")
    return " ".join(parts)


# ------------------------------------------------------------------ #
# geometric filtering (pure)
# ------------------------------------------------------------------ #
def box_geom_ok(spec: PromptSpec, box: Tuple[float, float, float, float],
                area_px: int, h: int, w: int) -> Tuple[bool, str]:
    x0, y0, x1, y1 = box
    bw, bh = x1 - x0, y1 - y0
    if area_px < spec.min_area:
        return False, "below_min_area"
    if spec.max_area_frac and area_px > spec.max_area_frac * h * w:
        return False, "above_max_area_frac"
    if spec.min_box_width and bw < spec.min_box_width:
        return False, "below_min_box_width"
    if spec.min_box_height and bh < spec.min_box_height:
        return False, "below_min_box_height"
    if bh > 0:
        ar = bw / bh
        if spec.min_aspect_ratio and ar < spec.min_aspect_ratio:
            return False, "below_min_aspect_ratio"
        if spec.max_aspect_ratio and ar > spec.max_aspect_ratio:
            return False, "above_max_aspect_ratio"
    if spec.edge_margin:
        m = spec.edge_margin
        if x0 < m and y0 < m and x1 > w - m and y1 > h - m:
            return False, "spans_full_frame_edges"
    cx, cy = (x0 + x1) / 2 / w, (y0 + y1) / 2 / h
    if spec.allowed_regions and not any(_in_region(cx, cy, r) for r in spec.allowed_regions):
        return False, "outside_allowed_region"
    if spec.forbidden_regions and any(_in_region(cx, cy, r) for r in spec.forbidden_regions):
        return False, "inside_forbidden_region"
    return True, "ok"


def _in_region(cx: float, cy: float, region: List[float]) -> bool:
    x0, y0, x1, y1 = region
    return x0 <= cx <= x1 and y0 <= cy <= y1
