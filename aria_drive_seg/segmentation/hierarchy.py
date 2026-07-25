"""Hierarchical parent->subclass classification (Phase 3 fix).

Fine subclasses (person/vehicle subtypes) must NOT compete on the full frame — that
turns a normal car into `emergency_vehicle` or an adult into `child` on a marginal
score. Instead: detect the PARENT (person / car / truck / …) with its mask, then run
the subclass prompts ONLY on the parent crop and relabel the parent's mask to a
subclass only when the top subclass score is high AND clearly beats the second
alternative. The mask geometry always comes from the parent detection.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple


def decide_subclass(scores: Dict[str, float], min_score: float,
                    min_margin: float) -> Tuple[Optional[str], str, float]:
    """Return (subclass_name|None, status, confidence).

    status: accepted | rejected | ambiguous | no_candidates.
    A subclass replaces the parent label only when accepted."""
    if not scores:
        return None, "no_candidates", 0.0
    items = sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))
    top_name, top = items[0]
    second = items[1][1] if len(items) > 1 else 0.0
    if top < min_score:
        return None, "rejected", float(top)
    if second > 0 and (top - second) < min_margin:
        return None, "ambiguous", float(top)
    return top_name, "accepted", float(top)


def load_hierarchy(doc: dict) -> dict:
    """Parse the `hierarchy` block of grounded_prompts.yaml (safe defaults)."""
    h = doc.get("hierarchy", {}) or {}
    parents = h.get("parents", {}) or {}
    subclasses = set()
    for subs in parents.values():
        subclasses.update(subs)
    return {
        "parents": parents,
        "subclasses": subclasses,
        "min_score": float(h.get("subclass_min_score", 0.35)),
        "min_margin": float(h.get("subclass_min_margin", 0.08)),
        "max_parents_per_frame": int(h.get("max_parents_per_frame", 12)),
        "min_parent_area_frac": float(h.get("min_parent_area_frac", 0.002)),
    }
