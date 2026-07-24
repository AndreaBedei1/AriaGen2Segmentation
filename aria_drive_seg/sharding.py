"""Disjoint frame sharding for multi-GPU segmentation (§13).

One independent process per GPU gets a DISJOINT shard of frame indices; a common
manifest tracks completion; outputs are keyed by frame_index so recomposition is
just an ordered merge. Round-robin keeps per-shard load balanced when frame cost
varies along the recording.
"""
from __future__ import annotations

from typing import Any, Dict, List, Sequence


def shard_indices(all_indices: Sequence[int], num_shards: int, shard_id: int) -> List[int]:
    if num_shards <= 0:
        raise ValueError("num_shards must be >= 1")
    if not (0 <= shard_id < num_shards):
        raise ValueError("shard_id out of range")
    return [all_indices[i] for i in range(shard_id, len(all_indices), num_shards)]


def assert_disjoint_cover(all_indices: Sequence[int], num_shards: int) -> bool:
    seen: set = set()
    for s in range(num_shards):
        part = set(shard_indices(all_indices, num_shards, s))
        if seen & part:
            return False
        seen |= part
    return seen == set(all_indices)


def recompose(shard_results: List[Dict[int, Any]]) -> List[Any]:
    """Merge per-shard {frame_index: result} maps into a frame_index-ordered list."""
    merged: Dict[int, Any] = {}
    for part in shard_results:
        merged.update(part)
    return [merged[k] for k in sorted(merged)]


def parse_devices(spec: str, available: int) -> List[int]:
    """'auto' -> all; '0' -> [0]; '0,1' -> [0,1]."""
    if spec in (None, "auto", ""):
        return list(range(available))
    return [int(x) for x in str(spec).split(",") if x.strip() != ""]
