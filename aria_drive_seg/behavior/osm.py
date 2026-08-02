"""OpenStreetMap road network: local cache, projection and feature extraction.

The network is fetched once per bounding box from the Overpass API and cached on
disk as the **raw** Overpass JSON, so every later run is reproducible offline from
the cache and the parsing can change without refetching.

Two rules the rest of the analysis depends on:

* a tag that is absent is `None`, never a default. OSM `maxspeed`, `lanes` and
  `oneway` are frequently unmapped, and treating "unmapped" as "one lane, two-way,
  50 km/h" would manufacture road context that nobody surveyed;
* the cache and everything derived from it hold **absolute coordinates**, so they
  live in the local output tree and are never committed.
"""
from __future__ import annotations

import hashlib
import json
import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

#: Highway values that carry vehicle traffic. Footways and cycleways are kept out
#: of the matchable network: snapping a car onto a footpath is a silent error.
DRIVEABLE_HIGHWAY = {
    "motorway", "motorway_link", "trunk", "trunk_link", "primary",
    "primary_link", "secondary", "secondary_link", "tertiary", "tertiary_link",
    "unclassified", "residential", "living_street", "service", "road",
}

#: Node tags that mark a control point on the road.
NODE_FEATURE_TAGS = {
    "highway": {"traffic_signals", "stop", "give_way", "crossing",
                "mini_roundabout", "turning_circle"},
    "railway": {"level_crossing", "crossing"},
}

MEAN_EARTH_RADIUS_M = 6_371_008.8


# --------------------------------------------------------------------------- #
# Fetch and cache
# --------------------------------------------------------------------------- #
def bbox_for_track(lat: Sequence[float], lon: Sequence[float],
                   margin_m: float = 300.0) -> Tuple[float, float, float, float]:
    """(south, west, north, east) around a track, with a metric margin."""
    lat = np.asarray(lat, dtype=float)
    lon = np.asarray(lon, dtype=float)
    if lat.size == 0:
        raise ValueError("cannot build a bounding box from an empty track")
    dlat = margin_m / 110_540.0
    dlon = margin_m / (111_320.0 * max(math.cos(math.radians(float(np.mean(lat)))), 1e-6))
    return (float(lat.min() - dlat), float(lon.min() - dlon),
            float(lat.max() + dlat), float(lon.max() + dlon))


def _cache_key(bbox: Tuple[float, float, float, float], query_version: str) -> str:
    raw = json.dumps({"bbox": [round(v, 5) for v in bbox], "q": query_version},
                     sort_keys=True)
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


OVERPASS_QUERY_VERSION = "drive_v1"


def _overpass_query(bbox: Tuple[float, float, float, float], timeout_s: int) -> str:
    s, w, n, e = bbox
    return (
        f"[out:json][timeout:{int(timeout_s)}];\n"
        f'(way["highway"]({s:.6f},{w:.6f},{n:.6f},{e:.6f}););\n'
        "(._;>;);\n"
        "out body;\n"
    )


def fetch_osm(bbox: Tuple[float, float, float, float],
              cache_dir: str | Path,
              endpoint: str | Sequence[str] = "https://overpass-api.de/api/interpreter",
              timeout_s: int = 180,
              allow_network: bool = True,
              retries: int = 2) -> Dict[str, Any]:
    """Return the raw Overpass payload for a bbox, fetching only on a cache miss.

    `endpoint` may be a list: the public Overpass instances rate-limit and return
    504 under load, and a research run should not fail because one mirror is busy.
    Mirrors are tried in order and the one that answered is recorded next to the
    cache, because which mirror served the data is part of provenance.

    Raises `FileNotFoundError` when the cache is cold and the network is refused,
    so a run without connectivity fails loudly instead of quietly analysing a
    route with no map.
    """
    cache_dir = Path(cache_dir)
    key = _cache_key(bbox, OVERPASS_QUERY_VERSION)
    path = cache_dir / f"overpass_{key}.json"
    if path.exists():
        return json.loads(path.read_text())
    if not allow_network:
        raise FileNotFoundError(
            f"no OSM cache at {path} and network access is disabled; run once "
            "with network access, or import a local .osm/.geojson extract")

    import urllib.request

    endpoints = [endpoint] if isinstance(endpoint, str) else list(endpoint)
    query = _overpass_query(bbox, timeout_s)
    failures: List[str] = []
    payload = None
    used = None
    for ep in endpoints:
        for attempt in range(retries):
            try:
                req = urllib.request.Request(
                    ep, data=query.encode("utf-8"),
                    headers={"User-Agent":
                             "aria-drive-seg/article1 (research pilot)"})
                with urllib.request.urlopen(req, timeout=timeout_s + 30) as resp:
                    payload = json.loads(resp.read().decode("utf-8"))
                used = ep
                break
            except Exception as exc:                          # noqa: BLE001
                failures.append(f"{ep} attempt {attempt + 1}: {exc}")
                if attempt < retries - 1:
                    time.sleep(5.0 * (attempt + 1))
        if payload is not None:
            break
    if payload is None:
        raise RuntimeError("Overpass fetch failed on every endpoint:\n  " +
                           "\n  ".join(failures))

    cache_dir.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(payload))
    tmp.replace(path)
    meta = cache_dir / f"overpass_{key}.meta.json"
    meta.write_text(json.dumps({
        "bbox": list(bbox), "endpoint_used": used, "endpoints_tried": endpoints,
        "failures": failures,
        "query_version": OVERPASS_QUERY_VERSION, "query": query,
        "elements": len(payload.get("elements", [])),
        "fetched_epoch_s": time.time(),
        "note": "raw Overpass payload; contains absolute coordinates, never commit",
    }, indent=2))
    return payload


# --------------------------------------------------------------------------- #
# Parsing
# --------------------------------------------------------------------------- #
def _tag_int(tags: Dict[str, str], key: str) -> Optional[int]:
    """Parse an integer tag, returning None for absent or non-numeric values."""
    raw = tags.get(key)
    if raw is None:
        return None
    try:
        return int(str(raw).split(";")[0].strip())
    except (TypeError, ValueError):
        return None


def parse_maxspeed_kph(tags: Dict[str, str]) -> Optional[float]:
    """`maxspeed` in km/h. Returns None when absent or not a plain speed.

    `walk`, `none`, `signals` and national-default codes are deliberately not
    converted to a number: they are not a surveyed speed limit.
    """
    raw = tags.get("maxspeed")
    if raw is None:
        return None
    text = str(raw).strip().lower()
    try:
        if text.endswith("mph"):
            return float(text.replace("mph", "").strip()) * 1.609344
        return float(text)
    except ValueError:
        return None


@dataclass
class Way:
    osm_id: int
    node_ids: List[int]
    tags: Dict[str, str]

    @property
    def highway(self) -> Optional[str]:
        return self.tags.get("highway")

    @property
    def is_roundabout(self) -> bool:
        return (self.tags.get("junction") in ("roundabout", "circular")
                or self.tags.get("highway") == "mini_roundabout")

    @property
    def oneway(self) -> Optional[bool]:
        raw = self.tags.get("oneway")
        if raw is None:
            # A roundabout is one-way by OSM convention even when untagged.
            return True if self.is_roundabout else None
        if raw in ("yes", "true", "1", "-1"):
            return True
        if raw in ("no", "false", "0"):
            return False
        return None

    @property
    def reversed_oneway(self) -> bool:
        return self.tags.get("oneway") == "-1"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "osm_way_id": self.osm_id,
            "highway": self.highway,
            "name_present": "name" in self.tags,
            "is_roundabout": self.is_roundabout,
            "junction": self.tags.get("junction"),
            "oneway": self.oneway,
            "lanes": _tag_int(self.tags, "lanes"),
            "maxspeed_kph": parse_maxspeed_kph(self.tags),
            "bridge": self.tags.get("bridge") not in (None, "no"),
            "tunnel": self.tags.get("tunnel") not in (None, "no"),
            "is_service": self.highway == "service",
            "surface": self.tags.get("surface"),
        }


@dataclass
class RoadNetwork:
    """A projected, segment-indexed driveable road network."""

    lat0: float
    lon0: float
    ways: List[Way]
    node_xy: Dict[int, Tuple[float, float]]
    node_tags: Dict[int, Dict[str, str]]

    # Segment table, filled by `_index`.
    seg_a: np.ndarray = field(default_factory=lambda: np.zeros((0, 2)))
    seg_b: np.ndarray = field(default_factory=lambda: np.zeros((0, 2)))
    seg_way: np.ndarray = field(default_factory=lambda: np.zeros(0, np.int64))
    seg_offset_m: np.ndarray = field(default_factory=lambda: np.zeros(0))
    seg_length_m: np.ndarray = field(default_factory=lambda: np.zeros(0))
    seg_heading_deg: np.ndarray = field(default_factory=lambda: np.zeros(0))
    way_length_m: np.ndarray = field(default_factory=lambda: np.zeros(0))

    # Point features of interest, in projected metres.
    feature_xy: Dict[str, np.ndarray] = field(default_factory=dict)

    def project(self, lat, lon) -> Tuple[np.ndarray, np.ndarray]:
        """Equirectangular projection about the network origin, in metres.

        Valid because a single drive spans a few kilometres: over that extent the
        distortion of an equirectangular projection is far below the GPS error.
        """
        lat = np.asarray(lat, dtype=float)
        lon = np.asarray(lon, dtype=float)
        x = np.radians(lon - self.lon0) * MEAN_EARTH_RADIUS_M * math.cos(
            math.radians(self.lat0))
        y = np.radians(lat - self.lat0) * MEAN_EARTH_RADIUS_M
        return x, y

    def unproject(self, x, y) -> Tuple[np.ndarray, np.ndarray]:
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        lat = self.lat0 + np.degrees(y / MEAN_EARTH_RADIUS_M)
        lon = self.lon0 + np.degrees(
            x / (MEAN_EARTH_RADIUS_M * math.cos(math.radians(self.lat0))))
        return lat, lon

    def summary(self) -> Dict[str, Any]:
        tagged = lambda k: int(sum(1 for w in self.ways if w.to_dict()[k] is not None))
        return {
            "ways": len(self.ways),
            "segments": int(self.seg_way.size),
            "total_length_km": float(self.way_length_m.sum() / 1000.0),
            "roundabout_ways": int(sum(1 for w in self.ways if w.is_roundabout)),
            "ways_with_lanes_tag": tagged("lanes"),
            "ways_with_maxspeed_tag": tagged("maxspeed_kph"),
            "ways_with_oneway_tag": tagged("oneway"),
            "highway_classes": _counts([w.highway for w in self.ways]),
            "point_features": {k: int(v.shape[0]) for k, v in self.feature_xy.items()},
            "unmapped_tags_are_none": True,
        }


def _counts(values: Sequence[Optional[str]]) -> Dict[str, int]:
    out: Dict[str, int] = {}
    for v in values:
        out[str(v)] = out.get(str(v), 0) + 1
    return dict(sorted(out.items(), key=lambda kv: -kv[1]))


def build_network(payload: Dict[str, Any], lat0: float, lon0: float,
                  driveable_only: bool = True) -> RoadNetwork:
    """Parse an Overpass payload into a projected, segment-indexed network."""
    nodes: Dict[int, Tuple[float, float]] = {}
    node_tags: Dict[int, Dict[str, str]] = {}
    raw_ways: List[Way] = []

    for el in payload.get("elements", []):
        if el.get("type") == "node":
            nodes[int(el["id"])] = (float(el["lat"]), float(el["lon"]))
            if el.get("tags"):
                node_tags[int(el["id"])] = dict(el["tags"])
        elif el.get("type") == "way":
            tags = dict(el.get("tags") or {})
            if "highway" not in tags:
                continue
            if driveable_only and tags["highway"] not in DRIVEABLE_HIGHWAY:
                continue
            ids = [int(n) for n in el.get("nodes", [])]
            if len(ids) >= 2:
                raw_ways.append(Way(int(el["id"]), ids, tags))

    net = RoadNetwork(lat0=lat0, lon0=lon0, ways=raw_ways,
                      node_xy={}, node_tags=node_tags)
    for nid, (la, lo) in nodes.items():
        x, y = net.project(la, lo)
        net.node_xy[nid] = (float(x), float(y))
    _index(net, nodes)
    return net


def _index(net: RoadNetwork, nodes: Dict[int, Tuple[float, float]]) -> None:
    """Build the segment table and the point-feature index."""
    a_list, b_list, way_list, off_list = [], [], [], []
    way_lengths = np.zeros(len(net.ways), dtype=float)

    for wi, way in enumerate(net.ways):
        pts = [net.node_xy[n] for n in way.node_ids if n in net.node_xy]
        if len(pts) < 2:
            continue
        arr = np.asarray(pts, dtype=float)
        seg_len = np.sqrt(((arr[1:] - arr[:-1]) ** 2).sum(axis=1))
        offsets = np.concatenate([[0.0], np.cumsum(seg_len)[:-1]])
        a_list.append(arr[:-1])
        b_list.append(arr[1:])
        way_list.append(np.full(seg_len.size, wi, dtype=np.int64))
        off_list.append(offsets)
        way_lengths[wi] = float(seg_len.sum())

    if a_list:
        net.seg_a = np.concatenate(a_list)
        net.seg_b = np.concatenate(b_list)
        net.seg_way = np.concatenate(way_list)
        net.seg_offset_m = np.concatenate(off_list)
        d = net.seg_b - net.seg_a
        net.seg_length_m = np.sqrt((d ** 2).sum(axis=1))
        net.seg_heading_deg = np.degrees(np.arctan2(d[:, 0], d[:, 1])) % 360.0
    net.way_length_m = way_lengths

    # --- point features -----------------------------------------------------
    features: Dict[str, List[Tuple[float, float]]] = {}
    for nid, tags in net.node_tags.items():
        if nid not in net.node_xy:
            continue
        for key, values in NODE_FEATURE_TAGS.items():
            v = tags.get(key)
            if v in values:
                features.setdefault(v, []).append(net.node_xy[nid])

    # Junctions: nodes shared by two or more distinct driveable ways. This is the
    # topological definition; an OSM node is not tagged "junction".
    use_count: Dict[int, set] = {}
    for wi, way in enumerate(net.ways):
        for nid in way.node_ids:
            use_count.setdefault(nid, set()).add(wi)
    features["junction"] = [net.node_xy[n] for n, ws in use_count.items()
                            if len(ws) >= 2 and n in net.node_xy]

    # Roundabouts: the centroid of each roundabout way, plus its nodes, so that
    # "distance to roundabout" means distance to the circle, not to one node.
    rb: List[Tuple[float, float]] = []
    for way in net.ways:
        if not way.is_roundabout:
            continue
        rb.extend(net.node_xy[n] for n in way.node_ids if n in net.node_xy)
    features["roundabout"] = rb

    net.feature_xy = {k: (np.asarray(v, dtype=float).reshape(-1, 2))
                      for k, v in features.items() if v}


def distance_to_features(net: RoadNetwork, x: np.ndarray, y: np.ndarray,
                         feature: str) -> np.ndarray:
    """Euclidean distance from each point to the nearest feature of a kind.

    Returns `inf` where the network has no such feature, which is honest: the
    absence of a mapped traffic signal is not the same as being far from one.
    """
    pts = net.feature_xy.get(feature)
    out = np.full(np.asarray(x).shape, np.inf, dtype=float)
    if pts is None or pts.size == 0:
        return out
    p = np.column_stack([np.asarray(x, float), np.asarray(y, float)])
    # Chunked to keep the pairwise matrix bounded for long tracks.
    chunk = max(1, int(4_000_000 // max(pts.shape[0], 1)))
    for i in range(0, p.shape[0], chunk):
        blk = p[i:i + chunk]
        d = np.sqrt(((blk[:, None, :] - pts[None, :, :]) ** 2).sum(axis=2))
        out[i:i + chunk] = d.min(axis=1)
    return out


# --------------------------------------------------------------------------- #
# Geometry helpers
# --------------------------------------------------------------------------- #
def point_segment_projection(px: np.ndarray, py: np.ndarray,
                             ax: np.ndarray, ay: np.ndarray,
                             bx: np.ndarray, by: np.ndarray
                             ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Perpendicular projection of points onto segments.

    Returns (distance, t, foot) with `t` clamped to [0, 1] so the foot never
    leaves the segment.
    """
    abx, aby = bx - ax, by - ay
    denom = abx * abx + aby * aby
    denom = np.where(denom <= 0, 1e-12, denom)
    t = ((px - ax) * abx + (py - ay) * aby) / denom
    t = np.clip(t, 0.0, 1.0)
    fx, fy = ax + t * abx, ay + t * aby
    return np.sqrt((px - fx) ** 2 + (py - fy) ** 2), t, np.stack([fx, fy], axis=-1)


def angular_difference_deg(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Smallest absolute difference between two headings, in [0, 180]."""
    d = np.abs(np.asarray(a, float) - np.asarray(b, float)) % 360.0
    return np.where(d > 180.0, 360.0 - d, d)


def menger_curvature(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Local curvature (1/m) from consecutive point triples.

    Endpoints get the curvature of their neighbour rather than a zero, because a
    zero would read as "straight" where the truth is "not computable".
    """
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    n = x.size
    k = np.full(n, np.nan)
    if n < 3:
        return k
    ax, ay = x[:-2], y[:-2]
    bx, by = x[1:-1], y[1:-1]
    cx, cy = x[2:], y[2:]
    area2 = np.abs((bx - ax) * (cy - ay) - (by - ay) * (cx - ax))
    d_ab = np.hypot(bx - ax, by - ay)
    d_bc = np.hypot(cx - bx, cy - by)
    d_ca = np.hypot(ax - cx, ay - cy)
    denom = d_ab * d_bc * d_ca
    with np.errstate(divide="ignore", invalid="ignore"):
        k[1:-1] = np.where(denom > 1e-9, 2.0 * area2 / denom, 0.0)
    k[0], k[-1] = k[1], k[-2]
    return k
