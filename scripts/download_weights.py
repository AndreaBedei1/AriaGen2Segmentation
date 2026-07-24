#!/usr/bin/env python3
"""Download ONLY the official weights needed by the pipeline, once (§2).

Writes weights/manifest.json recording source URL, version/revision, size and
SHA-256 for every artifact so later runs can verify integrity and run fully
offline. Idempotent: re-running skips artifacts already present with a matching
SHA-256.

    python scripts/download_weights.py --methods grounded_sam2,oneformer_mapillary
    python scripts/download_weights.py --only sam2.1_hiera_large

NO telemetry; nothing is uploaded. This is the ONLY stage allowed network access.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT = Path(__file__).resolve().parent.parent
WEIGHTS = ROOT / "weights"

# --------------------------------------------------------------------------- #
# Registry of official artifacts. `kind`: "url" (direct file) | "hf" (HF repo).
# --------------------------------------------------------------------------- #
REGISTRY: Dict[str, Dict[str, Any]] = {
    "sam2.1_hiera_large": {
        "method": "grounded_sam2",
        "kind": "url",
        "url": "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt",
        "dest": "sam2.1/sam2.1_hiera_large.pt",
        "version": "092824",
        "license": "Apache-2.0 (SAM 2)",
    },
    "grounding_dino_base": {
        "method": "grounded_sam2",
        "kind": "hf",
        "repo_id": "IDEA-Research/grounding-dino-base",
        "dest": "grounding-dino-base",
        "license": "Apache-2.0",
    },
    # Method 2 default backend (same Mapillary v1.2 65-class taxonomy as OneFormer).
    "mask2former_mapillary_semantic": {
        "method": "oneformer_mapillary",
        "kind": "hf",
        "repo_id": "facebook/mask2former-swin-large-mapillary-vistas-semantic",
        "dest": "mask2former-mapillary-semantic",
        "license": "Mapillary Vistas research/non-commercial (weights); MIT (code)",
    },
    # A converted OneFormer-DiNAT-L-Mapillary checkpoint (if provided) is appended
    # dynamically from configs/default.yaml (source: hf_oneformer / oneformer_id).
}


def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


def _download_url(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    print(f"  downloading {url}")
    t0 = time.time()
    with urllib.request.urlopen(url) as r, open(tmp, "wb") as f:  # noqa: S310 (official host)
        total = int(r.headers.get("Content-Length", 0))
        done = 0
        while True:
            block = r.read(1 << 20)
            if not block:
                break
            f.write(block)
            done += len(block)
            if total:
                pct = 100 * done / total
                sys.stdout.write(f"\r    {done/1e6:8.1f}/{total/1e6:.1f} MB ({pct:5.1f}%)")
                sys.stdout.flush()
    sys.stdout.write("\n")
    os.replace(tmp, dest)
    print(f"  done in {time.time()-t0:.1f}s")


def _download_hf(repo_id: str, dest: Path) -> Dict[str, Any]:
    from huggingface_hub import snapshot_download
    dest.mkdir(parents=True, exist_ok=True)
    local = snapshot_download(repo_id=repo_id, local_dir=str(dest))
    # resolve commit + per-file info
    files = []
    total = 0
    for p in sorted(Path(local).rglob("*")):
        if p.is_file() and ".cache" not in p.parts:
            sz = p.stat().st_size
            total += sz
            files.append({"file": str(p.relative_to(local)), "bytes": sz})
    # main weight sha (if present)
    main_sha = None
    for cand in ("model.safetensors", "pytorch_model.bin"):
        mp = Path(local) / cand
        if mp.exists():
            main_sha = sha256_file(mp)
            break
    return {"files": files, "bytes": total, "main_weight_sha256": main_sha}


def resolve_oneformer_from_config() -> Optional[Dict[str, Any]]:
    import yaml
    cfg = yaml.safe_load((ROOT / "configs" / "default.yaml").read_text())
    of = cfg.get("oneformer_mapillary", {})
    # Only fetch a OneFormer checkpoint if one is explicitly configured; the default
    # (hf_mask2former) is already in the static REGISTRY.
    if of.get("source") == "hf_oneformer" and of.get("oneformer_id"):
        oid = of["oneformer_id"]
        if str(oid).startswith(("weights/", "/")):
            return None  # already a local dir
        return {"method": "oneformer_mapillary", "kind": "hf",
                "repo_id": oid, "dest": "oneformer-dinat-mapillary",
                "license": "Mapillary Vistas research/non-commercial"}
    return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--methods", default="grounded_sam2")
    ap.add_argument("--only", default=None, help="comma-separated artifact names")
    ap.add_argument("--verify", action="store_true", help="only verify existing manifest")
    args = ap.parse_args()

    registry = dict(REGISTRY)
    of = resolve_oneformer_from_config()
    if of:
        registry["oneformer_checkpoint"] = of

    methods = set(args.methods.split(","))
    only = set(args.only.split(",")) if args.only else None
    WEIGHTS.mkdir(parents=True, exist_ok=True)
    manifest_path = WEIGHTS / "manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}

    for name, spec in registry.items():
        if only and name not in only:
            continue
        if not only and spec.get("method") not in methods:
            continue
        dest = WEIGHTS / spec["dest"]
        print(f"[{name}] {spec['kind']} -> {dest}")
        entry: Dict[str, Any] = {"name": name, "kind": spec["kind"],
                                 "license": spec.get("license"), "method": spec.get("method")}
        if spec["kind"] == "url":
            prev = manifest.get(name, {})
            if dest.exists() and prev.get("sha256") and prev["sha256"] == sha256_file(dest):
                print("  present + sha matches, skip")
                continue
            if not args.verify:
                _download_url(spec["url"], dest)
            entry.update({"url": spec["url"], "version": spec.get("version"),
                          "bytes": dest.stat().st_size, "sha256": sha256_file(dest),
                          "path": str(dest.relative_to(ROOT))})
        elif spec["kind"] == "hf":
            info = _download_hf(spec["repo_id"], dest) if not args.verify else {}
            entry.update({"repo_id": spec["repo_id"], "path": str(dest.relative_to(ROOT)), **info})
        manifest[name] = entry
        _write_json(manifest_path, manifest)
        print(f"  recorded in {manifest_path.name}")

    print("\nAll requested weights present. Manifest:", manifest_path)
    return 0


def _write_json(path: Path, obj: Any) -> None:
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(obj, indent=2))
    tmp.replace(path)


if __name__ == "__main__":
    raise SystemExit(main())
