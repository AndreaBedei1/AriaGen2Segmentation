"""Atomic writes, lossless uint16 mask I/O, JSONL, and stage manifests (§12, §14)."""
from __future__ import annotations

import contextlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, Optional

import numpy as np

from .hashing import stable_hash


# --------------------------------------------------------------------------- #
# Atomic writes
# --------------------------------------------------------------------------- #
@contextlib.contextmanager
def atomic_write(path: str | Path, mode: str = "wb", encoding: Optional[str] = None):
    """Write to a temp file in the same dir, fsync, then atomically rename.

    Guarantees a reader never sees a half-written file (§5, §14).
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), prefix=".tmp_", suffix=path.suffix)
    try:
        with os.fdopen(fd, mode, encoding=encoding) as f:
            yield f
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    except BaseException:
        with contextlib.suppress(FileNotFoundError):
            os.unlink(tmp)
        raise


def atomic_write_bytes(path: str | Path, data: bytes) -> None:
    with atomic_write(path, "wb") as f:
        f.write(data)


def atomic_write_text(path: str | Path, text: str) -> None:
    with atomic_write(path, "w", encoding="utf-8") as f:
        f.write(text)


def atomic_write_json(path: str | Path, obj: Any, indent: int = 2) -> None:
    atomic_write_text(path, json.dumps(obj, indent=indent, default=_json_default))


def _json_default(o: Any):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, Path):
        return str(o)
    return str(o)


# --------------------------------------------------------------------------- #
# Lossless uint16 mask PNG I/O (§12: numeric id maps must always exist)
# --------------------------------------------------------------------------- #
def write_mask_u16(path: str | Path, mask: np.ndarray) -> None:
    """Write a single-channel uint16 id-mask as a lossless PNG, atomically."""
    if mask.dtype != np.uint16:
        if mask.max() > 65535 or mask.min() < 0:
            raise ValueError("mask values out of uint16 range")
        mask = mask.astype(np.uint16)
    png = _encode_png_u16(mask)
    atomic_write_bytes(path, png)


def read_mask_u16(path: str | Path) -> np.ndarray:
    import cv2  # local import; only needed where masks are handled
    arr = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if arr is None:
        raise IOError(f"could not read mask {path}")
    return arr.astype(np.uint16)


def _encode_png_u16(mask: np.ndarray) -> bytes:
    """Encode uint16 grayscale PNG to bytes. Prefer cv2 (reliable 16-bit)."""
    try:
        import cv2
        ok, buf = cv2.imencode(".png", mask)
        if not ok:
            raise RuntimeError("cv2.imencode failed")
        return buf.tobytes()
    except Exception:
        # Pure-Pillow fallback (little-endian 16-bit grayscale)
        from PIL import Image
        im = Image.fromarray(mask.astype("<u2"), mode="I;16")
        import io as _io
        b = _io.BytesIO()
        im.save(b, format="PNG")
        return b.getvalue()


# --------------------------------------------------------------------------- #
# JSONL append (frame-by-frame metadata / error logs)
# --------------------------------------------------------------------------- #
def append_jsonl(path: str | Path, record: Dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(record, default=_json_default)
    with open(path, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def read_jsonl(path: str | Path) -> Iterator[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


# --------------------------------------------------------------------------- #
# Config fingerprint + stage manifest (resumability & cache invalidation, §14)
# --------------------------------------------------------------------------- #
def config_fingerprint(*parts: Any) -> str:
    """Stable short hash of everything that must invalidate a stage's cache."""
    return stable_hash(list(parts))


class Manifest:
    """A tiny JSON manifest tracking per-stage completion + fingerprint.

    Layout: {"stage": str, "fingerprint": str, "done": {key: meta}, "meta": {...}}
    """

    def __init__(self, path: str | Path, stage: str, fingerprint: str, meta: Optional[Dict] = None):
        self.path = Path(path)
        self.stage = stage
        self.fingerprint = fingerprint
        self.meta = meta or {}
        self.done: Dict[str, Any] = {}

    @classmethod
    def load_or_new(cls, path: str | Path, stage: str, fingerprint: str,
                    meta: Optional[Dict] = None) -> "Manifest":
        m = cls(path, stage, fingerprint, meta)
        p = Path(path)
        if p.exists():
            try:
                data = json.loads(p.read_text())
            except Exception:
                return m  # corrupt manifest -> start fresh
            if data.get("fingerprint") == fingerprint and data.get("stage") == stage:
                m.done = data.get("done", {})
                m.meta = data.get("meta", meta or {})
            # else: fingerprint changed -> stale cache, start fresh (invalidation)
        return m

    def is_done(self, key: str) -> bool:
        return str(key) in self.done

    def mark(self, key: str, value: Any = True) -> None:
        self.done[str(key)] = value

    def save(self) -> None:
        atomic_write_json(self.path, {
            "stage": self.stage,
            "fingerprint": self.fingerprint,
            "meta": self.meta,
            "done": self.done,
        })


def file_ok(path: str | Path, min_bytes: int = 1) -> bool:
    p = Path(path)
    return p.exists() and p.stat().st_size >= min_bytes
