"""Structured, dependency-free logging."""
from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Optional


class _KVFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        base = super().format(record)
        extra = getattr(record, "kv", None)
        if extra:
            kvs = " ".join(f"{k}={v}" for k, v in extra.items())
            return f"{base} | {kvs}"
        return base


def setup_logging(level: str = "INFO", logfile: Optional[str] = None) -> logging.Logger:
    root = logging.getLogger("aria_drive_seg")
    root.setLevel(getattr(logging, level.upper(), logging.INFO))
    root.handlers.clear()
    fmt = _KVFormatter("%(asctime)s %(levelname)-7s %(name)s: %(message)s",
                       datefmt="%H:%M:%S")
    sh = logging.StreamHandler(sys.stderr)
    sh.setFormatter(fmt)
    root.addHandler(sh)
    if logfile:
        Path(logfile).parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(logfile)
        fh.setFormatter(fmt)
        root.addHandler(fh)
    root.propagate = False
    return root


def get_logger(name: str = "aria_drive_seg") -> logging.Logger:
    return logging.getLogger(name if name.startswith("aria_drive_seg") else f"aria_drive_seg.{name}")


def log_kv(logger: logging.Logger, level: str, msg: str, **kv) -> None:
    logger.log(getattr(logging, level.upper()), msg, extra={"kv": kv})
