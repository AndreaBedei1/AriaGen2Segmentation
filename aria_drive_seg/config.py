"""Configuration loading with deep-merge overrides and project-root resolution."""
from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml


def _deep_merge(base: Dict, override: Dict) -> Dict:
    out = copy.deepcopy(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


class Config:
    def __init__(self, data: Dict[str, Any], project_root: Path):
        self._d = data
        self.project_root = project_root

    @classmethod
    def load(cls, path: Optional[str | Path] = None,
             overrides: Optional[Dict] = None,
             project_root: Optional[str | Path] = None) -> "Config":
        root = Path(project_root) if project_root else _find_project_root()
        default_path = root / "configs" / "default.yaml"
        data = yaml.safe_load(default_path.read_text()) if default_path.exists() else {}
        if path:
            user = yaml.safe_load(Path(path).read_text()) or {}
            data = _deep_merge(data, user)
        if overrides:
            data = _deep_merge(data, overrides)
        return cls(data, root)

    # dict-like access
    def __getitem__(self, key: str) -> Any:
        return self._d[key]

    def get(self, dotted: str, default: Any = None) -> Any:
        cur: Any = self._d
        for part in dotted.split("."):
            if not isinstance(cur, dict) or part not in cur:
                return default
            cur = cur[part]
        return cur

    def set(self, dotted: str, value: Any) -> None:
        parts = dotted.split(".")
        cur = self._d
        for p in parts[:-1]:
            cur = cur.setdefault(p, {})
        cur[parts[-1]] = value

    def resolve(self, path_like: str | Path) -> Path:
        """Resolve a possibly-relative config path against the project root."""
        p = Path(path_like)
        return p if p.is_absolute() else (self.project_root / p)

    @property
    def data(self) -> Dict[str, Any]:
        return self._d

    def as_dict(self) -> Dict[str, Any]:
        return copy.deepcopy(self._d)


def _find_project_root(start: Optional[Path] = None) -> Path:
    cur = (start or Path(__file__)).resolve()
    for parent in [cur, *cur.parents]:
        if (parent / "configs" / "default.yaml").exists() or (parent / "pyproject.toml").exists():
            if (parent / "configs").exists() or (parent / "pyproject.toml").exists():
                return parent
    return Path.cwd()


def apply_cli_overrides(cfg: Config, args) -> Config:
    """Fold common CLI flags (§15) into the config."""
    mapping = {
        "start_time": "frames.start_time_s",
        "end_time": "frames.end_time_s",
        "frame_step": "frames.frame_step",
        "max_frames": "frames.max_frames",
        "devices": "hardware.devices",
        "workers": "hardware.workers",
        "batch_size": "hardware.batch_size",
    }
    for attr, dotted in mapping.items():
        val = getattr(args, attr, None)
        if val is not None:
            cfg.set(dotted, val)
    return cfg
