"""`run-all` (§15): orchestrate inspect -> extract -> segment(each) -> align-gaze
-> analyze -> render.

Stages need different environments (VRS I/O needs projectaria_tools; segmentation
needs torch). run-all dispatches each stage to the right interpreter, resolved from
env vars so NO absolute paths live in code:

    ARIA_VRS_PYTHON  interpreter with projectaria_tools + opencv (inspect/extract/gaze/analyze/render)
    ARIA_ML_PYTHON   interpreter with torch (segment grounded_sam2 / oneformer_mapillary)
    ARIA_ONEFORMER_PYTHON  optional: interpreter for the OneFormer-DiNAT env (else ARIA_ML_PYTHON)

Any unset var falls back to the current interpreter. Each stage is resumable, so a
failed run can be re-invoked and it will skip completed work.
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from typing import List, Optional

from .config import Config
from .logging_utils import get_logger

log = get_logger("run-all")


def _py(var: str, default: Optional[str] = None) -> str:
    return os.environ.get(var) or default or sys.executable


def _stage(python: str, args: List[str]) -> int:
    cmd = [python, "-m", "aria_drive_seg", *args]
    log.info("stage: %s", " ".join(args[:2]))
    r = subprocess.run(cmd)
    if r.returncode != 0:
        log.error("stage failed (rc=%d): %s", r.returncode, " ".join(args))
    return r.returncode


def run_all(vrs: str, output: str, cfg: Config, methods: List[str],
            resume: bool = True, force: bool = False) -> int:
    vrs_py = _py("ARIA_VRS_PYTHON")
    ml_py = _py("ARIA_ML_PYTHON")
    of_py = _py("ARIA_ONEFORMER_PYTHON", ml_py)

    common: List[str] = []
    if not resume:
        common.append("--no-resume")
    if force:
        common.append("--force")

    # 1) inspect
    rc = _stage(vrs_py, ["inspect", "--vrs", vrs, "--output", str(Path(output) / "inspection")])
    if rc:
        return rc
    # 2) extract
    rc = _stage(vrs_py, ["extract", "--vrs", vrs, "--output", output, *common])
    if rc:
        return rc
    # 3) segmentation (each method, in its ML env)
    for m in methods:
        py = of_py if (m == "oneformer_mapillary" and cfg.get("oneformer_mapillary.source") == "hf_oneformer") else ml_py
        rc = _stage(py, ["segment", "--method", m, "--input", output, *common])
        if rc:
            log.warning("segmentation %s failed; continuing", m)
    # 4) gaze
    _stage(vrs_py, ["align-gaze", "--vrs", vrs, "--input", output])
    # 5) analyze
    _stage(vrs_py, ["analyze", "--input", output])
    # 6) render
    _stage(vrs_py, ["render", "--input", output])
    log.info("run-all complete -> %s", output)
    return 0
