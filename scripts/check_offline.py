#!/usr/bin/env python3
"""Run any command under the network guard and report outbound-connection attempts (§2).

    python scripts/check_offline.py -- python -m aria_drive_seg segment --method grounded_sam2 --input val10

Exit code 0 => no non-loopback connection was attempted. Non-zero => a violation
was recorded (the offending host is printed). Implemented via a sitecustomize shim
injected through PYTHONPATH so the guard is active from interpreter start-up,
before transformers/torch import.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

SHIM = """
import atexit, json, os
try:
    from aria_drive_seg.netguard import install_netguard, get_violations
    install_netguard(strict=True)
    @atexit.register
    def _dump():
        v = get_violations()
        p = os.environ.get("NETGUARD_REPORT")
        if p:
            with open(p, "w") as f:
                json.dump(v, f)
except Exception as e:
    import sys; print("[check_offline] shim error:", e, file=sys.stderr)
"""


def main() -> int:
    argv = sys.argv[1:]
    if argv and argv[0] == "--":
        argv = argv[1:]
    if not argv:
        print("usage: check_offline.py -- <command...>", file=sys.stderr)
        return 2

    tmpdir = Path(tempfile.mkdtemp(prefix="netguard_"))
    (tmpdir / "sitecustomize.py").write_text(SHIM)
    report = tmpdir / "violations.json"

    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join([str(tmpdir), env.get("PYTHONPATH", "")])
    env["NETGUARD_REPORT"] = str(report)
    for k in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_DATASETS_OFFLINE"):
        env[k] = "1"

    print(f"[check_offline] running under netguard: {' '.join(argv)}")
    proc = subprocess.run(argv, env=env)

    violations = []
    if report.exists():
        try:
            violations = json.loads(report.read_text())
        except Exception:
            pass
    if violations:
        print(f"\n[check_offline] ✗ {len(violations)} outbound connection attempt(s):")
        for kind, host in violations:
            print(f"    {kind} -> {host}")
        return 1
    print("\n[check_offline] ✓ no non-loopback connection attempts detected")
    return proc.returncode


if __name__ == "__main__":
    raise SystemExit(main())
