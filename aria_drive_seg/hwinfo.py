"""Hardware detection (§13) -> hardware_report.json. Works with or without torch."""
from __future__ import annotations

import json
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional


def _nvidia_smi_query() -> List[Dict[str, Any]]:
    fields = ["name", "memory.total", "compute_cap", "driver_version"]
    try:
        out = subprocess.check_output(
            ["nvidia-smi", f"--query-gpu={','.join(fields)}",
             "--format=csv,noheader,nounits"],
            stderr=subprocess.DEVNULL, text=True, timeout=15,
        )
    except Exception:
        return []
    gpus = []
    for line in out.strip().splitlines():
        vals = [v.strip() for v in line.split(",")]
        if len(vals) == len(fields):
            gpus.append({
                "name": vals[0],
                "vram_mb": _to_int(vals[1]),
                "compute_capability": vals[2],
                "driver_version": vals[3],
            })
    return gpus


def _to_int(x: str) -> Optional[int]:
    try:
        return int(float(x))
    except Exception:
        return None


def _cpu_mem() -> Dict[str, Any]:
    info: Dict[str, Any] = {
        "logical_cores": os.cpu_count(),
        "physical_cores": None,
        "ram_total_gb": None,
        "platform": platform.platform(),
        "python": sys.version.split()[0],
    }
    try:
        import psutil  # optional
        info["physical_cores"] = psutil.cpu_count(logical=False)
        info["ram_total_gb"] = round(psutil.virtual_memory().total / 1e9, 1)
    except Exception:
        # /proc fallbacks
        try:
            with open("/proc/meminfo") as f:
                for line in f:
                    if line.startswith("MemTotal:"):
                        info["ram_total_gb"] = round(int(line.split()[1]) * 1024 / 1e9, 1)
                        break
        except Exception:
            pass
        try:
            physical = set()
            core = phys = None
            with open("/proc/cpuinfo") as f:
                for line in f:
                    if line.startswith("physical id"):
                        phys = line.split(":")[1].strip()
                    elif line.startswith("core id"):
                        core = line.split(":")[1].strip()
                        if phys is not None:
                            physical.add((phys, core))
            if physical:
                info["physical_cores"] = len(physical)
        except Exception:
            pass
    return info


def _torch_info() -> Dict[str, Any]:
    out: Dict[str, Any] = {"available": False}
    try:
        import torch
        out["available"] = True
        out["torch_version"] = torch.__version__
        out["cuda_version"] = torch.version.cuda
        out["cudnn"] = torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else None
        out["cuda_available"] = torch.cuda.is_available()
        gpus = []
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                p = torch.cuda.get_device_properties(i)
                bf16 = False
                try:
                    bf16 = torch.cuda.is_bf16_supported()
                except Exception:
                    pass
                gpus.append({
                    "index": i, "name": p.name,
                    "vram_mb": round(p.total_memory / (1024 ** 2)),
                    "compute_capability": f"{p.major}.{p.minor}",
                    "multi_processor_count": p.multi_processor_count,
                    "bf16_supported": bf16,
                })
        out["gpus"] = gpus
    except Exception as e:  # torch not installed in this env
        out["error"] = str(e)
    return out


def collect_hardware() -> Dict[str, Any]:
    smi = _nvidia_smi_query()
    torch_info = _torch_info()
    cpu = _cpu_mem()
    disk = {}
    try:
        du = shutil.disk_usage(str(Path.cwd()))
        disk = {"total_gb": round(du.total / 1e9, 1), "free_gb": round(du.free / 1e9, 1)}
    except Exception:
        pass
    # Prefer torch GPU view; fall back to nvidia-smi
    gpus = torch_info.get("gpus") or smi
    return {
        "gpus": gpus,
        "num_gpus": len(gpus),
        "nvidia_smi": smi,
        "torch": torch_info,
        "cpu_mem": cpu,
        "disk": disk,
    }


def write_hardware_report(out_path: str | Path) -> Dict[str, Any]:
    from .io_utils import atomic_write_json
    info = collect_hardware()
    atomic_write_json(out_path, info)
    return info


if __name__ == "__main__":
    print(json.dumps(collect_hardware(), indent=2))
