"""Offline network guard (§2).

Monkeypatches the socket layer to BLOCK (and record) any outbound connection to a
non-loopback address. Use it to prove that inference performs no network I/O:

    from aria_drive_seg.netguard import install_netguard, get_violations
    install_netguard(strict=True)
    ...run inference...
    assert not get_violations()

Loopback / unix sockets are allowed (local IPC, e.g. CUDA, X11, torch workers).
"""
from __future__ import annotations

import ipaddress
import os
import socket
from typing import List, Optional, Tuple

_ORIG_CONNECT = socket.socket.connect
_ORIG_CONNECT_EX = socket.socket.connect_ex
_ORIG_CREATE_CONN = socket.create_connection
_VIOLATIONS: List[Tuple[str, str]] = []
_STRICT = True
_ALLOW_HOSTS = set()


def _is_local(address) -> bool:
    try:
        if not isinstance(address, tuple):
            return True  # AF_UNIX etc.
        host = address[0]
        if host in _ALLOW_HOSTS:
            return True
        if host in ("localhost", "", "::1"):
            return True
        ip = ipaddress.ip_address(host)
        return ip.is_loopback or ip.is_link_local or ip.is_unspecified
    except Exception:
        # Non-numeric host (a DNS name) -> treat as REMOTE (blocked/flagged).
        return False


def _record(address) -> None:
    host = address[0] if isinstance(address, tuple) else str(address)
    _VIOLATIONS.append(("connect", str(host)))


def install_netguard(strict: bool = True, allow: Optional[List[str]] = None) -> None:
    global _STRICT
    _STRICT = strict
    if allow:
        _ALLOW_HOSTS.update(allow)
    # also flip the offline env switches
    for k in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_DATASETS_OFFLINE",
              "HF_HUB_DISABLE_TELEMETRY", "DO_NOT_TRACK"):
        os.environ.setdefault(k, "1")

    def guarded_connect(self, address):
        if not _is_local(address):
            _record(address)
            if _STRICT:
                raise OSError(f"[netguard] blocked outbound connection to {address}")
            return
        return _ORIG_CONNECT(self, address)

    def guarded_connect_ex(self, address):
        if not _is_local(address):
            _record(address)
            if _STRICT:
                return 111  # ECONNREFUSED
        return _ORIG_CONNECT_EX(self, address)

    def guarded_create_connection(address, *a, **k):
        if not _is_local(address):
            _record(address)
            if _STRICT:
                raise OSError(f"[netguard] blocked outbound connection to {address}")
        return _ORIG_CREATE_CONN(address, *a, **k)

    socket.socket.connect = guarded_connect
    socket.socket.connect_ex = guarded_connect_ex
    socket.create_connection = guarded_create_connection


def uninstall_netguard() -> None:
    socket.socket.connect = _ORIG_CONNECT
    socket.socket.connect_ex = _ORIG_CONNECT_EX
    socket.create_connection = _ORIG_CREATE_CONN


def get_violations() -> List[Tuple[str, str]]:
    return list(_VIOLATIONS)


def reset_violations() -> None:
    _VIOLATIONS.clear()
