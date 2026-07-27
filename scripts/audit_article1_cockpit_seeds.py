#!/usr/bin/env python3
"""Audit reviewed cockpit seeds against final masks; emits no values without GT."""
from __future__ import annotations

import argparse
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", required=True)
    ap.add_argument("--reviewed", required=True)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()
    if not Path(args.reviewed).exists():
        raise SystemExit("reviewed annotations absent; audit intentionally not produced")
    raise SystemExit("CVAT mask matching will be enabled once a reviewed export is supplied")


if __name__ == "__main__":
    main()
