#!/usr/bin/env python3
"""Validate Article 1 CVAT exports without promoting seeds to ground truth."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--export", required=True)
    ap.add_argument("--labels", required=True)
    ap.add_argument("--reviewed-marker", required=True)
    args = ap.parse_args()
    export, marker = Path(args.export), Path(args.reviewed_marker)
    if not export.exists():
        raise SystemExit("CVAT export not found")
    if not marker.exists():
        raise SystemExit("review marker missing: export remains unreviewed and cannot be GT")
    labels = set(json.loads(Path(args.labels).read_text())["labels"])
    if not labels:
        raise SystemExit("empty label schema")
    print(f"reviewed export present: {export}; declared labels={len(labels)}")


if __name__ == "__main__":
    main()
