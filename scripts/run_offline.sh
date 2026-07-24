#!/usr/bin/env bash
# Force fully-offline inference (§2). Source this, or prefix a command with it:
#   source scripts/run_offline.sh
#   scripts/run_offline.sh python -m aria_drive_seg segment --method grounded_sam2 --input val10
#
# After the one-time weights download, NOTHING here should touch the network.
set -euo pipefail

# Hugging Face / transformers: never hit the hub or its telemetry.
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export HF_HUB_DISABLE_TELEMETRY=1
export HF_HUB_DISABLE_IMPLICIT_TOKEN=1

# Disable assorted third-party telemetry/analytics.
export DISABLE_TELEMETRY=1
export DO_NOT_TRACK=1
export NO_ALBUMENTATIONS_UPDATE=1
export YOLO_OFFLINE=1
export ULTRALYTICS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
# torch.hub must not fetch.
export TORCH_HOME="${TORCH_HOME:-$(pwd)/weights/torch_home}"
# Point HF cache at a local, pre-populated dir so offline loads resolve.
export HF_HOME="${HF_HOME:-$(pwd)/weights/hf_home}"

echo "[run_offline] HF_HUB_OFFLINE=$HF_HUB_OFFLINE TRANSFORMERS_OFFLINE=$TRANSFORMERS_OFFLINE telemetry disabled"

# If given a command, exec it under these settings; otherwise just export (when sourced).
if [ "$#" -gt 0 ]; then
    exec "$@"
fi
