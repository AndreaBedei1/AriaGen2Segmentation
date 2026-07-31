#!/usr/bin/env bash
# Phase 5: run the FROZEN Article 1 pipeline on the selected motorcycle segment.
#
# Frozen means: the checkpoints, prompts, thresholds, class mapping, fusion rules,
# temporal configuration and fallback policy are exactly the ones the car baseline
# was produced with. Nothing is tuned for the motorcycle before the frozen result
# has been observed. The configs are hashed before and after the run and the hashes
# are written next to the outputs.
#
# Stage environments:
#   VRS I/O env  : extraction and gaze alignment (projectaria_tools)
#   ML env       : Mask2Former, Grounding DINO + SAM 2.1, optical flow
set -euo pipefail
# Stages are piped through tee; without pipefail a crashed stage would report the
# exit status of tee and the run would march on to the next stage.
set -o pipefail

RUN_DIR="${RUN_DIR:?set RUN_DIR}"
VRS="${VRS:?set VRS}"
START_NS="${START_NS:?set START_NS}"
END_NS="${END_NS:?set END_NS}"
RECORDING_ID="${RECORDING_ID:?set RECORDING_ID}"
SESSION_ID="${SESSION_ID:-moto_session_01}"
PARTICIPANT_ID="${PARTICIPANT_ID:-participant_01}"
VEHICLE="${VEHICLE:-motorcycle}"

IO_PY="${IO_PY:-$HOME/projectaria_gen2_python_env/bin/python}"
ML_PY="${ML_PY:-$HOME/aria_seg_ml_env/bin/python}"

FROZEN_CONFIGS=(
  configs/default.yaml
  configs/classes.yaml
  configs/grounded_prompts.yaml
  configs/article1/classes_article1.yaml
  configs/article1/mapillary_to_article1.yaml
  configs/article1/external.yaml
  configs/article1/temporal_segmentation.yaml
  configs/article1/semantic_camera.yaml
  configs/article1/semantic_camera_video.yaml
  configs/article1/semantic_camera_final.yaml
)

mkdir -p "$RUN_DIR/logs"
LOG="$RUN_DIR/logs/frozen_run.log"

hash_configs() {
  sha256sum "${FROZEN_CONFIGS[@]}" | sha256sum | cut -d' ' -f1
}

BEFORE="$(hash_configs)"
{
  echo "frozen_config_bundle_sha256_before=$BEFORE"
  echo "run_dir=$RUN_DIR"
  echo "vrs=$VRS"
  echo "recording_id=$RECORDING_ID"
  echo "window_ns=[$START_NS,$END_NS]"
  echo "vehicle=$VEHICLE session=$SESSION_ID participant=$PARTICIPANT_ID"
  echo "started=$(date -Is)"
} | tee "$LOG"

step() { echo "=== $(date -Is) $* ===" | tee -a "$LOG"; }

step "1/7 timestamped extraction (VRS I/O env)"
"$IO_PY" scripts/extract_article1_segment.py \
  --vrs "$VRS" --recording-id "$RECORDING_ID" --domain "$VEHICLE" \
  --start-timestamp-ns "$START_NS" --end-timestamp-ns "$END_NS" \
  --output "$RUN_DIR" 2>&1 | tee -a "$LOG"

step "2/7 external Mask2Former + causal temporal (ML env)"
"$ML_PY" -m aria_drive_seg article1 segment-temporal \
  --input "$RUN_DIR" --vehicle-type "$VEHICLE" \
  --session-id "$SESSION_ID" --participant-id "$PARTICIPANT_ID" \
  --offline 2>&1 | tee -a "$LOG"

step "3/7 dense semantic camera fusion (ML env)"
"$ML_PY" -m aria_drive_seg article1 semantic-camera \
  --input "$RUN_DIR" --external "$RUN_DIR/article1_temporal" \
  --vehicle-type "$VEHICLE" --session-id "$SESSION_ID" \
  --participant-id "$PARTICIPANT_ID" --offline 2>&1 | tee -a "$LOG"

step "4/7 causal presentation stabilization (ML env)"
"$ML_PY" -m aria_drive_seg article1 stabilize-semantic-camera-video \
  --input "$RUN_DIR" --semantic-camera "$RUN_DIR/semantic_camera" \
  --vehicle-type "$VEHICLE" --session-id "$SESSION_ID" \
  --participant-id "$PARTICIPANT_ID" --offline 2>&1 | tee -a "$LOG"

step "5/7 bidirectional presentation final pass (ML env)"
"$ML_PY" scripts/run_article1_semantic_camera_final.py \
  --frames "$RUN_DIR" \
  --semantic-camera "$RUN_DIR/semantic_camera" \
  --previous-presentation "$RUN_DIR/semantic_camera_video_stabilized" \
  --output "$RUN_DIR/semantic_camera_final_pass" 2>&1 | tee -a "$LOG"

step "6/7 semantic gaze alignment (VRS I/O env)"
"$IO_PY" -m aria_drive_seg align-gaze --vrs "$VRS" --input "$RUN_DIR" \
  2>&1 | tee -a "$LOG"

step "7/7 frozen-config verification"
AFTER="$(hash_configs)"
echo "frozen_config_bundle_sha256_after=$AFTER" | tee -a "$LOG"
if [ "$BEFORE" != "$AFTER" ]; then
  echo "FROZEN CONFIG VIOLATION: a configuration changed during the run" | tee -a "$LOG"
  exit 3
fi

sha256sum "${FROZEN_CONFIGS[@]}" > "$RUN_DIR/logs/frozen_config_checksums.sha256"
echo "finished=$(date -Is)" | tee -a "$LOG"
echo "OK: frozen run complete, configs unchanged" | tee -a "$LOG"
