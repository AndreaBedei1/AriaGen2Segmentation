#!/usr/bin/env bash
# Everything that depends on the frozen motorcycle baseline having finished:
# hand audit, failure-mode QA, exploratory comparison, annotation selection and
# package rebuild, training plan, presentation video, and all reports.
set -euo pipefail

RUN="${RUN:-output/article1/motorcycle_baseline_30s}"
CAR_RUN="${CAR_RUN:-outputs/article1/auto_temporal_180_210}"
REPORTS="${REPORTS:-reports/article1_motorcycle_ingestion}"
MOTO_ID="${MOTO_ID:-motorcycle_5ab8604a14df}"
CAR_ID="${CAR_ID:-car_2e84f0c3e245}"
MOTO_SHA12="${MOTO_SHA12:-5ab8604a14df}"
VIDEO="${VIDEO:-$RUN/semantic_camera_moto_final.mp4}"
FPS="${FPS:-15.0012}"
START_S="${START_S:-925.0}"

IO_PY="${IO_PY:-$HOME/projectaria_gen2_python_env/bin/python}"
ML_PY="${ML_PY:-$HOME/aria_seg_ml_env/bin/python}"

step() { echo; echo "=== $(date -Is) $* ==="; }

step "1/9 hand visibility audit"
"$IO_PY" scripts/audit_article1_hands.py \
  --frames "$RUN" \
  --hand-tracking "output/article1/ingestion/hand_tracking/${MOTO_ID}.parquet" \
  --proxy "$RUN/hand_proxy/hand_proxy.parquet" \
  --reports "$REPORTS"

step "2/9 failure-mode analysis and QA sequences"
"$IO_PY" scripts/analyze_article1_motorcycle_failures.py \
  --frames "$RUN" --semantic-camera "$RUN/semantic_camera" \
  --source-sha12 "$MOTO_SHA12" \
  --hand-candidates "$REPORTS/hand_visibility_candidates.csv" \
  --reports "$REPORTS"

step "3/9 exploratory car vs motorcycle comparison"
"$IO_PY" scripts/compare_article1_domains.py \
  --car-run "$CAR_RUN" --motorcycle-run "$RUN" \
  --car-recording-id "$CAR_ID" --motorcycle-recording-id "$MOTO_ID" \
  --hand-summary "$REPORTS/hand_failure_summary.json" \
  --route-summary "$REPORTS/route_alignment_summary.json" \
  --reports "$REPORTS"

step "4/9 annotation selection, now including failure-mode frames"
"$IO_PY" scripts/select_article1_annotation_frames.py \
  --segment-dir "$RUN/semantic_camera" \
  --failure-json "$REPORTS/motorcycle_failure_modes.json" \
  --reports "$REPORTS"

step "5/9 CVAT package with pre-annotations"
"$IO_PY" scripts/build_article1_cvat_package.py \
  --reports "$REPORTS" \
  --preannotation-root "${MOTO_ID}=$RUN/semantic_camera"

step "6/9 cockpit training plan (does not train)"
"$IO_PY" scripts/prepare_article1_cockpit_training.py \
  --selection "$REPORTS/annotation_selection.csv" \
  --output "$REPORTS/cockpit_training_plan.json"

step "7/9 presentation video and final-pass QA package"
"$ML_PY" scripts/render_article1_semantic_camera_final.py \
  --frames "$RUN" \
  --semantic-camera "$RUN/semantic_camera" \
  --previous-presentation "$RUN/semantic_camera_video_stabilized" \
  --final "$RUN/semantic_camera_final_pass" \
  --video "$VIDEO" \
  --qa "$REPORTS/final_pass_qa" \
  --fps "$FPS" \
  --source-start-seconds "$START_S"

step "7b/9 side-by-side external / internal / fusion comparison video"
"$ML_PY" scripts/render_article1_semantic_camera.py \
  --frames "$RUN" \
  --semantic-camera "$RUN/semantic_camera" \
  --output "$RUN/videos_comparison" \
  --fps "$FPS"

step "8/9 reports"
"$IO_PY" scripts/report_article1_acquisition_qa.py --reports "$REPORTS"
"$IO_PY" scripts/report_article1_motorcycle_baseline.py --run "$RUN" \
  --reports "$REPORTS" \
  --segment-selection "$REPORTS/segment_selection_motorcycle.json" \
  --video "$VIDEO"
"$IO_PY" scripts/report_article1_hand_audit.py --reports "$REPORTS" \
  --proxy-summary "$RUN/hand_proxy/summary.json"
"$IO_PY" scripts/report_article1_comparison.py --reports "$REPORTS"
"$IO_PY" scripts/report_article1_annotation_dataset.py --reports "$REPORTS"

step "9/9 full test suite"
"$IO_PY" -m pytest tests/ -q 2>&1 | tail -3

echo
echo "=== $(date -Is) post-pipeline steps complete ==="
