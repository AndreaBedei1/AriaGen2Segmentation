# Article 1 reproducibility

All VRS processing is local with Hugging Face offline flags and pre-downloaded,
SHA-256-verified weights. Outputs live under `outputs/article1/<session_id>/` and are excluded
from Git. Stage manifests fingerprint model config, taxonomy, mapping, thresholds and
geometry; incompatible changes invalidate the relevant stage cache.

Example:

```bash
python -m aria_drive_seg article1 segment-external \
  --input outputs/article1/session01 --vehicle-type car \
  --session-id session01 --participant-id p01 \
  --config configs/article1/external.yaml --offline
```

Record VRS/checkpoint hashes, commit SHA, hardware report and exact CLI for every paper run.

Checkpoint-2 reuses saved native probabilities (no second model inference):

```bash
python -m aria_drive_seg article1 reprocess-external \
  --input outputs/article1/auto_30s \
  --source article1_external --output article1_external_checkpoint2 \
  --config configs/article1/external_segmentation.yaml
python scripts/render_article1_checkpoint2.py \
  --input outputs/article1/auto_30s \
  --output output/article1/checkpoint2_videos_30s
```

The 180–210 s source window is frames 1805–2100 sampled every fifth RGB frame: 60 frames,
2 fps and 30 seconds. Generated outputs remain ignored; manifests, configs and reports are
tracked.
