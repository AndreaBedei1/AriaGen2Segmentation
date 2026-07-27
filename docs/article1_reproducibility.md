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
