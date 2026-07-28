# Article 1 temporal reproducibility

Direct streaming:

```bash
python -m aria_drive_seg article1 segment-temporal \
  --input outputs/article1/auto_temporal_180_210 \
  --vehicle-type car --session-id auto_temporal_180_210 \
  --participant-id unknown \
  --config configs/article1/temporal_segmentation.yaml --offline --resume
```

Replay from a complete static output:

```bash
python -m aria_drive_seg article1 stabilize-temporal \
  --input <run>/article1_external --frames <run> \
  --vehicle-type car --session-id <id> --participant-id <id> \
  --config configs/article1/temporal_segmentation.yaml --resume
```

Replay fails if static probabilities or masks are absent. Resume loads a portable NPZ state
from the latest checkpoint common to every requested T0–T4 mode. Config/static fingerprints,
frame continuity, timestamps and geometry are checked. State checkpoints contain arrays and
scalars only, never Python objects.

The manifest records causal and gaze-independent flags, flow backend, temporal scale,
fingerprints, processed indices, timestamp range, resets and available outputs. Static masks
remain separate from temporal masks.
