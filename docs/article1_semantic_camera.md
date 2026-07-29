# Article 1 semantic camera

The semantic-camera stage fuses two independent, same-frame evidence streams into
one dense Article 1 mask. It is a static per-frame stage: it does not use gaze,
tracking, optical flow, previous frames or future frames.

## Model selection

The external stream is a saved Article 1 Mask2Former/Mapillary result. Reading a
temporal run intentionally selects its preserved raw `static_masks` and
`static_confidence`, not its stabilized output.

The internal provider is selected fail-closed:

1. use the reviewed cockpit SegFormer checkpoint configured at
   `semantic_camera.internal.segformer_checkpoint` when it exists;
2. in `auto` mode, otherwise use the explicitly marked local
   Grounding DINO + SAM2.1 cockpit proxy;
3. fail when `segformer` is explicitly requested but the checkpoint is absent.

The temporary proxy combines open-vocabulary cockpit detections with a low-confidence
bottom-band geometric prior. It is not a trained or reviewed cockpit semantic model.

## Fusion contract

External and internal masks must have identical geometry. Scores combine confidence,
normalized entropy and configurable class priorities. Detected cockpit evidence can
compete with the external stream; the geometric prior has a narrower whitelist and
may override only weak external evidence. Safety-relevant external classes receive
higher priority.

Any external `unknown` pixels that remain after fusion are assigned to
`other_environment` with provenance `dense_other_environment_fill`, confidence zero
and entropy one. This guarantees the output contract—class IDs 1 through 13 at every
pixel—without disguising the fill as confident model evidence.

Per-frame outputs include the dense mask, overlay, final confidence and entropy,
provenance, conflicts, external and internal masks/confidence, metadata and selected
float16 diagnostic probability maps. Manifests fingerprint configuration, taxonomy,
external evidence, internal provider and vehicle type. An incompatible resume fails;
completed frames are not rewritten.

## Reproduction

The Grounded-SAM2 fallback requires the repository ML environment and local weights:

```bash
PYTHONPATH=. /home/andreabedei/aria_seg_ml_env/bin/python \
  -m aria_drive_seg article1 semantic-camera \
  --input outputs/article1/auto_temporal_180_210 \
  --external outputs/article1/auto_temporal_180_210/article1_temporal \
  --output outputs/article1/auto_temporal_180_210/semantic_camera \
  --vehicle-type car \
  --session-id auto_temporal_180_210 \
  --participant-id checkpoint_semantic_camera \
  --config configs/article1/semantic_camera.yaml \
  --offline --resume
```

Render the dense and three-panel comparison videos:

```bash
PYTHONPATH=. /home/andreabedei/miniconda3/envs/aria-car/bin/python \
  scripts/render_article1_semantic_camera.py \
  --frames outputs/article1/auto_temporal_180_210 \
  --semantic-camera \
    outputs/article1/auto_temporal_180_210/semantic_camera \
  --output output/article1/semantic_camera_30s \
  --fps 10
```

