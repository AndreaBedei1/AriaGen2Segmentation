# Article 1 semantic-camera video stabilization

This stage consumes a completed scientific semantic-camera run without changing it.
It produces two additional outputs:

1. `improved_masks`: stronger but still frame-independent internal/external fusion;
2. `presentation_masks`: explicitly non-causal offline video stabilization.

The presentation product is for visualization and review. It must not silently
replace the scientific per-frame result.

## Improved fusion

The improved raw fusion keeps every internal-class win from the scientific source.
New mirror, instrument-display and control evidence competes using confidence,
top-1 entropy proxies, class priorities and a cockpit position prior. Strong
road-surface, lane, regulatory, vehicle, two-wheeler, pedestrian, light and sign
evidence is protected by per-class confidence gates.

`driver_hand` has no independent Article 1 final class and therefore remains rolled
into `control_and_ego_vehicle`.

## Offline presentation mode

Adjacent rectified RGB frames are aligned at 1008×756 with local OpenCV DIS flow.
Each pair stores forward and backward flow plus a validity mask in both target
coordinate systems. For every target frame, evidence from a symmetric ±5-frame
window is warped into the target and combined by confidence-weighted class voting.

Class-specific policy:

- mirror, instrument display and control/ego vehicle: TTL 5;
- lane and regulatory road marking: TTL 3;
- road surface, vehicles, two-wheelers, pedestrians, lights, signs and road
  boundaries: conservative TTL 1;
- other environment: no temporal propagation.

Activation/deactivation hysteresis retains current internal evidence, permits
supported neighbor activation with a relaxed margin and protects strong external
pixels again at full resolution. Presentation provenance distinguishes past,
future, bidirectional and protected-external decisions.

## Metrics

Two metric coordinate systems are reported:

- image-coordinate metrics measure literal per-pixel changes in the rendered image;
- flow-aligned metrics warp the previous mask into current geometry before measuring
  switches and persistence.

Both are pre-GT continuity diagnostics. Lower switch/flicker may preserve false
positives and is not an accuracy result.

## Reproduction

```bash
PYTHONPATH=. /home/andreabedei/miniconda3/envs/aria-car/bin/python \
  -m aria_drive_seg article1 stabilize-semantic-camera-video \
  --input outputs/article1/auto_temporal_180_210 \
  --semantic-camera \
    outputs/article1/auto_temporal_180_210/semantic_camera \
  --output \
    outputs/article1/auto_temporal_180_210/semantic_camera_video_stabilized \
  --vehicle-type car \
  --session-id auto_temporal_180_210 \
  --participant-id presentation_30s \
  --config configs/article1/semantic_camera_video.yaml \
  --offline --resume
```

The renderer supports one-based subsets so large products can be encoded
sequentially and resumed:

```bash
for video in 1 2 3 4 5; do
  PYTHONPATH=. /home/andreabedei/miniconda3/envs/aria-car/bin/python \
    scripts/render_article1_semantic_camera_video.py \
    --frames outputs/article1/auto_temporal_180_210 \
    --semantic-camera \
      outputs/article1/auto_temporal_180_210/semantic_camera \
    --presentation \
      outputs/article1/auto_temporal_180_210/semantic_camera_video_stabilized \
    --output \
      output/article1/semantic_camera_video_stabilized_30s \
    --fps 10 --videos "$video"
done
```
