# Article 1 temporal segmentation

The primary pipeline is causal and gaze-independent. Frame `t` consumes the static
checkpoint-2 probability map for `t`, rectified RGB at `t`, and portable state produced at
`t-1`. It has no API for future frames or gaze coordinates.

`static_raw` remains the full-resolution static composite and is never overwritten.
`temporal_stabilized` is a separate output. The temporal grid defaults to half resolution;
semantic masks and diagnostics are returned to full geometry for comparison.

The stages are rectified-RGB flow, flow validity/occlusion, backward warp of the previous
state, probability fusion, class TTL/hysteresis, and a separate thin-marking layer.
Bidirectional smoothing is not implemented.

The checkpoint compares:

- T0: static raw;
- T1: hysteresis without flow;
- T2: flow and probability fusion;
- T3: class-specific TTL/hysteresis;
- T4: T3 plus thin-marking stabilization.

All T0–T4 outputs use the same static evidence. Their pre-GT diagnostics cannot establish
accuracy.
