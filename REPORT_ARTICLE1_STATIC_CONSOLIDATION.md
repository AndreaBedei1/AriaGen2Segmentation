# Article 1 — static per-frame consolidation

Date: 2026-07-27  
Branch: `feature/auto-moto-semantic-gaze`  
Initial commit: `315bd3382e39e3c3c33f717aff013453ba442b11`

## Scope

This change consolidates the existing static pipeline only. It adds no optical flow,
tracking, temporal propagation, EMA, hysteresis, SAM2 video, gaze smoothing, SegFormer
training or complete-VRS execution. Existing outputs, `val10`, `run200` and A/B/C/D
configurations were not changed.

## Canonical pipeline

`article1 segment-external` now performs Mask2Former inference, native normalization,
probabilistic Mapillary→Article 1 aggregation, unknown policy, base mask, raw thin layer,
road-supported filtering, final composition and canonical diagnostics in one path.

`article1 reprocess-external` remains available for threshold/policy experiments without
model inference, but shares the exact policy, metadata and writer implementation. The legacy
`preserve_thin_markings` helper is explicitly checkpoint-1-only and is not called by either
production command.

The persisted native float16 representation is also the representation consumed by the
direct policy. Direct and reprocessed results with the same config are therefore identical
for base/raw/filtered/composite/reason masks and float16-equivalent for macro probabilities.

## Thin road support

Components are built before road clipping. Real overlap is:

`pixels(component AND final_road_support) / pixels(component)`.

At the default 0.60 threshold, 0%, 30% and 59% are rejected; 60%, 90% and 100% are accepted.
The default `supported_pixels_only` policy retains only supported, non-conflicting pixels.
The optional `full_component_if_supported` policy retains the whole component after it
passes the overlap gate, except high-confidence non-road conflicts when suppression is on.

Road support combines road probability ≥0.25 with the base road mask, applies a controlled
12-pixel dilation, then removes high-confidence vehicle, pedestrian, sign, traffic-light,
other-environment and cockpit-class regions. Raw, dilated and final supports are optional
diagnostic outputs.

## Fail-closed model loading and profiling

The loader accepts exactly the two diagnosed unused final-Swin-LayerNorm missing keys.
Unexpected keys must match the anchored legacy `relative_position_index` buffer pattern.
Any classifier/decoder/head/stage-norm key, shape mismatch or loader error aborts. Its report
includes all keys, allowed patterns, gate outcome, weight hashes and library versions.

CUDA profiling now synchronizes immediately before and after the timed forward pass and
before returning host results, preventing asynchronous kernel execution from understating
forward latency.

## Verification

The suite covers direct-vs-reprocess equivalence, both thin classes, six overlap boundary
cases, supported/full policies, large components, vehicle/cockpit conflicts, road-edge
components, road-support construction and all loading-gate failure modes. No scientific
accuracy claim is made: these are deterministic implementation tests without reviewed GT.
