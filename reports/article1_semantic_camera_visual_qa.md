# Article 1 semantic camera — visual QA

Date: 2026-07-29

Clip: VRS interval 180–210 s, frames 1801–2100, 10 fps, 2016×1512

Artifacts: `output/article1/semantic_camera_30s/contact_sheet.jpg` and the
four `preview_*.png` images in the same directory.

## Scope

This is a non-GT visual review. It checks readability, geometry, provenance behavior
and obvious failure modes. It does not establish accuracy, IoU, precision or recall.

## Observations

- The external Mask2Former stream remains dominant in the road scene. Road, road
  markings, vehicles and traffic signs remain visibly aligned with the preserved
  external baseline in the sampled frames.
- The Grounded-SAM2 cockpit proxy produces useful-looking evidence on the instrument
  cluster, A-pillars and some cockpit objects. The geometric prior provides continuous
  low-confidence coverage in the lower cockpit band.
- Fusion visibly corrects some external cockpit confusion while preserving
  high-priority external road users and traffic infrastructure.
- The final overlay is dense and readable. The legend, frame index, capture timestamp
  and per-frame provenance fractions remain legible in the encoded video.
- The automatically selected maximum-internal frame 1883 shows instrument-display
  evidence and the bottom cockpit prior. The final result retains external road-scene
  structure.
- The maximum-fill frame 1804 exposes the intended diagnostic behavior: unresolved
  pixels become `other_environment`; this is a zero-confidence dense fallback, not a
  semantic-model success.
- The maximum-conflict frame 1984 exposes a proxy limitation. Open-vocabulary evidence
  can be fragmented on textured interior/roof regions. The score and class-priority
  gates prevent most of that proposal from replacing stronger external evidence.
- The contact sheet covers the clip from frame 1801 through 2100 and shows no geometry
  shift between external, internal/proxy and final panels.

## Known limitations

- No reviewed cockpit SegFormer checkpoint is present, so every internal result in
  this experiment is fallback evidence.
- The geometric prior is car-specific and deliberately coarse. It must be replaced
  or calibrated before treating the output as a validated car/motorcycle semantic
  camera.
- Exact full external macro probabilities were not retained uniformly by the source
  run. Fusion therefore uses real saved top-1 confidence and the explicitly named
  `top1_maximum_entropy_proxy`.
- `other_environment` dominates the frame and includes both genuine environment and
  diagnostic dense-fill pixels; provenance separates the latter.
- There is no ground truth for this clip and no motorcycle experiment.

## Video validation

Both MP4 files were checked with `ffprobe` and decoded sequentially with OpenCV:

| file | codec | geometry | fps | duration | decoded frames |
|---|---|---:|---:|---:|---:|
| `01_semantic_camera_dense.mp4` | H.264/yuv420p | 1280×960 | 10 | 30.000 s | 300 |
| `02_external_internal_fusion.mp4` | H.264/yuv420p | 1920×560 | 10 | 30.000 s | 300 |

No frame had inconsistent rendered geometry.
