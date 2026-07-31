# Article 1 semantic-camera video stabilization — visual QA

Date: 2026-07-29

Clip: VRS interval 180–210 s, frames 1801–2100, 10 fps, 2016×1512

This is a non-GT review. Visual continuity is not semantic accuracy.

## Artifacts reviewed

- `output/article1/semantic_camera_video_stabilized_30s/contact_sheet.jpg`
- videos 01–05 in the same directory
- automatically selected internal-flicker previews: 1809, 1890, 1804, 1806
- automatically selected lane-flicker previews: 2034, 1916, 1918, 1938

Candidate frames were selected from isolated one-frame changes in the current
scientific masks at quarter resolution, before visual inspection.

## Observations

- The presentation cockpit is visibly more continuous than both current and
  improved-raw fusion across the contact sheet.
- Frame 1804 shows dashboard, steering/control and instrument regions retained over
  a dropout that is visible in the raw sequence.
- Frame 1890 shows the intended recall-oriented behavior clearly: display/control
  evidence is carried into the target from neighboring frames and remains aligned
  with the cockpit under flow.
- That same frame exposes the main residual risk: the fallback open-vocabulary model
  can assign an internal subclass incorrectly or produce an oversized region. Offline
  persistence can then retain the mistake. The raw scientific output remains
  available precisely for this reason.
- Strong external road users and road geometry remain visible in the reviewed
  cockpit frames. The full-run gate audit confirms that all configured
  high-confidence external pixels were preserved.
- Frame 1938 shows improved lane/road-marking continuity in presentation mode,
  especially across narrow gaps. Small additional fragments are also visible, which
  matches the requested presentation/recall bias.
- The focus videos keep a full-scene row and a magnified crop, making it possible to
  check both continuity and leakage into the external scene.

## Quantitative context

Relative to improved raw fusion:

- isolated one-frame flicker: −27.46%;
- flow-aligned switch rate: −66.08%;
- flow-aligned internal persistence: +60.42 percentage points;
- flow-aligned lane/marking continuity: +25.81 percentage points.

The literal image-coordinate switch rate drops only 0.53% relative to improved raw
and is higher than the sparse current baseline. This is expected from the much larger
internal coverage and camera-relative boundary motion; it is reported rather than
hidden.

## Video verification

Every MP4 was probed and decoded sequentially:

| video | geometry | codec | fps | duration | decoded frames |
|---|---:|---|---:|---:|---:|
| 01 current fusion | 1280×960 | H.264/yuv420p | 10 | 30.000 s | 300 |
| 02 improved fusion | 1280×960 | H.264/yuv420p | 10 | 30.000 s | 300 |
| 03 raw vs stabilized | 1920×560 | H.264/yuv420p | 10 | 30.000 s | 300 |
| 04 internal focus | 1920×960 | H.264/yuv420p | 10 | 30.000 s | 300 |
| 05 lane/marking focus | 1920×960 | H.264/yuv420p | 10 | 30.000 s | 300 |

No decoded frame changed geometry.
