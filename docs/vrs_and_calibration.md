# VRS, calibration, and gaze projection

How the pipeline reads a Project Aria Gen2 `.vrs`, rectifies the fisheye RGB, and aligns +
projects the on-device eye-gaze. All numbers below are from the validated test recording
`unknown_20260528_090955.vrs` and its `output/inspection/inspection_report.json`.

---

## Streams found

The test recording contains **17 streams**. `inspect` enumerates all of them with their
per-stream sample counts and first/last timestamps:

| Stream id | Label | Samples | Role |
|---|---|--:|---|
| 211-1 / 211-2 | camera-et-left / camera-et-right | 1880 / 1880 | eye-tracking cameras |
| **214-1** | **camera-rgb** | **3762** | scene RGB (segmentation input) |
| 246-1 | temperature | 7222 | device temperature |
| 247-1 | baro0 | 18940 | barometer |
| 248-1 | ppg | 96805 | photoplethysmography |
| 281-1 | gps-app | 374 | GPS |
| 371-1 | handtracking | 3761 | hand tracking |
| **373-1** | **eyegaze** | **11283** | on-device eye-gaze (gaze input) |
| 500-1 | als | 3549 | ambient light sensor |
| 1201-1..4 | slam-front-left/right, slam-side-left/right | 3761 each | 4 SLAM cameras |
| 1202-1 / 1202-2 | imu-left / imu-right | 302908 / 298497 | 2 IMUs |
| 1203-1 | mag0 | 37812 | magnetometer |

The pipeline uses **`camera-rgb`** (214-1) and **`eyegaze`** (373-1); the other streams are
reported for provenance but not consumed. Stream labels can differ across device generations,
so `AriaProvider.stream_id` resolves by label and falls back to scanning all streams.

---

## RGB geometry

| Property | Value |
|---|---|
| Resolution | 2016 × 1512 |
| Camera model | `FISHEYE624` |
| Focal length | 878.996 px (fx = fy) |
| Principal point | (999.60, 763.36) |
| Rate | 10.00 Hz (median Δt ≈ 99.99 ms) |
| Frames / duration | 3762 frames / 376.07 s |
| Timestamps | monotonic, 0 non-monotonic, **0 gaps**, 0 estimated dropped frames |

### Upright — no rotation needed

The raw **Gen2** RGB is **already upright**. Unlike Aria **Gen1** (which needs a 90° rotation),
Gen2 requires none — verified by viewing the raw frames. The rectification default
`rectify.rotate_ccw90: 0` encodes this. (The knob exists should a future recording need it;
`provider.make_pinhole` applies it via `rotate_camera_calib_cw90deg`.)

`inspect` writes evenly-spaced **raw + rectified sample JPEGs** to `inspection/samples/` so the
orientation and rectification can be eyeballed before committing to a full run.

---

## Rectification (fisheye624 → linear pinhole)

Segmentation models expect a standard perspective image, so each frame is rectified from the
native fisheye624 model to a **linear pinhole** camera. The approach (`vrs/provider.py`) uses
the **official `projectaria_tools` calibration** — no hand-rolled distortion math:

1. **Target camera.** `calibration.get_linear_camera_calibration(out_width, out_height, focal,
   label, T_device_camera)` builds a pinhole `CameraCalibration`. Defaults:
   `out_width=2016`, `out_height=1512`, `focal=879.0` px (lower focal → wider FOV, more black
   border). The pinhole inherits the RGB camera's device extrinsics.
2. **Fast remap.** For the exact model-correct mapping, `projectaria_tools` provides
   `distort_by_calibration`. Running that per frame over thousands of frames is slow, so the
   `Rectifier` precomputes a **one-time `cv2` remap**: for every destination pixel it
   unprojects a ray with the pinhole and reprojects it with the fisheye source
   (`dst.unproject → src.project`), building `map_x`/`map_y` once, then applies
   `cv2.remap(..., INTER_LINEAR, BORDER_CONSTANT=0)` to each frame.
3. **Numerically validated.** The fast remap was checked against `distort_by_calibration` and
   matches it to a mean absolute difference of **0.3 / 255** per pixel — visually and
   numerically equivalent, at a fraction of the cost.

The calibration actually used (source fisheye summary, rectify params, and resulting pinhole
summary) is persisted to `frames/calibration.json` for provenance and reverse mapping. The
`rectify` block is part of the segmentation cache fingerprint, so changing resolution or focal
invalidates and recomputes the affected masks.

---

## Eye-gaze temporal alignment (`align-gaze`)

The on-device gaze (373-1) runs at **30 Hz** (11283 samples in the test recording); RGB runs at
10 Hz. Alignment (`gaze/align.py`) matches each frame to gaze samples on the **`DEVICE_TIME`**
timeline (the timestamp domain used everywhere, `vrs.time_domain: DEVICE_TIME`):

- **Nearest sample.** For a frame at `capture_timestamp_ns`, the bracketing gaze samples are
  found by binary search; the nearest is used unless it is farther than
  `gaze.max_dt_ms` (default **20 ms**) on its side, which flags the frame `far_dt`.
- **Interpolation.** If `gaze.interp` is on (default) and **both** bracketing samples are valid
  and within `max_dt_ms`, yaw/pitch (and depth, when both valid) are **linearly interpolated**;
  otherwise the nearest sample is used directly.
- **Validity.** A frame's gaze is marked `valid` only when it is within `max_dt_ms`, the
  sample's `combined_gaze_valid` flag is set, projection succeeded, and the projected point
  lands **inside** the rectified image. Otherwise the reason is recorded (`far_dt`,
  `invalid_flag`, `projection_failed`, `out_of_image`).

Outputs: `gaze/raw_gaze.parquet` (every sample: yaw, pitch, depth, origin, validity flags),
`gaze/aligned_gaze.parquet` (per frame: chosen sample, `dt_ms`, yaw/pitch/depth, depth source,
original + rectified pixel `(u,v)`, in-image flags, validity reason), and `gaze/summary.json`.

On the test recording, `inspect` reports gaze validity of **93.2% combined-valid** and **88.1%
spatial-point-valid** across all 11283 samples.

---

## Gaze ray projection (CPF → Device → Camera)

Projection (`gaze/project.py`) uses `projectaria_tools`' official reprojection so the CPF
(Central Pupil Frame) → Device → Camera transforms and the camera distortion model are all
handled by the SDK. The gaze is projected into **both** geometries so masks in either can be
queried:

- **original** — the native fisheye624 RGB calibration.
- **rectified** — the linear pinhole calibration (this is what the segmentation masks use).

Two code paths:

- **Official path** (nearest sample): `mps.utils.get_gaze_vector_reprojection(eye_gaze,
  rgb_label, device_calib, target_calib, depth_m)`.
- **Manual path** (interpolated yaw/pitch): mirrors the SDK math —
  `get_eyegaze_point_at_depth(yaw, pitch, depth)` gives a CPF point, transformed by
  `T_device_cpf`, then by the inverse camera extrinsics, then projected with the target
  calibration.

A projected point is `in_image` only if it is finite and within that geometry's bounds.

### Depth handling and fallback

The gaze ray needs a depth to reproject to a pixel:

- If `eye_gaze.depth` is **valid (> 0)**, it is used and the source is tagged `device`.
- Otherwise a **configurable fallback** distance is used (`gaze.fallback_depth_m`, default
  **8 m**) and the source is tagged `fallback`.

The `depth_source` field is written per frame, so any pixel derived from the fallback is
**clearly flagged as an approximation**. A single projected gaze point cannot capture the full
eye-tracker uncertainty (see the limitations in the [README](../README.md#declared-limitations));
the downstream disc- and gaussian-weighted gaze queries (`analyze`) partially mitigate this.

---

## Timestamp domain

Everything operates on **`DEVICE_TIME`** (nanoseconds), read via
`get_timestamps_ns(stream, TimeDomain.DEVICE_TIME)`. RGB timestamps are fetched **without
decoding images** for speed, and `inspect` validates monotonicity and gap-freeness on this
timeline (a "gap" = an interval > 1.5× the median Δt; the test recording has none). Using one
device-time domain for RGB and gaze is what makes the 30 Hz-to-10 Hz alignment well-defined.
