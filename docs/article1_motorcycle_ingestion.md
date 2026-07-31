# Motorcycle ingestion and the multi-rate policy

How the motorcycle recording enters Article 1, and the rules that keep the
temporary 10 fps / 15 fps difference from contaminating the science.

## Why the frame rate is measured, never read

Both current recordings carry the identical profile name
`driver_dataset_v1_raw_for_ht` and the identical device serial, yet their RGB
streams run at 10.0008 Hz and 15.0012 Hz. The profile name is therefore not
evidence of anything. Every stage measures the rate from the stream's own capture
timestamps (`ingestion.timeline.measure_rate`) and reports the declared nominal
rate next to it rather than in place of it.

`measure_rate` returns two rates:

* `effective_fps_median` — `1 / median(dt)`, the instantaneous cadence. A pause
  does not change it, because a recording that stops and restarts still samples at
  its configured rate while it is running. This is the primary value.
* `effective_fps_span` — `(n - 1) / duration`. A pause drags it down, which is why
  it is not primary, but it is the honest number for a **bursty** stream where the
  median interval is meaningless (the device's `temperature` stream delivers a
  burst of samples and then waits; its median-based rate is several kHz and its
  real throughput is about 20 Hz).

## Temporal parameters live in seconds

No stage takes a window in frames. Windows are declared in seconds in
`configs/article1/dataset_protocol.yaml` and converted per recording with
`frames_for_seconds(seconds, effective_fps)`. The same configuration therefore
means the same duration at 10 fps and at 15 fps, and produces a different frame
count in each — which is the point.

Cross-recording counts go through `per_second(count, duration_s)`. Comparing raw
per-frame counts between recordings sampled at different rates is a unit error, not
a result.

## The domain is decided from the pixels

`ingestion.inventory.classify_domain` receives a `DomainEvidence` object that has
no rate, count or duration field. It structurally cannot see the fps difference.

The discriminator is the static ego structure around the camera, pooled over frames
sampled far apart in time so that scene content decorrelates:

| signal | car | motorcycle |
|---|---:|---:|
| top-band static fraction | 75.4% | 3.8% |
| top-band luminance | 0.16 | 0.39 |
| border static fraction | 42.5% | 6.8% |

A car cabin encloses the camera: the top of the frame is a dark, motionless
headliner and the border is pillars and door frames. A motorcycle leaves the camera
in the open: the top is bright, changing sky and only a small region low in the
frame stays fixed.

Two details matter for robustness. The Aria RGB camera is a fisheye, so the image
corners are an optically black vignette that is static and dark in *both* domains;
every statistic is computed over the valid region only. And auto-exposure drifts
over a multi-minute drive, so the "static" threshold is a tolerance on that drift
(20/255) rather than a noise floor.

## Gaps are judged against each stream's own cadence

A gap is an interval longer than 1.5 of the stream's own periods, or longer than the
250 ms absolute pause threshold **where that threshold applies**. It is disabled for
streams whose nominal period already exceeds it: a 1 Hz GPS samples every 1000 ms by
design, and applying the absolute rule there would report every normal interval as a
pause.

Continuous segments use the stricter of two periods and the absolute threshold, with
the same exemption.

## Duplicates

A duplicate frame requires an identical perceptual hash **and** a near-zero mean
pixel difference. Consecutive frames of a driving video are normally very similar —
at 15 fps a slow scene barely changes in 67 ms — so a hash-distance-only rule would
report roughly half of any recording as broken. The looser statistic is still
reported, as `near_identical_consecutive_pairs`, because it describes scene change
rather than a defect.

## Two analysis modes

**A, native** — each recording at its own rate against its own timestamps, windows
in seconds, metrics per second, no resampling. The only mode that may support a
scientific statement.

**B, comparable timeline** — a 10 Hz grid for preliminary QA only, built by taking
the nearest real frame to each grid point. It never interpolates, never creates a
frame, never upsamples the car, and reports every reuse with its temporal error. A
grid point whose nearest real frame is too far away is **dropped, not filled**.

Measured over a 60 s window: the car gives 601 grid points from 601 distinct frames
at 5.6 ms maximum error; the motorcycle gives 600 grid points from 600 distinct
frames at 33.3 ms maximum error, which is half a 15 fps period. Zero reuses in
either. One motorcycle grid point, at the single dropped frame, exceeded the 60 ms
tolerance and was dropped.

Mode B must not become the dataset format and must not train or evaluate the future
vehicle classifier. A classifier fed the derived timeline could learn the
resampling signature instead of the vehicle.

## Provenance carried by every extracted frame

`recording_id`, `domain`, `source_file_sha256`, `source_stream_id`,
`source_frame_index`, `timestamp_ns`, `timestamp_s`, `effective_fps`, `width`,
`height`, `valid`, `extraction_status`.

`frame_index` is always identical to `source_frame_index`; a frame is never
renumbered by a local counter, and the original index appears in the file name too.
The extraction summary records `resampled: false`, `interpolated: false`,
`synthetic_frames: 0` and whether a decimation was explicitly declared.

## Gaze and hand tracking

Both are associated to frames **by timestamp**, inside a window expressed in
seconds, and every association keeps the signed temporal distance. One-to-one
correspondence is never assumed: a 30 Hz gaze stream has several samples per 15 fps
frame and all of them are preserved.

Gaze is used strictly after segmentation and never as a segmentation input.

Hand tracking is a proxy. `tracked` is not `visible`: the device can track a hand
outside the RGB field of view and can fail on a perfectly visible one. On this data
the car has a hand tracked in 100% of samples (75% projecting into the RGB image)
and the motorcycle in 37% (12.5% projecting into the image) — a property of where
the grips sit relative to a head-mounted camera, not of any model.

## What the frozen first run is for

The pipeline is run on the motorcycle with every checkpoint, prompt, threshold,
mapping, fusion rule and fallback exactly as the car baseline left them. The config
bundle is hashed before and after and a mismatch aborts the run.

Nothing is tuned to make the motorcycle look better before the frozen result has
been seen. That result is the evidence for what the car-tuned pipeline actually does
in a new domain.
