# Article 1 — acquisition and stream QA (car + motorcycle)

Generated: 2026-07-31T16:48:47.559479+00:00

Branch: `feature/article1-motorcycle-ingestion`

## Scope

This report validates the newly added motorcycle recording and re-validates the existing car recording with the same instrument. It answers a single question: **is the motorcycle recording usable, and over which intervals?**

The car recording is currently sampled at ~10 fps and is an explicitly provisional development baseline. The final protocol is 15 fps for both vehicles. The rate difference is reported here as an acquisition fact and is never used as an analysis variable.

## How the recordings were identified

No file was identified by its name. The project tree was scanned for data files, each VRS was opened, and the domain was decided from the pixels:

- method: `static_ego_structure_v1`
- inputs: sampled RGB frames
- explicitly not used: file name, rgb frame rate, frame count, recording profile name

The discriminator is the static ego structure around the camera. A car cabin encloses the camera, so the top of the frame is a dark, motionless headliner and the border is dominated by pillars and door frames. A motorcycle leaves the camera in the open, so the top of the frame is bright, changing sky and only a small region low in the image stays fixed.

| recording | domain | confidence | top band static | top band luminance | border static | evidence |
|---|---|---:|---:|---:|---:|---|
| `car_2e84f0c3e245` | car | 1.00 | 75.44% | 0.16 | 42.54% | 14 frames sampled across the recording |
| `motorcycle_5ab8604a14df` | motorcycle | 1.00 | 3.81% | 0.39 | 6.84% | 14 frames sampled across the recording |

`car_2e84f0c3e245` rationale:
- top band is 75.44% static: consistent with a roof above the camera
- top band luminance 0.16 is dark: consistent with an interior headliner

`motorcycle_5ab8604a14df` rationale:
- top band is only 3.81% static: consistent with open sky above the rider
- top band luminance 0.39 is bright: consistent with sky
- only 6.84% of the image border is static: the camera is not enclosed by a cabin

Both recordings carry the identical profile name `driver_dataset_v1_raw_for_ht` and the identical device serial `1M0YDN5HB70983`, yet their RGB streams run at 10.001 Hz and 15.001 Hz respectively. This is the concrete reason the frame rate is measured from timestamps and never read from the profile name.

### Candidate selection

- **car**: `car_2e84f0c3e245` — exactly one car candidate: no ambiguity
- **motorcycle**: `motorcycle_5ab8604a14df` — exactly one motorcycle candidate: no ambiguity

No duplicate recording, partial export or truncated file was found. Both VRS files decoded end to end: the RGB scan read every frame the timestamp table declared, which is the completeness evidence used here.

## Motorcycle — `motorcycle_5ab8604a14df`

- source file: `unknown_20260731_162521.vrs` (1578.5 MB)
- local absolute path: `/home/andreabedei/Scrivania/AndreaSegmentazione/unknown_20260731_162521.vrs`
- SHA-256: `5ab8604a14dfafecb15fc58a86c8d99ac59e4c83b1cb6025e43d22efe264f6d7`
- last modified: 2026-07-31T15:00:35+00:00
- device serial: `1M0YDN5HB70983`, profile name: `driver_dataset_v1_raw_for_ht`
- streams present: 17 (none of the checked modalities is missing)

### RGB

| quantity | value |
|---|---|
| resolution | 2016 x 1512 |
| sensor | IMX681 (pixel format 5) |
| codec | not exposed by the provider API |
| frames | 15,069 |
| first / last timestamp (ns) | 2,174,826,479,905 / 3,179,342,675,749 |
| duration | 1004.516 s |
| nominal rate | 15.0 Hz |
| **measured rate (median interval)** | **15.00117 Hz** |
| measured rate (span) | 15.00026 Hz |
| median interval | 66.6615 ms |
| mean / std interval | 66.6655 / 0.5438 ms |
| min / max interval | 65.649 / 133.317 ms |
| interval p1 / p5 / p50 / p95 / p99 | 66.579 / 66.636 / 66.661 / 66.693 / 66.731 ms |
| non-monotonic intervals | 0 |
| duplicate timestamps | 0 |
| duplicate images | 14 |
| near-identical consecutive pairs | 7,288 |
| intervals > 1.5 periods | 1 |
| intervals > 2 periods | 0 |
| intervals > 250 ms | 0 |
| estimated missing frames | 1 |
| longest interval | 133.317 ms |
| continuous segments | 1 |
| longest segment | 1004.52 s |
| clock drift, max residual vs constant rate | 55.11 ms |
| clock drift, residual std | 14.08 ms |
| median frame luminance | 0.313 |
| median sharpness (Laplacian variance) | 500.4 |
| very dark / very bright frames | 0 / 0 |

14 frame pairs are byte-level near identical (identical perceptual hash AND mean pixel difference <= 0.002). Consecutive frames of a driving video are normally very similar, so the much larger 7,288 near-identical pairs is a scene-change statistic, not a defect.

#### RGB intervals flagged

| source index before | source index after | interval ms | periods | estimated missing | > 250 ms |
|---:|---:|---:|---:|---:|---|
| 827 | 828 | 133.317 | 2.000 | 1 | no |

### All streams

| stream | samples | nominal Hz | measured Hz | representative Hz | duration s | flagged intervals | >250 ms | segments | p95 dt vs RGB ms | note |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `als` | 9,480 | 9 | 9.437 | 9.44 | 1004.4 | 0 | 0 | 1 | 50.3 |  |
| `baro0` | 50,568 | n/a | 50.336 | 50.34 | 1004.6 | 0 | 0 | 1 | 9.4 |  |
| `camera-et-left` | 5,023 | 5 | 5.000 | 5.00 | 1004.4 | 0 | 0 | 1 | 94.2 |  |
| `camera-et-right` | 5,023 | 5 | 5.000 | 5.00 | 1004.4 | 0 | 0 | 1 | 94.2 |  |
| `camera-rgb` | 15,069 | 15 | 15.001 | 15.00 | 1004.5 | 1 | 0 | 1 | n/a |  |
| `eyegaze` | 30,138 | 30 | 30.000 | 30.00 | 1004.6 | 0 | 0 | 1 | 15.9 |  |
| `gps-app` | 1,004 | n/a | 1.000 | 1.00 | 1002.7 | 0 | 0 | 1 | 477.1 |  |
| `handtracking` | 10,046 | 10 | 10.000 | 10.00 | 1004.5 | 0 | 0 | 1 | 47.6 |  |
| `imu-left` | 809,000 | 800 | 805.301 | 805.30 | 1004.6 | 0 | 0 | 1 | 0.6 |  |
| `imu-right` | 797,183 | 800 | 793.520 | 793.52 | 1004.6 | 0 | 0 | 1 | 0.6 |  |
| `mag0` | 100,852 | 100 | 100.388 | 100.39 | 1004.6 | 0 | 0 | 1 | 4.7 |  |
| `ppg` | 258,434 | 256 | 257.255 | 257.26 | 1004.6 | 0 | 0 | 1 | 1.8 |  |
| `slam-front-left` | 10,046 | 10 | 10.000 | 10.00 | 1004.5 | 0 | 0 | 1 | 47.6 |  |
| `slam-front-right` | 10,046 | 10 | 10.000 | 10.00 | 1004.5 | 0 | 0 | 1 | 47.6 |  |
| `slam-side-left` | 10,046 | 10 | 10.000 | 10.00 | 1004.5 | 0 | 0 | 1 | 47.6 |  |
| `slam-side-right` | 10,046 | 10 | 10.000 | 10.00 | 1004.5 | 0 | 0 | 1 | 47.6 |  |
| `temperature` | 24,318 | 1 | 7843.199 | 24.23 | 1003.6 | 8570 | 1029 | 7443 | 468.7 | bursty |

### Usability checks

- estimated missing frames 0.0066% within budget
- no pause longer than 250 ms
- RGB is a single continuous segment
- every required stream is present
- RGB timestamps are strictly monotonic

**Usable: yes**

## Car — `car_2e84f0c3e245`

- source file: `unknown_20260528_090955.vrs` (434.0 MB)
- local absolute path: `/home/andreabedei/Scrivania/AndreaSegmentazione/unknown_20260528_090955.vrs`
- SHA-256: `2e84f0c3e245a7afc353e7c981ed44916df5d8c723e61aab1dd7735903490bf9`
- last modified: 2026-07-23T22:09:05.985264+00:00
- device serial: `1M0YDN5HB70983`, profile name: `driver_dataset_v1_raw_for_ht`
- streams present: 17 (none of the checked modalities is missing)

### RGB

| quantity | value |
|---|---|
| resolution | 2016 x 1512 |
| sensor | IMX681 (pixel format 5) |
| codec | not exposed by the provider API |
| frames | 3,762 |
| first / last timestamp (ns) | 1,055,278,217,953 / 1,431,349,516,120 |
| duration | 376.071 s |
| nominal rate | 10.0 Hz |
| **measured rate (median interval)** | **10.00076 Hz** |
| measured rate (span) | 10.00076 Hz |
| median interval | 99.9924 ms |
| mean / std interval | 99.9924 / 0.0532 ms |
| min / max interval | 98.551 / 100.928 ms |
| interval p1 / p5 / p50 / p95 / p99 | 99.873 / 99.942 / 99.992 / 100.049 / 100.128 ms |
| non-monotonic intervals | 0 |
| duplicate timestamps | 0 |
| duplicate images | 0 |
| near-identical consecutive pairs | 1,967 |
| intervals > 1.5 periods | 0 |
| intervals > 2 periods | 0 |
| intervals > 250 ms | 0 |
| estimated missing frames | 0 |
| longest interval | 0.000 ms |
| continuous segments | 1 |
| longest segment | 376.07 s |
| clock drift, max residual vs constant rate | 1.67 ms |
| clock drift, residual std | 0.42 ms |
| median frame luminance | 0.219 |
| median sharpness (Laplacian variance) | 963.8 |
| very dark / very bright frames | 0 / 0 |

No RGB interval was flagged: the stream is uninterrupted.

### All streams

| stream | samples | nominal Hz | measured Hz | representative Hz | duration s | flagged intervals | >250 ms | segments | p95 dt vs RGB ms | note |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `als` | 3,549 | 9 | 9.437 | 9.44 | 376.0 | 0 | 0 | 1 | 50.4 |  |
| `baro0` | 18,940 | n/a | 50.359 | 50.36 | 376.1 | 0 | 0 | 1 | 9.4 |  |
| `camera-et-left` | 1,880 | 5 | 5.000 | 5.00 | 375.8 | 0 | 0 | 1 | 96.3 |  |
| `camera-et-right` | 1,880 | 5 | 5.000 | 5.00 | 375.8 | 0 | 0 | 1 | 96.3 |  |
| `camera-rgb` | 3,762 | 10 | 10.001 | 10.00 | 376.1 | 0 | 0 | 1 | n/a |  |
| `eyegaze` | 11,283 | 30 | 30.000 | 30.00 | 376.1 | 0 | 0 | 1 | 15.9 |  |
| `gps-app` | 374 | n/a | 1.000 | 1.00 | 373.2 | 0 | 0 | 1 | 456.2 |  |
| `handtracking` | 3,761 | 10 | 10.000 | 10.00 | 376.0 | 0 | 0 | 1 | 39.5 |  |
| `imu-left` | 302,908 | 800 | 805.403 | 805.40 | 376.1 | 0 | 0 | 1 | 0.6 |  |
| `imu-right` | 298,497 | 800 | 793.651 | 793.65 | 376.1 | 0 | 0 | 1 | 0.6 |  |
| `mag0` | 37,812 | 100 | 100.534 | 100.53 | 376.1 | 0 | 0 | 1 | 4.7 |  |
| `ppg` | 96,805 | 256 | 257.438 | 257.44 | 376.1 | 6 | 0 | 5 | 1.8 |  |
| `slam-front-left` | 3,761 | 10 | 10.000 | 10.00 | 376.0 | 0 | 0 | 1 | 39.5 |  |
| `slam-front-right` | 3,761 | 10 | 10.000 | 10.00 | 376.0 | 0 | 0 | 1 | 39.5 |  |
| `slam-side-left` | 3,761 | 10 | 10.000 | 10.00 | 376.0 | 0 | 0 | 1 | 39.5 |  |
| `slam-side-right` | 3,761 | 10 | 10.000 | 10.00 | 376.0 | 0 | 0 | 1 | 39.5 |  |
| `temperature` | 7,222 | 1 | 7066.638 | 19.23 | 375.5 | 3190 | 385 | 2755 | 466.3 | bursty |

### Usability checks

- estimated missing frames 0.0000% within budget
- no pause longer than 250 ms
- RGB is a single continuous segment
- every required stream is present
- RGB timestamps are strictly monotonic

**Usable: yes**

## Conclusions

### Is the motorcycle recording usable?

**Yes.** 15,069 RGB frames over 1004.5 s, strictly monotonic timestamps, no duplicate timestamp, and a single continuous segment covering the whole recording.

### Is the recording continuous?

Effectively yes. Exactly one interval is anomalous: 133.317 ms between source frames 827 and 828, i.e. 2.00 nominal periods at t = 55.13 s from the start. That is one dropped frame out of 15,069 (0.0066%), it is shorter than the 250 ms pause threshold, and it does not split the recording.

There is no pause, no restart and no non-monotonic timestamp anywhere in the motorcycle recording.

### Which intervals are valid, and which must be excluded?

The whole recording is valid: [2,174,826,479,905 ns, 3,179,342,675,749 ns], 1004.5 s, source frame indices 0 to 15068.

**No interval has to be excluded on acquisition grounds.** The single dropped frame is recorded in `timestamp_gaps_moto.csv` so that any stage crossing it can account for it; it does not justify discarding a segment.

### Is the effective frame rate compatible with 15 fps?

Yes. Measured 15.00117 Hz from the median interval and 15.00026 Hz across the full span, against a declared nominal of 15.0 Hz: a deviation of 0.0078%. Interval jitter is 0.544 ms standard deviation around a 66.661 ms period, and the p1-p99 interval range is 66.58-66.73 ms. The recording is a genuine 15 fps acquisition.

### Which streams are available?

All 17 expected streams are present in **both** recordings: `als`, `baro0`, `camera-et-left`, `camera-et-right`, `camera-rgb`, `eyegaze`, `gps-app`, `handtracking`, `imu-left`, `imu-right`, `mag0`, `ppg`, `slam-front-left`, `slam-front-right`, `slam-side-left`, `slam-side-right`, `temperature`.

The motorcycle ingestion is therefore fully multimodal: RGB, on-device eye gaze, eye-tracking cameras, hand tracking, four SLAM cameras, two IMUs, magnetometer, barometer, GPS, PPG, ALS and temperature.

### What limitations remain?

- The `temperature` stream is bursty in both recordings: several samples arrive together and then the stream waits. Its median-interval rate is meaningless and its `representative Hz` (span rate) should be used. The flagged intervals for that stream are inter-burst waits, not data loss.
- GPS quality differs sharply between the two recordings: the motorcycle has a position fix in 94.9% of its samples with a median accuracy of 3.0 m, while the car has a fix in only 65.8% of its samples with a median accuracy of 27.5 m. Route alignment quality is bounded by the car, not by the motorcycle: a metal roof degrades reception in a way an open motorcycle does not.
- The car recording remains a provisional 10 fps baseline. Any comparison against the motorcycle is exploratory until the car is re-recorded at 15 fps.
- No reviewed ground truth exists yet, so nothing in this report is an accuracy statement.

## Machine-readable outputs

- `acquisition_manifest.json` / `.csv` — every candidate file with size, SHA-256, modification time, stream structure and domain evidence
- `stream_qa_moto.json` / `stream_qa_auto.json` — full per-stream QA
- `timestamp_gaps_moto.csv` / `timestamp_gaps_auto.csv` — every flagged interval of every stream

## Verdict

- motorcycle: **usable** over its full duration, with no excluded interval
- car: **usable**, but provisional at 10 fps

