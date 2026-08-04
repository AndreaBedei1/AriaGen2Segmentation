# Article 1 — final behaviour statistics: summary

**Status: `exploratory_pilot`.** One participant, one session per vehicle. The car
was recorded at ~10 fps and the motorcycle at ~15 fps. Nothing here generalises to
a population, nothing here is a medical measurement, and no event is labelled a
traffic violation.

Branch `feature/article1-final-behavior-statistics`, from
`154a041093cbae5eb40aead2a56d0f93d5b0558b`.

## What this stage closed

The behaviour analysis had one binding limitation: semantic gaze existed for two
30 s frozen blocks, one per vehicle, and **the two blocks did not overlap on the
shared route**, so gaze could not enter the car/motorcycle comparison at all.

That is resolved. The full-recording semantic gaze covers 93.2% (car) and 89.6%
(motorcycle) of each drive, **all 42 shared 50 m bins carry a gaze reading on both
sides**, and the paired comparison now includes gaze on equal footing with
dynamics and physiology.

Blink and pupil metrics were added from the eye-gaze record's own fields, which
`projectaria_tools` 2.1.2 does not fully expose.

| phase | module | runner |
|---|---|---|
| eye state: blink + pupil | `behavior/eye_state.py` | `extract_article1_eye_state.py` |
| blink events and metrics | `behavior/blink.py` | ″ |
| pupil metrics + light correction | `behavior/pupil.py` | ″ |
| semantic attention | `behavior/semantic_attention.py` | `analyze_article1_final_behavior.py` |
| head–eye coordination | `behavior/head_eye.py` | ″ |
| event-window metrics | `behavior/events.py` (windows) | ″ |
| visual lane-position proxy | `behavior/lane_position.py` | `measure_article1_lane_position.py` |
| paired statistics | `behavior/stats.py` | `compare_article1_final_paired.py` |
| figures | `article1/final_plots.py` | `visualise_article1_final_behavior.py` |

Outputs: 3 new reports, 6 updated ones, 12 figures (the nine publication figures
unchanged plus three additions), and per-recording parquet tables under
`output/article1/final_behavior_statistics/`.

Segmentation was not modified. Mask2Former was not re-run. The 5 Hz and native
runs are untouched — `tests/test_final_behavior_statistics.py` pins their content
hashes. Grounding DINO and SAM 2.1 were not used.

## Headline results

* **Road-relevant gaze mass does not differ between the two vehicles on the
  shared route** — −7.4 pp, 95% CI [−10.4, +0.7], Cliff's δ = −0.17, p_FDR = 0.73.
  Nor does foveal entropy, nor top-1 distribution entropy. This is the comparison
  the pilot was built to make; its answer is a null, and it constrains everything
  else.
* **Attention is deployed differently in time, not in space.** Fixation time falls
  by 28.6 pp on the motorcycle (81% → 51%, δ = −0.83), scanpath length rises by
  14.6 deg/s (+38%), blink rate by 18/min (+74%). Same targets, less dwelling.
  **Caveat that cannot be discharged here:** the motorcycle transmits 2.5× the
  head vibration, and a vibrating head produces gaze-velocity excursions an I-VT
  classifier reads as saccadic. "Fixated less" and "the measurement was shaken"
  are not separable without a classifier validated under vibration.
* **Nine of 21 bin-level metrics survive FDR**: heart rate +41 bpm, IBI −254 ms,
  head vibration +0.225 m/s², fixation time −28.6 pp, interior-cockpit gaze
  −0.74 pp, scanpath +14.6 deg/s, blink rate +18/min, head angular speed
  +0.05 rad/s, pupil residual −0.039 mm.
* **Blink rate is the one new finding that reproduces across two units of
  analysis**: +18/min on 50 m bins and +23/min on 30 s time blocks (δ = +0.87,
  p_FDR = 0.008), with very different quantisation in each.
* **Pupil diameter is dominated by light.** A `log10(lux)` term explains **84%** of
  its variance on the motorcycle and 35% on the car. Any raw-diameter comparison
  between the two vehicles would substantially have been comparing their glazing.
* **Segmenting at 5 Hz changes nothing.** The native-cadence robustness run agrees
  to the third significant figure on every recording-level attention metric.

## Five things the data still cannot do

1. **No VIO/SLAM pose stream exists.** Vehicle speed falls back to the 1 Hz GPS
   speed field in both recordings.
2. **Hard braking is not detectable.** A 1 Hz speed series averages a brake
   application over a whole second. Zero-braking counts are detection limits.
3. **A lane crossing cannot be resolved from GPS.** The image-based
   `visual_lane_position_proxy` can be measured (57.8% / 69.6% of frames) but it
   is a camera-between-markings measurement on a head-mounted camera, not the
   vehicle's metric position in its lane.
4. **Event-related responses are not testable.** 40 car events against 184
   motorcycle events, 3–10 per type for the car, one traffic signal in the whole
   pilot. Nothing survives FDR and nothing should.
5. **≈6 independent 30 s blocks.** 42 paired bins on one continuous stretch is the
   row count, not the sample size, and one session per vehicle confounds every
   between-vehicle difference with session.

## Method points worth carrying forward

* **The eye-gaze record carries more than its Python binding does.** Pupil
  diameter, its validity flag and the entrance-pupil position are all in the VRS
  `DataLayout` and none is exposed by `projectaria_tools` 2.1.2. Reading the
  payload directly is safe *if* the layout is read from the file and the decode is
  verified against the binding on every field it does expose — here, ten fields
  spanning buffer offsets 0 to 158, with zero mismatches on 41 421 samples.
* **The `blink` field's polarity is inverted relative to its name** in both
  recordings: `blink == False` is the eye closure. That was established from the
  run-length distribution, not assumed, and corroborated independently by 96% /
  95% agreement with gaze invalidation. A pipeline that trusted the field name
  would have reported a 90% eye-closure fraction.
* **A fixed image band cannot serve both vehicles.** The rows that are near-field
  road on a motorcycle are 85% dashboard in a car, because a head-mounted camera
  behind a windscreen sees the bonnet where the motorcycle sees tarmac. The
  lane-position band is measured per recording from the segmentation's own
  row-wise road profile; the first fixed-band attempt measured the car's dashboard
  and returned 0.1% coverage.
* **A lane-position rule that picks one marking either side of the camera axis
  bounds its own output to ±1** and can never detect a crossing. Selecting the
  adjacent marking *pair* whose lane centre is nearest the axis removes that
  structural bound.
* **Foveal probability mass, not top-1 share.** A driver looking down the road
  puts the fovea on the vanishing point, where the road subtends less than the
  foveal window, so the argmax goes to the surrounding scene. Both are reported;
  the mass figures carry every comparison.
* **A lateral head check is not a mirror check.** Head turns to a mirror, to a
  side road and to scenery are identical in this signal. No output is allowed to
  be relabelled as a mirror check, and `tests/` pins that.

## Verdict

Scale: **A** usable for exploratory analysis · **B** usable with exclusions ·
**C** descriptive only · **D** must be repeated.

| # | item | grade | reasoning |
|---|---|:--:|---|
| 1 | blink / pupil extraction | **A** | layout read from file, stream identified by content, decode verified field-by-field with zero mismatches on 41 421 samples |
| 2 | blink polarity determination | **A** | data-driven, evidence stored, independently corroborated |
| 3 | **semantic gaze coverage** | **A** | 93.2% / 89.6% of the recordings; 42/42 shared bins carry gaze on both sides |
| 4 | semantic attention metrics | **B** | complete, geometry-robust primary metric, `other_environment` still absorbs a large share |
| 5 | head–eye coordination | **C — descriptive only** | measured cleanly, but head checks cannot be attributed to a target |
| 6 | visual lane-position proxy | **C — descriptive only** | 58% / 70% coverage, band measured per recording, but it is a camera position on a moving head |
| 7 | paired statistics | **A** | three named units, block bootstrap, block permutation, FDR within family, effective sample size ahead of row count |
| 8 | car–motorcycle gaze comparison | **B — usable with exclusions** | now possible on every shared bin; ≈6 independent blocks is the binding limit |
| 9 | event-related comparison | **D — must be repeated** | nothing survives FDR; counts are 3–10 per type for the car |
| 10 | any medical or clinical reading | **not attempted, and not supportable** | |

## What still has to be acquired

1. **Multiple sessions per vehicle and more than one participant.** Every
   between-vehicle difference here is confounded with session. This is the single
   thing that would change the most.
2. **A fixation classifier validated under vibration**, or a head-referenced gaze
   signal. Without it the largest new gaze finding stays ambiguous.
3. **A higher-rate vehicle speed source** (device pose or CAN/OBD). Unblocks
   braking, acceleration and jerk.
4. **A blink reference** (EOG or high-speed video) for one session, to calibrate
   the on-device detector's absolute rate.
5. **Re-record the car at 15 fps**, and **better GPS for the car** (27.5 m median
   accuracy is the root cause of several **B** grades).
6. **Human review of the lane-position crossing candidates** (10 car, 34
   motorcycle) so a labelled set exists.

## Closing statement

The pilot's semantic gaze is now complete, the paired car–motorcycle comparison
includes it on every shared bin, and blink and pupil metrics exist where before
there were none. The headline gaze result is a null — attention goes to the same
places in both vehicles — and the differences that survive are about *how*
attention is distributed in time, with a vibration confound that this data cannot
discharge.

Everything in this directory is marked `exploratory_pilot` and must be cited that
way.
