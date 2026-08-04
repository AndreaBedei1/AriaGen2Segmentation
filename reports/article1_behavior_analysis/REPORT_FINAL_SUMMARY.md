# Article 1 — multimodal driver behaviour: final summary

**Status: `exploratory_pilot`.** One participant, one session per vehicle. The
car was recorded at ~10 fps and the motorcycle at ~15 fps. Nothing here
generalises to a population, nothing here is a medical measurement, and no event
is labelled as a traffic violation.

Branch `feature/article1-final-behavior-statistics`, on top of
`feature/article1-multimodal-behavior-analysis` and the frozen semantic-camera
baseline `a9ee61bc77cdf61f364eb0eec98c2975fd933829`.

> **Superseded content removed.** Earlier revisions of this file reported that
> semantic gaze covered 8% and 3% of the two recordings and that no shared route
> bin carried gaze for both vehicles. Both statements described two 30 s frozen
> blocks and are no longer true of any artefact in this repository: semantic gaze
> now covers **93.2%** (car) and **89.6%** (motorcycle) of the recordings, and all
> **42 / 42** shared bins carry a gaze reading on both sides.

## What was built

A multimodal analysis pipeline over the two pilot recordings, on top of the frozen
Article 1 semantic-camera baseline, which it reads and never modifies.

| phase | module | runner |
|---|---|---|
| sensor readers | `behavior/sensors.py` | `build_article1_behavior_timeline.py` |
| synchronised timeline | `behavior/multimodal.py` | ″ |
| OSM + map matching | `behavior/osm.py`, `mapmatch.py` | `match_article1_routes.py` |
| spatial binning and pairing | `behavior/route.py` | `build_article1_shared_route.py` |
| vehicle + head dynamics | `behavior/dynamics.py` | `analyze_article1_dynamics.py` |
| semantic gaze (full run) | `behavior/gaze_semantics.py`, `semantic_attention.py` | `analyze_article1_final_behavior.py` |
| blink and pupil | `behavior/eye_state.py`, `blink.py`, `pupil.py` | `extract_article1_eye_state.py` |
| head–eye coordination | `behavior/head_eye.py` | `analyze_article1_final_behavior.py` |
| visual lane-position proxy | `behavior/lane_position.py` | `measure_article1_lane_position.py` |
| PPG quality and HRV | `behavior/ppg.py` | `analyze_article1_ppg.py` |
| road events and responses | `behavior/events.py` | `analyze_article1_road_events.py`, `analyze_article1_final_behavior.py` |
| line-crossing candidates | `behavior/solid_line.py` | `detect_article1_solid_line_candidates.py` |
| indices and statistics | `behavior/indices.py`, `stats.py` | `compare_article1_paired_route.py`, `compare_article1_final_paired.py` |
| privacy, figures, video | `behavior/privacy.py`, `video.py`, `article1/final_plots.py` | `visualise_...`, `render_..._videos.py` |

Outputs: 12 reports across this directory and
`../article1_final_behavior_statistics/`, 12 publication figures, a local HTML
dashboard, 3 analysis videos, a human-review package, and per-recording parquet
tables.

## Headline results

* **42 paired 50 m route bins, 2 100 m of shared road** driven by both vehicles in
  the same direction, at 2.6 m median separation and 0° median heading
  difference. **All 42 carry a semantic-gaze reading on both sides.**
* **Road-relevant gaze mass does not differ between the two vehicles** on that
  shared stretch (−7.4 pp, 95% CI [−10.4, +0.7], Cliff's δ = −0.17, p_FDR = 0.73),
  and neither does foveal entropy. This is the comparison the pilot was built to
  make and its answer is a null.
* **What differs is how attention is distributed in time.** Fixation time falls
  by 28.6 percentage points on the motorcycle (81% → 51%, δ = −0.83), scanpath
  length rises by 14.6 deg/s and blink rate by 18/min. The vibration confound
  below means this cannot yet be read as a behavioural difference.
* **Nine of 21 bin-level metrics survive FDR**: heart rate +42 bpm, IBI −254 ms,
  head vibration +0.22 m/s² (2.5×), fixation time −28.6 pp, interior-cockpit gaze
  −0.74 pp, scanpath +14.6 deg/s, blink rate +18/min, head angular speed
  +0.05 rad/s, light-adjusted pupil residual −0.039 mm. Speed is *not* different
  on the shared stretch, which is exactly what spatial pairing is for.
* **Blink and pupil metrics now exist**, read from fields the eye-gaze record
  carries but `projectaria_tools` 2.1.2 does not expose. The decode is verified
  against the official binding on 41 421 samples with zero mismatches.
* **The visual lane-position proxy covers 58% (car) and 70% (motorcycle) of the
  segmented frames.** Its p05–p95 band is 1.63 lane half-widths for the car and
  1.87 for the motorcycle — a 1.15× wider band, not distinguishable on the shared
  route (p_FDR = 0.98).
* **Heart rate rises around roundabouts and curves and falls on straights** in
  both vehicles — consistent in direction, but on single-digit event counts, and
  no event-level comparison survives FDR.
* **PPG is unusually clean**: 97.4% / 95.0% of windows usable, median SQI 0.92 /
  0.80.

## Five things the data cannot do, established by measurement

These are among the most useful results in the pilot, because they define the
protocol.

1. **No VIO/SLAM pose stream exists.** Vehicle speed falls back to the 1 Hz GPS
   speed field in both recordings.
2. **Hard braking is not detectable.** A 1 Hz speed series averages a brake
   application over a whole second; the car's most negative resolvable
   acceleration is −1.32 m/s² against a −2.5 m/s² threshold. Zero-braking counts
   are detection limits, and are labelled as such.
3. **A lane crossing cannot be resolved from GPS.** Median lateral error is 24 m
   (car) and 3 m (motorcycle) against the ~1.2 m displacement to be measured. All
   61 GPS-based lateral-excursion candidates are correctly `not_evaluable`. The
   image-based `visual_lane_position_proxy` *can* be measured, but it is the
   position of a head-mounted camera between two markings, not the vehicle's
   metric position in its lane.
4. **The GPS cadence sets the usable route bin size.** At 1 Hz a fix lands every
   10–15 m, so a 10 m bin holds 0.7 fixes and cannot satisfy a two-fix pairing
   rule. 50 m is the smallest size the cadence can fill.
5. **Event-related responses are not testable at this event count.** 40 car
   events against 184 motorcycle events, 3–10 per type for the car, one traffic
   signal in the whole pilot. Nothing survives FDR and nothing should.

## The confound that limits the largest new finding

The motorcycle transmits **2.5× the head vibration** of the car (+0.225 m/s²,
δ = +0.99, the cleanest effect in the pilot). A vibrating head produces
gaze-velocity excursions that an I-VT fixation classifier reads as saccadic, so
*less measured fixation time on a vibrating platform is expected for instrumental
reasons alone*.

"The rider fixated less" and "the platform shook the measurement" are not
separable in this data. Separating them needs a fixation classifier validated
under vibration, or a head-referenced gaze signal. Until then the fixation-time,
scanpath and transition-rate results stay descriptive.

## Methodological points worth carrying forward

* **Semantic gaze top-1 share is biased by viewing geometry.** A rider looking
  down the road puts the fovea on the vanishing point, where the road subtends
  less than the foveal window, so `road_surface` reads low for correct driving.
  Foveal probability mass does not have that failure mode and is the primary
  metric throughout (34.8% / 33.2% road-relevant).
* **The eye-gaze record carries more than its Python binding does.** Pupil
  diameter and its validity flag are in the VRS `DataLayout` and are not exposed
  by `projectaria_tools` 2.1.2. Reading the payload directly is safe *if* the
  layout is read from the file and the decode is verified against the binding on
  every field it does expose.
* **The `blink` field's polarity is inverted relative to its name** in both
  recordings. A pipeline trusting the field name would have reported a 90%
  eye-closure fraction. The polarity was determined from run lengths and
  corroborated by 96% / 95% agreement with gaze invalidation.
* **A fixed image band cannot serve both vehicles.** The rows that are near-field
  road on a motorcycle are 85% dashboard in a car. The lane-position band is
  measured per recording from the segmentation's own row-wise road profile.
* **The dicrotic-notch check matters.** A 123 bpm motorcycle median against 80 bpm
  is exactly the shape of a double-detection artefact; it was tested for (lag-1
  interval autocorrelation +0.53 / +0.10, not negative) and rejected rather than
  assumed away.
* **OSM roundabout nodes are also junction nodes**, which blankets every
  roundabout with generic junctions and destroys its own baseline unless excluded.
* **`curve` and `straight` partition the whole drive**, so they cannot be treated
  as baseline contaminants — and the straight before a curve is the baseline that
  curve wants.
* **Effective sample size, not row count.** 42 paired bins on one continuous
  stretch resolve to ~6 independent 30 s blocks.
* **A metric that reproduces across two units of analysis is worth more than one
  that does not.** Blink rate survives on 50 m bins *and* on 30 s time blocks with
  very different quantisation; the pupil residual survives on bins and not on
  blocks, and is reported as the weaker of the two.

## Required verdict

Scale: **A** usable for exploratory analysis · **B** usable with exclusions ·
**C** descriptive only · **D** must be repeated.

| # | item | grade | reasoning |
|---|---|:--:|---|
| 1 | car recording quality | **B** | every sensor clean except GPS, which loses 34% of its records and reports 27.5 m median accuracy |
| 2 | motorcycle recording quality | **A** | all 17 streams complete, GPS at 3 m, 94.9% valid fixes |
| 3 | semantic gaze quality | **A** | method and projection sound, and coverage is now 93.2% / 89.6% with 42/42 shared bins on both sides; the missing fraction is blinks, measured rather than assumed |
| 4 | blink / pupil extraction | **A** | layout read from the file, decode verified field-by-field with zero mismatches on 41 421 samples |
| 5 | map matching quality | **B** | motorcycle **A** (99.2%, 1.5 m); car **B** (85%, 13.5 m) — adequate for 50 m bins, not finer |
| 6 | PPG quality | **A** | 97.4% / 95.0% usable, SQI 0.92 / 0.80, artefact check passed; HRV **B** |
| 7 | speed metric reliability | **C** | 1 Hz GPS speed field, no pose; cross-check r = 0.98 (moto) / 0.85 (car) |
| 8 | acceleration metric reliability | **C** | 1 Hz cannot resolve the events that matter; every zero count is a detection limit |
| 9 | lane-position measurement | **D** for the GPS route, **C** for the visual proxy | GPS is 2.5–20× too coarse; the image proxy is measurable but is a camera position on a moving head |
| 10 | car–motorcycle comparison quality | **B** | pairing, statistics and gaze coverage are all sound; ≈6 independent blocks and one session per vehicle are the binding limits |
| 11 | event-related comparison | **D — must be repeated** | full coverage in every window, but 3–10 car events per type and nothing survives FDR |
| 12 | data still needed | — | see below |

### 12. What still has to be acquired or computed

**Acquisition, in priority order**

1. **Multiple sessions per vehicle, and more than one participant.** Every
   between-vehicle difference in this pilot is confounded with session. Without
   repeats, no inferential statistic is ever appropriate. This is now the single
   highest-value acquisition, ahead of everything below.
2. **A higher-rate vehicle speed source.** Either enable the device pose stream or
   log vehicle CAN/OBD speed. This alone unblocks braking, acceleration and jerk
   — three of the **C** grades above.
3. **Re-record the car at 15 fps** with the same configuration on the same route.
4. **Better GPS for the car**, or an alternative positioning source. 27.5 m
   median is the root cause of its **B** grades throughout.
5. **Counterbalanced ordering** (car-then-motorcycle and the reverse) across
   sessions, at comparable times of day.
6. **A blink reference** (EOG or high-speed video) for one session, to calibrate
   the on-device detector's absolute rate. The between-vehicle contrast does not
   need it; the absolute rate does.

**Computation, achievable on the data already held**

7. **A fixation classifier validated under vibration**, or a head-referenced gaze
   signal. Without it the largest new gaze finding stays ambiguous.
8. **Human review of the lane-position crossing candidates** (10 car, 34
   motorcycle) and of the 61 GPS line-crossing candidates, using
   `datasets/article1_action_review/`, so a labelled set exists. No classifier may
   be trained below 20 confirmed events per class.
9. **A finer environment taxonomy.** `other_environment` still absorbs a large
   share of foveal gaze mass even after the fast taxonomy split sky, built
   environment and vegetation apart.

## Closing statement

This pilot's purpose was to define and validate the scientific pipeline for
Article 1, and it does that. The pipeline runs end to end on real data, every
stage records its own provenance and quality, and every limitation above was
found by measurement rather than assumed.

The gaze coverage limitation that dominated the previous revision is gone: the
semantic-gaze analysis now covers both recordings in full and enters the paired
comparison on every shared bin. Its headline answer is a null — attention goes to
the same places in both vehicles — and the differences that survive are about the
temporal distribution of attention, with a vibration confound this data cannot
discharge.

The substantive findings — vibration, illuminance, PPG quality, blink rate — are
real and reproducible on this data. They are also, with the possible exception of
blink rate, largely properties of riding an open vehicle rather than discoveries
about driver behaviour. The behavioural questions the article is actually about
need the acquisitions listed above before they can be asked properly.

Everything in this directory is marked `exploratory_pilot` and must be cited that
way.
