# Article 1 — multimodal driver behaviour: final summary

**Status: `exploratory_pilot`.** One participant, one session per vehicle. The
car was recorded at ~10 fps and the motorcycle at ~15 fps. Nothing here
generalises to a population, nothing here is a medical measurement, and no event
is labelled as a traffic violation.

Branch `feature/article1-multimodal-behavior-analysis`, from baseline commit
`a9ee61bc77cdf61f364eb0eec98c2975fd933829`.

## What was built

A multimodal analysis pipeline over the two pilot recordings, on top of the
frozen Article 1 semantic-camera baseline, which it reads and never modifies.

| phase | module | runner |
|---|---|---|
| sensor readers | `behavior/sensors.py` | `build_article1_behavior_timeline.py` |
| synchronised timeline | `behavior/multimodal.py` | ″ |
| OSM + map matching | `behavior/osm.py`, `mapmatch.py` | `match_article1_routes.py` |
| spatial binning and pairing | `behavior/route.py` | `build_article1_shared_route.py` |
| vehicle + head dynamics | `behavior/dynamics.py` | `analyze_article1_dynamics.py` |
| semantic gaze | `behavior/gaze_semantics.py` | `analyze_article1_semantic_gaze.py` |
| PPG quality and HRV | `behavior/ppg.py` | `analyze_article1_ppg.py` |
| road events and responses | `behavior/events.py` | `analyze_article1_road_events.py` |
| line-crossing candidates | `behavior/solid_line.py` | `detect_article1_solid_line_candidates.py` |
| indices and statistics | `behavior/indices.py`, `stats.py` | `compare_article1_paired_route.py` |
| privacy, figures, video | `behavior/privacy.py`, `video.py` | `visualise_...`, `render_..._videos.py` |

Outputs: 9 reports, 21 figures, a local HTML dashboard, 3 analysis videos, a
human-review package, and per-recording parquet tables. **479 tests pass, 15
skipped, no regressions.**

## Headline results

* **42 paired 50 m route bins, 2 100 m of shared road** driven by both vehicles in
  the same direction, at 2.6 m median separation and 0° median heading
  difference.
* **Five differences survive FDR** on that shared stretch, all in the direction
  the physical setup predicts for an open vehicle: head vibration +0.22 m/s²
  (2.5×), illuminance +4 036 lux, PPG quality −0.12, head angular speed
  +0.05 rad/s, heart rate +42 bpm. Speed is *not* different on the shared
  stretch, which is exactly what spatial pairing is for.
* **The motorcycle occupies a lateral band 2.3× wider inside its lane** than the
  car (2.46 m vs 1.07 m, p05–p95), measured from the image rather than from GPS.
* **Heart rate rises around roundabouts and curves and falls on straights** in
  both vehicles — consistent in direction, but on single-digit event counts.
* **PPG is unusually clean**: 97.4% / 95.0% of windows usable, median SQI 0.92 /
  0.80.

## Five things the data cannot do, established by measurement

These are the most useful results in the pilot, because they define the protocol.

1. **No VIO/SLAM pose stream exists.** Vehicle speed falls back to the 1 Hz GPS
   speed field in both recordings.
2. **Hard braking is not detectable.** A 1 Hz speed series averages a brake
   application over a whole second; the car's most negative resolvable
   acceleration is −1.32 m/s² against a −2.5 m/s² threshold. Zero-braking counts
   are detection limits, and are labelled as such.
3. **A lane crossing cannot be resolved from GPS.** Median lateral error is 24 m
   (car) and 3 m (motorcycle) against the ~1.2 m displacement to be measured. All
   61 lateral-excursion candidates are correctly `not_evaluable`.
4. **The GPS cadence sets the usable route bin size.** At 1 Hz a fix lands every
   10–15 m, so a 10 m bin holds 0.7 fixes and cannot satisfy a two-fix pairing
   rule. 50 m is the smallest size the cadence can fill.
5. **Semantic gaze covers 8% and 3% of the recordings, and none of the shared
   route.** The frozen pipeline costs ~27 s/frame; full coverage would be ~140
   GPU-hours. Gaze consequently drops out of the paired comparison entirely.

## Methodological points worth carrying forward

* **Semantic gaze top-1 share is biased by viewing geometry.** A rider looking
  down the road puts the fovea on the vanishing point, where the road subtends
  less than the foveal window, so `road_surface` reads 4–9% for correct driving.
  Foveal probability mass does not have that failure mode and should be the
  primary metric (37% / 24% road-relevant).
* **The dicrotic-notch check matters.** A 123 bpm motorcycle median against 80
  bpm is exactly the shape of a double-detection artefact; it was tested for
  (lag-1 interval autocorrelation +0.53 / +0.10, not negative) and rejected
  rather than assumed away.
* **OSM roundabout nodes are also junction nodes**, which blankets every
  roundabout with generic junctions and destroys its own baseline unless
  excluded.
* **`curve` and `straight` partition the whole drive**, so they cannot be treated
  as baseline contaminants — and the straight before a curve is the baseline that
  curve wants.
* **Effective sample size, not row count.** 42 paired bins on one continuous
  stretch resolve to ~6 independent 30 s blocks.

## Required verdict

Scale: **A** usable for exploratory analysis · **B** usable with exclusions ·
**C** descriptive only · **D** must be repeated.

| # | item | grade | reasoning |
|---|---|:--:|---|
| 1 | car recording quality | **B** | every sensor clean except GPS, which loses 34% of its records and reports 27.5 m median accuracy |
| 2 | motorcycle recording quality | **A** | all 17 streams complete, GPS at 3 m, 94.9% valid fixes |
| 3 | semantic gaze quality | **C** | method and projection are sound (95% / 94% valid readings), but coverage is 8% / 3% and not on the shared route → the *coverage* alone is **D** |
| 4 | map matching quality | **B** | motorcycle **A** (99.2%, 1.5 m); car **B** (85%, 13.5 m) — adequate for 50 m bins, not finer |
| 5 | PPG quality | **A** | 97.4% / 95.0% usable, SQI 0.92 / 0.80, artefact check passed; HRV **B** |
| 6 | speed metric reliability | **C** | 1 Hz GPS speed field, no pose; cross-check r = 0.98 (moto) / 0.85 (car) |
| 7 | acceleration metric reliability | **C** | 1 Hz cannot resolve the events that matter; every zero count is a detection limit |
| 8 | line-crossing candidate reliability | **D** for the GPS route, **B** for the image-based lateral offset | GPS is 2.5–20× too coarse; the image measurement works and resolves a real 2.3× difference |
| 9 | car–motorcycle comparison quality | **C** | pairing and statistics are sound, but ≈6 independent blocks, one session per vehicle, and no gaze at all |
| 10 | data still needed | — | see below |

### 10. What still has to be acquired or computed

**Acquisition, in priority order**

1. **Re-record the car at 15 fps** with the same configuration on the same route.
   The current 10 fps car is the stated temporary artefact and half the analysis
   carries a caveat because of it.
2. **A higher-rate vehicle speed source.** Either enable the device pose stream or
   log vehicle CAN/OBD speed. This alone unblocks braking, acceleration and jerk
   — three of the four **C** grades above.
3. **Multiple sessions per vehicle, and more than one participant.** Every
   between-vehicle difference in this pilot is confounded with session. Without
   repeats, no inferential statistic is ever appropriate.
4. **Better GPS for the car**, or an alternative positioning source. 27.5 m
   median is the root cause of its **B** grades throughout.
5. **Counterbalanced ordering** (car-then-motorcycle and the reverse) across
   sessions, at comparable times of day.

**Computation, achievable on the data already held**

6. **Run the frozen pipeline on a block inside the shared route.** This is the
   single highest-value next step: ~2–3.5 GPU-hours per domain buys the entire
   paired semantic-gaze comparison, which is currently absent. Target bins are
   listed in `route/paired_route_segments.csv`.
7. **Extend semantic coverage generally** — even 4 × 30 s blocks per domain would
   take gaze coverage from 3% to 12% on the motorcycle.
8. **Human review of the 61 line-crossing candidates** using
   `datasets/article1_action_review/`, so a labelled set exists. No classifier
   may be trained below 20 confirmed events per class.
9. **A finer environment taxonomy.** `other_environment` absorbs 63–76% of foveal
   gaze mass; splitting sky / built / vegetation would make the gaze breakdown
   considerably more informative.

## Closing statement

This pilot's purpose was to define and validate the scientific pipeline for
Article 1, and it does that. The pipeline runs end to end on real data, every
stage records its own provenance and quality, and every limitation above was
found by measurement rather than assumed.

The substantive findings — vibration, illuminance, PPG quality, in-lane lateral
band — are real and reproducible on this data. They are also, with the exception
of the lateral band, largely properties of riding an open vehicle rather than
discoveries about driver behaviour. The behavioural questions the article is
actually about need the acquisitions listed above before they can be asked
properly.

Everything in this directory is marked `exploratory_pilot` and must be cited that
way.
