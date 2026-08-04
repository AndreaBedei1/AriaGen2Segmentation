# Article 1 — final behaviour statistics

How to reproduce the stage that integrates the full-recording semantic gaze, adds
blink and pupil metrics, and produces the final paired car/motorcycle comparison.

Everything here is **exploratory pilot** work: one participant, one session per
vehicle. Nothing is a medical measurement, and no eye or pupil metric may be
relabelled as drowsiness, workload or a clinical sign.

## What this stage consumes, and never writes to

| input | role |
|---|---|
| `output/article1/fast_semantic_gaze/` | primary semantic gaze, 5 Hz segmentation, **read-only** |
| `output/article1/fast_semantic_gaze_native/` | robustness check, native cadence, **read-only** |
| `output/article1/behavior_analysis/` | route bins, dynamics, PPG, ALS |
| `reports/article1_behavior_analysis/route|events|solid_line/` | shared-route pairing, road events, line candidates |
| the two VRS recordings | blink and pupil, read directly from the eye-gaze record |

The two semantic-gaze runs are inputs and are never modified. `output/` is not
version-controlled, so the guard is a checked-in manifest of content hashes,
`reports/article1_final_behavior_statistics/frozen_run_manifest.json`, which
`tests/test_final_behavior_statistics.py` verifies. Regenerate it only when a run
has been deliberately re-executed:

```bash
python scripts/write_article1_frozen_run_manifest.py
```

## Running the stage

Everything runs in **env A** (`~/projectaria_gen2_python_env`); no ML environment
and no GPU are needed, because the segmentation already exists.

```bash
PY=~/projectaria_gen2_python_env/bin/python

# 1. Blink and pupil, straight from the eye-gaze record's own DataLayout.
#    Refuses to write anything if the decode does not reproduce projectaria_tools.
$PY scripts/extract_article1_eye_state.py

# 2. Visual lane-position proxy over the full run (~40 s; reads 6 904 masks).
$PY scripts/measure_article1_lane_position.py

# 3. Semantic attention, event windows and head-eye coordination.
$PY scripts/analyze_article1_final_behavior.py

# 4. Paired statistics: bin, event and time-block families.
$PY scripts/compare_article1_final_paired.py

# 5. Nine publication figures plus the three additions.
$PY scripts/visualise_article1_final_behavior.py

# 6. The legacy paired comparison, now also consuming the full gaze.
$PY scripts/compare_article1_paired_route.py
```

Steps 1–2 are independent of each other; 3 needs both; 4 and 5 need 3.

## Reading pupil diameter out of the VRS

`projectaria_tools` 2.1.2 binds the eye-gaze record into `mps.EyeGaze` /
`mps.EyeGazeVergence`. Those expose the blink flags and their validity flags but
**not** the pupil diameter, which the record nonetheless carries:

```
left_eye/pupil_diameter_valid   Bool    offset 55
left_eye/pupil_diameter_meter   float   offset 56
left_eye/blink_valid            Bool    offset 60
left_eye/blink                  Bool    offset 61
```

`behavior/eye_state.py` decodes the record payload directly. Three rules make that
safe, and all three are enforced in code rather than documented and hoped for:

1. **the layout is read from the file.** Field names, types and byte offsets come
   from the descriptor the recording itself embeds. A different firmware layout is
   decoded correctly or refused — never decoded wrongly;
2. **the stream is identified by content.** The only record type accepted is the
   one whose decoded tracker timestamp reproduces its own VRS record-header
   timestamp;
3. **the decode is verified against the binding on every sample.** Ten fields
   spanning buffer offsets 0 to 158 — including the four blink fields at 60, 61,
   106, 107, which bracket the two pupil fields at 56 and 102. Zero mismatches on
   41 421 samples. `extract_article1_eye_state.py` exits non-zero if this fails.

If a future `projectaria_tools` binds the pupil fields, prefer it and keep the raw
reader as the cross-check.

## The blink flag's polarity

In both recordings the field named `blink` is `True` for ~90% of samples. That is
not an eye shut for 90% of a drive; the polarity is inverted relative to the name.

`determine_blink_polarity` decides from the data: the closed state is the polarity
that is **both rarer and shorter-running**. If the two are not clearly separated it
returns `undetermined` and the pipeline refuses to emit events rather than
guessing. The raw values are stored unchanged; polarity is applied only when
deriving closure.

Independent corroboration: 96.7% (car) and 95.1% (motorcycle) of gaze-invalid
samples coincide with a derived closure. Gaze cannot be estimated through a closed
eyelid, so that agreement is what makes the decision safe to build on.

## Conventions this stage holds to

* **The unit of analysis is a bin, an event or a time block — never a frame.**
  Every statistics row states its unit and its effective sample size, which is the
  number of independent blocks, not the row count.
* **Every window is a duration in seconds** taken from
  `configs/article1/behavior_analysis.yaml`, and is applied to each signal family
  on that family's **own real samples**.
* **Nothing is interpolated.** Blink events are runs of consecutive real samples;
  a sampling gap ends a run. Light is attached from the nearest real ALS sample
  inside a tolerance, or not at all.
* **Foveal probability mass is the primary gaze metric**, top-1 share the
  secondary one. Top-1 is biased by viewing geometry: gaze at the vanishing point
  lands where the road subtends less than the foveal window.
* **A lateral head check is never a mirror check.** No artefact may relabel it,
  and the tests pin that.
* **The visual lane-position proxy is never a vehicle position.** Every artefact
  carries `visual_lane_position_proxy` and `is_vehicle_metric_position: false`.

## Two measurement traps worth remembering

**A fixed image band cannot serve both vehicles.** The rows that are near-field
road on a motorcycle are 85% dashboard in a car, because a head-mounted camera
behind a windscreen sees the bonnet where the motorcycle sees tarmac. A fixed
0.72–0.95 band produced 0.1% lane-position coverage on the car. The band is now
measured per recording from the segmentation's own row-wise road profile:
0.663–0.720 for the car, 0.725–1.000 for the motorcycle.

**A lane-position rule that takes one marking from either side of the camera axis
bounds its own output to ±1** and can never report a crossing, because the axis
then always lies between the two boundaries. Selecting the adjacent marking
*pair* whose lane centre is nearest the axis removes that structural bound.

## Outputs

| path | content |
|---|---|
| `output/article1/final_behavior_statistics/<rec>/eye_state_samples.parquet` | one row per real gaze record: blink, pupil, validity flags, timestamps |
| ″ `blink_events.parquet`, `blink_rate_windows.parquet` | events with their category and reason; 60 s / 10 s windows |
| ″ `pupil_samples.parquet` | diameter, lux, local baseline, light-adjusted residual |
| ″ `lane_position_frames.parquet`, `lane_position_crossing_candidates.parquet` | per-frame proxy and its candidates |
| `semantic_attention_bins.parquet` | every metric per (domain, shared bin) |
| `semantic_attention_bin_classes.parquet` | per-class dwell, fixation, revisit, time-to-first |
| `event_response.parquet` | every metric per (event, window), with deltas |
| `paired_statistics.parquet` | the three comparison families |
| `reports/article1_final_behavior_statistics/` | three reports, twelve figures, the frozen-run manifest |

## Tests

```bash
~/projectaria_gen2_python_env/bin/python -m pytest tests/test_final_behavior_statistics.py -q
```

The file mixes behavioural tests on synthetic data whose answer is known by
construction with repository guards: no interpolated blink, no frame as a
statistical unit, no medical claim, no head check relabelled as a mirror check,
the frozen runs unchanged, and no report still carrying the retired 8% / 3% gaze
coverage claim.
