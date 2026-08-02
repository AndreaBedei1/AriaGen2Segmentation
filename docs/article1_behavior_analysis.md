# Article 1 multimodal behaviour analysis

How to run the pilot pipeline, and the rules it is built under.

Everything it produces is marked `exploratory_pilot`. One participant, one
session per vehicle, a car at ~10 fps and a motorcycle at ~15 fps.

## Environment

Every stage runs in the VRS I/O environment, which is the only one with
`projectaria_tools` **and** scipy/matplotlib/OpenCV:

```bash
IO_PY=$HOME/projectaria_gen2_python_env/bin/python
```

No stage needs a GPU. The semantic camera it consumes does, but that is the
frozen baseline and is not re-run here.

## Order of stages

```bash
$IO_PY scripts/build_article1_behavior_timeline.py        # ~5 min, decodes every sensor sample
$IO_PY scripts/match_article1_routes.py                   # ~20 s (first run fetches OSM)
$IO_PY scripts/build_article1_shared_route.py             # ~5 s
$IO_PY scripts/analyze_article1_dynamics.py               # ~5 s
$IO_PY scripts/analyze_article1_ppg.py                    # ~2 s
$IO_PY scripts/analyze_article1_semantic_gaze.py          # ~45 s
$IO_PY scripts/analyze_article1_road_events.py            # ~2 s
$IO_PY scripts/detect_article1_solid_line_candidates.py   # ~90 s
$IO_PY scripts/compare_article1_paired_route.py           # ~10 s
$IO_PY scripts/visualise_article1_behavior.py             # ~30 s
$IO_PY scripts/render_article1_behavior_videos.py         # ~8 min
$IO_PY scripts/write_article1_behavior_manifest.py        # ~2 s
```

Each stage reads the previous stage's parquet from
`output/article1/behavior_analysis/<recording_id>/` and writes its summary to
`reports/article1_behavior_analysis/`.

`match_article1_routes.py --no-network` runs from the on-disk Overpass cache
only, and fails loudly if the cache is cold rather than analysing a route with no
map.

## Configuration

`configs/article1/behavior_analysis.yaml`. Every temporal parameter is a
**duration in seconds**; nothing in the file is a frame count. The Overpass
`endpoint` is a list because the public instances answer 504 under load.

## The rules, and where they are enforced

| rule | enforced by |
|---|---|
| rates measured from timestamps, never assumed | `ingestion.timeline.measure_rate`, `sensors.SensorSeries.rate` |
| windows in seconds, converted per recording | `configs/.../behavior_analysis.yaml`, `frames_for_seconds` |
| counts normalised per second or per minute | `dynamics._per_minute`, `gaze_semantics` |
| no interpolation, no synthetic sample | `multimodal.derive_view` drops, never fills |
| association tolerance from measured cadences | `multimodal.association_tolerance_s` |
| gaze never feeds segmentation | `gaze_semantics` has no inference or mask-write call |
| vehicle motion never from the head IMU | separate functions; `compute_vehicle_dynamics` has no IMU reference |
| gravity removed and checked | `compute_head_dynamics`, magnitude vs 9.80665 |
| PPG gated before it becomes a rate | `ppg.assess_quality` → `tag_beats_with_quality` |
| HRV refused on short windows | `ppg.hrv_window` returns `refusals` |
| events from the map, not from the signals | `events.detect_events` reads `map_matched` only |
| line crossing needs several evidence sources | `solid_line._decide_state` |
| no verdict without human review | only four `CANDIDATE_STATES` exist |
| bins are the unit of analysis, not frames | `compare_article1_paired_route.py` |
| block bootstrap and block permutation | `stats.py` |
| no coordinate in a committed file | `privacy.assert_no_absolute_coordinates` |
| all twenty planned figures produced | `visualise_....REQUIRED_FIGURES`, checked before the run exits 0 |

`tests/test_behavior_guards.py` and `tests/test_behavior_analysis.py` fail the
build if any of these stops being true.

## Reading the outputs

Start at `reports/article1_behavior_analysis/REPORT_FINAL_SUMMARY.md`. It carries
the ten required verdicts and the list of what still has to be acquired.

`DATA_MANIFEST.json` says which committed tables are complete and which are a
slice, and where the full versions live locally.

The local dashboard is `output/article1/behavior_analysis/dashboard/index.html`.

Figures are numbered after the analysis plan: `01`–`20` are the twenty it asks
for, `21`–`24` are supplementary. `visualise_article1_behavior.py` exits non-zero
if any of the twenty is missing, so a partial gallery cannot pass unnoticed.

## Adding semantic coverage

This is the highest-value follow-up. The frozen pipeline costs ~27 s/frame, and
gaze currently covers 8% (car) and 3% (motorcycle) of the recordings. The car
block reaches 8 of the 42 paired bins and the motorcycle block reaches none of
them, so no bin carries gaze for both vehicles and gaze is absent from the paired
comparison. A motorcycle block inside those 8 bins is the cheapest thing that
unblocks it.

To fix it, pick a bin from `reports/article1_behavior_analysis/route/paired_route_segments.csv`,
find the frame range that covers it in `frame_route_bins.parquet`, and run the
existing frozen driver unmodified:

```bash
RUN_DIR=output/article1/<name> VRS=<recording>.vrs \
  START_NS=<...> END_NS=<...> RECORDING_ID=<...> VEHICLE=<car|motorcycle> \
  scripts/run_article1_motorcycle_baseline.sh
```

Then add the run directory to `semantic_blocks` in the config and re-run the gaze
stage. Roughly 2–3.5 GPU-hours per 30 s block.
