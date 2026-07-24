# Architecture

How `aria_drive_seg` is organized, how data flows between stages, why it runs across three
Python environments, and the design of resumability, overlap resolution, and parallelism.

---

## Module map

```
aria_drive_seg/
├── cli.py            # argparse CLI: inspect|extract|segment|align-gaze|analyze|render|run-all|hwinfo
├── __main__.py       # `python -m aria_drive_seg`
├── pipeline.py       # run-all orchestrator: dispatches each stage to the right interpreter
├── config.py         # Config: deep-merge YAML load, dotted get/set, project-root path resolution
├── taxonomy.py       # Taxonomy: canonical id<->name, deterministic palette, colorize()
├── io_utils.py       # atomic writes, uint16 mask PNG I/O, JSONL, Manifest, config_fingerprint
├── hashing.py        # sha256_file, canonical_json, stable_hash (cache keys, weights manifest)
├── hwinfo.py         # hardware_report.json (nvidia-smi + torch + CPU/RAM/disk)
├── netguard.py       # socket monkeypatch that blocks/records non-loopback connections
├── logging_utils.py  # setup_logging / get_logger
│
├── vrs/
│   ├── provider.py   # AriaProvider: projectaria_tools wrapper; streams, RGB, gaze, calib, Rectifier
│   ├── inspect.py    # `inspect`: stream enumeration, timestamp health, calibration, samples
│   └── extract.py    # `extract`: original + rectified frames, frame index (parquet/csv)
├── gaze/
│   ├── align.py      # `align-gaze`: temporal alignment of gaze to frames
│   └── project.py    # GazeProjector: CPF->Device->Camera reprojection into both geometries
├── segmentation/
│   ├── base.py       # shared FrameOutput schema, SegLayout paths, write/iter/aggregate
│   ├── grounded_sam2.py  # Method 1: Grounding DINO + SAM 2.1
│   ├── oneformer.py      # Method 2: Mask2Former / OneFormer on Mapillary + MapillaryMapper
│   ├── overlap.py        # class-aware NMS + pixel-level overlap resolution
│   └── run.py            # resumable runner, per-frame error isolation, OOM guard
├── analyze/
│   ├── gaze_labels.py    # per-frame gaze-to-class assignment (pixel, disc, gaussian, boundary)
│   ├── compare.py        # per-method + between-method (agreement, IoU) + gaze metrics
│   └── run.py            # `analyze`: orchestrates the above, writes metrics + markdown
└── render/
    ├── overlays.py       # overlay primitives (blend, gaze marker, legend, HUD)
    └── run.py            # `render`: overlay + comparison + gaze videos and diagnostics
```

Configuration lives in `configs/` (`default.yaml`, `classes.yaml`, `grounded_prompts.yaml`,
`mapillary_to_canonical.yaml`); one-time setup scripts in `scripts/`.

---

## Data flow

```
              env A (VRS I/O)                    env B (torch)                 env A (VRS I/O)
   ┌───────────────────────────────┐   ┌──────────────────────────┐   ┌──────────────────────────────┐
.vrs ─▶ inspect ─▶ extract ─▶ frames/ ─┬─▶ segment grounded_sam2 ─▶ grounded_sam2/ ─┐
                     │                  └─▶ segment oneformer_mapillary ─▶ oneformer_mapillary/ ─┤
                     │                                                                           ├─▶ analyze ─▶ comparison/, reports/
                     └────────────────────────▶ align-gaze ─▶ gaze/ ──────────────────────────┘         │
                                                                                                          └─▶ render ─▶ videos/, figures/
```

- **inspect** validates the recording (streams, RGB/gaze health, calibration) and writes sample images.
- **extract** decodes RGB sequentially, writes `frames/original` and `frames/rectified` plus a
  `frames.parquet`/`frames.csv` index. It never loads the whole video into RAM.
- **segment** (once per method) reads the rectified frames and writes canonical (+native, for
  Method 2) id-masks, confidence, and per-frame metadata under `<method>/`.
- **align-gaze** reads the on-device eye-gaze and the frame index, and writes raw + per-frame
  aligned gaze parquet.
- **analyze** consumes both methods' masks + the aligned gaze and writes all no-GT metrics.
- **render** composites overlays and comparison/gaze videos from masks + gaze.

The **rectified** frame is the canonical input to segmentation, so both methods and the gaze
projection share one geometry.

---

## The three-environment design

### Why three environments

The stages have **conflicting dependency stacks**:

- **VRS I/O** needs `projectaria-tools` (+ opencv) to open the `.vrs`, read calibration, and do
  the official gaze reprojection. It does **not** need torch.
- **ML inference** needs a specific torch/CUDA stack plus `transformers` and the official
  `sam2`. Pulling torch into the VRS env is unnecessary and version-fragile.
- **OneFormer DiNAT-L** additionally needs the `natten` custom op, which is pinned to *older*
  torch versions than the primary ML env (see [model_selection.md](model_selection.md)). This
  cannot coexist with the modern torch in env B, so it gets its own optional env.

Keeping them separate means each stack stays reproducible and neither contaminates the other.
The `aria_drive_seg` package installs editable (`pip install -e .`) in all three; its *core*
dependencies are deliberately light (numpy, pyyaml, pyarrow, pandas, pillow) so it imports in
the VRS env without torch. `projectaria_tools`, torch, and cv2 are all imported **lazily**
inside the functions that use them.

| Env | Var | Stages |
|---|---|---|
| A `~/projectaria_gen2_python_env` | `ARIA_VRS_PYTHON` | inspect, extract, align-gaze, analyze, render |
| B `~/aria_seg_ml_env` | `ARIA_ML_PYTHON` | segment (grounded_sam2, oneformer_mapillary/Mask2Former) |
| C *(optional)* | `ARIA_ONEFORMER_PYTHON` | segment (oneformer_mapillary with `source: hf_oneformer`) |

### How run-all dispatches

`pipeline.run_all` does not import torch or projectaria_tools itself. It shells out to
`python -m aria_drive_seg <stage>` using the interpreter resolved from the env vars
(`_py(var)` falls back to the current interpreter if the var is unset). The OneFormer segment
stage is routed to `ARIA_ONEFORMER_PYTHON` **only** when
`oneformer_mapillary.source == hf_oneformer`; otherwise it uses `ARIA_ML_PYTHON`. If a
segmentation method fails, `run-all` logs it and continues with the rest.

### The on-disk schema is the interface between environments

Because stages run in different interpreters (and could run on different machines), they
**never share Python objects** — they communicate only through files:

- `frames/frames.parquet` is the contract produced by `extract` and consumed by `segment`,
  `align-gaze`, and `render` (`base.iter_frames` reads it). Columns: `frame_index`,
  `capture_timestamp_ns`, `original_path`, `rectified_path`, `rotate_ccw90`, and image sizes.
- Each method writes the identical `SegLayout` (`canonical_masks/`, `confidence/`,
  `metadata/`, `metadata.parquet`, `manifest.json`, plus `native/` for Method 2). Downstream
  code is method-agnostic because it only knows this layout.
- `gaze/aligned_gaze.parquet` keys gaze to `frame_index`, joining cleanly to masks.

All structured outputs are Parquet/JSON with explicit, documented columns; masks are lossless
uint16 PNG id-maps. Nothing downstream depends on a method's in-memory representation.

---

## Resumability, caching, and the config fingerprint

Long runs must survive interruption and avoid recomputing unchanged work. The mechanism
(`io_utils.Manifest` + `io_utils.config_fingerprint`, backed by `hashing.stable_hash`):

1. **Fingerprint.** At the start of `extract` and each `segment`, the stage hashes everything
   that must invalidate its cache into a short stable string:
   - `extract`: rectify params, JPEG quality, rectify-enabled flag, the `frames` selection, and the VRS path.
   - `grounded_sam2`: the whole `grounded_sam2` config block, the **full text** of
     `grounded_prompts.yaml`, and the `rectify` block.
   - `oneformer_mapillary`: the `oneformer_mapillary` config block, the **full text** of
     `mapillary_to_canonical.yaml`, and the `rectify` block.
2. **Manifest.** `Manifest.load_or_new` loads the existing `manifest.json` **only if** its
   stored fingerprint and stage match; otherwise it starts fresh (stale cache invalidated).
   The manifest records each completed frame and small per-frame stats.
3. **Skip rule.** A frame is skipped only if the manifest marks it done **and** its output
   files actually exist on disk (`is_frame_done` checks the mask + metadata, and the native
   mask for Method 2). This tolerates partially-deleted outputs.
4. **`--force`** clears the done-set (recompute all); **`--no-resume`** disables skipping.

So changing the **checkpoint, prompts, thresholds, taxonomy mapping, calibration, or
resolution** changes the fingerprint and transparently invalidates exactly the affected
stage's cache — while an unrelated edit (e.g. render colors) does not.

### Atomicity and error isolation

- **Atomic writes** everywhere (`io_utils.atomic_write*`): write to a temp file in the same
  directory, `flush` + `fsync`, then `os.replace`. A reader never sees a half-written mask,
  index, or manifest. Parquet/CSV indexes are written to `.tmp_*` then renamed.
- **Per-frame error isolation.** A failing frame is caught, appended to
  `logs/<stage>_errors.jsonl` with its index and error, and the run continues — one bad frame
  never aborts the batch.
- **CUDA OOM guard.** `torch.cuda.OutOfMemoryError` is caught, `torch.cuda.empty_cache()` is
  called, and the frame is retried once.

---

## Overlap resolution (Method 1)

Grounded-SAM2 produces many overlapping box-prompted masks that must be composited into one
canonical id-map. Two steps (`segmentation/overlap.py`):

1. **Class-aware NMS** on boxes (`class_aware_nms`): greedy per-class suppression of boxes with
   IoU above `grounded_sam2.nms_iou` (default 0.7), keeping the highest-scoring box per cluster.
2. **Pixel-level resolution** (`resolve_overlaps`): each surviving mask is an `Instance` with a
   `(priority, combined_score)`. For every pixel, the instance with the **higher priority**
   wins; at **equal priority**, the **higher combined score** wins. Painting is done in a
   deterministic low→high order so runs are reproducible.

The priorities (set per class in `grounded_prompts.yaml`) encode the intended precedence:

- **Thin / safety-critical classes beat large background.** lane_marking, crosswalk, person,
  rider, bicycle, motorcycle, traffic_light, traffic_sign carry high priority so they are not
  overwritten by road_surface, sidewalk, building, sky, or vegetation regardless of area.
- **`windshield` is deliberately low priority** (a transparent backdrop): road/sky/cars/lanes
  seen *through* the glass win, and windshield only fills where nothing else was detected.
- **Per-class `max_area_frac` caps** stop a prompt from swallowing the frame — this is how the
  ego-cabin over-capture bug (a single "car" mask covering 87.8% of the frame) was fixed:
  `car` is capped at 0.30 of the image, `other_cockpit` at 0.45.
- **Unassigned pixels stay `unknown` (id 0)** — a class is never invented from a residual mask.

The combined score itself is `sqrt(gdino_box_score * sam_iou)` (geometric mean, configurable
via `grounded_sam2.score_combine`), so a low-confidence detection cannot overwrite a
high-confidence one at the same priority.

Method 2 needs none of this: it is already a dense argmax over class×mask queries
(`einsum("bqc,bqhw->bchw")`), so its canonical map is produced by a native-id → canonical-id
lookup table (`MapillaryMapper`), keeping the native map untouched alongside.

---

## Parallelization and hardware

- **Single GPU (validated).** One RTX 6000 Ada (48 GB) runs both models at full resolution.
  Frames are processed sequentially per method. `VrsDataProvider` is treated as **not
  thread-safe**, so VRS reads are sequential by design.
- **CPU workers** (`--workers`, `hardware.workers: auto`) are available for decode/rectify/I-O.
- **AMP / perf knobs** (`hardware.*` in `configs/default.yaml`): `amp: bf16` (bf16 on Ada),
  `tf32: true`, `channels_last: true`, `torch_compile: false`, `vram_margin_mb`.
- **Multi-GPU (designed).** The intended scale-out is **disjoint frame sharding**: each GPU
  runs `segment` over a distinct subset of frame indices into the same `SegLayout`, sharing a
  common manifest, and the per-frame outputs are reassembled by `frame_index` (the metadata
  aggregation and every downstream stage sort by `frame_index`, so shard order is irrelevant).
  Because the manifest keys on frame index and writes are atomic, shards do not conflict.

---

## Adding a third segmentation method

The schema makes this local. Implement a segmenter that returns a `base.FrameOutput`
(canonical `uint16` mask, optional confidence, detections, timings), write it through
`base.write_frame_output` into a new `SegLayout(input_dir, "<name>")`, register it in
`segmentation/run.py`, and add its native→canonical mapping. `analyze` and `render` discover
any method whose `canonical_masks/` directory is populated, so no downstream change is needed
for per-method metrics; between-method agreement is currently specific to the two shipped
methods.
