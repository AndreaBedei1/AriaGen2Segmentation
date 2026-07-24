# aria_drive_seg

Fully-local, dual-method semantic segmentation + eye-gaze analysis for **Meta Project Aria Gen2** driving recordings.

The pipeline reads an Aria Gen2 `.vrs`, rectifies the fisheye RGB to a linear pinhole,
runs **two independent segmentation methods**, aligns the on-device eye-gaze to each frame,
and produces per-frame class masks, gaze-to-class labels, comparison metrics, and overlay
videos. Everything runs **offline** after a one-time weights download — there are no cloud
inference calls and no telemetry (see [docs/privacy.md](docs/privacy.md)).

The two methods are complementary:

| Method | Model(s) | Nature | Taxonomy |
|---|---|---|---|
| **Method 1 — Grounded-SAM2** | Grounding DINO (base) + SAM 2.1 Hiera-Large | Open-vocabulary, **prompted**, sparse (not an exhaustive partition) | canonical prompts incl. cockpit |
| **Method 2 — Mask2Former / OneFormer (Mapillary)** | Mask2Former Swin-L Mapillary (default) or OneFormer DiNAT-L (optional) | Dense semantic partition | Mapillary Vistas v1.2 (65 classes) → canonical |

Model choice and the survey behind it: [docs/model_selection.md](docs/model_selection.md).

---

## Contents

- [Hardware requirements](#hardware-requirements)
- [Installation (three environments)](#installation-three-environments)
- [Download weights](#download-weights)
- [Offline execution](#offline-execution)
- [Running the pipeline](#running-the-pipeline)
- [Output structure](#output-structure)
- [Resuming and caching](#resuming-and-caching)
- [Changing the classes / prompts](#changing-the-classes--prompts)
- [Viewing the masks](#viewing-the-masks)
- [Interpreting the metrics](#interpreting-the-metrics)
- [Declared limitations](#declared-limitations)
- [Further documentation](#further-documentation)

---

## Hardware requirements

Reference machine (the one this pipeline was validated on):

| Component | Value |
|---|---|
| GPU | 1× NVIDIA RTX 6000 Ada Generation, 48 GB VRAM |
| Compute capability | 8.9 (Ada Lovelace), bf16 supported |
| Driver / CUDA | driver 580 / CUDA 13 |
| CPU / RAM | 30 logical cores / ~64 GB |
| Disk | hundreds of GB free (frames, masks, and videos are large) |

A single GPU is sufficient. Both models fit comfortably in 48 GB at full resolution;
CUDA out-of-memory is caught and retried after `torch.cuda.empty_cache()`. Multi-GPU is
possible via disjoint frame sharding (see [docs/architecture.md](docs/architecture.md)).

Generate a machine report at any time:

```bash
python -m aria_drive_seg hwinfo --output output/reports
# -> output/reports/hardware_report.json (GPU, driver, CUDA, torch, CPU/RAM/disk)
```

`hwinfo` prefers torch's GPU view and falls back to `nvidia-smi`, so it works even in an
environment without torch installed.

---

## Installation (three environments)

Dependencies are **split across three Python 3.10 environments** for isolation; the
`aria_drive_seg` package is installed editable in each. `run-all` selects the right
interpreter per stage from environment variables (below), so no absolute paths live in code.

| Env | Purpose | Key deps | Used by stages | Interpreter var |
|---|---|---|---|---|
| **A** `~/projectaria_gen2_python_env` | VRS I/O | Python 3.10.20, `projectaria-tools` 2.1.2, opencv | inspect, extract, align-gaze, analyze, render | `ARIA_VRS_PYTHON` |
| **B** `~/aria_seg_ml_env` | ML inference | torch 2.13.0+cu130, transformers 5.14, official `sam2` | segment (grounded_sam2 **and** oneformer_mapillary/Mask2Former) | `ARIA_ML_PYTHON` |
| **C** *(optional)* | OneFormer DiNAT-L | older torch + `natten` | segment (oneformer_mapillary with `source: hf_oneformer`) | `ARIA_ONEFORMER_PYTHON` |

Env C is only needed if you switch Method 2 to the DiNAT-L OneFormer backbone (which needs
the `natten` custom op). The default Method 2 backend (Mask2Former) has **zero custom ops**
and runs in env B.

### Env A — VRS I/O

```bash
uv venv --python 3.10 ~/projectaria_gen2_python_env
source ~/projectaria_gen2_python_env/bin/activate
pip install "projectaria-tools==2.1.2"
pip install -e ".[io]"      # core deps + opencv-python-headless, matplotlib, imageio-ffmpeg
deactivate
```

### Env B — ML inference (Grounded-SAM2 + Mask2Former-Mapillary)

```bash
uv venv --python 3.10 ~/aria_seg_ml_env
source ~/aria_seg_ml_env/bin/activate
# torch matched to your CUDA (reference stack: torch 2.13.0+cu130)
pip install torch torchvision
pip install "transformers" accelerate timm huggingface_hub
# official SAM 2 (no CUDA extension build needed for image inference)
SAM2_BUILD_CUDA=0 pip install "git+https://github.com/facebookresearch/sam2.git"
pip install -e ".[io]"
deactivate
```

### Env C — optional OneFormer DiNAT-L

Only if you set `oneformer_mapillary.source: hf_oneformer`. Follow the natten install notes in
[docs/model_selection.md](docs/model_selection.md) (a torch-matched `natten` wheel), then
`pip install -e ".[io]"` into that env. Point `ARIA_ONEFORMER_PYTHON` at it.

---

## Download weights

Fetch the official checkpoints **once**; every later run is offline. Each artifact is recorded
in `weights/manifest.json` with its source URL/repo and SHA-256 for integrity + provenance.

```bash
# in env B (needs huggingface_hub)
python scripts/download_weights.py --methods grounded_sam2
```

This downloads and records:

| Artifact | Source | Size | License |
|---|---|---|---|
| SAM 2.1 Hiera-Large (`sam2.1_hiera_large.pt`) | `dl.fbaipublicfiles.com/segment_anything_2/092824/` | ~898 MB | Apache-2.0 |
| Grounding DINO base | HF `IDEA-Research/grounding-dino-base` | ~930 MB | Apache-2.0 |

**Method 2 weights** (Mask2Former Swin-L Mapillary, `facebook/mask2former-swin-large-mapillary-vistas-semantic`,
~866 MB) are loaded locally from `weights/mask2former-mapillary-semantic/`. Fetch them once with
`huggingface_hub` into that directory, for example:

```bash
python -c "from huggingface_hub import snapshot_download; \
snapshot_download('facebook/mask2former-swin-large-mapillary-vistas-semantic', \
local_dir='weights/mask2former-mapillary-semantic')"
```

> Note: `download_weights.py` currently fetches only the Grounded-SAM2 artifacts (SAM 2.1 +
> Grounding DINO); the Method 2 checkpoint is obtained separately as shown above. The loader
> (`oneformer.py`) prefers the local `weights/…` directory whenever it exists, so once the
> files are present all inference is offline. Mapillary-trained weights inherit Mapillary
> Vistas' research/non-commercial terms — see the licensing caveat in
> [docs/model_selection.md](docs/model_selection.md).

The SAM 2.1 model **config** referenced in `configs/default.yaml`
(`configs/sam2.1/sam2.1_hiera_l.yaml`) is a config *name* resolved by the installed `sam2`
package (its bundled Hydra config search path), not a file inside this repo.

---

## Offline execution

After the one-time download, force fully-offline inference by sourcing the helper (it exports
the Hugging Face offline switches and disables assorted telemetry):

```bash
source scripts/run_offline.sh
# ... then run any pipeline command in this shell ...

# or prefix a single command:
scripts/run_offline.sh python -m aria_drive_seg segment --method grounded_sam2 --input output
```

Every CLI subcommand also accepts `--offline`, which sets `HF_HUB_OFFLINE`,
`TRANSFORMERS_OFFLINE`, and `HF_DATASETS_OFFLINE` for that process.

**Verify** that a run makes zero non-loopback connections, using the strict socket netguard:

```bash
python scripts/check_offline.py -- python -m aria_drive_seg segment --method grounded_sam2 --input output
# exit 0 + "no non-loopback connection attempts detected"  => offline-clean
```

Details in [docs/privacy.md](docs/privacy.md).

---

## Running the pipeline

The pipeline is a sequence of stages. Two conventions matter:

- **`inspect` / `extract` / `run-all`** take `--output` (the run directory to create).
- **`segment` / `align-gaze` / `analyze` / `render`** take `--input` (that same run directory).

So you `extract` into a directory, then point the later stages at it.

### Option A — one command (recommended): `run-all`

`run-all` orchestrates every stage and dispatches each to the correct interpreter:

```bash
export ARIA_VRS_PYTHON=~/projectaria_gen2_python_env/bin/python
export ARIA_ML_PYTHON=~/aria_seg_ml_env/bin/python
# optional, only for the DiNAT-L OneFormer backend:
export ARIA_ONEFORMER_PYTHON=~/aria_oneformer_env/bin/python

python -m aria_drive_seg run-all \
    --vrs unknown_20260528_090955.vrs \
    --output output \
    --methods grounded_sam2,oneformer_mapillary
```

Any unset `ARIA_*_PYTHON` var falls back to the current interpreter. Because every stage is
resumable, a failed `run-all` can simply be re-invoked and it skips completed work.

### Option B — stage by stage

Run each command with the interpreter for its environment (VRS-I/O stages in env A,
`segment` in env B):

```bash
# env A
python -m aria_drive_seg inspect    --vrs unknown_20260528_090955.vrs --output output/inspection
python -m aria_drive_seg extract    --vrs unknown_20260528_090955.vrs --output output
# env B
python -m aria_drive_seg segment    --method grounded_sam2       --input output
python -m aria_drive_seg segment    --method oneformer_mapillary --input output
# env A
python -m aria_drive_seg align-gaze --vrs unknown_20260528_090955.vrs --input output
python -m aria_drive_seg analyze    --input output
python -m aria_drive_seg render     --input output
```

### Common flags

All subcommands accept:

| Flag | Meaning |
|---|---|
| `--start-time` / `--end-time` | frame-selection window, seconds from recording start |
| `--frame-step` | keep every Nth RGB frame |
| `--max-frames` | cap the number of frames |
| `--devices` | `auto` \| `"0"` \| `"0,1"` |
| `--workers` | CPU workers for decode/rectify/I-O |
| `--batch-size` | inference batch size |
| `--offline` | force HF/transformers offline |
| `--resume` / `--no-resume` | resume from cache (default on) / recompute |
| `--force` | ignore cache, recompute everything |
| `--config` | extra YAML merged over `configs/default.yaml` |
| `--log-level` | e.g. `INFO`, `DEBUG` |

Stage-specific: `inspect --samples N`, `segment --method …`, `analyze --method …`,
`render --what all`, `run-all --methods …`.

To process a short clip, for example the first 30 s at 2 Hz:

```bash
python -m aria_drive_seg extract --vrs recording.vrs --output out --end-time 30 --frame-step 5
```

---

## Output structure

Every stage writes into one run directory:

```
output/
├── inspection/
│   ├── inspection_report.json         # streams, RGB/gaze health, calibration
│   └── samples/                       # raw + rectified sample JPEGs
├── frames/
│   ├── original/frame_XXXXXX.jpg      # native-geometry RGB
│   ├── rectified/frame_XXXXXX.jpg     # pinhole-rectified RGB (fed to segmentation)
│   ├── frames.parquet | frames.csv    # frame index (timestamps, paths, sizes)
│   ├── calibration.json               # source + rectified calibration used
│   └── manifest.json                  # resumability + config fingerprint
├── gaze/
│   ├── raw_gaze.parquet               # every on-device gaze sample
│   ├── aligned_gaze.parquet           # gaze aligned + projected per frame
│   └── summary.json
├── grounded_sam2/
│   ├── canonical_masks/frame_XXXXXX.png   # uint16 canonical id-map (lossless)
│   ├── confidence/frame_XXXXXX.png        # uint8 per-pixel confidence
│   ├── metadata/frame_XXXXXX.json         # detections, scores, timings
│   ├── metadata.parquet | manifest.json | summary.json
├── oneformer_mapillary/
│   ├── native/frame_XXXXXX.png            # native Mapillary id-map (never deleted)
│   ├── canonical_masks/ confidence/ metadata/ …   # same schema as above
├── comparison/
│   ├── gaze/*.parquet                 # per-method gaze labels + method agreement
│   ├── metrics/metrics_no_gt.json     # all no-ground-truth metrics
│   └── figures/diag_frame_XXXXXX.png  # side-by-side diagnostics
├── videos/*.mp4                       # overlays, comparison, gaze-agreement
├── logs/                              # *_errors.jsonl, stage summaries
└── reports/analysis_summary.md        # human-readable metric summary
```

Both segmentation methods share the **same on-disk schema**, so `analyze` and `render` are
method-agnostic. This schema is also the interface between environments (see
[docs/architecture.md](docs/architecture.md)).

---

## Resuming and caching

`extract` and both `segment` stages are **resumable**. Each keeps a `manifest.json` recording:

- a **config fingerprint** — a stable hash of everything that must invalidate the cache, and
- the set of already-completed frames.

On re-run, completed frames whose output files exist on disk are skipped. If the fingerprint
changes — because you edited the **checkpoint, prompts, thresholds, taxonomy, calibration, or
resolution** — the affected stage's cache is treated as stale and recomputed. Force a full
recompute with `--force`; disable resume with `--no-resume`. All writes are atomic (temp file
+ fsync + rename), so an interrupted run never leaves a half-written mask or index.

`align-gaze`, `analyze`, and `render` recompute from the current masks each time they run.

---

## Changing the classes / prompts

The canonical taxonomy is the single source of truth; both methods map their native labels
onto it. To change what gets segmented, edit the YAML in `configs/` and re-run the affected
stage (the fingerprint change triggers recomputation):

| To change… | Edit | Re-run |
|---|---|---|
| Canonical classes, ids, colors, eval flags | `configs/classes.yaml` | both `segment`, then `analyze`/`render` |
| Grounding DINO prompts, thresholds, priorities, area caps | `configs/grounded_prompts.yaml` | `segment --method grounded_sam2` |
| Mapillary → canonical mapping | `configs/mapillary_to_canonical.yaml` | `segment --method oneformer_mapillary` |
| Rectification, gaze, hardware, model defaults | `configs/default.yaml` (or `--config extra.yaml`) | the affected stage |

Full explanation of the taxonomy, palette, and mappings: [docs/class_taxonomy.md](docs/class_taxonomy.md).

---

## Viewing the masks

`canonical_masks/frame_XXXXXX.png` are **single-channel uint16 id-maps**, not colored images:
each pixel value is a canonical class id (0 = `unknown`, black). Read them **losslessly**, then
colorize with the deterministic palette from `configs/classes.yaml`:

```python
import cv2
from aria_drive_seg.taxonomy import Taxonomy

tax  = Taxonomy.load("configs/classes.yaml")
ids  = cv2.imread("output/grounded_sam2/canonical_masks/frame_000000.png", cv2.IMREAD_UNCHANGED)
rgb  = tax.colorize(ids)          # (H, W, 3) uint8, palette is shared by both methods
```

Because the palette is shared, a class is drawn with the same color in both methods' overlays,
making them directly comparable. The `render` stage produces ready-made overlay and comparison
videos in `videos/` and high-res diagnostics in `comparison/figures/`.

Method 2 also keeps the untouched **native** Mapillary id-map in `native/`; the canonical mask
is an additional view, never a replacement.

---

## Interpreting the metrics

`analyze` writes `comparison/metrics/metrics_no_gt.json` and a readable
`reports/analysis_summary.md`. **There is no ground truth**, so these are consistency and
descriptive statistics, **not accuracy**:

- **Per-method:** coverage (fraction of assigned pixels), unknown fraction, classes/frame,
  ms/frame and fps.
- **Between methods:** pixelwise agreement on the common eval taxonomy, mean IoU between
  methods, per-class IoU between methods.
- **Gaze:** per-method time-on-class, gaze-on-unknown fraction, temporal stability, and
  between-method gaze-class agreement/disagreement.

Example numbers from the **10-frame validation subset (`val10`)** — illustrative only, not the
full recording, and **not accuracy**:

| Metric | Grounded-SAM2 | Mask2Former-Mapillary |
|---|--:|--:|
| Speed | ~1.9 s/frame (0.53 fps) | ~0.16 s/frame (6.4 fps) |
| Coverage | ~34.5% (sparse, prompted) | ~99.6% (dense partition) |
| Unknown | ~65.5% | ~0.4% |
| Classes/frame | ~11.3 | ~17.3 |

Between-method (common eval classes): pixelwise agreement **47.7%**, mean IoU **0.08**,
gaze-class agreement **33% / disagreement 67%**.

For **accuracy**, a ground-truth workflow is implemented (it just needs labels):
`scripts/prepare_annotation.py` builds a stratified, CVAT-compatible annotation package, and
`scripts/evaluate_gt.py` scores mIoU, per-class IoU, pixel/mean-class accuracy, boundary F1,
accuracy-on-assigned, and gaze-class accuracy — refusing to fabricate numbers when no labels
exist. No ground truth has been created yet, so **no accuracy is claimed**. See
[docs/evaluation.md](docs/evaluation.md) for every metric.

---

## Declared limitations

These are inherent to the methods and must be kept in mind when reading any output:

- **Grounded-SAM2 is not an exhaustive partition.** It segments only what the text prompts
  detect; unprompted or undetected pixels remain `unknown` (id 0). Coverage is expected to be
  well below 100% (~34.5% on `val10`).
- **Mask2Former / OneFormer cannot recognize classes outside its taxonomy.** It is a dense
  classifier over Mapillary Vistas' 65 classes; anything with no equivalent is mapped to
  `unknown`. It cannot invent cockpit-interior classes.
- **In-cabin viewpoint is a domain shift.** The camera sits inside the vehicle looking out
  through the windshield, whereas Mapillary/road datasets are street-level exterior views.
  Expect degraded reliability on cabin structure and through-glass content.
- **Model agreement is not accuracy.** All between-method agreement/IoU numbers measure
  *consistency* on shared classes only; two models can agree and both be wrong. Real accuracy
  requires human-labeled ground truth (not yet produced — see
  [docs/evaluation.md](docs/evaluation.md)).
- **A single gaze point does not capture eye-tracker uncertainty.** Gaze is reduced to one
  projected pixel per frame; it ignores calibration error, angular spread, and depth
  ambiguity. Disc- and gaussian-weighted queries mitigate but do not remove this. Depth uses
  the device estimate when valid, else a configurable fallback (default 8 m), always flagged
  as an approximation.

---

## Further documentation

| Doc | Contents |
|---|---|
| [docs/architecture.md](docs/architecture.md) | Module map, data flow, 3-env design, caching, overlap resolution, parallelism |
| [docs/privacy.md](docs/privacy.md) | Local-only guarantee, netguard mechanism, no telemetry |
| [docs/vrs_and_calibration.md](docs/vrs_and_calibration.md) | Streams, fisheye624 rectification, gaze alignment + projection |
| [docs/class_taxonomy.md](docs/class_taxonomy.md) | Canonical groups, palette, per-method mappings |
| [docs/evaluation.md](docs/evaluation.md) | No-GT metrics (with `val10` example numbers) and the with-GT plan |
| [docs/troubleshooting.md](docs/troubleshooting.md) | Common failures and fixes |
| [docs/reproducibility.md](docs/reproducibility.md) | What is logged, cache invalidation, exact repro commands |
| [docs/model_selection.md](docs/model_selection.md) | Model survey and the Method 2 choice (pre-existing) |
