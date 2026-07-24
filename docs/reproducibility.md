# Reproducibility

What the pipeline records for provenance, what makes a run reproducible, how the cache is
invalidated, and the exact commands to reproduce a result.

---

## What is logged

### Environment and hardware

`python -m aria_drive_seg hwinfo --output output/reports` writes `hardware_report.json`:

- **Python** version, **platform**, logical/physical cores, RAM, and free disk.
- **torch** view when available: `torch_version`, `cuda_version`, cuDNN, and per-GPU name,
  VRAM, compute capability, multiprocessor count, and `bf16_supported`.
- **`nvidia-smi`** view as a fallback (GPU name, VRAM, compute capability, driver version), so
  the report is still useful in an environment without torch (e.g. the VRS env).

Record `hardware_report.json` from **each** environment you run stages in, so the torch/CUDA
stack of the ML env and the projectaria-tools version of the VRS env are both captured. The
reference machine: RTX 6000 Ada (48 GB, compute 8.9), driver 580 / CUDA 13, Python 3.10.20.

The **library versions** that matter per environment (record them alongside the report):

| Env | Pinned stack |
|---|---|
| A (VRS I/O) | Python 3.10.20, `projectaria-tools` 2.1.2, opencv |
| B (ML) | Python 3.10, torch 2.13.0+cu130, transformers 5.14, official `sam2` |
| C (optional) | older torch + `natten` (for OneFormer DiNAT-L) |

### Weights provenance

`weights/manifest.json` (written by `scripts/download_weights.py`) records, per artifact:
source **URL or HF repo id**, version/revision, byte size, and **SHA-256**
(`main_weight_sha256` for HF repos). This pins exactly which checkpoint produced a result and
lets a later run **verify integrity** (`download_weights.py --verify`) before trusting a cache.
Example (test machine): SAM 2.1 Hiera-Large `sha256 2647878d…dd318`, version `092824`.

### Config fingerprints (per stage)

Each resumable stage stores its cache key in `<stage>/manifest.json` as `fingerprint` — a
stable hash (`hashing.stable_hash` → sorted-key canonical JSON → SHA-256, truncated) of exactly
the inputs that must invalidate that stage:

| Stage | Fingerprint inputs |
|---|---|
| `extract` | rectify params, JPEG quality, rectify-enabled, `frames` selection, VRS path |
| `grounded_sam2` | full `grounded_sam2` config block, full text of `grounded_prompts.yaml`, `rectify` block |
| `oneformer_mapillary` | full `oneformer_mapillary` config block, full text of `mapillary_to_canonical.yaml`, `rectify` block |

The manifest also records which model/checkpoint was used (e.g. the resolved Mask2Former model
path) and per-frame stats (coverage, timing). Method summaries (`summary.json`) capture frame
counts, error counts, peak VRAM (Method 1), and unmapped native labels (Method 2).

### Seeds and determinism

`configs/default.yaml` exposes `project.seed: 1234` and `project.deterministic: true`, and
`hardware`: `amp: bf16`, `tf32: true`. In practice the inference is deterministic **by
construction** rather than by RNG seeding:

- Both methods run under `torch.inference_mode()` with **no stochastic sampling** — Method 2
  decodes by argmax over class×mask queries; Method 1 uses fixed thresholds, greedy NMS, and
  argmax box→mask selection.
- **Overlap compositing paints in a deterministic order** (`resolve_overlaps` sorts by
  `(priority, score)`), so the result does not depend on detection order.
- All writes are atomic, so an interrupted run cannot leave a partially-written mask that would
  differ on resume.

> Caveat: `tf32: true` and bf16 autocast allow tiny numerical differences in the logits across
> different GPUs/driver builds. Because decoding is argmax-based, this is virtually always
> invisible in the class ids, but pin the GPU/driver (via `hardware_report.json`) if you need
> bit-exact confidence maps. Set `hardware.tf32: false` / `hardware.amp: fp32` for the strictest
> numerical reproducibility at a speed cost.

---

## How cache invalidation works

A stage recomputes a frame only when needed:

1. On start, the stage computes its fingerprint (above).
2. `Manifest.load_or_new` loads the existing `manifest.json` **only if** its stored fingerprint
   and stage name match; otherwise it starts fresh — a changed fingerprint transparently
   invalidates the whole stage's cache.
3. A frame is skipped only if it is marked done **and** its output files exist on disk.
4. `--force` clears the done-set; `--no-resume` disables skipping entirely.

So editing the **checkpoint, prompts, thresholds, taxonomy mapping, calibration, or
resolution** invalidates exactly the affected stage, while unrelated changes (render colors,
analysis) do not force resegmentation. See
[architecture.md](architecture.md#resumability-caching-and-the-config-fingerprint).

To force a clean, fully-recomputed run:

```bash
python -m aria_drive_seg run-all --vrs recording.vrs --output output --force --no-resume
```

---

## Exact commands to reproduce a result

```bash
# 0) capture provenance from each environment
~/aria_seg_ml_env/bin/python           -m aria_drive_seg hwinfo --output output/reports
~/projectaria_gen2_python_env/bin/python -m aria_drive_seg hwinfo --output output/reports_vrs

# 1) verify weights integrity against the manifest (no re-download)
python scripts/download_weights.py --verify

# 2) run fully offline, deterministic config, from a known VRS
source scripts/run_offline.sh
export ARIA_VRS_PYTHON=~/projectaria_gen2_python_env/bin/python
export ARIA_ML_PYTHON=~/aria_seg_ml_env/bin/python
python -m aria_drive_seg run-all \
    --vrs unknown_20260528_090955.vrs \
    --output output \
    --methods grounded_sam2,oneformer_mapillary

# 3) (optional) prove the run made no external connections
python scripts/check_offline.py -- \
    ~/aria_seg_ml_env/bin/python -m aria_drive_seg segment --method oneformer_mapillary --input output
```

To reproduce a **specific stage** with an alternative config while keeping the rest cached,
pass `--config extra.yaml` (deep-merged over `configs/default.yaml`); only the stages whose
fingerprint changes will recompute.

The artifacts that together pin a result: the input `.vrs`, `weights/manifest.json` (checkpoint
SHA-256s), the `configs/*.yaml` in effect, each stage's `manifest.json` (`fingerprint`), and the
`hardware_report.json`(s).

---

## Environment pinning strategy

- **Three isolated venvs**, one per dependency stack (see
  [architecture.md](architecture.md#the-three-environment-design)). Each pins its own torch /
  transformers / projectaria-tools / natten versions; none contaminates another.
- **`aria_drive_seg` installed editable** (`pip install -e .`) in each, with deliberately light
  *core* deps (numpy, pyyaml, pyarrow, pandas, pillow) so the package imports without torch in
  the VRS env; heavy deps (torch, transformers, sam2, natten) are per-env.
- **`run-all` binds stages to interpreters** via `ARIA_VRS_PYTHON` / `ARIA_ML_PYTHON` /
  `ARIA_ONEFORMER_PYTHON` — no absolute interpreter paths live in code, so the same repo is
  portable across machines by re-exporting those vars.
- **Weights are content-addressed** by SHA-256 in the manifest, and **configs are
  fingerprinted** into each stage's cache — the two provenance anchors that make a rerun
  verifiably the same computation.
