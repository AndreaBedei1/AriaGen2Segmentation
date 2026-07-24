# Troubleshooting

Common failures and how the pipeline already handles or avoids them. Most issues stem from the
three-environment split or from offline loading.

---

## `TypeError: ... bfloat16` when converting model output to numpy

**Symptom.** A crash while post-processing Method 2 output, e.g. numpy refusing a `bfloat16`
tensor.

**Cause.** Inference runs under `torch.autocast` in bf16 (or fp16). NumPy has no bfloat16
dtype, so converting an autocast tensor directly fails.

**Fix (already in `segmentation/oneformer.py`).** All post-processing is done in **float32,
outside the autocast block**: the class and mask query logits are cast with `.float()` before
the `softmax`/`sigmoid`/`einsum` and the final `.cpu().numpy()`. If you add a new
mask-classification head, cast to float32 before leaving the GPU/autocast context the same way.

---

## OneFormer DiNAT-L: `natten` / `detectron2` errors (why env C exists)

**Symptom.** With `oneformer_mapillary.source: hf_oneformer`, import/build errors from
`natten`, or a `TypeError: natten2dqkrpb() missing 1 required positional argument: 'dilation'`,
or a `detectron2` build failure.

**Cause.** The DiNAT-L backbone needs the `natten` custom op, whose wheels are pinned to
**older** torch versions than the primary ML env (env B, torch 2.13). The original OneFormer
repo additionally needs `detectron2`, which has no modern wheels.

**Fix.** Use the **default Method 2 backend** — `source: hf_mask2former`
(`facebook/mask2former-swin-large-mapillary-vistas-semantic`) — which has the **same Mapillary
65-class taxonomy** and **zero custom ops**, so it runs in env B. Only if you specifically need
DiNAT-L, create the optional **env C** with a torch-matched `natten` wheel (no detectron2; the
HF OneFormer implementation is already updated for the new natten API) and point
`ARIA_ONEFORMER_PYTHON` at it. Full install notes and wheel indices are in
[model_selection.md](model_selection.md#natten-specifics-critical-for-the-dinat-path).

---

## CUDA out of memory

**Symptom.** `torch.cuda.OutOfMemoryError` mid-run.

**Handled.** Both runners wrap inference in an OOM guard: the error is caught,
`torch.cuda.empty_cache()` is called, and the frame is retried **once**. The failing frame,
if it still fails, is isolated to `logs/<method>_errors.jsonl` and the run continues.

**If it persists.** Lower `oneformer_mapillary.input_max_size` (default 2048, longest side),
reduce `--batch-size`, keep `hardware.amp: bf16`, or raise `hardware.vram_margin_mb`. At full
2016×1512 both models fit comfortably in 48 GB, so persistent OOM usually means another process
is holding VRAM (`nvidia-smi`).

---

## Offline load failures (model tries to reach the hub)

**Symptom.** A `segment` run hangs or errors trying to contact `huggingface.co`, especially
under `check_offline.py` or with no network.

**Cause.** A model id is a **hub id** rather than a local path, and the HF cache is not
pre-populated.

**Fix.**
1. Ensure the weights exist locally (see [README → Download weights](../README.md#download-weights)).
   The loaders already **prefer a local `weights/…` directory** when one exists — Grounding
   DINO from `weights/grounding-dino-base`, Mask2Former from `weights/mask2former-mapillary-semantic`.
2. Point the config ids at local dirs explicitly if needed:
   `grounded_sam2.grounding_dino_id`, `oneformer_mapillary.mask2former_id` (or
   `oneformer_id`) → a local path.
3. `source scripts/run_offline.sh` first — it sets `HF_HUB_OFFLINE=1` (etc.) and pins `HF_HOME`
   / `TORCH_HOME` to local pre-populated caches, so offline resolution succeeds.
4. Verify with `python scripts/check_offline.py -- <command>`; a printed `connect -> host` tells
   you which artifact still resolves to a remote id.

---

## Grounding DINO is slow

**Symptom.** Method 1 spends most of its per-frame time in the `grounding_dino` timing (it is
the slower method overall, ~1.9 s/frame on the validation subset).

**Cause / design.** Each Grounding DINO call is expensive, and there are ~39 canonical prompts.

**Mitigation (already in `configs/grounded_prompts.yaml`).** `prompt_strategy: grouped` batches
classes sharing a `group_tag` into **one** GDINO call (e.g. all `background` classes together),
instead of one call per class. Difficult classes that need isolation (e.g. `lane_marking`) are
marked `solo: true`. Options are `grouped` (default, fastest safe), `concatenated` (a single
call with every prompt), or `individual` (one call per class, slowest). Raising
`box_threshold`/`text_threshold` also trims spurious low-confidence detections and downstream
SAM work.

---

## VRS reading is slow / just want timestamps

**Symptom.** Enumerating frames or timestamps is slow if you decode every image.

**Fix (already used).** Use `AriaProvider.rgb_timestamps_ns` /
`timestamps_ns`, which call `get_timestamps_ns(stream, DEVICE_TIME)` and return all timestamps
**without decoding any images**. `inspect` and `extract` use this for frame selection and
timestamp-health checks; only frames actually selected are decoded.

---

## `VrsDataProvider is not thread-safe`

**Symptom.** Corrupt reads or crashes if you parallelize VRS access with threads.

**Cause / rule.** `projectaria_tools`' `VrsDataProvider` is treated as **not thread-safe**.

**Fix.** Read the VRS **sequentially** (the pipeline does). For scale-out, shard across
**processes/GPUs by frame index** (each process opens its own provider) rather than threading a
single provider — see
[architecture.md → parallelization](architecture.md#parallelization-and-hardware).

---

## A mask file "looks black" / wrong colors

**Cause.** `canonical_masks/*.png` are **single-channel uint16 id-maps**, not RGB. Opened in a
normal viewer they look near-black because ids are small integers.

**Fix.** Read losslessly (`cv2.IMREAD_UNCHANGED`) and colorize with the shared palette
(`Taxonomy.colorize`) — see [README → Viewing the masks](../README.md#viewing-the-masks). The
`render` stage already produces colored overlay videos and diagnostics. `confidence/*.png` are
uint8 (0–255) confidence, not class ids; Method 2's `native/*.png` are native Mapillary ids.

---

## `analyze` / `segment` find nothing

- **`frames.parquet` missing / `iter_frames` fails** → run `extract` first; the frame index it
  writes is the contract every later stage reads.
- **`analyze` says "no segmentation outputs found"** → run at least one `segment`; `analyze`
  only picks up a method whose `canonical_masks/` contains `frame_*.png`.
- **Gaze skipped ("no eyegaze stream")** → the VRS lacks a gaze stream (or a different label);
  `align-gaze` writes `summary.json` with `"eyegaze": false` and downstream gaze metrics are
  simply omitted. Check the label with `inspect` and, if needed, set `vrs.eyegaze_label`.

---

## `run-all` used the wrong interpreter / a stage "can't import torch" or "can't import projectaria_tools"

**Cause.** `ARIA_VRS_PYTHON` / `ARIA_ML_PYTHON` (/ `ARIA_ONEFORMER_PYTHON`) are unset, so a
stage ran under the current interpreter, which lacks that stack.

**Fix.** Export the three vars to the right venvs before `run-all` (see
[README → Running the pipeline](../README.md#running-the-pipeline)). Unset vars fall back to the
current interpreter by design, which is only correct if that interpreter happens to have the
needed deps.
