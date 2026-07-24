# Privacy and offline guarantee

This pipeline is designed to process sensitive first-person driving recordings — including
**on-device eye-gaze** — with **no data ever leaving the machine**. This document states the
guarantee and describes exactly how it is enforced and verified.

---

## The guarantee

- **All processing is local.** Reading the `.vrs`, rectification, both segmentation methods,
  gaze alignment, analysis, and rendering run entirely on the local GPU/CPU.
- **No cloud inference.** There are no calls to hosted model APIs (no MPS cloud, no hosted
  segmentation/detection endpoints). Every model runs from local weights.
- **Network access happens exactly once**, during the one-time weights download, and nowhere
  else in the pipeline.
- **No telemetry.** Hugging Face / framework telemetry and third-party analytics are disabled.
- **Nothing is uploaded.** The recording, extracted frames, masks, gaze, and metrics stay on
  disk in the run directory.

---

## What the one-time download fetches (`scripts/download_weights.py`)

This is the **only** stage permitted to touch the network. It downloads the official
checkpoints and records each in `weights/manifest.json` with source URL/repo, size, and
**SHA-256**, so later runs can verify integrity and run fully offline. It is idempotent
(re-running skips artifacts already present with a matching SHA-256) and it **uploads nothing**.

| Artifact | Source | Recorded |
|---|---|---|
| SAM 2.1 Hiera-Large | `dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt` | URL, version, bytes, SHA-256 |
| Grounding DINO base | HF `IDEA-Research/grounding-dino-base` | repo id, per-file bytes, main-weight SHA-256 |

The Method 2 checkpoint (Mask2Former Swin-L Mapillary) is placed under
`weights/mask2former-mapillary-semantic/` and loaded locally (see the
[README](../README.md#download-weights)). After weights are present, **no stage needs the
network again.**

---

## How offline mode is enforced

### `scripts/run_offline.sh`

Source it (or prefix a command with it) to export the offline switches for the whole shell.
After the download, nothing here should touch the network. It sets, among others:

| Variable | Effect |
|---|---|
| `HF_HUB_OFFLINE=1`, `TRANSFORMERS_OFFLINE=1`, `HF_DATASETS_OFFLINE=1` | Hugging Face never contacts the hub |
| `HF_HUB_DISABLE_TELEMETRY=1`, `HF_HUB_DISABLE_IMPLICIT_TOKEN=1` | no HF telemetry / implicit tokens |
| `DISABLE_TELEMETRY=1`, `DO_NOT_TRACK=1`, `NO_ALBUMENTATIONS_UPDATE=1`, `YOLO_OFFLINE=1`, `ULTRALYTICS_OFFLINE=1` | disable assorted third-party telemetry/update checks |
| `TOKENIZERS_PARALLELISM=false` | quiet, deterministic tokenizers |
| `TORCH_HOME=$(pwd)/weights/torch_home`, `HF_HOME=$(pwd)/weights/hf_home` | pin caches to local, pre-populated dirs so offline loads resolve |

```bash
source scripts/run_offline.sh
scripts/run_offline.sh python -m aria_drive_seg segment --method grounded_sam2 --input output
```

The CLI's `--offline` flag independently sets `HF_HUB_OFFLINE`, `TRANSFORMERS_OFFLINE`, and
`HF_DATASETS_OFFLINE` (and disables HF telemetry) for that single process. Model loaders also
**prefer local `weights/…` directories** whenever they exist, so inference resolves without the
hub regardless.

### `aria_drive_seg/netguard.py` — the socket guard

Environment variables ask libraries to stay offline; the netguard **enforces** it. It
monkeypatches `socket.socket.connect`, `connect_ex`, and `socket.create_connection` to
**block and record** any outbound connection to a non-loopback address:

- Loopback / link-local / unix sockets are allowed (local IPC: CUDA, X11, torch workers).
- A numeric non-loopback IP, or any DNS hostname, is treated as **remote** → recorded, and in
  `strict=True` mode the connection is refused (`OSError` / `ECONNREFUSED`).
- Violations are retrievable with `get_violations()`; installing the guard also sets the HF
  offline env switches as a belt-and-braces measure.

### `scripts/check_offline.py` — proving a run is clean

Runs any command under the netguard and reports every outbound-connection attempt. The guard
is injected via a `sitecustomize.py` on `PYTHONPATH`, so it is active **from interpreter
start-up, before `transformers`/`torch` import**:

```bash
python scripts/check_offline.py -- python -m aria_drive_seg segment --method grounded_sam2 --input output
```

- **Exit 0** and `✓ no non-loopback connection attempts detected` → the run made no external
  connections.
- **Non-zero** → at least one attempt was recorded; the offending `kind -> host` is printed.

Both segmentation methods have been verified to run under `check_offline.py` with
`HF_HUB_OFFLINE=1` and produce **zero non-loopback connection attempts**.

---

## Eye-gaze and personal data

- The **on-device eye-gaze** is read locally from the `373-1` stream in the VRS via
  `projectaria_tools`; it is never transmitted. Raw and per-frame aligned gaze are written only
  to the local run directory (`gaze/raw_gaze.parquet`, `gaze/aligned_gaze.parquet`).
- Extracted frames, masks, overlays, and gaze remain in the run directory under your control.
  Overlay videos in `videos/` visualize gaze on the driving scene and may contain identifying
  content; treat them as sensitive and share deliberately.
- The recording can contain bystanders, license plates, and location-correlated signals (the
  VRS also carries GPS and other sensors). Nothing in this pipeline anonymizes them —
  downstream handling and redaction are the operator's responsibility.

---

## Summary

| Concern | Status |
|---|---|
| Cloud inference APIs | none |
| Network during inference | none (enforced by netguard, verifiable via `check_offline.py`) |
| Telemetry / analytics | disabled by `run_offline.sh` and `--offline` |
| Weights | downloaded once, pinned by SHA-256 in `weights/manifest.json` |
| Eye-gaze / frames / masks | local files only, never uploaded |
