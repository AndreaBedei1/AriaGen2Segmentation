# Model Selection — Road-Scene Semantic/Panoptic Segmentation (fully local)

**Project:** Aria Gen2 driving-footage semantic segmentation pipeline
**Target hardware:** 1× NVIDIA RTX 6000 Ada (48 GB, compute capability **8.9 / Ada Lovelace**), driver 580 / CUDA 13 capable (CUDA 12.x wheels run fine).
**Constraint:** everything runs **offline** after a one-time weights download. **No cloud inference APIs.**
**Date of survey:** 2026-07-24.

This document compares public, locally-runnable checkpoints for road-scene segmentation and confirms the default model choice. All performance numbers are quoted from official repos/papers with inline links; do not trust a single blog.

---

## TL;DR recommendation

**Keep the default: OneFormer with a DiNAT-L backbone trained on Mapillary Vistas** — it has the richest public road taxonomy (Mapillary Vistas v1.2, **65 classes**: lane markings, crosswalk, curb, guardrail, barrier, rider, etc.), does semantic **and** panoptic, and has an official checkpoint (mIoU 64.0 s.s. / 64.9 ms). No newer model beats it on the decisive criterion — *equivalent-or-richer road taxonomy with a public checkpoint that runs fully locally and reproducibly on this GPU*. Newer/stronger models (EoMT, InternImage, ViT-Adapter) only ship **Cityscapes 19-class** semantic checkpoints (narrower taxonomy) and/or need custom-op compilation.

**Practical caveat that shapes the install:** the OneFormer *original* repo pulls in **detectron2** (fragile on modern torch/CUDA) and a **natten** version whose API no longer matches OneFormer's DiNAT code. Two robust ways to run the exact same Mapillary weights locally:

- **Path A (recommended primary):** convert the DiNAT-L Mapillary `.pth` to Hugging Face format and run via `transformers` `OneFormerForUniversalSegmentation` + a modern `natten` wheel — **no detectron2**.
- **Path B (zero-friction fallback, identical taxonomy):** use the officially-hosted **`facebook/mask2former-swin-large-mapillary-vistas-semantic`** — same Mapillary v1.2 **65-class** taxonomy, native in `transformers`, **no custom ops at all** (mIoU 63.2 s.s. / 64.7 ms). Use this if the DiNAT/natten build is not worth the time.

Both OneFormer-Mapillary and Mask2Former-Mapillary weights inherit the **Mapillary Vistas research/non-commercial license** — see [Licensing caveat](#licensing-caveat).

> ### ⚠️ Verified reproducibility finding (this exact hardware, 2026-07-24)
>
> **Path A (OneFormer DiNAT-L via HF `transformers`) does NOT work out-of-the-box on this stack, and the delivered Method 2 is Path B (Mask2Former-Mapillary).** This was tested empirically, not assumed:
>
> - A dedicated env C was built: `~/aria_oneformer_env` with **torch 2.8.0+cu126** and **natten 0.21.1** (the *only* natten release with a prebuilt cp310 wheel for Ada / torch-2.x — from `https://whl.natten.org`).
> - `transformers` 5.14 `models/dinat/modeling_dinat.py` still imports the **old natten ≤0.14 API**: `from natten.functional import natten2dav, natten2dqkrpb`, and calls `natten2dqkrpb(q, k, rpb, kernel, dilation)` + `natten2dav(probs, v, kernel, dilation)`.
> - natten **0.21** exposes only the **fused** op `na2d(q, k, v, kernel, dilation, …)` (no separate `na2d_qk`/`na2d_av`, no `natten2d*`). natten **0.17** exposes `na2d_qk`/`na2d_av` — matching *neither* what transformers imports. **No single natten build has both the API `transformers` DiNAT expects AND an Ada/torch-2.x wheel.** Result: `ImportError: cannot import name 'natten2dav' from 'natten.functional'`.
> - The original SHI-Labs repo path needs **detectron2** (no numpy-2, build breakage on CUDA 13), so it is not viable here either.
>
> **To actually run OneFormer-DiNAT-L-Mapillary you must do ONE of:** (a) **port** `transformers`' DiNAT `NeighborhoodAttention.forward` to natten 0.21's fused `na2d` (correctness-critical — the relative-positional-bias `rpb` must be folded in correctly) and then convert the `.pth`; or (b) build the **legacy** detectron2 + natten-0.14 stack inside a **CUDA-11.x container** with an older torch. Both are out-of-scope of a single reproducible install.
>
> **Decision:** ship **Mask2Former Swin-L Mapillary** as Method 2 — it is native in `transformers`, needs **zero custom ops**, and produces the **identical Mapillary v1.2 65-class taxonomy** (so the canonical mapping, the gaze study, and the method-vs-method comparison are unaffected; only ~0.8 mIoU is left on the table: 63.2 vs 64.0 single-scale). The pipeline keeps a `source: hf_oneformer` switch + `oneformer_id` slot so a correctly-converted OneFormer checkpoint drops straight in.
>
> The paragraphs below (Path A / "How to obtain") describe the *intended* OneFormer route and remain useful once the natten port is done, but the claim "HF transformers DiNAT is already updated for the new natten API" is **empirically false for transformers 5.14** on this box.

---

## Candidate comparison

### Summary table

| Model / backbone | Train set (taxonomy) | Reported mIoU (dataset) | Checkpoint (public) | Custom op that must compile | HF-native | Road taxonomy | Verdict |
|---|---|---|---|---|---|---|---|
| **OneFormer DiNAT-L** | Mapillary Vistas v1.2 (**65 cls**) | **64.0** s.s / 64.9 ms (Mapillary val) | shi-labs.com `.pth` | natten (see notes) | via conversion | **Richest** (lane/crosswalk/curb/guardrail…) + panoptic | **DEFAULT** |
| OneFormer Swin-L | Mapillary Vistas v1.2 (65 cls) | 62.9 s.s / 64.1 ms (Mapillary val) | shi-labs.com `.pth` | none (Swin) | via conversion | Same 65-cls | Alt (no natten) |
| **Mask2Former Swin-L** | Mapillary Vistas v1.2 (**65 cls**) | 63.2 s.s / 64.7 ms (Mapillary val) | `facebook/…mapillary-vistas-semantic` | **none** | **yes** | Same 65-cls | **Fallback / easiest** |
| Mask2Former Swin-L | Cityscapes (19 cls) | 83.3 s.s / 84.3 ms (Cityscapes val) | `facebook/…cityscapes-semantic` | none | yes | Narrow (19) | Cityscapes only |
| InternImage-H + Mask2Former | Mapillary-pretrain → Cityscapes (19 cls out) | **87.0** ms val / 86.1 test (Cityscapes) | OpenGVLab release | **DCNv3** | no | Narrow (19 out) | SOTA but heavy |
| ViT-Adapter (BEiT-L) + Mask2Former | Mapillary-pretrain → Cityscapes (19 cls out) | 84.9 s.s / 85.8 ms val; 85.2 test (Cityscapes) | czczup release | **MSDeformAttn** | no | Narrow (19 out) | Strong, mmseg-based |
| **EoMT-L (DINOv2 ViT-L)** | Cityscapes (19 cls) | 84.2 (Cityscapes val, 1024²) | `tue-mps/cityscapes_semantic_eomt_large_1024` | **none** | **yes** | Narrow (19) | Newest/fastest, no Mapillary ckpt |
| SegFormer-B5 | Cityscapes (19 cls) | ~82–84 (Cityscapes val) | `nvidia/segformer-b5-finetuned-cityscapes-1024-1024` | none | yes | Narrow (19), no panoptic | Lightweight baseline |

s.s = single-scale, ms = multi-scale + flip.

---

### 1. OneFormer (DEFAULT) — DiNAT-L / Swin-L on Mapillary Vistas

- **Repo / paper:** [SHI-Labs/OneFormer](https://github.com/SHI-Labs/OneFormer) (CVPR 2023), [paper 2211.06220](https://huggingface.co/papers/2211.06220).
- **Training set & taxonomy:** Mapillary Vistas **v1.2**, **65 evaluation classes** (66 incl. `unlabeled`/void). This is the key differentiator: it natively distinguishes lane markings (general vs crosswalk), curb, curb-cut, guardrail, barrier, wall, fence, several rider/vehicle types, poles/utility poles, traffic-sign front/back, etc. Class list is identical to the HF Mask2Former-Mapillary `id2label` (65 entries) — see [Mapillary → canonical](#mapillary-vistas--canonical-taxonomy).
- **Reported mIoU (Mapillary val, from repo model-zoo table):**
  - DiNAT-L: PQ 47.8, **mIoU 64.0 (s.s.) / 64.9 (ms+flip)**
  - Swin-L: PQ 46.7, mIoU 62.9 / 64.1
  - ConvNeXt-L: PQ 47.9, mIoU 63.2 / 63.8
- **Checkpoints + configs (official, exact):**
  - DiNAT-L config `configs/mapillary_vistas/dinat/oneformer_dinat_large_bs16_300k.yaml`, weights `https://shi-labs.com/projects/oneformer/mapillary/250_16_dinat_l_oneformer_mapillary_300k.pth`
  - Swin-L config `configs/mapillary_vistas/swin/oneformer_swin_large_bs16_300k.yaml`, weights `https://shi-labs.com/projects/oneformer/mapillary/250_16_swin_l_oneformer_mapillary_300k.pth`
  - ConvNeXt-L config `configs/mapillary_vistas/convnext/oneformer_convnext_large_bs16_300k.yaml`, weights `https://shi-labs.com/projects/oneformer/mapillary/250_16_convnext_l_oneformer_mapillary_300k.pth`
  - **Note:** there is **no Mapillary checkpoint on the HF Hub** for OneFormer (any backbone). The HF `shi-labs/*` set is ADE20K, Cityscapes, COCO only (see §2).
- **License:** OneFormer code is MIT ([repo](https://github.com/SHI-Labs/OneFormer)); **Mapillary-trained weights inherit Mapillary Vistas' research/non-commercial terms** ([Licensing caveat](#licensing-caveat)).
- **CUDA / custom ops:**
  - DiNAT backbone **requires `natten`** (Neighborhood Attention). Swin/ConvNeXt backbones do not.
  - Original repo also compiles **MSDeformAttn** (`oneformer/modeling/pixel_decoder/ops/make.sh`) and needs **detectron2**.
  - RTX 6000 Ada = sm_89; all custom ops build for it (set `NATTEN_CUDA_ARCH=8.9` / `TORCH_CUDA_ARCH_LIST=8.9` if compiling from source).
- **Reproducibility on this hardware:** the *original* stack (Python 3.8 / torch 1.10.1 / CUDA 11.3 / detectron2-v0.6, per [INSTALL.md](https://github.com/SHI-Labs/OneFormer/blob/main/INSTALL.md)) is dated. On torch 2.4–2.6/CUDA 12.x you hit two frictions — detectron2 build fragility and a natten API mismatch — both solved by the HF-conversion path (see [How to obtain locally](#how-to-obtain-the-default-locally)).
- **Real road scenes:** trained directly on street-level imagery (Mapillary); best taxonomy match for driving footage. Panoptic head gives instance IDs for cars/persons/etc.
- **Compute cost:** DiNAT-L OneFormer ≈ 223 M params; comfortably fits 48 GB for inference at full res.

### 2. OneFormer on Hugging Face `transformers` (what actually exists)

Confirmed set under [`shi-labs`](https://huggingface.co/shi-labs) — **no Mapillary variant**:

| Repo id | Dataset | Backbone |
|---|---|---|
| `shi-labs/oneformer_ade20k_swin_large` | ADE20K | Swin-L |
| `shi-labs/oneformer_ade20k_dinat_large` | ADE20K | DiNAT-L (needs natten) |
| `shi-labs/oneformer_ade20k_swin_tiny` | ADE20K | Swin-T |
| `shi-labs/oneformer_cityscapes_swin_large` | Cityscapes | Swin-L |
| `shi-labs/oneformer_cityscapes_dinat_large` | Cityscapes | DiNAT-L (needs natten) |
| `shi-labs/oneformer_coco_swin_large` | COCO | Swin-L |
| `shi-labs/oneformer_coco_dinat_large` | COCO | DiNAT-L (needs natten) |

Classes: `OneFormerForUniversalSegmentation` + `OneFormerProcessor` ([docs](https://huggingface.co/docs/transformers/model_doc/oneformer)). The architecture is dataset-agnostic — it will load a **converted** Mapillary checkpoint (num_labels=65) with either a Swin-L or DiNAT-L backbone. There is no *pre-converted* Mapillary checkpoint, so we must convert the `.pth` ourselves (transformers ships `convert_oneformer_original_pytorch_checkpoint_to_pytorch.py`), or use Mask2Former-Mapillary (§3).

### 3. Mask2Former (fallback with identical Mapillary taxonomy)

- **Repo / paper:** [facebookresearch/Mask2Former](https://github.com/facebookresearch/Mask2Former) (CVPR 2022), [paper 2112.01527](https://arxiv.org/abs/2112.01527).
- **HF checkpoints (native, no conversion):**
  - `facebook/mask2former-swin-large-mapillary-vistas-semantic` — **Mapillary v1.2, 65 classes**, mIoU **63.2 s.s / 64.7 ms** ([MODEL_ZOO](https://github.com/facebookresearch/Mask2Former/blob/main/MODEL_ZOO.md), [card](https://huggingface.co/facebook/mask2former-swin-large-mapillary-vistas-semantic)).
  - `facebook/mask2former-swin-large-mapillary-vistas-panoptic` — panoptic head.
  - `facebook/mask2former-swin-large-cityscapes-semantic` — Cityscapes, mIoU 83.3 / 84.3.
- **Classes:** `Mask2FormerForUniversalSegmentation` + `AutoImageProcessor`.
- **Custom ops:** **none needed** — HF `transformers` re-implements the deformable-attention pixel decoder in pure PyTorch; no detectron2, no natten, no MSDeformAttn compile.
- **License:** code MIT; **Mapillary weights inherit Mapillary's non-commercial terms** (model card license = "other").
- **Why it's the fallback, not the upgrade:** same taxonomy, ~0.8 mIoU lower than OneFormer DiNAT-L, no DiNAT/panoptic-vs-semantic flexibility — but *trivially* reproducible on the target GPU. This is the safe default if the DiNAT build is deprioritized.

### 4. InternImage-H + Mask2Former (SOTA Cityscapes, heavy)

- **Repo:** [OpenGVLab/InternImage](https://github.com/OpenGVLab/InternImage) (CVPR 2023 Highlight), MIT.
- **Numbers:** Mapillary-pretrained then Cityscapes-finetuned → **87.0 ms mIoU (Cityscapes val), 86.1 test** — the strongest here.
- **But:** the released segmentation checkpoint outputs **Cityscapes 19 classes** (narrower than Mapillary-65 for road furniture / lane detail). Core operator **DCNv3 must be compiled** (custom CUDA), mmsegmentation-based, **not HF-native**, ~1.08 B params (heavier inference). Overkill and lower reproducibility for our taxonomy needs.

### 5. ViT-Adapter (BEiT-L) + Mask2Former

- **Repo:** [czczup/ViT-Adapter](https://github.com/czczup/ViT-Adapter) (ICLR 2023 Spotlight), Apache-2.0 code.
- **Numbers:** Mapillary-pretrained → **Cityscapes val 84.9 s.s / 85.8 ms**, test 85.2 (config `mask2former_beit_adapter_large_896_80k_cityscapes_ss.py`).
- **But:** outputs **Cityscapes 19 classes**, requires **MSDeformAttn** compilation, mmcv/mmseg stack, **not HF-native**, 571 M params. Strong Cityscapes model, wrong taxonomy for our target.

### 6. EoMT — "Your ViT is Secretly an Image Segmentation Model" (newest)

- **Repo / paper:** [tue-mps/eomt](https://github.com/tue-mps/eomt) (CVPR 2025 Highlight). License **MIT**.
- **Why it's attractive:** DINOv2 ViT-L, **encoder-only** (no adapter, no pixel-decoder) → **no custom CUDA ops**, and up to **~4× faster** than ViT-Adapter+Mask2Former at similar accuracy. **Native in `transformers`** as `EomtForUniversalSegmentation` + `AutoImageProcessor`.
- **Cityscapes:** `tue-mps/cityscapes_semantic_eomt_large_1024`, **84.2 mIoU** (val, 1024²) — comparable to ViT-Adapter+Mask2Former.
- **Disqualifier for *default*:** released checkpoints are **ADE20K / Cityscapes / COCO only — no Mapillary Vistas checkpoint**. So its road taxonomy tops out at Cityscapes-19, *narrower* than OneFormer-Mapillary-65. Excellent candidate to revisit **if/when** a Mapillary EoMT checkpoint appears, or as a fast Cityscapes-taxonomy model. Does **not** meet the "at-least-equivalent road taxonomy" bar today.

### 7. SegFormer / SegNeXt (lightweight baselines)

- `nvidia/segformer-b5-finetuned-cityscapes-1024-1024` (HF-native, `SegformerForSemanticSegmentation`), ~82–84 mIoU Cityscapes val, **19 classes, per-pixel only (no panoptic/instance)**. Great as a cheap sanity baseline; taxonomy too coarse to be the primary model.

---

## Custom-op compatibility matrix (the part that actually breaks)

| Op | Needed by | Prebuilt wheel for torch 2.4–2.6 / CUDA 12.x / cp310 / sm_89? | Notes |
|---|---|---|---|
| **natten** | OneFormer/DiNAT (original **and** HF DiNAT checkpoints) | **Yes** — natten `0.17.3/0.17.4` (torch 2.4/2.5) and `0.17.5` (torch 2.6) | Index `https://whl.natten.org/old` (0.17.x). Latest 0.21.x needs torch ≥ 2.8 — too new. |
| **MSDeformAttn** | OneFormer *original* repo; ViT-Adapter | n/a (source build `make.sh`) | **HF `transformers` Mask2Former/OneFormer do NOT need it** (pure-PyTorch fallback). |
| **DCNv3** | InternImage | n/a (source build) | Adds friction; InternImage not HF-native. |
| **detectron2** | OneFormer/Mask2Former *original* repos | **No PyPI wheels** — source build only | Last release v0.6 (2021). Build breakage reported on new stacks ([issue #5503](https://github.com/facebookresearch/detectron2/issues/5503): fails on CUDA 13 + torch 2.9.1). Pin **numpy < 2**. Avoidable via HF path. |

### natten specifics (critical for the DiNAT path)

- Wheel index (legacy 0.14–0.17 series): **`https://whl.natten.org/old`** (current series: `https://whl.natten.org`; the old `https://shi-labs.com/natten/wheels/` still mirrors).
- Exact install commands (Linux, cp310, sm_89):
  - torch 2.5 + CUDA 12.4: `pip install natten==0.17.4+torch250cu124 -f https://whl.natten.org/old`
  - torch 2.4 + CUDA 12.4: `pip install natten==0.17.4+torch240cu124 -f https://whl.natten.org/old`
  - torch 2.6 + CUDA 12.4: `pip install natten==0.17.5+torch260cu124 -f https://whl.natten.org/old`
  - (cu121 variants exist too; swap `cu124`→`cu121`.)
- **API pitfall:** OneFormer's DiNAT code was written for the **natten 0.14.x** API (`natten2dqkrpb`, `natten2dav`). natten **0.17** renamed these to `na2d_qk`/`na2d_av` and now requires explicit `dilation`/`kernel_size`, so the *original* repo throws `TypeError: natten2dqkrpb() missing 1 required positional argument: 'dilation'` ([HF discussion](https://huggingface.co/shi-labs/oneformer_coco_dinat_large/discussions/1), [OneFormer #117](https://github.com/SHI-Labs/OneFormer/issues/117)). The **HF `transformers` DiNAT/OneFormer implementation is already updated for the new natten API** — another reason to prefer the HF path. INSTALL.md's pinned `natten==0.14.4` has **no wheel for torch 2.x / sm_89**, so don't use it on this box.

---

## How to obtain the default locally

### Path A — DiNAT-L Mapillary via HF `transformers` (recommended, no detectron2)

```bash
# 1) env (example: torch 2.5 + CUDA 12.4)
pip install "torch==2.5.*" torchvision --index-url https://download.pytorch.org/whl/cu124
pip install "transformers>=4.44" accelerate timm huggingface_hub
# 2) natten (DiNAT backbone) — prebuilt wheel, no compile
pip install natten==0.17.4+torch250cu124 -f https://whl.natten.org/old

# 3) fetch the official Mapillary DiNAT-L weights + config (one-time, then offline)
wget https://shi-labs.com/projects/oneformer/mapillary/250_16_dinat_l_oneformer_mapillary_300k.pth
#   config: configs/mapillary_vistas/dinat/oneformer_dinat_large_bs16_300k.yaml (from the OneFormer repo)

# 4) convert .pth -> HF format with transformers' converter, then load:
#    python transformers/models/oneformer/convert_oneformer_original_pytorch_checkpoint_to_pytorch.py \
#      --checkpoint_path 250_16_dinat_l_oneformer_mapillary_300k.pth \
#      --config_file oneformer_dinat_large_bs16_300k.yaml \
#      --pytorch_dump_folder_path ./oneformer_mapillary_dinat_large_hf
```

```python
from transformers import OneFormerForUniversalSegmentation, OneFormerProcessor
proc  = OneFormerProcessor.from_pretrained("./oneformer_mapillary_dinat_large_hf")
model = OneFormerForUniversalSegmentation.from_pretrained("./oneformer_mapillary_dinat_large_hf")
# inputs = proc(images=img, task_inputs=["semantic"], return_tensors="pt")  # or "panoptic"
```

> Verify the converter accepts the Mapillary config (num_labels = 65) and the DiNAT backbone (the ADE20K/COCO DiNAT checkpoints were produced by this same script). If conversion is problematic, use Path B.

### Path B — Mask2Former Swin-L Mapillary (zero custom ops, same 65-class taxonomy)

```python
from transformers import AutoImageProcessor, Mask2FormerForUniversalSegmentation
proc  = AutoImageProcessor.from_pretrained("facebook/mask2former-swin-large-mapillary-vistas-semantic")
model = Mask2FormerForUniversalSegmentation.from_pretrained("facebook/mask2former-swin-large-mapillary-vistas-semantic")
# no detectron2, no natten, no MSDeformAttn build
```

### Path C — OneFormer *original* detectron2 repo (only if you need exact repo parity)

Follow [INSTALL.md](https://github.com/SHI-Labs/OneFormer/blob/main/INSTALL.md), but expect to: build detectron2 from source (pin `numpy<2`), install a modern natten (0.17.x) **and patch** the DiNAT modeling call for the new API, and run `sh make.sh` for MSDeformAttn with `TORCH_CUDA_ARCH_LIST=8.9`. Highest fidelity, lowest reproducibility — not recommended as the primary route.

---

## Companion models (already selected elsewhere in the pipeline)

- **SAM 2.1 (Hiera-Large)** — mask refinement / promptable segmentation.
  - Install: `pip install "git+https://github.com/facebookresearch/sam2.git"`.
  - Checkpoint `sam2.1_hiera_large.pt`: `https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt` (script `checkpoints/download_ckpts.sh`).
  - Config name: `configs/sam2.1/sam2.1_hiera_l.yaml`.
  - HF-native alternative: `transformers` `Sam2Model`/`Sam2Processor` (video: `Sam2VideoModel`/`Sam2VideoProcessor`), load `facebook/sam2.1-hiera-large` ([docs](https://huggingface.co/docs/transformers/model_doc/sam2)).
- **Grounding DINO** — open-vocabulary / zero-shot detection prompts.
  - `IDEA-Research/grounding-dino-base` (or lighter `IDEA-Research/grounding-dino-tiny`) via `AutoProcessor` + `AutoModelForZeroShotObjectDetection` ([docs](https://huggingface.co/docs/transformers/model_doc/grounding-dino)). No custom ops.

---

## Mapillary Vistas → canonical taxonomy

Mapillary Vistas **v1.2** = 66 labels (65 evaluated + `unlabeled`/void); **v2.0** = 124 labels. The OneFormer and Mask2Former checkpoints above are **v1.2 / 65-class** ([Mapillary v2.0 blog](https://blog.mapillary.com/update/2021/01/18/vistas-2-dataset.html)). Authoritative class list = `id2label` of [`facebook/mask2former-swin-large-mapillary-vistas-semantic/config.json`](https://huggingface.co/facebook/mask2former-swin-large-mapillary-vistas-semantic/blob/main/config.json) (mirrors OneFormer's `register_mapillary_vistas.py`).

Collapse of the 65 native classes into the canonical pipeline groups:

| Canonical group | Mapillary v1.2 native classes |
|---|---|
| `road_surface` | Road, Service Lane, Bike Lane, Parking |
| `lane_marking` | Lane Marking - General |
| `crosswalk` | Crosswalk - Plain, Lane Marking - Crosswalk |
| `sidewalk` | Sidewalk, Pedestrian Area |
| `curb` | Curb, Curb Cut |
| `person` | Person |
| `rider` | Bicyclist, Motorcyclist, Other Rider |
| `car` | Car |
| `truck` | Truck, Trailer, Caravan |
| `bus` | Bus |
| `motorcycle` | Motorcycle |
| `bicycle` | Bicycle |
| `traffic_light` | Traffic Light |
| `traffic_sign` | Traffic Sign (Front), Traffic Sign (Back), Traffic Sign Frame |
| `pole` | Pole, Utility Pole, Street Light |
| `guardrail` | Guard Rail |
| `barrier` | Barrier |
| `wall` | Wall |
| `fence` | Fence |
| `building` | Building, Bridge, Tunnel |
| `vegetation` | Vegetation |
| `terrain` | Terrain, Sand, Snow, Mountain, Water |
| `sky` | Sky |
| `rail` | Rail Track, On Rails |
| `street_object` (misc furniture; map or ignore) | Banner, Bench, Bike Rack, Billboard, Catch Basin, CCTV Camera, Fire Hydrant, Junction Box, Mailbox, Manhole, Phone Booth, Pothole, Trash Can |
| `other_vehicle` | Boat, Other Vehicle, Wheeled Slow |
| `animal` (usually ignore) | Bird, Ground Animal |
| `ego` / ignore | Car Mount, Ego Vehicle, (void = `unlabeled`) |

Notes for implementation:
- Mapillary separates **lane markings** and **crosswalk** as their own semantic classes — a major advantage over Cityscapes-19 (where these fold into `road`). Keep `lane_marking`/`crosswalk` distinct in the canonical schema to preserve this signal.
- `Traffic Sign (Front)` vs `(Back)` lets you drop rear-facing signs if only front-facing signage matters.
- `Pothole`, `Manhole`, `Catch Basin` sit *on* the road surface — decide whether to merge into `road_surface` or keep as hazards.
- If you later switch to a **v2.0 / 124-class** checkpoint, expect finer splits (e.g. lane-marking sub-types, dashed/solid) that must be re-grouped; there is no official v1.2↔v2.0 map, only image-aligned inference.

---

## Final recommendation

**Default confirmed: OneFormer DiNAT-L trained on Mapillary Vistas.** It uniquely combines (a) the richest *public* road taxonomy (Mapillary v1.2, 65 classes with lane/crosswalk/curb/guardrail granularity), (b) an official checkpoint, (c) semantic + panoptic, and (d) the best road-val mIoU among directly-comparable universal models (64.0 s.s.). No surveyed newer model is *clearly better* on the required bar — EoMT/InternImage/ViT-Adapter ship only Cityscapes-19 checkpoints (narrower taxonomy) and/or need custom-op compilation and are not HF-native. We do **not** switch on publication year or non-reproducible ensemble/test-set numbers.

**Obtain it locally (primary):** convert `250_16_dinat_l_oneformer_mapillary_300k.pth` (config `oneformer_dinat_large_bs16_300k.yaml`) to HF format and serve via `OneFormerForUniversalSegmentation` + `natten==0.17.4+torch250cu124` (or `0.17.5+torch260cu124`) from `https://whl.natten.org/old` — **no detectron2**.

**If the DiNAT/natten conversion is not worth it:** fall back to **`facebook/mask2former-swin-large-mapillary-vistas-semantic`** — identical 65-class taxonomy, native `transformers`, zero custom ops, ~0.8 mIoU lower. This is the lowest-risk fully-local path and is a perfectly good production default.

### Licensing caveat

Both OneFormer-Mapillary and Mask2Former-Mapillary **code** are MIT, but the **weights were trained on Mapillary Vistas**, whose dataset license is **research/non-commercial**. Treat any Mapillary-trained checkpoint as **research-use** unless MBS obtains a commercial license from Mapillary. For unrestricted-license needs, prefer Cityscapes/ADE20K/COCO checkpoints (also non-commercial for Cityscapes; ADE20K is BSD-3, COCO CC-BY-4.0) or a model trained on permissively-licensed data — at the cost of the Mapillary road taxonomy.

---

## Sources

- OneFormer repo & model zoo — https://github.com/SHI-Labs/OneFormer , INSTALL — https://github.com/SHI-Labs/OneFormer/blob/main/INSTALL.md
- OneFormer HF docs — https://huggingface.co/docs/transformers/model_doc/oneformer ; org — https://huggingface.co/shi-labs
- natten install/releases — https://github.com/SHI-Labs/NATTEN/blob/main/docs/install.md , https://github.com/SHI-Labs/NATTEN/releases , index https://whl.natten.org/old
- natten API pitfall — https://huggingface.co/shi-labs/oneformer_coco_dinat_large/discussions/1 , https://github.com/SHI-Labs/OneFormer/issues/117
- detectron2 modern-stack build issue — https://github.com/facebookresearch/detectron2/issues/5503
- Mask2Former repo/zoo — https://github.com/facebookresearch/Mask2Former/blob/main/MODEL_ZOO.md ; Mapillary card — https://huggingface.co/facebook/mask2former-swin-large-mapillary-vistas-semantic
- Mapillary Vistas v2.0 — https://blog.mapillary.com/update/2021/01/18/vistas-2-dataset.html
- InternImage — https://github.com/OpenGVLab/InternImage ; ViT-Adapter — https://github.com/czczup/ViT-Adapter
- EoMT — https://github.com/tue-mps/eomt , https://huggingface.co/tue-mps/cityscapes_semantic_eomt_large_1024 , https://huggingface.co/docs/transformers/model_doc/eomt
- SAM 2 — https://github.com/facebookresearch/sam2 , https://huggingface.co/docs/transformers/model_doc/sam2
- Grounding DINO — https://huggingface.co/docs/transformers/model_doc/grounding-dino , https://huggingface.co/IDEA-Research/grounding-dino-base
