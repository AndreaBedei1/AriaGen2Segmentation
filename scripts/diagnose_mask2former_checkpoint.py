#!/usr/bin/env python3
"""Diagnose the local Mask2Former conversion and prove deterministic loading."""
from __future__ import annotations

import argparse
import json
import platform
import random
from pathlib import Path

import numpy as np

from aria_drive_seg.hashing import sha256_file
from aria_drive_seg.io_utils import atomic_write_json, atomic_write_text
from aria_drive_seg.segmentation.oneformer import load_verified_mask2former


def seed_all(seed):
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="weights/mask2former-mapillary-semantic")
    ap.add_argument("--image", required=True)
    ap.add_argument("--output", default="reports")
    ap.add_argument("--seed", type=int, default=2026)
    args = ap.parse_args()
    import cv2
    import torch
    import transformers
    from transformers import AutoImageProcessor

    ckpt, out = Path(args.checkpoint), Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    image = cv2.cvtColor(cv2.imread(args.image), cv2.COLOR_BGR2RGB)
    proc = AutoImageProcessor.from_pretrained(ckpt)
    inputs = proc(images=image, return_tensors="pt")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    runs = []
    tensors = []
    params = []
    for _ in range(2):
        seed_all(args.seed)
        model = load_verified_mask2former(str(ckpt)).to(device).eval()
        norm = model.model.pixel_level_module.encoder.swin.layernorm
        params.append((norm.weight.detach().cpu().clone(), norm.bias.detach().cpu().clone()))
        with torch.inference_mode():
            result = model(**{k: v.to(device) for k, v in inputs.items()})
        logits = result.class_queries_logits.float().cpu()
        masks = result.masks_queries_logits.float().cpu()
        tensors.append((logits, masks))
        runs.append({"loading_info": model._aria_loading_info,
                     "logits_shape": list(logits.shape), "masks_shape": list(masks.shape)})
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Prove the new final Swin norm is outside the Mask2Former feature-map path.
    seed_all(args.seed)
    model = load_verified_mask2former(str(ckpt)).to(device).eval()
    with torch.inference_mode():
        baseline = model(**{k: v.to(device) for k, v in inputs.items()}).class_queries_logits.float().cpu()
        model.model.pixel_level_module.encoder.swin.layernorm.weight.fill_(7.0)
        model.model.pixel_level_module.encoder.swin.layernorm.bias.fill_(13.0)
        perturbed = model(**{k: v.to(device) for k, v in inputs.items()}).class_queries_logits.float().cpu()
    deterministic = {
        "parameters_equal": bool(torch.equal(params[0][0], params[1][0]) and
                                 torch.equal(params[0][1], params[1][1])),
        "class_logits_exact_equal": bool(torch.equal(tensors[0][0], tensors[1][0])),
        "mask_logits_exact_equal": bool(torch.equal(tensors[0][1], tensors[1][1])),
        "predictions_equal": bool(torch.equal(tensors[0][0].argmax(-1),
                                               tensors[1][0].argmax(-1))),
        "final_layernorm_perturbation_max_abs_logit_diff":
            float((baseline - perturbed).abs().max()),
    }
    config = json.loads((ckpt / "config.json").read_text())
    report = {
        "checkpoint": str(ckpt.resolve()),
        "checkpoint_files_sha256": {
            p.name: sha256_file(p) for p in sorted(ckpt.iterdir()) if p.is_file()
        },
        "model_architecture": config.get("architectures"),
        "model_type": config.get("model_type"),
        "num_labels": len(config.get("id2label", {})),
        "backbone_config": config.get("backbone_config"),
        "checkpoint_revision": config.get("_commit_hash"),
        "versions": {"python": platform.python_version(), "torch": torch.__version__,
                     "transformers": transformers.__version__, "cuda": torch.version.cuda},
        "seed": args.seed,
        "device": device,
        "runs": runs,
        "deterministic": deterministic,
        "diagnosis": {
            "cause": "Transformers v5 SwinBackbone wrapper adds final SwinModel LayerNorm absent from the converted checkpoint; legacy relative_position_index buffers are no longer persistent model state.",
            "affected_component": "final SwinModel LayerNorm, outside SwinBackbone feature_maps path",
            "solution": "fail-closed verified loader; allow only the two diagnosed keys and initialize the unused final LayerNorm explicitly to identity",
        },
        "gate_passed": (
            deterministic["parameters_equal"]
            and deterministic["class_logits_exact_equal"]
            and deterministic["mask_logits_exact_equal"]
            and deterministic["predictions_equal"]
            and deterministic["final_layernorm_perturbation_max_abs_logit_diff"] == 0.0
        ),
    }
    atomic_write_json(out / "article1_mask2former_checkpoint_diagnosis.json", report)
    d = deterministic
    md = f"""# Article 1 Mask2Former checkpoint diagnosis

- Checkpoint: `{report['checkpoint']}`
- Architecture: `{report['model_architecture']}`
- PyTorch: `{torch.__version__}`; Transformers: `{transformers.__version__}`
- Seed: `{args.seed}`; device: `{device}`
- Missing keys: `{runs[0]['loading_info']['missing_keys']}`
- Unexpected keys: {len(runs[0]['loading_info']['unexpected_keys'])} legacy relative-position buffers
- Solution: explicit identity initialization of the final unused Swin LayerNorm; fail closed on any other mismatch.

## Determinism

- Parameters equal across loads: **{d['parameters_equal']}**
- Class logits exactly equal: **{d['class_logits_exact_equal']}**
- Mask logits exactly equal: **{d['mask_logits_exact_equal']}**
- Predictions equal: **{d['predictions_equal']}**
- Max class-logit difference after setting the final LayerNorm to weight=7/bias=13:
  **{d['final_layernorm_perturbation_max_abs_logit_diff']}**

Gate passed: **{report['gate_passed']}**.

The final `SwinModel.layernorm` is applied only to `last_hidden_state`; `SwinBackbone`
constructs Mask2Former feature maps from pre-final `reshaped_hidden_states` and its trained
`hidden_states_norms`. The two absent parameters are therefore not used by segmentation.
All trained backbone blocks, stage norms, pixel decoder and classifier weights load.
"""
    atomic_write_text(out / "article1_mask2former_checkpoint_diagnosis.md", md)


if __name__ == "__main__":
    main()
