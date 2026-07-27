"""SegFormer-B2 cockpit interface for Article 1.

Training is intentionally not started until manually reviewed annotations from both
car and motorcycle domains are available. The schema lives here so annotation and
downstream fusion code can target one stable contract without fabricating weights.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json

import numpy as np

from ..hashing import sha256_file, stable_hash


COCKPIT_CLASSES = (
    "background_internal", "mirror", "instrument_display",
    "control_and_ego_vehicle",
)


@dataclass
class CockpitOutput:
    probabilities: np.ndarray  # 4×H×W, normalized
    mask: np.ndarray           # H×W uint16 training-space ids
    confidence: np.ndarray     # H×W float32
    domain: str                # car | motorcycle
    checkpoint: str
    entropy: np.ndarray | None = None  # H×W normalized predictive entropy

    def validate(self) -> None:
        if self.domain not in {"car", "motorcycle"}:
            raise ValueError("domain must be car or motorcycle")
        if self.probabilities.ndim != 3 or self.probabilities.shape[0] != len(COCKPIT_CLASSES):
            raise ValueError("probabilities must have shape 4xHxW")
        if self.mask.shape != self.probabilities.shape[1:]:
            raise ValueError("mask/probability geometry mismatch")
        if self.entropy is not None and self.entropy.shape != self.mask.shape:
            raise ValueError("entropy/probability geometry mismatch")
        if not np.allclose(self.probabilities.sum(axis=0), 1, atol=2e-3):
            raise ValueError("cockpit probabilities must sum to one")


def require_reviewed_annotations(dataset_dir: str | Path) -> None:
    """Fail closed rather than treating pseudo-labels as ground truth."""
    marker = Path(dataset_dir) / "REVIEWED_ANNOTATIONS.json"
    if not marker.exists():
        raise RuntimeError(
            "SegFormer training blocked: manually reviewed car+motorcycle "
            "cockpit annotations are required.")


class CockpitDataset:
    """CVAT-derived image/mask dataset with explicit domain metadata."""

    def __init__(self, manifest, image_dir, mask_dir):
        import pandas as pd
        self.rows = pd.read_csv(manifest)
        required = {"sample_id", "vehicle_type", "frame_index", "review_status"}
        missing = required - set(self.rows)
        if missing:
            raise ValueError(f"cockpit manifest missing {sorted(missing)}")
        if not set(self.rows.vehicle_type).issubset({"car", "motorcycle"}):
            raise ValueError("invalid vehicle domain")
        self.image_dir, self.mask_dir = Path(image_dir), Path(mask_dir)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        import cv2
        row = self.rows.iloc[index]
        image = cv2.imread(str(self.image_dir / f"{row.sample_id}.jpg"))
        mask = cv2.imread(str(self.mask_dir / f"{row.sample_id}.png"), cv2.IMREAD_UNCHANGED)
        if image is None or mask is None:
            raise FileNotFoundError(row.sample_id)
        if mask.max() >= len(COCKPIT_CLASSES):
            raise ValueError("cockpit mask id outside schema")
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB), mask.astype(np.int64), row.to_dict()


class DomainBalancedSampler:
    """Deterministic alternating car/motorcycle indices."""

    def __init__(self, rows, seed=2026):
        self.rows = rows.reset_index(drop=True)
        self.seed = int(seed)

    def __iter__(self):
        rng = np.random.default_rng(self.seed)
        groups = {d: np.where(self.rows.vehicle_type.to_numpy() == d)[0]
                  for d in ("car", "motorcycle")}
        if any(len(v) == 0 for v in groups.values()):
            raise RuntimeError("domain-balanced sampling requires car and motorcycle")
        n = max(map(len, groups.values()))
        sampled = {d: rng.choice(v, n, replace=len(v) < n) for d, v in groups.items()}
        order = np.column_stack([sampled["car"], sampled["motorcycle"]]).ravel()
        return iter(order.tolist())

    def __len__(self):
        counts = self.rows.vehicle_type.value_counts()
        return 2 * int(counts.max())


def weighted_cross_entropy(logits, targets, class_weights=None, ignore_index=255):
    import torch
    return torch.nn.functional.cross_entropy(
        logits, targets, weight=class_weights, ignore_index=ignore_index)


def checkpoint_metadata(path: str | Path, config: dict) -> dict:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    return {"checkpoint": str(path), "sha256": sha256_file(path),
            "config_fingerprint": stable_hash(config)}


def save_training_checkpoint(path, model, optimizer, epoch: int, config: dict):
    import torch
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                "epoch": int(epoch), "config_fingerprint": stable_hash(config)}, tmp)
    tmp.replace(path)


def resume_training_checkpoint(path, model, optimizer, config: dict) -> int:
    import torch
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload["config_fingerprint"] != stable_hash(config):
        raise RuntimeError("cockpit checkpoint/config fingerprint mismatch")
    model.load_state_dict(payload["model"])
    optimizer.load_state_dict(payload["optimizer"])
    return int(payload["epoch"]) + 1


class CockpitSegFormer:
    def __init__(self, checkpoint: str | Path, device="cuda"):
        self.checkpoint = Path(checkpoint)
        if not self.checkpoint.exists():
            raise RuntimeError("valid reviewed SegFormer cockpit checkpoint is absent")
        self.device = device
        self.model = None
        self.processor = None

    def load(self):
        from transformers import AutoImageProcessor, SegformerForSemanticSegmentation
        self.processor = AutoImageProcessor.from_pretrained(self.checkpoint)
        self.model = SegformerForSemanticSegmentation.from_pretrained(
            self.checkpoint, local_files_only=True).to(self.device).eval()
        if int(self.model.config.num_labels) != len(COCKPIT_CLASSES):
            raise RuntimeError("SegFormer checkpoint class-count mismatch")
        return self

    def infer(self, image_rgb: np.ndarray, domain: str) -> CockpitOutput:
        import torch
        import torch.nn.functional as F
        if self.model is None:
            self.load()
        inputs = self.processor(images=image_rgb, return_tensors="pt")
        with torch.inference_mode():
            logits = self.model(**{k: v.to(self.device) for k, v in inputs.items()}).logits
            logits = F.interpolate(logits, size=image_rgb.shape[:2], mode="bilinear",
                                   align_corners=False)
            prob = logits.softmax(1)[0].float().cpu().numpy()
        entropy = -(prob * np.log(np.clip(prob, 1e-8, 1))).sum(0) / np.log(len(COCKPIT_CLASSES))
        out = CockpitOutput(prob, prob.argmax(0).astype(np.uint16),
                            prob.max(0).astype(np.float32), domain,
                            str(self.checkpoint), entropy.astype(np.float32))
        out.validate()
        return out


def domain_iou(predictions, targets, domains, num_classes=4, ignore_index=255):
    """Per-domain IoU; returns NaN for classes absent from prediction and GT."""
    result = {}
    for domain in ("car", "motorcycle"):
        keep = np.asarray(domains) == domain
        if not keep.any():
            result[domain] = [float("nan")] * num_classes
            continue
        pred, gt = np.asarray(predictions)[keep], np.asarray(targets)[keep]
        scores = []
        for cid in range(num_classes):
            valid = gt != ignore_index
            inter = ((pred == cid) & (gt == cid) & valid).sum()
            union = (((pred == cid) | (gt == cid)) & valid).sum()
            scores.append(float(inter / union) if union else float("nan"))
        result[domain] = scores
    return result


def train_cockpit(train_dataset: CockpitDataset, valid_dataset: CockpitDataset,
                  cfg: dict):
    """Local deterministic training loop; blocked unless both datasets are reviewed."""
    import random
    import torch
    from transformers import AutoImageProcessor, SegformerForSemanticSegmentation

    require_reviewed_annotations(train_dataset.mask_dir.parent)
    domains = set(train_dataset.rows.vehicle_type)
    if domains != {"car", "motorcycle"}:
        raise RuntimeError("training requires both car and motorcycle domains")
    seed = int(cfg.get("seed", 2026))
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    pretrained = Path(cfg["pretrained_local"])
    if not pretrained.exists():
        raise RuntimeError("local SegFormer-B2 pretrained checkpoint absent")
    processor = AutoImageProcessor.from_pretrained(pretrained, local_files_only=True)
    model = SegformerForSemanticSegmentation.from_pretrained(
        pretrained, num_labels=len(COCKPIT_CLASSES), ignore_mismatched_sizes=True,
        local_files_only=True).to("cuda")
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(cfg.get("learning_rate", 6e-5)),
                                  weight_decay=float(cfg.get("weight_decay", .01)))
    start_epoch = 0
    resume = Path(cfg.get("checkpoint_dir", "checkpoints/article1_cockpit")) / "last.pt"
    if cfg.get("resume", True) and resume.exists():
        start_epoch = resume_training_checkpoint(resume, model, optimizer, cfg)
    weights = torch.tensor(cfg.get("class_weights", [1] * len(COCKPIT_CLASSES)),
                           device="cuda", dtype=torch.float32)
    sampler = DomainBalancedSampler(train_dataset.rows, seed)
    model.train()
    for epoch in range(start_epoch, int(cfg.get("epochs", 50))):
        for index in sampler:
            image, mask, _ = train_dataset[index]
            inputs = processor(images=image, return_tensors="pt")
            target = torch.from_numpy(mask)[None, None].float()
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.bfloat16,
                                enabled=cfg.get("mixed_precision") == "bf16"):
                logits = model(**{k:v.to("cuda") for k,v in inputs.items()}).logits
                target_low = torch.nn.functional.interpolate(
                    target, size=tuple(logits.shape[-2:]), mode="nearest")[0, 0].long().to("cuda")
                loss = weighted_cross_entropy(logits, target_low[None], weights)
            loss.backward(); optimizer.step()
        save_training_checkpoint(resume, model, optimizer, epoch, cfg)
    return model
