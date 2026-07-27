import numpy as np
import pytest

from aria_drive_seg.article1.cockpit_segformer import (
    CockpitOutput,
    DomainBalancedSampler,
    domain_iou,
    require_reviewed_annotations,
    resume_training_checkpoint,
    save_training_checkpoint,
    weighted_cross_entropy,
)


def test_cockpit_output_schema():
    p = np.full((4, 3, 5), .25, np.float32)
    out = CockpitOutput(p, p.argmax(0).astype(np.uint16), p.max(0), "car", "local")
    out.validate()


def test_training_fails_closed_without_reviewed_annotations(tmp_path):
    with pytest.raises(RuntimeError, match="manually reviewed"):
        require_reviewed_annotations(tmp_path)


def test_synthetic_loss_and_checkpoint_resume(tmp_path):
    torch = pytest.importorskip("torch")
    model = torch.nn.Conv2d(3, 4, 1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    x = torch.randn(2, 3, 4, 5)
    target = torch.randint(0, 4, (2, 4, 5))
    loss = weighted_cross_entropy(model(x), target)
    assert loss.isfinite()
    loss.backward()
    optimizer.step()
    path = tmp_path / "checkpoint.pt"
    cfg = {"seed": 2026, "classes": 4}
    save_training_checkpoint(path, model, optimizer, 3, cfg)
    assert resume_training_checkpoint(path, model, optimizer, cfg) == 4
    with pytest.raises(RuntimeError, match="fingerprint"):
        resume_training_checkpoint(path, model, optimizer, {"seed": 7})


def test_domain_balanced_sampler_and_metrics():
    pd = pytest.importorskip("pandas")
    rows = pd.DataFrame({"vehicle_type": ["car", "car", "motorcycle"]})
    sampled = list(DomainBalancedSampler(rows, seed=7))
    domains = rows.iloc[sampled].vehicle_type.tolist()
    assert domains[::2] == ["car", "car"]
    assert domains[1::2] == ["motorcycle", "motorcycle"]
    pred = np.array([[[0, 1]], [[0, 2]]])
    gt = np.array([[[0, 1]], [[0, 2]]])
    scores = domain_iou(pred, gt, ["car", "motorcycle"])
    assert scores["car"][0] == 1
    assert scores["motorcycle"][2] == 1
