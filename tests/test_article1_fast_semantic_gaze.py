"""Contracts for the compact full-recording semantic-gaze path."""
from __future__ import annotations

import json
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from aria_drive_seg.article1.fast_benchmark import temporal_reuse_agreement
from aria_drive_seg.article1.fast_io import nearest_indices, timestamp_grid_indices
from aria_drive_seg.article1.fast_plots import apply_shared_limits, shared_limits
from aria_drive_seg.article1.fast_semantic_gaze import (
    FastMapillaryMapper, apply_interior_bottom_contact,
    apply_interior_upper_contact)
from aria_drive_seg.config import Config
from aria_drive_seg.taxonomy import Taxonomy

ROOT = Path(__file__).resolve().parent.parent
CONFIG = ROOT / "configs/article1/fast_semantic_gaze.yaml"
CLASSES = ROOT / "configs/article1/classes_fast_semantic_gaze.yaml"
MAPPING = ROOT / "configs/article1/mapillary_to_fast_semantic_gaze.yaml"

EXPECTED_ORDER = [
    "unknown", "road_surface", "lane_marking", "regulatory_road_marking",
    "vehicle", "two_wheeler", "pedestrian", "traffic_sign", "traffic_light",
    "road_boundary_or_sidewalk", "vegetation", "built_environment", "sky",
    "interior_cockpit", "mirror", "other_environment",
]


def test_fast_taxonomy_has_stable_requested_order():
    taxonomy = Taxonomy.load(CLASSES)
    assert taxonomy.names() == EXPECTED_ORDER
    assert taxonomy.by_name["mirror"].eval is False
    assert taxonomy.by_name["interior_cockpit"].group == "cockpit"


def test_every_native_mapillary_label_is_mapped():
    id2 = json.loads((ROOT / "weights/mask2former-mapillary-semantic" /
                      "config.json").read_text())["id2label"]
    taxonomy = Taxonomy.load(CLASSES)
    mapper = FastMapillaryMapper({int(k): v for k, v in id2.items()}, MAPPING,
                                 taxonomy)
    assert len(mapper.id2label) == 65
    assert mapper.unmapped_native_labels == []
    assert not np.any(mapper.native_to_fast == 0)
    assert mapper.matrix().shape == (65, len(EXPECTED_ORDER))


def test_other_environment_is_a_small_declared_residual_mapping():
    mapping = yaml.safe_load(MAPPING.read_text())["mapping"]
    residual = {name for name, entry in mapping.items() if entry.get("residual")}
    assert residual == {"Bird", "Ground Animal", "Sand", "Snow", "Water", "Boat"}
    assert all(mapping[name]["fast_class"] == "other_environment" for name in residual)


def test_frequency_is_timestamp_selected_and_configurable():
    ts = np.arange(0, 2_000_000_001, 100_000_000, dtype=np.int64)
    native = timestamp_grid_indices(ts, "native")
    hz5 = timestamp_grid_indices(ts, 5.0)
    hz25 = timestamp_grid_indices(ts, 2.5)
    assert len(native) == 21 and len(hz5) == 11 and len(hz25) == 6
    assert np.all(ts[hz5] == np.arange(0, 2_000_000_001, 200_000_000))
    assert len(set(hz5)) == len(hz5)


def test_nearest_mask_association_keeps_signed_real_time_error():
    source = np.array([0, 200, 400], np.int64)
    target = np.array([20, 170, 390], np.int64)
    index, delta = nearest_indices(source, target)
    assert index.tolist() == [0, 1, 2]
    assert delta.tolist() == [-20, 30, 10]


def test_temporal_ablation_reuses_only_selected_real_masks():
    masks = np.arange(5, dtype=np.uint16)[:, None, None]
    ts = np.arange(5, dtype=np.int64) * 100_000_000
    assert temporal_reuse_agreement(masks, ts, "native") == 1.0
    assert temporal_reuse_agreement(masks, ts, 5.0) < 1.0


def test_bottom_contact_proxy_is_high_precision_and_cockpit_only():
    taxonomy = Taxonomy.load(CLASSES)
    mask = np.full((20, 20), taxonomy.id_of("road_surface"), np.uint16)
    mask[17:20, 8:12] = taxonomy.id_of("two_wheeler")  # ego, touches bottom
    mask[15:17, 2:5] = taxonomy.id_of("vehicle")       # external, no contact
    changed = apply_interior_bottom_contact(
        mask, taxonomy, {"enabled": True, "lower_start_fraction": .7,
                         "contact_rows": 2,
                         "eligible_classes": ["vehicle", "two_wheeler"]})
    assert changed == 12
    assert np.all(mask[17:20, 8:12] == taxonomy.id_of("interior_cockpit"))
    assert np.all(mask[15:17, 2:5] == taxonomy.id_of("vehicle"))


def test_upper_contact_proxy_is_car_only_and_crop_limited():
    taxonomy = Taxonomy.load(CLASSES)
    built = taxonomy.id_of("built_environment")
    mask = np.full((20, 20), taxonomy.id_of("sky"), np.uint16)
    mask[:12, :4] = built       # headliner-like component touches top
    mask[4:7, 10:13] = built    # exterior component does not touch top
    policy = {"enabled_domains": ["car"], "upper_end_fraction": .4,
              "minimum_upper_fraction": .55,
              "contact_rows": 2, "eligible_classes": ["built_environment"]}
    assert apply_interior_upper_contact(mask, taxonomy, policy, "car") == 48
    assert np.all(mask[:12, :4] == taxonomy.id_of("interior_cockpit"))
    assert np.all(mask[4:7, 10:13] == built)
    moto = np.full((20, 20), built, np.uint16)
    assert apply_interior_upper_contact(moto, taxonomy, policy, "motorcycle") == 0


def test_full_run_config_disables_slow_and_mirror_primary_paths():
    cfg = Config.load(CONFIG)
    assert cfg.get("semantic_gaze_fast_external.segmentation_frequency_hz") == 5.0
    assert cfg.get("semantic_gaze_fast_external.mirror.include_in_primary_mask") is False
    assert cfg.get("semantic_gaze_fast_external.save_macro_probabilities") is False
    assert cfg.get("semantic_gaze_fast_external.temporal_stabilization.enabled") is False


def test_fast_sources_do_not_import_slow_full_frame_models_or_use_rate_feature():
    sources = [ROOT / "aria_drive_seg/article1/fast_semantic_gaze.py",
               ROOT / "aria_drive_seg/article1/fast_io.py"]
    text = "\n".join(path.read_text() for path in sources)
    for banned in ("GroundedSAM2Segmenter", "grounded_sam2 import",
                   "segmentation.grounded_sam2", "from sam2", "import sam2"):
        assert banned not in text
    assert "features =" not in text and "features=" not in text


def test_requested_branch_descends_from_stable_baseline_without_full_fov_path():
    """Guard the stable lineage and keep retired full-FOV code out of fast mode."""
    baseline = "feature/article1-multimodal-behavior-analysis"
    has_baseline = subprocess.run(
        ["git", "show-ref", "--verify", "--quiet", f"refs/heads/{baseline}"],
        cwd=ROOT, check=False).returncode == 0
    if has_baseline:
        assert subprocess.run(
            ["git", "merge-base", "--is-ancestor", baseline, "HEAD"],
            cwd=ROOT, check=False).returncode == 0
    branch = subprocess.run(
        ["git", "branch", "--show-current"], cwd=ROOT, check=True,
        capture_output=True, text=True).stdout.strip().lower()
    if branch:  # CI may use a detached HEAD.
        assert "full-fov" not in branch and "full_fov" not in branch
    tracked = subprocess.run(
        ["git", "ls-files", "aria_drive_seg/article1/fast_*",
         "configs/article1/*fast*", "scripts/*fast*"],
        cwd=ROOT, check=True, capture_output=True, text=True).stdout.lower()
    assert "full_fov" not in tracked and "full-fov" not in tracked


def test_shared_plot_limits_are_identical_for_comparison_panels():
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2)
    limits = apply_shared_limits(axes, [[1, 2, 3], [10, 20]], include_zero=True)
    assert axes[0].get_ylim() == axes[1].get_ylim() == limits
    assert shared_limits([[1], [2]], include_zero=True)[0] < 0
    plt.close(fig)


def test_clean_semantic_camera_renderer_writes_video(tmp_path):
    import cv2
    from aria_drive_seg.article1.fast_render import render_domain

    root = tmp_path / "run"
    for directory in ("frames/rectified", "segmentation/masks",
                      "segmentation/confidence", "segmentation/normalized_entropy",
                      "gaze"):
        (root / directory).mkdir(parents=True, exist_ok=True)
    frame_rows = []; seg_rows = []
    for i in range(3):
        stem = f"frame_{i:06d}"
        image = np.full((24, 32, 3), 40 + i * 10, np.uint8)
        mask = np.full((24, 32), 1 + i, np.uint16)
        cv2.imwrite(str(root / "frames/rectified" / f"{stem}.jpg"), image)
        cv2.imwrite(str(root / "segmentation/masks" / f"{stem}.png"), mask)
        cv2.imwrite(str(root / "segmentation/confidence" / f"{stem}.png"),
                    np.full((24, 32), 220, np.uint8))
        cv2.imwrite(str(root / "segmentation/normalized_entropy" / f"{stem}.png"),
                    np.full((24, 32), 20, np.uint8))
        frame_rows.append({"frame_index": i, "capture_timestamp_ns": i * 200_000_000,
                           "rectified_path": f"frames/rectified/{stem}.jpg",
                           "width": 32, "height": 24})
        seg_rows.append({"frame_index": i, "capture_timestamp_ns": i * 200_000_000,
                         "mask_path": f"segmentation/masks/{stem}.png",
                         "confidence_path": f"segmentation/confidence/{stem}.png",
                         "normalized_entropy_path":
                             f"segmentation/normalized_entropy/{stem}.png",
                         "mean_confidence": .8, "mean_normalized_entropy": .1,
                         "fraction_other_environment": 0.0})
    pd.DataFrame(frame_rows).to_parquet(root / "frames/frames.parquet", index=False)
    pd.DataFrame(seg_rows).to_parquet(
        root / "segmentation/segmentation_index.parquet", index=False)
    pd.DataFrame(columns=["semantic_valid", "segmentation_frame_index",
                          "rect_u", "rect_v"]).to_parquet(
        root / "gaze/semantic_gaze.parquet", index=False)
    output = tmp_path / "semantic.mp4"; preview = tmp_path / "preview.mp4"
    info = render_domain(root, output, Config.load(CONFIG), preview)
    assert output.stat().st_size > 0 and preview.stat().st_size > 0
    assert info["layout"] == "single_full_frame_clean_semantic_overlay"
    assert info["diagnostic_panels"] is False
