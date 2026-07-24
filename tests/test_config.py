"""Config parsing + CLI overrides (§18)."""
from aria_drive_seg.config import Config


def test_load_defaults(config):
    assert config.get("vrs.rgb_label") == "camera-rgb"
    assert config.get("rectify.out_width") == 2016


def test_dotted_get_missing_returns_default(config):
    assert config.get("nope.nope", 42) == 42


def test_deep_merge_override(project_root):
    cfg = Config.load(project_root=project_root,
                      overrides={"rectify": {"focal": 500.0}})
    assert cfg.get("rectify.focal") == 500.0
    # untouched sibling keys survive the merge
    assert cfg.get("rectify.out_width") == 2016


def test_set_and_resolve(config, project_root):
    config.set("frames.max_frames", 7)
    assert config.get("frames.max_frames") == 7
    p = config.resolve("configs/classes.yaml")
    assert p.is_absolute() and p.exists()


def test_apply_cli_overrides(project_root):
    import argparse
    from aria_drive_seg.config import apply_cli_overrides
    cfg = Config.load(project_root=project_root)
    ns = argparse.Namespace(start_time=1.0, end_time=None, frame_step=5,
                            max_frames=None, devices="0", workers=4, batch_size=None)
    apply_cli_overrides(cfg, ns)
    assert cfg.get("frames.start_time_s") == 1.0
    assert cfg.get("frames.frame_step") == 5
    assert cfg.get("hardware.devices") == "0"
