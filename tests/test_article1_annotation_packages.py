import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
VAL = ROOT / "validation/article1_auto_moto"


def test_external_dev_manifest_contains_real_auto_only_until_moto_available():
    df = pd.read_csv(VAL / "external_dev40_manifest.csv")
    assert len(df) == 20
    assert set(df.vehicle_type) == {"car"}
    assert df.frame_index.nunique() == 20
    labels = json.loads((VAL / "external_dev40/cvat_package/labels.json").read_text())
    assert labels["vehicle_frames"] == {"car": 20, "motorcycle": 0}
    assert "control_and_ego_vehicle" not in labels["labels"]


def test_cockpit_package_schema_and_no_fake_moto_rows():
    df = pd.read_csv(VAL / "cockpit120_manifest.csv")
    assert len(df) == 60
    assert set(df.vehicle_type) == {"car"}
    assert set(df.review_status) == {"not_annotated"}
    labels = json.loads((VAL / "cockpit120/cvat_package/labels.json").read_text())
    assert labels["vehicle_frames"]["motorcycle"] == 0
    assert labels["labels"] == [
        "mirror", "instrument_display", "control_and_ego_vehicle",
        "background_internal", "ignore"]
