import os
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
VRS = os.environ.get("ARIA_TEST_VRS", str(ROOT / "unknown_20260528_090955.vrs"))


@pytest.fixture(scope="session")
def project_root():
    return ROOT


@pytest.fixture(scope="session")
def taxonomy():
    from aria_drive_seg.taxonomy import Taxonomy
    return Taxonomy.load(ROOT / "configs" / "classes.yaml")


@pytest.fixture(scope="session")
def config():
    from aria_drive_seg.config import Config
    return Config.load(project_root=ROOT)


@pytest.fixture(scope="session")
def vrs_path():
    if not Path(VRS).exists():
        pytest.skip(f"test VRS not found at {VRS}")
    return VRS


def has_projectaria():
    try:
        import projectaria_tools  # noqa
        return True
    except Exception:
        return False
