import numpy as np
import pytest

from aria_drive_seg.article1.cockpit_segformer import (
    CockpitOutput,
    require_reviewed_annotations,
)


def test_cockpit_output_schema():
    p = np.full((4, 3, 5), .25, np.float32)
    out = CockpitOutput(p, p.argmax(0).astype(np.uint16), p.max(0), "car", "local")
    out.validate()


def test_training_fails_closed_without_reviewed_annotations(tmp_path):
    with pytest.raises(RuntimeError, match="manually reviewed"):
        require_reviewed_annotations(tmp_path)
