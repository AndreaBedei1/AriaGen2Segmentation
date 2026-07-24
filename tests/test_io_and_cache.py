"""Atomic writes, uint16 masks, manifest resumability + cache invalidation (§14, §18)."""
import numpy as np
import pytest

from aria_drive_seg.io_utils import (Manifest, atomic_write_text,
                                     config_fingerprint, read_mask_u16,
                                     write_mask_u16)


def test_atomic_write_text(tmp_path):
    p = tmp_path / "a.txt"
    atomic_write_text(p, "hello")
    assert p.read_text() == "hello"
    # no stray temp files left
    assert list(tmp_path.glob(".tmp_*")) == []


def test_uint16_mask_roundtrip(tmp_path):
    m = np.array([[0, 1, 65535], [39, 300, 12]], dtype=np.uint16)
    p = tmp_path / "m.png"
    write_mask_u16(p, m)
    back = read_mask_u16(p)
    assert back.dtype == np.uint16
    assert np.array_equal(back, m)


def test_mask_rejects_out_of_range(tmp_path):
    with pytest.raises(ValueError):
        write_mask_u16(tmp_path / "x.png", np.array([[70000]], dtype=np.int32))


def test_manifest_resume_and_invalidation(tmp_path):
    fp1 = config_fingerprint("stageX", {"thr": 0.3})
    m = Manifest.load_or_new(tmp_path / "man.json", "stageX", fp1)
    m.mark(5, {"ok": True})
    m.save()
    # same fingerprint -> remembers done
    m2 = Manifest.load_or_new(tmp_path / "man.json", "stageX", fp1)
    assert m2.is_done(5)
    # changed fingerprint (e.g. threshold) -> cache invalidated
    fp2 = config_fingerprint("stageX", {"thr": 0.5})
    m3 = Manifest.load_or_new(tmp_path / "man.json", "stageX", fp2)
    assert not m3.is_done(5)


def test_fingerprint_stable_and_sensitive():
    a = config_fingerprint("s", {"x": 1, "y": 2})
    b = config_fingerprint("s", {"y": 2, "x": 1})
    assert a == b  # order independent
    assert a != config_fingerprint("s", {"x": 1, "y": 3})
