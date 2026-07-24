"""`extract` command (§5): sequentially read the VRS, write original + rectified
RGB frames with atomic writes, a resumable manifest, and a frame index in
Parquet + CSV. Never loads the whole video into RAM."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from ..config import Config
from ..io_utils import (Manifest, append_jsonl, atomic_write_bytes,
                        atomic_write_json, config_fingerprint, file_ok)
from ..logging_utils import get_logger
from .provider import AriaProvider, RectifyParams

log = get_logger("extract")


def _select_indices(rgb_ts: np.ndarray, cfg: Config) -> List[int]:
    n = rgb_ts.size
    t0 = rgb_ts[0]
    start = cfg.get("frames.start_time_s")
    end = cfg.get("frames.end_time_s")
    step = int(cfg.get("frames.frame_step", 1) or 1)
    maxf = cfg.get("frames.max_frames")
    idxs = list(range(0, n, step))
    if start is not None:
        s_ns = t0 + float(start) * 1e9
        idxs = [i for i in idxs if rgb_ts[i] >= s_ns]
    if end is not None:
        e_ns = t0 + float(end) * 1e9
        idxs = [i for i in idxs if rgb_ts[i] <= e_ns]
    if maxf is not None:
        idxs = idxs[: int(maxf)]
    return idxs


def _encode_jpeg(img: np.ndarray, quality: int) -> bytes:
    import cv2
    ok, buf = cv2.imencode(".jpg", img[:, :, ::-1], [cv2.IMWRITE_JPEG_QUALITY, int(quality)])
    if not ok:
        raise RuntimeError("jpeg encode failed")
    return buf.tobytes()


def run_extract(vrs_path: str, out_dir: str, cfg: Config,
                resume: bool = True, force: bool = False) -> Dict[str, Any]:
    out = Path(out_dir)
    frames_dir = out / "frames"
    orig_dir = frames_dir / "original"
    rect_dir = frames_dir / "rectified"
    for d in (orig_dir, rect_dir):
        d.mkdir(parents=True, exist_ok=True)

    prov = AriaProvider(vrs_path, cfg.get("vrs.time_domain", "DEVICE_TIME"))
    rgb_label = cfg.get("vrs.rgb_label", "camera-rgb")
    quality = int(cfg.get("frames.jpeg_quality", 92))

    rp = RectifyParams(
        out_width=int(cfg.get("rectify.out_width", 2016)),
        out_height=int(cfg.get("rectify.out_height", 1512)),
        focal=float(cfg.get("rectify.focal", 879.0)),
        rotate_ccw90=int(cfg.get("rectify.rotate_ccw90", 0)),
    )
    do_rect = bool(cfg.get("rectify.enabled", True))
    rect = prov.rectifier(rp, rgb_label) if do_rect else None

    # persist calibration used (for reverse mapping & provenance)
    calib_doc = {
        "source_calib": prov.calib_summary(rgb_label),
        "rectify": rp.__dict__,
        "rectified_pinhole": _pinhole_summary(rect.pinhole) if rect else None,
    }
    atomic_write_json(frames_dir / "calibration.json", calib_doc)

    rgb_ts = prov.rgb_timestamps_ns(rgb_label)
    idxs = _select_indices(rgb_ts, cfg)
    fp = config_fingerprint("extract", rp.__dict__, quality, do_rect,
                            cfg.get("frames"), str(vrs_path))
    manifest = Manifest.load_or_new(frames_dir / "manifest.json", "extract", fp,
                                    meta={"vrs": str(vrs_path), "num_selected": len(idxs)})
    if force:
        manifest.done.clear()

    log.info("extracting %d frames (of %d) rect=%s -> %s", len(idxs), rgb_ts.size, do_rect, out)
    records: List[Dict[str, Any]] = []
    errors = 0
    for k, i in enumerate(idxs):
        name = f"frame_{i:06d}.jpg"
        op = orig_dir / name
        rpth = rect_dir / name
        rec = {
            "frame_index": int(i),
            "capture_timestamp_ns": int(rgb_ts[i]),
            "original_path": f"frames/original/{name}",
            "rectified_path": f"frames/rectified/{name}" if do_rect else None,
            "rotate_ccw90": rp.rotate_ccw90,
        }
        already = (manifest.is_done(i) and file_ok(op) and (not do_rect or file_ok(rpth)))
        if already and not force:
            r = manifest.done[str(i)]
            if isinstance(r, dict):
                records.append(r)
            else:
                records.append(rec)
            continue
        try:
            raw, ts = prov.rgb_by_index(i, rgb_label)
            rec["capture_timestamp_ns"] = int(ts)
            rec["orig_w"], rec["orig_h"] = int(raw.shape[1]), int(raw.shape[0])
            atomic_write_bytes(op, _encode_jpeg(raw, quality))
            if do_rect:
                r_img = rect.rectify(raw)
                rec["rect_w"], rec["rect_h"] = int(r_img.shape[1]), int(r_img.shape[0])
                atomic_write_bytes(rpth, _encode_jpeg(r_img, quality))
            manifest.mark(i, rec)
            records.append(rec)
        except Exception as e:  # never abort the whole run for one bad frame (§14)
            errors += 1
            append_jsonl(out / "logs" / "extract_errors.jsonl",
                         {"frame_index": int(i), "error": str(e)})
            log.warning("frame %d failed: %s", i, e)
        if (k + 1) % 100 == 0:
            manifest.save()
            log.info("  %d/%d frames", k + 1, len(idxs))
    manifest.save()

    # frame index -> parquet + csv
    _write_index(frames_dir, records)
    result = {"num_selected": len(idxs), "num_written": len(records),
              "errors": errors, "output": str(frames_dir)}
    atomic_write_json(out / "logs" / "extract_summary.json", result)
    log.info("extract done: %d frames, %d errors", len(records), errors)
    return result


def _write_index(frames_dir: Path, records: List[Dict[str, Any]]) -> None:
    import pandas as pd
    df = pd.DataFrame(sorted(records, key=lambda r: r["frame_index"]))
    # atomic parquet + csv
    tmp_pq = frames_dir / ".tmp_frames.parquet"
    df.to_parquet(tmp_pq, index=False)
    tmp_pq.replace(frames_dir / "frames.parquet")
    tmp_csv = frames_dir / ".tmp_frames.csv"
    df.to_csv(tmp_csv, index=False)
    tmp_csv.replace(frames_dir / "frames.csv")


def _pinhole_summary(calib) -> Dict[str, Any]:
    return {
        "model": str(calib.get_model_name()),
        "image_size": [int(x) for x in calib.get_image_size()],
        "focal_lengths": [float(x) for x in calib.get_focal_lengths()],
        "principal_point": [float(x) for x in calib.get_principal_point()],
        "T_device_camera": np.asarray(calib.get_transform_device_camera().to_matrix()).tolist(),
    }
