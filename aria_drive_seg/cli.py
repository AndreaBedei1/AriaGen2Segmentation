"""Command-line interface (§15).

    python -m aria_drive_seg inspect     --vrs <file> [--output DIR]
    python -m aria_drive_seg extract     --vrs <file> --output DIR
    python -m aria_drive_seg segment     --method {grounded_sam2,oneformer_mapillary} --input DIR
    python -m aria_drive_seg align-gaze  --vrs <file> --input DIR
    python -m aria_drive_seg analyze     --input DIR
    python -m aria_drive_seg render      --input DIR
    python -m aria_drive_seg run-all     --vrs <file> --output DIR
    python -m aria_drive_seg hwinfo      [--output DIR]
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Optional


def _add_common(p: argparse.ArgumentParser) -> None:
    p.add_argument("--start-time", type=float, default=None, dest="start_time",
                   help="seconds from recording start")
    p.add_argument("--end-time", type=float, default=None, dest="end_time")
    p.add_argument("--frame-step", type=int, default=None, dest="frame_step")
    p.add_argument("--max-frames", type=int, default=None, dest="max_frames")
    p.add_argument("--devices", type=str, default=None, help="'auto' | '0' | '0,1'")
    p.add_argument("--workers", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=None, dest="batch_size")
    p.add_argument("--offline", action="store_true", help="force HF/transformers offline")
    p.add_argument("--resume", dest="resume", action="store_true", default=True)
    p.add_argument("--no-resume", dest="resume", action="store_false")
    p.add_argument("--force", action="store_true", help="ignore cache, recompute")
    p.add_argument("--config", type=str, default=None, help="extra YAML config")
    p.add_argument("--log-level", type=str, default="INFO", dest="log_level")


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser("aria_drive_seg", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="command", required=True)

    p = sub.add_parser("inspect", help="open + validate a VRS (§4)")
    p.add_argument("--vrs", required=True)
    p.add_argument("--output", default=None, help="output dir (default <output>/inspection)")
    p.add_argument("--samples", type=int, default=6)
    _add_common(p)

    p = sub.add_parser("extract", help="extract original + rectified frames (§5)")
    p.add_argument("--vrs", required=True)
    p.add_argument("--output", required=True)
    _add_common(p)

    p = sub.add_parser("segment", help="run a segmentation method (§7/§8)")
    p.add_argument("--method", required=True,
                   choices=["grounded_sam2", "oneformer_mapillary"])
    p.add_argument("--input", required=True, help="output dir from extract")
    _add_common(p)

    p = sub.add_parser("align-gaze", help="align + project on-device eyegaze (§9)")
    p.add_argument("--vrs", required=True)
    p.add_argument("--input", required=True)
    _add_common(p)

    p = sub.add_parser("analyze", help="gaze-to-class assignment + metrics (§10/§17)")
    p.add_argument("--input", required=True)
    p.add_argument("--method", default=None, choices=["grounded_sam2", "oneformer_mapillary"])
    _add_common(p)

    p = sub.add_parser("render", help="overlay + comparison videos (§11)")
    p.add_argument("--input", required=True)
    p.add_argument("--what", default="all")
    _add_common(p)

    p = sub.add_parser("run-all", help="inspect->extract->segment->gaze->analyze->render")
    p.add_argument("--vrs", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--methods", default="grounded_sam2,oneformer_mapillary")
    _add_common(p)

    p = sub.add_parser("hwinfo", help="write hardware_report.json (§13)")
    p.add_argument("--output", default=".")
    _add_common(p)

    p = sub.add_parser("article1", help="Article 1 car-vs-motorcycle pipeline")
    a1 = p.add_subparsers(dest="article1_command", required=True)
    pext = a1.add_parser("segment-external",
                         help="probabilistic Mask2Former Mapillary macro segmentation")
    pext.add_argument("--input", required=True, help="run directory containing frames/")
    pext.add_argument("--vehicle-type", required=True, choices=["car", "motorcycle"])
    pext.add_argument("--session-id", required=True)
    pext.add_argument("--participant-id", required=True)
    _add_common(pext)
    prend = a1.add_parser("render-external", help="render native and Article 1 external videos")
    prend.add_argument("--input", required=True)
    prend.add_argument("--output", default=None)
    prend.add_argument("--vehicle-type", required=True, choices=["car", "motorcycle"])
    prend.add_argument("--session-id", required=True)
    prend.add_argument("--participant-id", required=True)
    prend.add_argument("--fps", type=float, default=None)
    _add_common(prend)
    return ap


def _load_cfg(args):
    from .config import Config, apply_cli_overrides
    cfg = Config.load(path=getattr(args, "config", None))
    return apply_cli_overrides(cfg, args)


def main(argv: Optional[list] = None) -> int:
    args = build_parser().parse_args(argv)
    if getattr(args, "offline", False):
        for k in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_DATASETS_OFFLINE"):
            os.environ[k] = "1"
        os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")

    from .logging_utils import setup_logging
    setup_logging(getattr(args, "log_level", "INFO"))
    cfg = _load_cfg(args)

    cmd = args.command
    if cmd == "inspect":
        from .vrs.inspect import run_inspect
        out = args.output or str(Path(args.vrs).with_suffix("")) + "_out/inspection"
        rep = run_inspect(args.vrs, out, cfg, num_samples=args.samples)
        print(f"readable={rep.get('readable')} rgb={rep.get('rgb_label')} "
              f"eyegaze={rep.get('eyegaze_label')} -> {out}")
        return 0 if rep.get("readable") else 2

    if cmd == "extract":
        from .vrs.extract import run_extract
        res = run_extract(args.vrs, args.output, cfg, resume=args.resume, force=args.force)
        print(res)
        return 0

    if cmd == "hwinfo":
        from .hwinfo import write_hardware_report
        info = write_hardware_report(Path(args.output) / "hardware_report.json")
        print(f"gpus={info['num_gpus']} -> {Path(args.output) / 'hardware_report.json'}")
        return 0

    if cmd == "segment":
        from .segmentation.run import run_segment
        return run_segment(args.method, args.input, cfg, resume=args.resume, force=args.force)

    if cmd == "align-gaze":
        from .gaze.align import run_align_gaze
        run_align_gaze(args.vrs, args.input, cfg)
        return 0

    if cmd == "analyze":
        from .analyze.run import run_analyze
        run_analyze(args.input, cfg, method=args.method)
        return 0

    if cmd == "render":
        from .render.run import run_render
        run_render(args.input, cfg, what=args.what)
        return 0

    if cmd == "run-all":
        from .pipeline import run_all
        return run_all(args.vrs, args.output, cfg, methods=args.methods.split(","),
                       resume=args.resume, force=args.force)

    if cmd == "article1":
        cfg.set("article1.vehicle_type", args.vehicle_type)
        cfg.set("article1.session_id", args.session_id)
        cfg.set("article1.participant_id", args.participant_id)
        if args.article1_command == "segment-external":
            from .article1.external import run_external
            return run_external(args.input, cfg, resume=args.resume, force=args.force)
        if args.article1_command == "render-external":
            from .article1.render import run_render_external
            result = run_render_external(args.input, cfg, output_dir=args.output, fps=args.fps)
            print(result)
            return 0

    return 1


if __name__ == "__main__":
    sys.exit(main())
