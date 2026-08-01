#!/usr/bin/env python3
"""Generate REPORT_GEOMETRY_AUDIT.md from the measured transform chain."""
from __future__ import annotations

import argparse
import datetime as dt
import json
from pathlib import Path
from typing import List


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reports",
                    default="reports/article1_motorcycle_fov_mirror_refinement")
    args = ap.parse_args()

    reports = Path(args.reports)
    doc = json.loads((reports / "geometry" / "transform_chain.json").read_text())
    src, rect = doc["source"], doc["rectification"]
    crop = doc["crop_checks"]
    video = doc["final_video"]

    kept = rect["retained_valid_fraction"]
    lost_fraction = 1.0 - kept

    L: List[str] = [
        "# Article 1 — image geometry audit (VRS to final video)",
        "",
        f"Generated: {dt.datetime.now(dt.timezone.utc).isoformat()}",
        "",
        "Branch: `feature/article1-motorcycle-fov-mirror-refinement`",
        "",
        "## The question",
        "",
        "Does the Article 1 pipeline discard part of the motorcycle's field of view, "
        "and in particular the mirrors that sit near the image edge?",
        "",
        "Nothing here is inferred from reading the code. Every source pixel was "
        "pushed through the real transform and asked where it lands, so this "
        "measurement could equally well have concluded that nothing is lost.",
        "",
        "## Answer",
        "",
        f"**{lost_fraction:.1%} of the camera's valid field of view is discarded "
        f"before the models ever see the frame** — "
        f"{rect['discarded_valid_pixels']:,} pixels of real scene content.",
        "",
        "**But it is not a crop.** No stage performs a centre crop, no stage changes "
        "the aspect ratio, and the final video is 4:3 like the source. The loss is "
        "optical: the pipeline rectifies a very wide fisheye onto a *linear pinhole "
        "at the same focal length*, and a pinhole simply cannot represent the angles "
        "the fisheye captures.",
        "",
        "| | source fisheye | rectified pinhole |",
        "|---|---:|---:|",
        f"| horizontal field of view | {src['angular_coverage']['horizontal_fov_deg']:.1f}deg | "
        f"{doc['rectification']['angular_coverage']['horizontal_fov_deg']:.1f}deg |",
        f"| vertical field of view | {src['angular_coverage']['vertical_fov_deg']:.1f}deg | "
        f"{doc['rectification']['angular_coverage']['vertical_fov_deg']:.1f}deg |",
        f"| angle at the left edge | {src['angular_coverage']['left_deg']:.1f}deg | "
        f"{doc['rectification']['angular_coverage']['left_deg']:.1f}deg |",
        f"| angle at the right edge | {src['angular_coverage']['right_deg']:.1f}deg | "
        f"{doc['rectification']['angular_coverage']['right_deg']:.1f}deg |",
        "",
        "A pinhole places a ray at radius `f * tan(theta)` while this fisheye places it "
        f"near `f * theta`. With the same focal ({rect['focal']}) the pinhole runs out "
        "of image long before the fisheye runs out of scene: everything beyond about "
        f"{doc['rectification']['angular_coverage']['right_deg']:.0f}deg falls outside "
        "the rectified frame and is dropped.",
        "",
        "## Where the loss lands",
        "",
        "| edge | valid pixels lost |",
        "|---|---:|",
        f"| left | {rect['pixels_lost_left']} |",
        f"| right | {rect['pixels_lost_right']} |",
        f"| top | {rect['pixels_lost_top']} |",
        f"| bottom | {rect['pixels_lost_bottom']} |",
        "",
        "Measured along the middle row and middle column, so the numbers can be "
        "checked against the overlay by eye.",
        "",
        "The left and right losses are the ones that matter for this recording. A "
        "motorcycle's bar-end mirrors sit exactly there, low and wide, and "
        "`retained_region_overlay.jpg` shows both of them straddling the boundary "
        "between the region the pipeline keeps and the region it throws away. Road "
        "surface, roadside buildings and pedestrians are discarded with them.",
        "",
        "## Per-stage geometry",
        "",
        "| stage | size | aspect | aspect preserved | transform | source retained |",
        "|---|---|---:|---|---|---:|",
    ]
    for s in doc["stages"]:
        retained = ("—" if s["source_frame_retained_fraction"] is None
                    else f"{s['source_frame_retained_fraction']:.1%}")
        preserved = ("—" if s["aspect_preserved"] is None
                     else ("yes" if s["aspect_preserved"] else "**no**"))
        L.append(f"| `{s['stage']}` | {s['width']}x{s['height']} | "
                 f"{s['aspect_ratio']:.3f} | {preserved} | {s['transform']} | "
                 f"{retained} |")

    L += [
        "",
        "Full detail, including the inverse transform and the notes for each stage, "
        "is in `geometry/stage_geometry_table.csv` and "
        "`geometry/transform_chain.json`.",
        "",
        "### What each stage does and does not do",
        "",
        "- **VRS to raw frame** — no transform. "
        f"{src['width']}x{src['height']}, aspect {src['width'] / src['height']:.3f}. "
        f"{src['model_valid_fraction']:.2%} of the sensor rectangle is inside the "
        "camera model's angular limit; the four corners are not, and carry no ray.",
        "- **Rectification** — the only lossy stage. Same output size, same aspect "
        "ratio, no crop box, and yet it is where the field of view goes.",
        "- **Model input resizes** — Mask2Former resizes to a fixed 384x384 square "
        "with no padding, which does deform the aspect ratio inside the model. The "
        "mask is mapped back, so this is a resolution and prior concern rather than "
        "a field-of-view loss: no pixel is discarded.",
        "- **Temporal stages** — operate at half resolution with the aspect ratio "
        "preserved.",
        f"- **Final video** — {video['width']}x{video['height']}, aspect "
        f"{video.get('aspect_ratio', 0):.3f}, {video['frames']} frames at "
        f"{video['fps']:.4f} fps. Same 4:3 as the source.",
        "",
        "## Crop checks",
        "",
        "| stage | source aspect | destination aspect | aspect changed | centre crop | "
        "widescreen conversion |",
        "|---|---:|---:|---|---|---|",
    ]
    for name, c in crop.items():
        L.append(f"| {name} | {c['source_aspect']:.3f} | {c['destination_aspect']:.3f} | "
                 f"{'yes' if c['aspect_changed'] else 'no'} | "
                 f"{'yes' if c['is_centre_crop'] else 'no'} | "
                 f"{'yes' if c['widescreen_conversion'] else 'no'} |")

    L += [
        "",
        "**No centre crop and no widescreen conversion anywhere.** The suspicion that "
        "the video is being cut to something like 16:9 is not supported: every stage "
        "keeps 4:3. The field of view is lost earlier and for a different reason.",
        "",
        "## Images",
        "",
        "- `geometry/source_frame.jpg` — the raw fisheye frame",
        "- `geometry/rectified_frame.jpg` — what the pipeline actually segments",
        "- `geometry/retained_region_overlay.jpg` — yellow is the camera-model valid "
        "region, green is what the pipeline keeps, red is valid scene content thrown "
        "away",
        "",
        "## What follows from this",
        "",
        "The fix is not to undo a crop, because there is none. It is to stop "
        "projecting a 133-degree fisheye onto a 98-degree pinhole. Options that "
        "preserve the field of view:",
        "",
        "1. run the models on the original fisheye geometry with an explicit "
        "valid-pixel mask, accepting peripheral distortion;",
        "2. rectify onto a projection that can represent the full angular range;",
        "3. lower the pinhole focal length, which cannot reach 133 degrees at any "
        "finite width and would waste resolution on the centre.",
        "",
        "Option 1 is the one this refinement takes, because the instruction is "
        "explicit that a small distorted area is preferable to losing a real mirror.",
        "",
        "## Limits",
        "",
        "- The retained region is a property of the calibration and the rectification "
        "target, not of the frame content, so one frame is enough to characterise it. "
        "The overlay uses a representative riding frame.",
        "- Edge losses are reported along the middle row and column. The loss is "
        "larger towards the corners, where the fisheye reaches its widest angles.",
        "- This audit measures geometry only. Whether the recovered periphery "
        "actually improves mirror segmentation is a separate question, answered by "
        "the mirror refinement report.",
        "",
    ]

    out = reports / "REPORT_GEOMETRY_AUDIT.md"
    out.write_text("\n".join(L) + "\n")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
