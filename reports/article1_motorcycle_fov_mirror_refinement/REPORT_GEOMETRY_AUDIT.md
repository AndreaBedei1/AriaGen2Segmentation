# Article 1 — image geometry audit (VRS to final video)

Generated: 2026-08-01T19:16:03.160120+00:00

Branch: `feature/article1-motorcycle-fov-mirror-refinement`

## The question

Does the Article 1 pipeline discard part of the motorcycle's field of view, and in particular the mirrors that sit near the image edge?

Nothing here is inferred from reading the code. Every source pixel was pushed through the real transform and asked where it lands, so this measurement could equally well have concluded that nothing is lost.

## Answer

**43.0% of the camera's valid field of view is discarded before the models ever see the frame** — 1,262,087 pixels of real scene content.

**But it is not a crop.** No stage performs a centre crop, no stage changes the aspect ratio, and the final video is 4:3 like the source. The loss is optical: the pipeline rectifies a very wide fisheye onto a *linear pinhole at the same focal length*, and a pinhole simply cannot represent the angles the fisheye captures.

| | source fisheye | rectified pinhole |
|---|---:|---:|
| horizontal field of view | 132.7deg | 97.8deg |
| vertical field of view | 98.9deg | 81.4deg |
| angle at the left edge | 65.7deg | 48.9deg |
| angle at the right edge | 66.9deg | 48.9deg |

A pinhole places a ray at radius `f * tan(theta)` while this fisheye places it near `f * theta`. With the same focal (879.0) the pinhole runs out of image long before the fisheye runs out of scene: everything beyond about 49deg falls outside the rectified frame and is dropped.

## Where the loss lands

| edge | valid pixels lost |
|---|---:|
| left | 252 |
| right | 269 |
| top | 145 |
| bottom | 126 |

Measured along the middle row and middle column, so the numbers can be checked against the overlay by eye.

The left and right losses are the ones that matter for this recording. A motorcycle's bar-end mirrors sit exactly there, low and wide, and `retained_region_overlay.jpg` shows both of them straddling the boundary between the region the pipeline keeps and the region it throws away. Road surface, roadside buildings and pedestrians are discarded with them.

## Per-stage geometry

| stage | size | aspect | aspect preserved | transform | source retained |
|---|---|---:|---|---|---:|
| `01_vrs_rgb` | 2016x1512 | 1.333 | yes | identity | 96.3% |
| `02_rectify_pinhole` | 2016x1512 | 1.333 | yes | unproject(fisheye) then project(pinhole) | 54.8% |
| `03_mask2former_input` | 384x384 | 1.000 | **no** | resize | — |
| `04_grounding_dino_input` | 1333x1000 | 1.333 | **no** | resize | — |
| `05_sam2_input` | 2016x1512 | 1.333 | yes | resize | — |
| `06_semantic_camera_masks` | 2016x1512 | 1.333 | yes | identity | — |
| `07_temporal_processing_scale` | 1008x756 | 1.333 | yes | resize | — |
| `08_final_video` | 1280x960 | 1.333 | yes | resize | — |

Full detail, including the inverse transform and the notes for each stage, is in `geometry/stage_geometry_table.csv` and `geometry/transform_chain.json`.

### What each stage does and does not do

- **VRS to raw frame** — no transform. 2016x1512, aspect 1.333. 96.25% of the sensor rectangle is inside the camera model's angular limit; the four corners are not, and carry no ray.
- **Rectification** — the only lossy stage. Same output size, same aspect ratio, no crop box, and yet it is where the field of view goes.
- **Model input resizes** — Mask2Former resizes to a fixed 384x384 square with no padding, which does deform the aspect ratio inside the model. The mask is mapped back, so this is a resolution and prior concern rather than a field-of-view loss: no pixel is discarded.
- **Temporal stages** — operate at half resolution with the aspect ratio preserved.
- **Final video** — 1280x960, aspect 1.333, 450 frames at 15.0000 fps. Same 4:3 as the source.

## Crop checks

| stage | source aspect | destination aspect | aspect changed | centre crop | widescreen conversion |
|---|---:|---:|---|---|---|
| rectification | 1.333 | 1.333 | no | no | no |
| final_video | 1.333 | 1.333 | no | no | no |

**No centre crop and no widescreen conversion anywhere.** The suspicion that the video is being cut to something like 16:9 is not supported: every stage keeps 4:3. The field of view is lost earlier and for a different reason.

## Images

- `geometry/source_frame.jpg` — the raw fisheye frame
- `geometry/rectified_frame.jpg` — what the pipeline actually segments
- `geometry/retained_region_overlay.jpg` — yellow is the camera-model valid region, green is what the pipeline keeps, red is valid scene content thrown away

## What follows from this

The fix is not to undo a crop, because there is none. It is to stop projecting a 133-degree fisheye onto a 98-degree pinhole. Options that preserve the field of view:

1. run the models on the original fisheye geometry with an explicit valid-pixel mask, accepting peripheral distortion;
2. rectify onto a projection that can represent the full angular range;
3. lower the pinhole focal length, which cannot reach 133 degrees at any finite width and would waste resolution on the centre.

Option 1 is the one this refinement takes, because the instruction is explicit that a small distorted area is preferable to losing a real mirror.

## Limits

- The retained region is a property of the calibration and the rectification target, not of the frame content, so one frame is enough to characterise it. The overlay uses a representative riding frame.
- Edge losses are reported along the middle row and column. The loss is larger towards the corners, where the fisheye reaches its widest angles.
- This audit measures geometry only. Whether the recovered periphery actually improves mirror segmentation is a separate question, answered by the mirror refinement report.

