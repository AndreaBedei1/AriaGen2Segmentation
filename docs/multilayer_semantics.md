# Multilayer semantics + gaze resolution (Grounded-SAM2, Phase 3)

A single canonical mask cannot express "road **through** the windshield" vs "the
windshield glass" vs "a mirror". `aria_drive_seg/segmentation/layers.py` splits the
SAM instances into four functional layers, each resolved independently and saved as a
uint16 mask under `<method>/layers/<layer>/frame_XXXXXX.png`:

| layer | contents |
|---|---|
| `exterior` | outdoor scene classes (road, cars, people, buildings, sky, signs, …), **including what is seen through the glass** |
| `cockpit` | vehicle-interior objects (wheel, dashboard, cluster, hands, trim, …) |
| `transparent` | windshield + side windows (a see-through backdrop, not the content) |
| `mirror` | rear-view + side mirrors (functional regions, may reflect exterior) |

Layer membership: `transparent = {windshield,left_window,right_window}`,
`mirror = {rear_view_mirror,side_mirror,left/right_side_mirror}`, `cockpit` = the rest of
group `cockpit`, `exterior` = group `road`. The canonical mask is still written for
backward compatibility.

## Structured gaze resolution (§3)
`resolve_gaze_target()` returns a **structured** target, not a single string:
`primary_target`, `secondary_target`, `resolution_reason`, and per-layer observations
(`exterior_content`, `cockpit_object`, `transparent_surface`, `mirror_region`) with
per-layer confidence + coverage. Rules, in priority order:

1. **mirror present** → primary = the mirror class; `secondary` = any exterior content
   (reflected / behind); reason `gaze_on_mirror`.
2. **exterior content present** → primary = the exterior class; reason `through_glass`
   if a transparent surface is also there, else `direct_exterior`. *The glass never
   masks the content the driver is actually looking at.*
3. **cockpit object** → primary = the cockpit class; reason `cockpit_object`.
4. **transparent only, no content** → primary = **`unknown_exterior`** (never silently
   `windshield`); reason `transparent_only_no_content`.
5. nothing → `unknown`.

These fields are written per frame into
`comparison/gaze/gaze_labels_grounded_sam2.parquet` by the `analyze` stage.

## Validation
On the 3-frame layer check, a gaze pixel on the windshield/road resolved to
`stop_line` with reason `through_glass` (exterior wins over the transparent surface),
and the four layers cleanly separated exterior scene / cockpit / glass / mirrors
(verified visually). Rules covered in `tests/test_layers.py`.
