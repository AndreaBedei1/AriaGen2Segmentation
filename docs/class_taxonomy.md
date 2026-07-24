# Canonical class taxonomy

`configs/classes.yaml` is the **single source of truth** for classes. Both methods map their
native labels onto these canonical ids, and the deterministic palette is shared so colored
overlays are directly comparable. The `uint16` id written into every `canonical_masks/*.png` is
a canonical id from this file.

---

## The three groups

| Group | ids | Meaning | Produced by |
|---|---|---|---|
| **special** | 0 | `unknown` / unassigned pixels. Never invented from a residual mask. | both (as the residual) |
| **road** | 1–27 | Outdoor road-scene classes. | **both** methods (where Mapillary has an equivalent) |
| **cockpit** | 28–39 | Vehicle-interior classes. | **Grounded-SAM2 only** (open-vocabulary); Method 2 is not forced to produce these |

Per-class fields in `classes.yaml`: `id` (the integer written to the mask), `group`, `color`
(RGB palette), and `eval` (whether the class is part of the **common set** used for direct
method-vs-method comparison). `eval: true` is set only for classes both methods can produce.

### Road classes (1–27, `eval: true`)

`road_surface, lane_marking, crosswalk, sidewalk, curb, parking_area, person, rider, car,
truck, bus, motorcycle, bicycle, traffic_light, traffic_sign, pole, guardrail, barrier, wall,
fence, building, vegetation, terrain, sky, bridge, tunnel, rail_track`

### Cockpit classes (28–39, `eval: false`)

`steering_wheel, instrument_cluster, dashboard, infotainment_screen, rear_view_mirror,
side_mirror, windshield, a_pillar, driver_hand, smartphone, other_cockpit, unknown_cockpit`

These are excluded from between-method comparison because Method 2's Mapillary taxonomy has no
equivalent — they are studied only inside Grounded-SAM2.

---

## The palette

`Taxonomy.palette()` builds a `(max_id+1, 3)` uint8 lookup table from the `color` fields;
`Taxonomy.colorize(mask)` maps a uint16 id-mask to RGB. It is **deterministic** (fixed in
`classes.yaml`) and **shared by both methods**, so a class renders identically in both
overlays. `unknown` (id 0) is black `[0,0,0]`. Selected colors:

| id | class | RGB |
|--:|---|---|
| 0 | unknown | 0,0,0 |
| 1 | road_surface | 128,64,128 |
| 2 | lane_marking | 255,255,255 |
| 7 | person | 220,20,60 |
| 9 | car | 0,0,142 |
| 14 | traffic_light | 250,170,30 |
| 24 | sky | 70,130,180 |
| 28 | steering_wheel | 255,140,0 |
| 34 | windshield | 173,216,230 |
| 38 | other_cockpit | 128,128,128 |

(The road colors follow the familiar Cityscapes/Mapillary convention.) See
[README → Viewing the masks](../README.md#viewing-the-masks) for how to colorize.

---

## Method 1 (Grounded-SAM2): prompts → canonical

Method 1 is **open-vocabulary**: every canonical class (road **and** cockpit) has a Grounding
DINO text prompt in `configs/grounded_prompts.yaml`. A returned detection phrase is matched
back to a canonical class by token overlap, then SAM 2.1 turns its box into a mask. Each prompt
entry carries knobs that shape detection and overlap resolution:

| Field | Purpose |
|---|---|
| `prompt` / `synonyms` | text query (and OR'd extra phrases) fed to Grounding DINO |
| `box_threshold` / `text_threshold` | GDINO confidence gates |
| `priority` | overlap precedence — **higher wins** a pixel conflict |
| `min_area` / `max_area_frac` | drop masks too small / too large (fraction of frame) |
| `morph_close` | morphological closing kernel (px) |
| `group_tag` / `solo` | how classes are batched into GDINO calls |

Because it is prompted, Method 1 is **sparse and not an exhaustive partition** — unprompted or
undetected pixels stay `unknown`. Design points encoded in the prompts:

- **Thin/safety-critical classes get high priority** (e.g. `driver_hand` 85, `person` 80,
  `lane_marking` 70) so they are not overwritten by large low-priority backgrounds
  (`sky` 5, `building` 7, `vegetation` 8, `road_surface` 10).
- **`car` is area-capped** (`max_area_frac: 0.30`) to reject ego-cabin over-capture; real lead
  cars occupy a small fraction of the frame. `other_cockpit` is capped at 0.45.
- **`windshield` is low priority** (6): objects seen *through* the glass win; windshield only
  fills where nothing else was detected.

Cockpit classes (steering_wheel, dashboard, instrument_cluster, windshield, a_pillar,
driver_hand, smartphone, …) exist **only here**.

---

## Method 2 (Mask2Former / OneFormer): Mapillary-65 → canonical

Method 2 is a **dense** classifier over **Mapillary Vistas v1.2 (65 classes)**. It keeps the
**native id-mask** (`native/`) untouched and additionally writes the canonical view. The
mapping (`configs/mapillary_to_canonical.yaml`, applied by `MapillaryMapper` as a native-id →
canonical-id lookup table built from the model's `id2label`) tags each native class with a
**support level**:

| support | meaning | in strict eval? |
|---|---|---|
| `supported` | direct canonical equivalent | **yes** |
| `partially_supported` | approximate / coarser mapping | no |
| `unsupported` | no canonical equivalent → `unknown` | no |

Mapillary distinguishes **"Ego Vehicle" / "Car Mount"** from **"Car"**, both mapped to
`other_cockpit`. This is why Method 2 labels the driver's own cabin correctly and does **not**
call the ego vehicle a "car".

### Full mapping (Mapillary v1.2 native → canonical)

| Native (Mapillary) | Canonical | Support |
|---|---|---|
| Road | road_surface | supported |
| Service Lane | road_surface | partially_supported |
| Bike Lane | road_surface | partially_supported |
| Pothole | road_surface | partially_supported |
| Lane Marking - General | lane_marking | supported |
| Crosswalk - Plain | crosswalk | supported |
| Lane Marking - Crosswalk | crosswalk | partially_supported |
| Sidewalk | sidewalk | supported |
| Pedestrian Area | sidewalk | partially_supported |
| Curb | curb | supported |
| Curb Cut | curb | partially_supported |
| Parking | parking_area | supported |
| Person | person | supported |
| Bicyclist / Motorcyclist / Other Rider | rider | supported |
| Car | car | supported |
| Caravan / Other Vehicle | car | partially_supported |
| Truck | truck | supported |
| Trailer | truck | partially_supported |
| Bus | bus | supported |
| Motorcycle | motorcycle | supported |
| Bicycle | bicycle | supported |
| Traffic Light | traffic_light | supported |
| Traffic Sign (Front) | traffic_sign | supported |
| Traffic Sign (Back) / Traffic Sign Frame | traffic_sign | partially_supported |
| Pole | pole | supported |
| Street Light / Utility Pole | pole | partially_supported |
| Guard Rail | guardrail | supported |
| Barrier | barrier | supported |
| Wall | wall | supported |
| Fence | fence | supported |
| Building | building | supported |
| Bridge | bridge | supported |
| Tunnel | tunnel | supported |
| Vegetation | vegetation | supported |
| Terrain | terrain | supported |
| Mountain / Sand / Snow | terrain | partially_supported |
| Sky | sky | supported |
| Rail Track | rail_track | supported |
| **Ego Vehicle / Car Mount** | **other_cockpit** | unsupported |
| Bird, Ground Animal, Water, Banner, Bench, Bike Rack, Billboard, Catch Basin, CCTV Camera, Fire Hydrant, Junction Box, Mailbox, Manhole, Phone Booth, Trash Can, Boat, On Rails, Wheeled Slow | **unknown** | unsupported |

Any native label with no mapping entry is logged (`unmapped_native_labels` in the method's
`summary.json`) and mapped to `unknown` — on the test recording this list is empty.

> The narrative grouping in [model_selection.md](model_selection.md#mapillary-vistas--canonical-taxonomy)
> was an earlier sketch (it folds Bridge/Tunnel under `building`, Water/Mountain under
> `terrain`, and uses a `rail` name). The **authoritative** machine-applied mapping is
> `configs/mapillary_to_canonical.yaml`, tabulated above (Bridge/Tunnel are their own canonical
> classes; Water is `unknown`; the class is `rail_track`).

---

## What "direct comparison" uses

Between-method agreement and IoU are computed **only on the common eval set** — the road
classes with `eval: true` that both methods can produce (`Taxonomy.eval_ids()`). Concretely:

- **Cockpit classes are Grounded-only** and therefore excluded from between-method metrics.
- On the Method-2 side, only `supported` native classes land on those eval canonical ids;
  `partially_supported` and `unsupported` map to non-eval ids or `unknown`.
- `unknown` (id 0) is never counted as a class in the comparison.

This keeps the comparison honest: it measures consistency where the two methods are actually
comparable, not where one is structurally unable to answer. It is still **consistency, not
accuracy** — see [docs/evaluation.md](evaluation.md).

---

## Changing the taxonomy

Edit `configs/classes.yaml` (ids/colors/eval), `configs/grounded_prompts.yaml` (Method 1
prompts), and/or `configs/mapillary_to_canonical.yaml` (Method 2 mapping), then re-run the
affected `segment` stage. These files are folded into the segmentation cache fingerprint, so an
edit transparently invalidates and recomputes exactly the affected masks (see
[architecture.md](architecture.md#resumability-caching-and-the-config-fingerprint)). Keep
`id: 0` (`unknown`) defined — `Taxonomy` requires it.
