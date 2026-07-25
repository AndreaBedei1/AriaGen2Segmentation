# Grounded-SAM2 recall improvement — progress report

Branch: `feature/grounded-sam2-recall` (from `main` @ `3ddddf2`).
Commits so far:
- `69dea65` Fix Grounded-SAM2 prompt engine: variants/synonyms, per-class thresholds, robust mapping
- `5a03550` Implement Grounded-SAM2 instance + geometric filtering and recall-tuned prompts

Method 1 remains **open-vocabulary prompted segmentation** (not dense semantic).
All work is local/offline; baseline `val10` and `run200` were **not** modified (Phase-1
validation ran in the git-ignored `runs/p1_val10`). Push is pending — no git credentials
are available in this environment, so the branch must be pushed manually.

## Phase 1 — bugs fixed (with evidence)

| # | Bug in the first version | Fix |
|---|---|---|
| 1 | `synonyms`/variants were **never sent** to Grounding DINO (`_run_gdino` used only `spec.prompt`) | new `prompt_engine.py` builds captions from `prompt + prompt_variants + synonyms` (deduplicated); `prompt_mode` = combined \| variants \| individual |
| 2 | a group used the **min threshold** of its classes and never re-applied the class threshold | permissive **candidate** thresholds for the call, then each detection re-filtered by its **own** `box_threshold` + `min_map_confidence`; rejections + per-class stats are logged |
| 3 | `max_instances` was read but **never applied** | applied per class after NMS + assignment, keeping the top combined-score instances; dropped count recorded |
| 4 | only `min_area`/`max_area_frac` geometric filters | added `min/max_aspect_ratio`, `min_box_width/height`, `allowed/forbidden_regions`, `edge_margin`, per-class `nms_iou`; all config-driven and logged |
| 5 | phrase→class was a fragile token Jaccard | deterministic mapping: exact / IDF-weighted token overlap (down-weights shared tokens like "car") / singular-plural / ambiguity rejection / `negative_contexts`; stores the **native GDINO phrase** + mapping confidence |

**Key root-cause finding (from real GDINO output):** a big *combined* caption makes GDINO
return "run-on" phrases spanning several class prompts (e.g. `road surface … parking lot
surface parking space`, `speedometer … car dashboard … center console display`), which are
then genuinely ambiguous. Switching the default to **per-class calls** (`prompt_mode:
variants`, one caption per class) makes each returned phrase map to a single class cleanly.
A second false-positive mode — an *absent* class making GDINO emit a low-score full-frame
box that SAM fills to ~90 % — is killed by a complete per-class `max_area_frac` policy.

## Phase 1 — validation (10 distributed frames, `runs/p1_val10` vs baseline `val10`)

| metric | baseline | Phase 1 |
|---|--:|--:|
| total detections (10 frames) | 217 | **380** |
| avg classes / frame | 11.5 | **21.2** |
| avg coverage | 0.35 | 0.45 |
| masks covering >50 % of frame (FP) | several (up to 88 %) | **0** |
| classes newly detected | — | **+10**: side_mirror, traffic_sign, rider, truck, bus, fence, wall, barrier, sky, terrain |
| classes lost vs baseline | — | **0** |
| tunnel false positives | — | 0 (was 8/10) |
| GDINO ms/frame | ~1.9 s (grouped) | ~5.5 s (per-class) |

Visually verified (frame 2256): the exterior scene **through the windshield** is now
segmented (two lead cars, road, sidewalk, vegetation, sky) alongside the cockpit
(steering wheel, instrument cluster, driver hand, A-pillar, both mirrors, dashboard) —
the "recognise objects through the windshield/mirrors" goal. No huge false masks remain.

## What works well now
road_surface, car, vegetation, sky, building, sidewalk, both mirrors, instrument cluster,
dashboard, driver hand, A-pillar, windshield; newly reliable: traffic_sign, person/rider,
truck/bus, fence/wall/barrier.

## Still imperfect / to watch
- **Speed**: per-class calls cost ~5–6.5 s/frame (≈0.15 fps). Acceptable for validation;
  a hybrid grouped+per-class or fewer variant phrases is a future speed/recall ablation.
- Fine thin classes (lane_marking, small distant signs/lights) need the Phase-5 multiscale
  pass to lift recall further.
- Cockpit sub-parts overlap semantically (a_pillar vs windshield vs trim) — Phase-3
  multilayer output + Phase-4 ROI will disambiguate.
- No accuracy claim: needs the Phase-7 ground truth.

## Tests
`51 passed` (env A), including new `test_prompt_engine.py` (variants really sent, dedup,
per-class thresholds read, exact/synonym/singular-plural mapping, ambiguity rejection,
negative context, aspect-ratio/region/box-dim geometric filters). Baseline 41 stay green.

## Phases 2–6 (implemented, tested, committed)

**Phase 2 — extended taxonomy** (`a9dccf2`): +60 fine classes (ids 40-99; 0-39 unchanged
for backward compat) across surfaces/geometry, vulnerable road users, vehicles,
signage/infra and interior parts. Each declares a `parent` generic (pedestrian→person,
stop_sign→traffic_sign, left_side_mirror→side_mirror, …) so analysis can roll subclasses
up to the common comparison taxonomy; deterministic auto-colours. `eval=false` (Grounded-
only). On 10 frames the extended taxonomy reached ~122 detections/frame but at ~15 s/frame.

**Phase 3 — multilayer semantics** (`38f9964`): four functional layer masks
(exterior_content / cockpit_object / transparent_surface / mirror_region) + structured
gaze resolution. A gaze pixel on the glass resolves to the exterior class behind it
(`through_glass`); if none was detected → `unknown_exterior` (never silently `windshield`);
mirrors are primary with an optional reflected secondary. Verified visually (gaze→`stop_line`
through glass; clean layer separation). See docs/multilayer_semantics.md.

**Phase 4 & 5 — hierarchical ROI + multiscale** (`6564859`): ROIs derived from structural
detections (no hardcoded coords; optional EMA stabilisation); targeted class subsets
re-detected on upscaled windshield/window/mirror/interior crops + overlapping windshield
tiles for tiny objects; cross-pass box+mask NMS fusion with provenance. Validated: crop
passes ~doubled detections (83 full-frame + ~125 crop → 208), all ROIs derived, 0 errors,
~47–50 s/frame (max-recall). Default OFF. See docs/hierarchical_segmentation.md.

**Phase 6 — calibration/ablation** (`a4d32c7`): `scripts/calibrate_prompts.py` +
`configs/ablation/{baseline_grouped,per_class_variants,hierarchical_roi,
hierarchical_roi_multiscale}.yaml` to pick the speed/recall operating point on metrics +
visual QA. See docs/ablation_grounded_sam2.md.

Tests: **73 passed** (added test_taxonomy_extended, test_layers, test_roi, test_prompt_engine).

## Speed/recall tradeoff (measured on this hardware)
Per-class full-frame ≈ 5–6.5 s/frame (original taxonomy) / ≈15 s/frame (extended);
+ROI+multiscale ≈ 47–50 s/frame. The operating point is chosen by the ablation (user
directive), balancing recall vs cost vs false positives — not fixed a priori.

## Remaining phases
7 minimal ground truth (infra already present: `scripts/prepare_annotation.py` +
`scripts/evaluate_gt.py` — needs a labelled 60-frame set) · 8 optional SAM2 temporal
propagation · 9 richer layer/gaze diagnostic videos · 10 further tests · 11 run the
ablation to choose a config, then 60–100 then 200 consecutive frames · 12 final report.
The full 3762-frame run stays **gated** on explicit confirmation (Phase 11.13).

## Git / push
9 thematic commits on `feature/grounded-sam2-recall`. **Push is pending** — no git
credentials are available in this environment; push the branch manually:
`git push -u origin feature/grounded-sam2-recall`.

## Reproduce the Phase-1 validation
```bash
git checkout feature/grounded-sam2-recall
cp -r val10/frames runs/p1_val10/frames && cp -r val10/gaze runs/p1_val10/gaze
~/aria_seg_ml_env/bin/python -m aria_drive_seg segment --method grounded_sam2 --input runs/p1_val10 --offline
~/projectaria_gen2_python_env/bin/python -m pytest tests/ -q
```
