# Prompt engine (Grounded-SAM2, Phase 1)

`aria_drive_seg/segmentation/prompt_engine.py` turns the per-class config in
`configs/grounded_prompts.yaml` into Grounding DINO captions and maps the returned
phrases back to canonical classes. It fixes three real bugs in the first version.

## Per-class config fields
`prompt`, `prompt_variants`, `synonyms`, `negative_contexts`, `prompt_mode`
(`combined` | `variants` | `individual`), `box_threshold`, `text_threshold`,
`candidate_box_threshold`, `candidate_text_threshold`, `min_map_confidence`,
`priority`, geometric filters (`min_area`, `max_area_frac`, `min/max_aspect_ratio`,
`min_box_width/height`, `allowed_regions`, `forbidden_regions`, `edge_margin`),
`max_instances`, `nms_iou`, `morph_close`, `group_tag`, `solo`.

## Captions actually include variants + synonyms
`group_captions()` builds the caption from `prompt + prompt_variants + synonyms`
(deduplicated by normalized token key). The old code sent only `spec.prompt`, so
synonyms/variants were dead config. `prompt_mode` controls batching:
- `variants` (default): **one GDINO call per class** with all its phrases. This avoids
  the "run-on phrase" failure a big combined caption causes, where GDINO returns a span
  covering several adjacent class prompts (`road surface … parking lot surface parking
  space`) that maps ambiguously. Per-class calls keep each returned phrase unambiguous.
- `combined`: classes sharing a `group_tag` share one caption (fewer calls, faster, but
  more ambiguity). `individual`: main prompt only, per class.

## Candidate vs final thresholds
A grouped/combined call uses the permissive `candidate_*` thresholds; **after** the
phrase→class mapping each detection is re-filtered by its **own** class `box_threshold`
and `min_map_confidence`. The old code applied the group's minimum threshold to every
class and never re-applied the per-class one. Every rejection is counted by reason
(`below_box_threshold`, `low_map_confidence`, `ambiguous(...)`, `negative_context`,
geometric reasons, `max_instances`, `mask_dedup`) and surfaced in the frame metadata
and in the segmenter's per-class stats.

## Robust phrase→class mapping (`PhraseMapper`)
1. exact normalized (singularized) phrase match → confidence 1.0;
2. **IDF-weighted** token overlap: tokens shared by many classes (e.g. `car`, which
   appears in most cockpit prompts) get little weight; distinctive tokens (`dashboard`)
   dominate — this stopped the mass "ambiguous" rejections that shared tokens caused;
3. singular/plural handling; deterministic tie-break by name;
4. **ambiguity rejection** when two candidate classes explain the phrase within a tight
   margin (only possible in combined mode; per-class mode has one candidate);
5. **negative_contexts**: a phrase best-matching a decoy (`billboard` for `traffic_sign`)
   is dropped.
The **native GDINO phrase** and the **mapping confidence** are stored per detection.

## Geometric + instance filtering
`box_geom_ok()` enforces the per-class geometric filters; `max_instances` keeps the
top-N combined-score instances per class after NMS. A complete `max_area_frac` policy
kills the "absent class → GDINO emits a low-score full-frame box → SAM fills 90 % of the
frame" false positive (observed at 88–94 % before the fix).

## Validation (10 distributed frames vs baseline)
detections 217 → 380, classes/frame 11.5 → 21.2, +10 new classes, 0 lost, 0 masks
>50 % of the frame, tunnel false-positives gone. Tested in `tests/test_prompt_engine.py`.
