# Article 1 models

External scene segmentation uses the local Mask2Former Swin-L Mapillary checkpoint.
Native query/mask scores are normalized into native per-pixel probabilities, then summed
by the mapping in `configs/article1/mapillary_to_article1.yaml`.
The verified loader accepts only the documented Transformers-v5 final-Swin-layernorm
compatibility case, initializes that unused final normalization to identity, and fails closed
for every other missing, unexpected or mismatched parameter. The evidence and hashes are in
`reports/article1_mask2former_checkpoint_diagnosis.md`.

Cockpit segmentation will use one SegFormer-B2 trained jointly on car and motorcycle for
background_internal, mirror, instrument_display and control_and_ego_vehicle. Training fails
closed until manually reviewed annotations from both domains exist. A separate model per
vehicle is intentionally disallowed.

Grounded-SAM2 B/C can seed manual annotations; D is not the primary annotation source.
Mapillary Ego Vehicle and Car Mount are external `other_environment` plus an explicit
provenance flag; they do not supervise the cockpit model.
