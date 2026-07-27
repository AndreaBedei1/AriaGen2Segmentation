# Article 1 models

External scene segmentation uses the local Mask2Former Swin-L Mapillary checkpoint.
Native query/mask scores are normalized into native per-pixel probabilities, then summed
by the mapping in `configs/article1/mapillary_to_article1.yaml`.

Cockpit segmentation will use one SegFormer-B2 trained jointly on car and motorcycle for
background_internal, mirror, instrument_display and control_and_ego_vehicle. Training fails
closed until manually reviewed annotations from both domains exist. A separate model per
vehicle is intentionally disallowed.

Grounded-SAM2 B/C can seed manual annotations; D is not the primary annotation source.
