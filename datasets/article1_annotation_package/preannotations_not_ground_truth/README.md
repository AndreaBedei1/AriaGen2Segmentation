# PRE-ANNOTATIONS — NOT GROUND TRUTH

Every mask in `SegmentationClass/` is automatic model output:

- external scene: Mask2Former Swin-L trained on Mapillary Vistas, aggregated into
  the Article 1 macro taxonomy;
- cockpit: an unreviewed Grounding DINO + SAM 2.1 proxy plus a geometric
  bottom-of-frame prior.

Neither has been validated on this data, and the geometric cockpit prior in
particular was shaped around a car interior and does not describe a handlebar.

These masks exist to save the annotator time. They are wrong often enough that
they must be checked pixel by pixel, and where they disagree with the image the
image wins. Nothing here may be used as a reference, a metric target, or training
supervision until a human has reviewed it.
