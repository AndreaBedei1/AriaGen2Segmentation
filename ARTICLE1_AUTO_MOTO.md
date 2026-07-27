# Article 1 — Auto vs moto

## Scientific claim

The driven vehicle can change visual-attention strategy even when route and road context
are comparable. The unit of inference is therefore participant/session/route segment or
matched pair, never an independent video frame.

## Pipeline

1. Mask2Former Swin-L Mapillary external probabilities.
2. Probability aggregation into the 14-class Article 1 taxonomy.
3. Separate thin layers for lane and regulatory road markings.
4. One jointly trained SegFormer-B2 for symmetric car/motorcycle cockpit functions.
5. Probabilistic fusion, semantic gaze distributions and temporal probability smoothing.
6. Route-progress-based car/motorcycle pairing.

Grounded-SAM2 remains available only for reviewed pre-annotation, diagnostics, difficult
example search and active learning.

## Status

| Component | Status |
|---|---|
| Reduced taxonomy and Mapillary mapping | Implemented and unit-tested |
| Probabilistic external segmentation | Implemented; 10-frame auto smoke test completed |
| Thin road-marking preservation | Implemented and unit-tested; calibration pending GT |
| External diagnostic renderer | Implemented |
| Cockpit SegFormer schema/config | Implemented; training blocked by reviewed annotations |
| Fusion, temporal gaze, route pairing | Planned for later operational steps |
| Motorcycle and paired results | Blocked: no motorcycle VRS supplied |

No pseudo-label is treated as ground truth and no accuracy is reported without reviewed GT.
