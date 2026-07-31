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
| Probabilistic external segmentation | Checkpoint-2 stop gate completed on 60 auto frames |
| Thin road-marking preservation | Raw/filtered layers and rejection provenance implemented; calibration pending GT |
| External diagnostic renderer | Implemented |
| Cockpit SegFormer schema/config | Dataset, balanced sampler, loss, entropy, resume, split and anti-leakage checks implemented; training blocked by reviewed annotations |
| Fusion, temporal gaze | Implemented; dense semantic camera with causal stabilisation and a separate presentation final pass |
| Motorcycle acquisition | Ingested and validated: 15069 RGB frames at a measured 15.0012 Hz over 1004.5 s, one continuous segment, all 17 streams present |
| Multi-rate policy | Rates measured from timestamps, temporal windows declared in seconds, metrics normalised per second; target protocol declared at 15 fps for both vehicles |
| Frozen motorcycle baseline | Run on a selected 30 s segment with the car-tuned configuration unchanged, to expose domain shift before adapting anything |
| Route pairing | Preliminary and GPS-based: the car route coincides with the final ~2.2 km of the motorcycle ride, same direction |
| Annotation package | Prepared, balanced 70 frames per domain, with pre-annotations marked as not ground truth |
| Paired scientific results | Blocked: the car is still a provisional 10 fps baseline and no reviewed ground truth exists |

See `docs/article1_motorcycle_ingestion.md` and
`reports/article1_motorcycle_ingestion/`.

No pseudo-label is treated as ground truth and no accuracy is reported without reviewed GT.
The Mapillary Ego Vehicle/Car Mount labels are external provenance only and never scientific
`control_and_ego_vehicle`. See `REPORT_ARTICLE1_CHECKPOINT2.md`.
