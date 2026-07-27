# Article 1 evaluation

External segmentation: macro mIoU and per-class IoU, precision/recall, boundary F1, thin
marking F1, unknown rate and temporal stability. Cockpit evaluation is reported overall and
separately for car/motorcycle, including cross-domain experiments.

Semantic gaze evaluation reports top-1/top-2, GT probability, unknown-at-gaze, raw/smoothed
accuracy, flicker, switch error, mirror and instrument accuracy. Scientific comparisons use
paired estimates, confidence intervals, effect sizes and mixed-effects models when sample
size permits; frames are not independent replicates.

Pre-GT checkpoint-2 unknown-rate, pixel-count, agreement and timing tables are behavioral
diagnostics only. Profile selection and threshold claims require reviewed external GT.
Cockpit splits are grouped by participant/session/matched pair/route segment and reject
near-frame leakage; metrics must always be reported separately for car and motorcycle.
