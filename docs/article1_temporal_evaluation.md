# Article 1 temporal evaluation

Before reviewed GT, the pipeline reports behavior only: switch rates, unknown rates,
recoveries, propagation age, thin continuity, resets, current-evidence-free persistence,
flow validity, raw/temporal difference and per-class appearances/disappearances.

Lower flicker is not necessarily higher accuracy. Lower unknown is not necessarily higher
accuracy. Persistence may preserve either a true class or an error.

Post-hoc gaze diagnostics sample static and temporal results only after segmentation has
finished. They report class, confidence, unknown, provenance and propagation age at gaze.
They do not feed segmentation and are not gaze accuracy without GT.

Future evaluation must compare static and temporal outputs independently against the same
reviewed masks, then compare their gaze classes against reviewed semantic-gaze targets.
Frames are not independent statistical replicates.
