# Solid-line crossing review

One row per candidate in `review_form.csv`. Fill `human_label` and
`human_notes`; leave `human_label` empty for anything you cannot decide from the
material provided, and say why in the notes.

## Permitted values for `human_label`

| value | meaning |
|---|---|
| `compliant` | the manoeuvre was lawful (broken line, junction turn, directed around an obstacle) |
| `noncompliant` | a continuous line was crossed with no lawful reason visible |
| `uncertain` | the evidence genuinely does not settle it |
| `not_evaluable` | the material does not show what is needed (no clip, marking not visible, camera obscured) |

## Before labelling `noncompliant`

Check all of these, because each makes crossing a continuous line lawful:

* an obstacle, parked vehicle, roadworks or an emergency vehicle;
* a junction, private access or roundabout inside the manoeuvre;
* a police or marshal direction;
* the line being a different marking altogether (edge line, hatching, bus lane).

## What the automatic state means

`state` is the detector's four-way candidate label. It is **not** a verdict and
`candidate_noncompliant` in particular means only that the evidence pattern is
consistent with crossing a continuous line. The detector cannot see obstacles,
signage or instructions.

## Positioning caveat

Read `evidence_median_gps_accuracy_m` on every row. Where it exceeds the lateral
displacement a lane change produces, the crossing cannot be resolved from the
track at all and the honest label is `not_evaluable` however suggestive the
numbers look.
