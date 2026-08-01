# Article 1 — full-FOV and mirror refinement: results

Generated: 2026-08-01T22:22:14.796278+00:00

Branch: `feature/article1-motorcycle-fov-mirror-refinement`  
Base commit: `a9ee61bc77cdf61f364eb0eec98c2975fd933829`

## Summary

The geometry audit found a real and large field-of-view loss. The refinement built to recover it **did not improve the mirrors, and made the cockpit classes worse**. This report states that plainly, because the measurement is what it is.

## What was changed

Two things at once, which is the main methodological weakness of this run:

1. **Geometry** — the pipeline now runs in the source fisheye geometry instead of the pinhole-rectified one, recovering the 43.0% of valid field of view the audit showed was discarded.
2. **Cockpit configuration** — shared car+motorcycle mirror prompts, mirror area bounds tightened, and the geometric bottom prior cut from bottom-32%/conf-0.40/four-overridable-classes to bottom-14%/conf-0.15/`unknown`-only.

Because both changed together, the attribution below is partly inferential.

## Measured outcome, same 450-frame window

Refined figures are measured over valid pixels only; the baseline over its whole rectified frame.

| class | baseline share | refined share | baseline presence | refined presence | baseline lateral band | refined lateral band |
|---|---:|---:|---:|---:|---:|---:|
| `mirror` | 0.0195% | 0.0019% | 22% | 4% | 0.0056% | 0.0000% |
| `instrument_display` | 0.0202% | 0.0177% | 49% | 26% | 0.0271% | 0.0026% |
| `control_and_ego_vehicle` | 0.7700% | 0.0294% | 99% | 49% | 1.1270% | 0.0148% |
| `two_wheeler` | 0.0428% | 1.8209% | 20% | 73% | 0.0546% | 0.3436% |
| `road_surface` | 28.5883% | 26.3384% | 97% | 99% | 22.7149% | 22.2529% |
| `lane_marking` | 3.5202% | 3.1857% | 99% | 100% | 2.4418% | 2.4783% |
| `regulatory_road_marking` | 1.6453% | 1.4237% | 45% | 49% | 1.1536% | 0.9909% |
| `traffic_sign` | 0.2334% | 0.1567% | 78% | 76% | 0.2724% | 0.1552% |
| `vehicle` | 0.6712% | 0.6746% | 84% | 87% | 0.9724% | 0.9873% |
| `road_boundary_or_obstacle` | 7.1464% | 7.2970% | 100% | 100% | 10.4249% | 11.3476% |
| `other_environment` | 57.3341% | 59.0428% | 100% | 100% | 60.7970% | 61.4117% |

| quantity | baseline | refined |
|---|---:|---:|
| mean confidence | 0.9615 | 0.9502 |
| mean entropy | 0.0671 | 0.0860 |
| internal/external conflict | 0.3427 | 0.0174 |
| geometric prior share | 0.6997% | 0.0000% |
| switch rate per second | 1.1469 | 1.1728 |
| dense coverage | 1.0000 | 1.0000 |
| invalid class ids | 0 | 0 |

## What this says

### The mirrors got worse, not better

Mirror presence fell from 22% of frames to 4%, and mirror pixels by roughly a factor of ten. In the lateral band — the very region the recovered field of view was supposed to expose — mirror coverage went from a small non-zero value to **exactly zero**.

Recovering the field of view therefore did not recover the mirrors. The most likely reason is that the fisheye compresses its periphery hard: the mirrors are now present but small and strongly warped, which is worse for an open-vocabulary detector than being absent from a clean pinhole view. The tightened mirror area cap may also have rejected proposals the baseline accepted; that is not separable from the geometry change in this run.

### The cockpit collapse is partly by design and partly not

`control_and_ego_vehicle` fell from 0.7700% to 0.0294%. The geometric prior contributed 0.6997% in the baseline and 0% now, which accounts for most of the drop. That part is the intended correction: the prior was painting the bottom of the frame as cockpit on position alone. What is not intended is that `instrument_display` presence also fell from 49% to 26%: a real instrument cluster is being found less often, not just a false one being suppressed.

### A new failure mode appeared

`two_wheeler` rose from 0.0428% to 1.8209% and its presence from 20% to 73%. In the wider view the ego motorcycle's own visible structure is plausibly being labelled as an external two-wheeler. That would be a misclassification of ego parts, and it needs human inspection before the wider view is trusted.

### The external scene held up

Road surface, lane markings, road boundaries, vehicles and pedestrians are broadly comparable, with small losses in road surface, markings and traffic signs and small gains in boundaries, vehicles and traffic lights. Dense coverage stays at 100% and invalid class ids at zero in both runs. Confidence is slightly lower and entropy slightly higher in the refined run, consistent with a harder, wider, more distorted image.

## Limits

- No reviewed ground truth exists, so none of the above is accuracy. These are coverage, presence, stability and provenance diagnostics.
- Geometry and cockpit configuration changed together. A clean attribution needs a third run: full field of view with the frozen cockpit configuration unchanged.
- The two runs live in different geometries, so no per-pixel difference is computed.
- The full-recording mirror scan and the annotation supplement were not produced in this session; the negative mirror result makes both premature.

## Verdict

### D. The refinement is not reliable and the previous baseline must be kept.

The field-of-view loss is real and is now measured and documented: 43.0% of the valid view, 1,262,087 pixels, with 252 and 269 pixels lost on the left and right edges where the bar-end mirrors sit. That finding stands and is the useful outcome of this work.

But the refinement built on it does not deliver the improvement it was meant to. Mirrors are detected less often, the instrument cluster is detected less often, and a new ego-as-two-wheeler confusion appears. Verdict C would claim a partial correction that the numbers do not support.

The frozen baseline therefore remains the reference. The next attempt should keep the field of view but change one thing at a time, and should treat the peripheral compression of the fisheye as the problem to solve rather than assuming that showing the detector more pixels is enough.

