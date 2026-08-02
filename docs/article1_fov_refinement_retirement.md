# Retirement of the full-FOV / mirror refinement

The experimental branch `feature/article1-motorcycle-fov-mirror-refinement` is
retired. It is not part of the project any more and nothing in Article 1 depends
on it.

## Why

The branch tried to widen the retained RGB field of view through the model
transform chain and to refine the shared `mirror` / minimal-cockpit proposals. Its
own validation commit (`92a448f`, *"Validate the full-FOV refinement against the
frozen baseline: it regresses"*) recorded that the result was **worse** than the
frozen baseline. A refinement that regresses against the baseline it is measured
against cannot be carried forward, so it is dropped rather than patched.

## Verification performed before deletion

| check | result |
|---|---|
| merged into the baseline? | no — `git branch --contains` listed only the full-FOV branch itself for `97d165d`, `ab00789`, `e300cee`, `92a448f` |
| baseline still intact? | yes — `feature/article1-motorcycle-ingestion` still at `a9ee61b` |
| needed by another branch? | no — no other local or remote branch contained any of the four commits |
| uncommitted local work to preserve? | no — the working tree was clean at retirement time |

## The active baseline

The single active semantic-camera baseline is now:

* branch `feature/article1-motorcycle-ingestion`
* commit `a9ee61bc77cdf61f364eb0eec98c2975fd933829`

The behaviour-analysis work starts exactly from that commit, on
`feature/article1-multimodal-behavior-analysis`. No code, configuration or report
was imported from the full-FOV branch.

## What was deleted

* the local branch `feature/article1-motorcycle-fov-mirror-refinement`;
* `output/article1/motorcycle_fov_mirror_refinement_30s/` (7.2 GB of local
  refinement output, including `semantic_camera_moto_full_fov_final.mp4`);
* the stale `aria_drive_seg/geometry/` bytecode cache left behind by the branch
  switch.

`reports/article1_motorcycle_fov_mirror_refinement/` and the tracked full-FOV
sources (`aria_drive_seg/geometry/`, `configs/article1/semantic_camera_full_fov.yaml`,
the four `scripts/*fov*|*geometry*` entry points and the two geometry test
modules) disappeared with the branch checkout, because they existed only on it.

## What was deliberately NOT deleted

* both VRS recordings;
* the frozen baseline runs `outputs/article1/auto_temporal_180_210/` (car) and
  `output/article1/motorcycle_baseline_30s/` (motorcycle), **including the valid
  baseline motorcycle video** `semantic_camera_moto_final.mp4`;
* every annotation package, validation set and ingestion export.

## Manual git steps still required

Neither of these could be done from this environment: `git push` has no
credential helper here and the GitHub CLI is not installed.

```
fatal: could not read Username for 'https://github.com': Device o indirizzo non esistente
```

Run both from an authenticated checkout.

**1. Delete the retired remote branch.**

```bash
git push origin --delete feature/article1-motorcycle-fov-mirror-refinement
```

Equivalent with the GitHub CLI:

```bash
gh api -X DELETE repos/AndreaBedei1/AriaGen2Segmentation/git/refs/heads/feature/article1-motorcycle-fov-mirror-refinement
```

Until that runs, `origin/feature/article1-motorcycle-fov-mirror-refinement` still
exists on GitHub. It is inert — no local branch tracks it and nothing builds on
it.

**2. Push the behaviour-analysis branch.**

```bash
git push -u origin feature/article1-multimodal-behavior-analysis
```

The branch is ready to push: the full test suite passes (480 passed, 15 skipped)
and the working tree is clean. It must **not** be merged into `main`.
