# Evaluation

Evaluation has **two levels**: metrics that need **no ground truth** (produced by `analyze`),
and metrics that need **human-labeled ground truth** (the scoring code and annotation tooling
are implemented, but **no ground-truth data has been created yet**). **Because no ground truth
exists, no accuracy is claimed anywhere** — every number produced today is a consistency or
descriptive statistic, and the GT scorer refuses to fabricate accuracy when no labels are
present.

---

## Level 1 — no-ground-truth metrics (§17.1, implemented)

`analyze` writes `comparison/metrics/metrics_no_gt.json` and a readable
`reports/analysis_summary.md`. The metrics fall into three families.

### Per-method (`compare.per_method_metrics`)

Computed over each method's `canonical_masks/`:

| Metric | Definition |
|---|---|
| `coverage_mean` | mean fraction of pixels assigned a class (id > 0) |
| `unknown_frac_mean` | mean fraction of `unknown` (id 0) pixels |
| `classes_per_frame_mean` | mean number of distinct non-unknown classes per frame |
| `mean_inference_ms` / `fps` | mean per-frame inference time and its reciprocal |

### Between methods (`compare.method_agreement`)

Restricted to the **common eval taxonomy** (road classes with `eval: true` that both methods
can produce):

| Metric | Definition |
|---|---|
| `pixelwise_agreement_common` | of pixels where **both** methods assigned a common eval class, the fraction with the **same** class |
| `mean_iou_between_methods` | mean over eval classes of IoU(method1 == c, method2 == c) |
| `per_class_iou_between_methods` | that IoU per class (null where a class is absent in both) |

> These measure **consistency, not accuracy.** Two models agreeing does not mean either is
> right; the JSON carries this note inline.

### Gaze (`compare.gaze_temporal_distribution`, `compare.gaze_agreement`, `analyze.gaze_labels`)

Per method, and between methods, using `gaze/aligned_gaze.parquet`:

| Metric | Definition |
|---|---|
| `valid_frame_frac` | fraction of frames with a valid projected gaze |
| `gaze_on_unknown_frac` | fraction of frames whose gaze lands on `unknown` |
| `time_per_class_s` | observed time attributed to each gaze class (Δt between sampled frames) |
| `class_temporal_stability` | fraction of consecutive frames with the same gaze class |
| `gaze_class_agreement_frac` / `disagreement_frac` | between methods, over frames both-valid, on the common eval taxonomy |

Per-frame gaze labels (`comparison/gaze/gaze_labels_<method>.parquet`) additionally record, for
each frame's gaze pixel: the **class at the exact pixel**, the **dominant class in a disc**
(radius `gaze.disc_radius_px`, default 40), the **class distribution under a 2-D gaussian**
(`gaze.gauss_sigma_px` default 60, or derived from `gauss_sigma_deg` via the rectified focal)
with its top class and unknown weight, the observed-class **confidence**, the **distance to the
class-region boundary**, and `is_unknown` / `is_unsupported` flags.

---

## Example numbers — 10-frame validation subset (`val10`)

**These are from `val10`, a 10-frame validation subset sampled across the recording — NOT the
full 3762-frame recording, and NOT accuracy.** They exist to sanity-check the pipeline
end-to-end. (The 10 frames are spread roughly every ~37.6 s, so `time_per_class_s` values are
coarse and dominated by the sampling interval.)

### Per-method (`val10`)

| Metric | Grounded-SAM2 | Mask2Former-Mapillary |
|---|--:|--:|
| Frames | 10 | 10 |
| Coverage | 34.5% (sparse, prompted) | 99.6% (dense partition) |
| Unknown | 65.5% | 0.4% |
| Classes / frame | 11.3 | 17.3 |
| ms / frame | 1902.81 | 156.69 |
| fps | 0.53 | 6.38 |

The coverage split is expected and by design: Method 1 is a sparse prompted method, Method 2 a
dense partition. Method 2 labels the ego cabin as `other_cockpit` (via Mapillary "Ego Vehicle")
rather than "car".

### Between methods (`val10`, common eval taxonomy)

| Metric | Value |
|---|--:|
| Pixelwise agreement (common) | 47.7% |
| Mean IoU between methods | 0.08 |
| Gaze-class agreement / disagreement (9 both-valid frames) | 33% / 67% |

Highest per-class IoU-between-methods on `val10`: `traffic_light` 0.75, `car` 0.45, `curb`
0.28, `pole` 0.18, `sidewalk` 0.12 — many classes are 0.0 on only 10 frames. The low mean IoU
and modest agreement reflect that a sparse prompted method and a dense classifier partition the
scene very differently; **this is consistency information, not an accuracy verdict.**

### Gaze time-on-class (`val10`)

Both methods report `valid_frame_frac` 90% and `class_temporal_stability` 22.2%.
`gaze_on_unknown_frac` is 20% (Grounded-SAM2) vs 10% (Mask2Former) — consistent with
Method 1's larger unknown region. On this subset Grounded-SAM2 attributes the most gaze time to
`windshield` and `car`, Mask2Former to `car` and `vegetation`.

---

## Level 2 — with-ground-truth evaluation (§17.2, implemented, awaiting labels)

The infrastructure to produce and score ground truth **exists in the repo**; what is missing is
the human-labeled data. No labels → **no accuracy metric is reported today**, and the scorer is
built to refuse to invent any.

### Sampling + annotation package (`scripts/prepare_annotation.py`)

```bash
python scripts/prepare_annotation.py --input output --num 60
```

Builds `output/annotation/` by drawing a **stratified sample across the whole recording**: the
frames are split into `--num` time bins (default 60, in the 50–100 range) and, per bin, the
frame **richest in rare / safety-critical classes** is chosen (person, rider, bicycle,
motorcycle, traffic_light, traffic_sign, crosswalk, lane_marking, truck, bus — measured on the
seed method's masks). It exports:

- `images/` — the rectified frames to label;
- `preannotation/*_id.png` + `*_color.png` — the seed method's canonical masks (default
  `oneformer_mapillary`) as an **editable starting point**;
- `labels_cvat.json` — the **CVAT** segmentation label schema (canonical classes + palette colors);
- `manifest.json` + `README.md` — the sampled frame list, provenance, and annotation instructions.

Two annotation routes are supported: **CVAT** (upload `images/`, import the label schema,
optionally seed from the pre-annotations, export as *Segmentation mask 1.1* / *CVAT for images*,
convert to canonical uint16 id-PNGs `frame_XXXXXX.png`), or **local correction** of the
`preannotation/*_id.png` masks saved into a `gt/` directory. Labeling stays local, consistent
with the [privacy guarantee](privacy.md).

### Scoring (`scripts/evaluate_gt.py` → `analyze/metrics_gt.py`)

```bash
python scripts/evaluate_gt.py --input output --gt output/gt
```

With ground-truth id-masks present it writes `comparison/metrics/metrics_gt.json` for every
method; **with an empty `--gt` it prints
`NO GROUND TRUTH YET — infrastructure ready, no accuracy claimed` and exits 0** without
fabricating numbers. Scoring is restricted to `unknown` + the common eval taxonomy, so both
methods are comparable. Implemented metrics (`metrics_gt.py`):

| Metric (JSON key) | Definition |
|---|---|
| `mIoU_eval` | mean IoU over eval classes (from a confusion matrix) |
| `per_class_iou` | IoU per class |
| `pixel_accuracy` | global accuracy — **`unknown` is a scored class, so `unknown` predictions count as errors** |
| `mean_class_accuracy` | mean per-class recall over eval classes |
| `per_class_precision` / `per_class_recall` | per class (covers rare / safety-critical classes) |
| `boundary_f1` | per-class boundary F1 with a tolerance dilation (edge quality) |
| `accuracy_on_assigned` | accuracy over pixels the method **committed** to (excludes `pred == unknown`) — separates "wrong" from "abstained", important for the sparse Method 1 |
| `gaze_class_accuracy` | predicted vs GT class at the gaze pixel (`gaze_frames_scored` counts) |

Reporting both `accuracy_on_assigned` and the unknown-as-error `pixel_accuracy` is what makes
the comparison fair: Method 1 (sparse) and Method 2 (dense) trade coverage against commitment,
and a single number would flatter one and punish the other.

**Still future work (not yet in `metrics_gt.py`):** gaussian-gaze-**weighted** segmentation
metrics (accuracy weighted by the gaze gaussian, i.e. how right the method is *where the driver
looked*) — today's gaze scoring is the single-pixel `gaze_class_accuracy`. Per-method coverage
is available from the Level-1 no-GT metrics.

---

## How to read all of this

- Everything currently produced is **consistency / descriptive**, not accuracy.
- The two methods are **structurally different** (sparse open-vocab vs dense fixed-taxonomy);
  low agreement is expected and is not evidence that either is wrong.
- Accuracy claims wait for Level 2. Until then, treat gaze-on-class, coverage, and agreement as
  diagnostics, and see the [declared limitations](../README.md#declared-limitations).
