# Article 1 unknown pre-calibration

**This is a behavior ablation, not an accuracy evaluation.** No profile is selected as optimal before reviewed Article 1 GT.

| profile | mean unknown | min | max |
|---|---:|---:|---:|
| permissive | 0.930% | 0.200% | 2.564% |
| balanced | 1.883% | 0.434% | 4.425% |
| conservative | 3.631% | 0.682% | 13.629% |

Reason maps use a bitmask: 1 low probability, 2 low margin, 4 high normalized entropy, 8 unsupported dominant native class.
The balanced profile is used only to produce checkpoint-2 diagnostics.
