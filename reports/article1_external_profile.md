# Article 1 external profiling

**Pre-GT speed/behavior profile; not an accuracy evaluation.**

| profile | total s/frame | forward s | VRAM MB | unknown | thin px | gaze-assisted |
|---|---:|---:|---:|---:|---:|---|
| P0_current | 17.840 | 0.100 | 1049.4 | 2.11% | 6169 | False |
| P1_reduced | 21.766 | 0.061 | 1314.7 | 2.03% | 7968 | False |
| P2_reduced_road_crop | 22.140 | 0.059 | 2621.9 | 14.11% | 10292 | False |
| P3_reduced_gaze_crop | 21.708 | 0.058 | 1314.3 | 1.94% | 9353 | True |

P2 uses a high-resolution lower road crop. P3 is explicitly gaze-assisted and is not eligible for unbiased global comparison. No final profile is selected.
