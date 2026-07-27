# Unknown calibration

Unknown is decided from top-1 probability, top-1/top-2 margin, normalized entropy and an
unsupported dominant native class. Reason maps are bitmasks: 1 low probability, 2 low
margin, 4 high entropy and 8 unsupported native. Multiple reasons may coexist.

Permissive, balanced and conservative profiles are behavior probes only. Their pre-GT
unknown rates must not be interpreted as accuracy or selected to reach a desired percentage.
Final thresholds require the reviewed external development set, with particular attention
to unknown-at-gaze and confusion with known `other_environment`.
