# Article 1 semantic gaze

The gaze output will retain pixel class, disc distribution, Gaussian-weighted probability
mass, top-1/top-2, entropy, confidence, unknown probability and boundary distance.
Temporal stabilization operates on probability vectors (EMA plus hysteresis, margin and
minimum dwell), never majority voting alone. Raw and smoothed probabilities/classes are
always stored side by side.
