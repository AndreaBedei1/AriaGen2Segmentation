# Article 1 car–motorcycle annotation space

This directory is separate from the existing 60-frame automobile ground truth.

Target: at least 60 reviewed car frames and 60 reviewed motorcycle frames, preferably
matched/comparable by route segment. Stratify route geometry, traffic, road users, signage,
illumination and internal/external/mirror gaze. Do not split randomly by frame.

Grounded-SAM2 output is a proposal only. Annotation audit must count seeds retained,
modified and deleted, masks added, seed/final IoU and per-class statistics.

No motorcycle frames or reviewed symmetric cockpit annotations are currently available;
no placeholder ground truth has been created.
