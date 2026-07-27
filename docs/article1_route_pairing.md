# Article 1 route pairing

Car and motorcycle recordings must be paired by route position, not timestamp. Planned
evidence priority is GPS, SLAM trajectory, route progress, heading, landmarks and optional
manual anchors. Output pairs carry spatial distance, heading difference and matching
confidence. Splits operate on sessions, participants, route segments or complete matched
pairs to prevent near-frame leakage.
