# Article 1 auto–moto — checkpoint 2 stop gate

Date: 2026-07-27  
Branch: `feature/auto-moto-semantic-gaze`

## Outcome

The external auto stop gate is complete on the existing 180–210 s clip only. No complete
VRS run, training, statistics or cross-domain experiment was started. `val10`, `run200`,
previous A–D outputs and prior Article 1 results were not modified.

The clip contains 60 identical rectified frames (source indices 1805–2100, every fifth RGB
frame), timestamps 1235764312058–1265262269475 ns, rendered at 2 fps for 30.0 s.

## Mask2Former checkpoint gate

Transformers 5.14 reports two missing final Swin layernorm tensors and 24 legacy
`relative_position_index` buffers. The missing normalization is not on the feature path used
by Mask2Former: two seeded loads, class logits, mask logits, final predictions and parameters
were identical; perturbing that normalization also changed class logits by exactly 0.0.
The loader now accepts only this exact case, sets it to identity and fails closed otherwise.
Full versions, tensor names, hashes and evidence are in
`reports/article1_mask2former_checkpoint_diagnosis.md`.

## Scientific corrections and diagnostics

- Mapillary Ego Vehicle/Car Mount no longer become scientific
  `control_and_ego_vehicle`; they map to `other_environment` with a binary ego-region flag.
- Native probabilities are aggregated before macro decisions. Top-1/top-2, confidence,
  margin, normalized entropy, unknown reason bitmask and native provenance are retained.
- Permissive/balanced/conservative smoke ablation unknown rates were respectively 0.930%,
  1.883% and 3.631%. These are not accuracy results and no profile is selected before GT.
- On all 60 frames, checkpoint-2 balanced unknown was 1.064% versus checkpoint-1 0.879%.
- Raw thin candidates averaged 26,112 pixels/frame; road-supported filtered layers averaged
  6,040. This reduction is not evidence of improved accuracy.
- Visual QA covered 10 lane, 5 regulatory, 5 reflection, 5 ego and 5 vehicle frames. Filtering
  removes many cockpit/reflection responses but also some true fragments; regulatory markings
  still need reviewed GT calibration.

## Profiling

P0–P3 were measured on the same ten frames. P0 totaled 17.840 s/frame; only 0.100 s was model
forward, while native-result serialization dominated. P1–P3 reduced forward time but did not
reduce total time because full-resolution aggregation/export remained. P3 is gaze-assisted
and cannot be used for unbiased global comparison. No production profile is selected.
See `reports/article1_external_profile.md`.

## Deliverables

Six H.264 files were generated and reopened at first/middle/last frame:

- `output/article1/checkpoint2_videos_30s/01_external_checkpoint1.mp4`
- `output/article1/checkpoint2_videos_30s/02_external_checkpoint2.mp4`
- `output/article1/checkpoint2_videos_30s/03_thin_raw.mp4`
- `output/article1/checkpoint2_videos_30s/04_thin_filtered.mp4`
- `output/article1/checkpoint2_videos_30s/05_unknown_reason.mp4`
- `output/article1/checkpoint2_videos_30s/06_checkpoint1_vs_checkpoint2.mp4`

All have 60 frames, 2 fps and 30.0 s. Videos 1–5 are 2432×1512; comparison video 6 is
3024×752. Preview PNGs, a 30-frame contact sheet and the render manifest are beside them.

External-dev40 and cockpit120 CVAT packages are prepared under
`validation/article1_auto_moto/`. They contain 20 and 60 real car images respectively,
correct label schemas and no fabricated masks. Motorcycle slots remain empty.

SegFormer-B2 infrastructure now includes a joint four-class output contract with probability,
confidence and normalized entropy maps; reviewed-annotation and two-domain gates; a
domain-balanced sampler; weighted loss; deterministic checkpoint/resume; grouped split
leakage checks; and per-domain IoU. Training remains correctly blocked.

## Remaining blockers

- No motorcycle VRS was supplied, so balanced packages, pairing and cross-domain results
  cannot be completed.
- No manually reviewed external or cockpit GT exists, so no accuracy, calibration winner,
  model winner or paper-level statistical claim is valid.
- SegFormer training and fusion must wait for reviewed car+motorcycle annotations.
