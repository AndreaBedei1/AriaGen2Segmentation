# Article 1 Mask2Former checkpoint diagnosis

- Checkpoint: `/home/andreabedei/Scrivania/AndreaSegmentazione/weights/mask2former-mapillary-semantic`
- Architecture: `['Mask2FormerForUniversalSegmentation']`
- PyTorch: `2.13.0+cu130`; Transformers: `5.14.1`
- Seed: `2026`; device: `cuda`
- Missing keys: `['model.pixel_level_module.encoder.swin.layernorm.bias', 'model.pixel_level_module.encoder.swin.layernorm.weight']`
- Unexpected keys: 24 legacy relative-position buffers
- Solution: explicit identity initialization of the final unused Swin LayerNorm; fail closed on any other mismatch.

## Determinism

- Parameters equal across loads: **True**
- Class logits exactly equal: **True**
- Mask logits exactly equal: **True**
- Predictions equal: **True**
- Max class-logit difference after setting the final LayerNorm to weight=7/bias=13:
  **0.0**

Gate passed: **True**.

The final `SwinModel.layernorm` is applied only to `last_hidden_state`; `SwinBackbone`
constructs Mask2Former feature maps from pre-final `reshaped_hidden_states` and its trained
`hidden_states_norms`. The two absent parameters are therefore not used by segmentation.
All trained backbone blocks, stage norms, pixel decoder and classifier weights load.
