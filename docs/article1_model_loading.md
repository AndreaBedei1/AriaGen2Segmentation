# Mask2Former checkpoint loading

The local Mapillary checkpoint is a converted Hugging Face Mask2Former Swin-L model.
With PyTorch 2.13.0+cu130 and Transformers 5.14.1, loading reports:

- missing `model.pixel_level_module.encoder.swin.layernorm.weight`;
- missing `model.pixel_level_module.encoder.swin.layernorm.bias`;
- 24 unexpected legacy `relative_position_index` buffers.

Transformers v5 represents the converted backbone through `SwinBackbone`. Its internal
`SwinModel` applies the missing final LayerNorm only to `last_hidden_state`, while
`SwinBackbone.forward` constructs Mask2Former feature maps from pre-final
`reshaped_hidden_states` and trained `hidden_states_norms`. The absent final norm is not on
the segmentation path. Legacy relative-position indices are deterministic derived buffers,
not learned parameters.

The verified loader:

1. fails closed unless the missing-key set is exactly those two keys;
2. fails on shape mismatch or loader errors;
3. explicitly initializes the unused final norm to identity;
4. records all unexpected keys.

Two seeded loads produced bit-identical class logits, mask logits and predictions.
Changing the final norm to weight 7 and bias 13 changed class logits by exactly 0.0.
See `reports/article1_mask2former_checkpoint_diagnosis.{json,md}` for hashes and evidence.
