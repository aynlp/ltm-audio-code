# External Adapter Contract

An external adapter connects the generic PCA-memory code to a locally installed
audio-language model. The adapter and its model files are not bundled here.

## Required interface

Subclass `ltm_ae.adapters.AudioTokenAdapter` and implement:

```python
extract(audio_paths, prompts) -> TokenBatch
generate_from_tokens(token_batch, enhanced_tokens, prompts, max_new_tokens=...) -> list[str]
```

`extract` returns the continuous audio token tensor of shape
`[batch, time, dimension]` together with opaque model inputs needed for
reinjection. `generate_from_tokens` must use the enhanced token tensor in the
same positions that the original model used for audio tokens.

`generate_native(audio_paths, prompts, max_new_tokens=...)` is optional. An
adapter that implements it can run the absent-category control's
`--verify-native-baseline` preflight, which compares the native decoding path
with decoding after re-injecting unmodified extracted tokens.

## Non-negotiable invariants

- Extract and inject at the same model interface.
- Use a separate memory for every model interface and target category.
- Preserve batch order, audio token count, and audio token positions.
- Preserve all text-token embeddings.
- Keep the external model in evaluation mode and freeze all of its parameters.
- Validate that a memory's `model_id` metadata matches the adapter's `model_id`.
- For paired controls, call `extract` once per audio/prompt pair, then decode
  both the original and enhanced tokens from that same token batch.

`replace_masked_embeddings` in `ltm_ae.adapters.injection` performs a
differentiable replacement without in-place writes. It checks that every
example exposes exactly as many audio placeholders as enhanced tokens.

## Interface locations

| Model family | Memory interface |
| --- | --- |
| Qwen2-Audio | projected audio tokens after the multimodal projector |
| Kimi-Audio | fused audio representation after the audio adaptor |
| Step-Audio-2 | continuous outputs after the audio adaptor |

The supplied helpers identify the replacement boundary but do not hard-code a
model checkout, model revision, device, token length, embedding dimension, or
third-party implementation detail.

## Gate factory

For speech experiments, the external factory used by `train_token_gate.py`
must return `ltm_ae.gate.training.GateTrainingBundle`. It must:

1. create the `TokenGate` and its optimizer;
2. freeze every external model module before returning it;
3. yield batches with `mixed_tokens`, `pca_tokens`, `clean_tokens`, and an
   optional boolean `token_mask`; and
4. expose a teacher-forcing callback that maps gated audio tokens to a scalar
   transcription loss through the frozen decoder.

The trainer refuses to run if any supplied external module has trainable
parameters.
