"""Qwen2-Audio integration helpers.

The external adapter must extract projected audio tokens after the multimodal
projector, create the text/audio input sequence, and expose the boolean mask
for audio placeholder positions. This module keeps the differentiable token
replacement rule independent of a particular Transformers revision.
"""

from __future__ import annotations

from typing import Any

from ltm_ae.adapters.injection import replace_masked_embeddings


def qwen_audio_slot_mask(input_ids: Any, audio_token_id: int) -> Any:
    """Return the placeholder mask used to inject projected Qwen audio tokens."""

    return input_ids == int(audio_token_id)


def inject_qwen_audio_tokens(model: Any, input_ids: Any, enhanced_tokens: Any) -> Any:
    """Build decoder input embeddings with enhanced projected audio tokens."""

    embeddings = model.get_input_embeddings()(input_ids)
    mask = qwen_audio_slot_mask(input_ids, model.config.audio_token_id)
    return replace_masked_embeddings(embeddings, mask, enhanced_tokens)
