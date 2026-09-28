"""Kimi-Audio adapter conventions.

An external adapter should return the fused semantic-token and continuous
speech-feature sequence *after* the model's audio adaptor. The output must be
the sequence directly consumed by its frozen language model; apply LTM-AE at
that point and retain the original token count and ordering.
"""

from __future__ import annotations

from typing import Any

from ltm_ae.adapters.injection import replace_masked_embeddings


def inject_kimi_audio_tokens(input_embeddings: Any, audio_slot_mask: Any, enhanced_tokens: Any) -> Any:
    """Replace Kimi audio slots after external preprocessing has constructed them."""

    return replace_masked_embeddings(input_embeddings, audio_slot_mask, enhanced_tokens)
