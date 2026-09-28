"""Step-Audio-2 adapter conventions.

An external adapter should expose the continuous outputs of the audio adaptor
that enter the language model. LTM-AE operates on those vectors, preserving
their temporal positions and all non-audio text-token embeddings.
"""

from __future__ import annotations

from typing import Any

from ltm_ae.adapters.injection import replace_masked_embeddings


def inject_step_audio_tokens(input_embeddings: Any, audio_slot_mask: Any, enhanced_tokens: Any) -> Any:
    """Replace Step-Audio-2 audio slots after external input construction."""

    return replace_masked_embeddings(input_embeddings, audio_slot_mask, enhanced_tokens)
