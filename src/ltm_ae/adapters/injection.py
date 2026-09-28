"""Framework-level helpers for safe replacement of audio embedding slots."""

from __future__ import annotations

from typing import Any


def replace_masked_embeddings(input_embeddings: Any, audio_slot_mask: Any, audio_tokens: Any) -> Any:
    """Return embeddings with audio slots replaced without in-place autograd writes.

    The function requires a common number of audio slots per batch item. This
    is intentional: adapters should bucket or pad audio examples before calling
    it, rather than silently mixing incompatible token lengths.
    """

    try:
        import torch
    except ImportError as error:  # pragma: no cover - depends on optional torch install
        raise ImportError("Audio-token injection requires the optional torch dependency.") from error

    if input_embeddings.ndim != 3:
        raise ValueError("input_embeddings must have shape [batch, sequence, dimension].")
    if audio_slot_mask.shape != input_embeddings.shape[:2]:
        raise ValueError("audio_slot_mask must have shape [batch, sequence].")
    if audio_tokens.ndim != 3:
        raise ValueError("audio_tokens must have shape [batch, audio_time, dimension].")
    if audio_tokens.shape[0] != input_embeddings.shape[0] or audio_tokens.shape[2] != input_embeddings.shape[2]:
        raise ValueError("audio_tokens batch or embedding dimension is incompatible with input embeddings.")

    mask = audio_slot_mask.to(device=input_embeddings.device, dtype=torch.bool)
    slots_per_item = mask.sum(dim=1)
    expected = int(audio_tokens.shape[1])
    if not torch.all(slots_per_item == expected):
        raise ValueError(
            "Each batch item must expose exactly audio_tokens.shape[1] audio slots; "
            f"found {slots_per_item.detach().cpu().tolist()} for expected {expected}."
        )

    positions = mask.nonzero(as_tuple=False)
    flat_positions = positions[:, 0] * input_embeddings.shape[1] + positions[:, 1]
    replacement = torch.zeros_like(input_embeddings).reshape(-1, input_embeddings.shape[-1])
    source = audio_tokens.to(device=input_embeddings.device, dtype=input_embeddings.dtype).reshape(
        -1, input_embeddings.shape[-1]
    )
    replacement = replacement.scatter(
        0,
        flat_positions[:, None].expand(-1, source.shape[-1]),
        source,
    ).reshape_as(input_embeddings)
    return input_embeddings * (~mask).unsqueeze(-1) + replacement
