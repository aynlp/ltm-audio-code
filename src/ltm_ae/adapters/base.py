"""Contracts for connecting LTM-AE to an external audio-language model.

The project intentionally does not vendor model source code or weights. An
adapter owns model-specific preprocessing and specifies where continuous audio
tokens enter the frozen language backbone.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class TokenBatch:
    """Continuous audio tokens and opaque model inputs needed for reinjection.

    ``tokens`` must have `[batch, time, dimension]` layout. It may be a NumPy
    array or a torch tensor; the adapter retains framework-specific objects in
    ``model_inputs`` and validates their layout before decoding.
    """

    tokens: Any
    model_inputs: Any
    metadata: dict[str, Any] = field(default_factory=dict)


class AudioTokenAdapter(ABC):
    """Model-specific token extraction and frozen-backbone decoding interface."""

    model_id: str

    @abstractmethod
    def extract(self, audio_paths: Sequence[str | Path], prompts: Sequence[str]) -> TokenBatch:
        """Return pre-language-backbone audio tokens for an equal-sized batch."""

    @abstractmethod
    def generate_from_tokens(
        self,
        token_batch: TokenBatch,
        enhanced_tokens: Any,
        prompts: Sequence[str],
        *,
        max_new_tokens: int,
    ) -> list[str]:
        """Inject enhanced audio tokens and decode with all backbone weights frozen."""

    def validate_memory_metadata(self, metadata: dict[str, object]) -> None:
        """Reject a memory explicitly built for a different model interface."""

        memory_model = metadata.get("model_id")
        if memory_model is not None and str(memory_model) != self.model_id:
            raise ValueError(f"Memory belongs to model_id={memory_model!r}, not adapter {self.model_id!r}.")

    def generate_native(
        self,
        audio_paths: Sequence[str | Path],
        prompts: Sequence[str],
        *,
        max_new_tokens: int,
    ) -> list[str] | None:
        """Optionally decode through the model's untouched native audio path.

        Adapters that expose this method can use it for a preflight check that
        decoding unmodified injected tokens agrees with the native model path.
        Returning ``None`` declares that the adapter cannot perform this check.
        """

        return None
