"""Learned token-level gate for the speech-transcription extension."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

try:  # Keep the NumPy-only PCA workflow usable without torch.
    import torch
    import torch.nn as nn
except ImportError:  # pragma: no cover - exercised only in torch-free environments
    torch = None
    nn = None


def _require_torch() -> None:
    if torch is None:
        raise ImportError("Token gating requires the optional torch dependency.")


if nn is not None:

    class TokenGate(nn.Module):
        """Predict one `[0, 1]` memory-use weight per temporal audio token.

        The gate receives the original mixture token and its calibrated PCA
        correction. It uses layer normalization, a 256-unit hidden layer, GELU,
        and a scalar sigmoid output, matching the paper's speech extension.
        """

        def __init__(self, hidden_size: int, bottleneck: int = 256) -> None:
            super().__init__()
            if hidden_size < 1 or bottleneck < 1:
                raise ValueError("hidden_size and bottleneck must be positive.")
            self.norm = nn.LayerNorm(hidden_size * 2)
            self.network = nn.Sequential(
                nn.Linear(hidden_size * 2, bottleneck),
                nn.GELU(),
                nn.Linear(bottleneck, 1),
            )
            nn.init.zeros_(self.network[-1].bias)

        def forward(self, mixed_tokens: Any, pca_tokens: Any) -> Any:
            if mixed_tokens.shape != pca_tokens.shape or mixed_tokens.ndim != 3:
                raise ValueError("mixed_tokens and pca_tokens must both have shape [batch, time, dimension].")
            correction = pca_tokens.float() - mixed_tokens.float()
            features = torch.cat((mixed_tokens.float(), correction), dim=-1)
            return torch.sigmoid(self.network(self.norm(features)))

        def apply(self, mixed_tokens: Any, pca_tokens: Any) -> tuple[Any, Any]:
            """Return gated enhanced tokens and scalar gate values per position."""

            values = self(mixed_tokens, pca_tokens)
            enhanced = mixed_tokens.float() + values * (pca_tokens.float() - mixed_tokens.float())
            return enhanced, values

else:

    class TokenGate:  # type: ignore[no-redef]
        """Placeholder that fails clearly when optional torch is absent."""

        def __init__(self, *_: Any, **__: Any) -> None:
            _require_torch()


@dataclass(frozen=True)
class GateLoss:
    """Detached losses reported by a gate optimization step."""

    total: float
    token_mse: float
    transcription: float | None


def freeze_module(module: Any) -> None:
    """Freeze an external audio-language model before gate-only optimization."""

    _require_torch()
    for parameter in module.parameters():
        parameter.requires_grad_(False)
    module.eval()


def _masked_mse(prediction: Any, target: Any, token_mask: Any | None) -> Any:
    _require_torch()
    if prediction.shape != target.shape:
        raise ValueError("prediction and target must have equal shape.")
    squared = (prediction.float() - target.float()).square()
    if token_mask is None:
        return squared.mean()
    if token_mask.shape != prediction.shape[:2]:
        raise ValueError("token_mask must have shape [batch, time].")
    weights = token_mask.to(device=prediction.device, dtype=prediction.dtype).unsqueeze(-1)
    denominator = weights.sum() * prediction.shape[-1]
    if float(denominator.detach().cpu()) == 0.0:
        raise ValueError("token_mask contains no valid positions.")
    return (squared * weights).sum() / denominator


def gate_training_loss(
    gate: Any,
    mixed_tokens: Any,
    pca_tokens: Any,
    clean_tokens: Any,
    *,
    token_mask: Any | None = None,
    transcription_loss: Callable[[Any], Any] | None = None,
    lambda_token: float = 1.0,
    lambda_text: float = 1.0,
) -> tuple[Any, GateLoss, Any]:
    """Compute the token MSE plus optional frozen-decoder transcription loss.

    ``transcription_loss`` receives the gated tokens and must return a scalar
    tensor produced by a frozen external decoder under teacher forcing. It is
    deliberately injected as a callback so model source, model paths, and data
    remain external to this anonymous release.
    """

    _require_torch()
    if lambda_token < 0.0 or lambda_text < 0.0:
        raise ValueError("Loss weights must be non-negative.")
    enhanced_tokens, gate_values = gate.apply(mixed_tokens, pca_tokens)
    token_loss = _masked_mse(enhanced_tokens, clean_tokens, token_mask)
    text_loss = transcription_loss(enhanced_tokens) if transcription_loss is not None else None
    if text_loss is not None and text_loss.ndim != 0:
        raise ValueError("transcription_loss must return a scalar tensor.")
    total = lambda_token * token_loss
    if text_loss is not None:
        total = total + lambda_text * text_loss
    report = GateLoss(
        total=float(total.detach().cpu()),
        token_mse=float(token_loss.detach().cpu()),
        transcription=float(text_loss.detach().cpu()) if text_loss is not None else None,
    )
    return total, report, gate_values


def assert_only_gate_trainable(gate: Any, external_modules: Mapping[str, Any]) -> None:
    """Fail fast if a frozen model accidentally exposes trainable parameters."""

    _require_torch()
    if not any(parameter.requires_grad for parameter in gate.parameters()):
        raise ValueError("The gate has no trainable parameters.")
    offenders = [
        name
        for name, module in external_modules.items()
        if any(parameter.requires_grad for parameter in module.parameters())
    ]
    if offenders:
        raise ValueError(f"External modules are not frozen: {', '.join(sorted(offenders))}.")
