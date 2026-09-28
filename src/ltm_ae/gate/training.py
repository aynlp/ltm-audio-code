"""Generic gate-only training loop with an external frozen-decoder loss hook."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from typing import Any

from ltm_ae.gate.token_gate import GateLoss, assert_only_gate_trainable, gate_training_loss

Batch = Mapping[str, Any]
TextLoss = Callable[[Any, Batch], Any]


@dataclass
class GateTrainingBundle:
    """Caller-owned model/data bindings for token-gate optimization.

    A factory outside this repository creates this object after loading an
    external audio-language model. The factory must freeze that model and give
    ``transcription_loss`` a differentiable teacher-forcing implementation.
    """

    gate: Any
    optimizer: Any
    train_batches: Callable[[], Iterable[Batch]]
    external_modules: Mapping[str, Any]
    transcription_loss: TextLoss | None = None
    validation_batches: Callable[[], Iterable[Batch]] | None = None


def _required(batch: Batch, name: str) -> Any:
    if name not in batch:
        raise ValueError(f"Gate training batch is missing {name!r}.")
    return batch[name]


def run_epoch(
    bundle: GateTrainingBundle,
    batches: Iterable[Batch],
    *,
    lambda_token: float,
    lambda_text: float,
    train: bool,
) -> GateLoss:
    """Run one train or validation epoch and return example-weighted averages."""

    assert_only_gate_trainable(bundle.gate, bundle.external_modules)
    bundle.gate.train(mode=train)
    total = token_mse = transcription_sum = 0.0
    transcription_count = examples = 0
    for batch in batches:
        mixed = _required(batch, "mixed_tokens")
        pca = _required(batch, "pca_tokens")
        clean = _required(batch, "clean_tokens")
        mask = batch.get("token_mask")

        def text_loss(enhanced_tokens: Any) -> Any:
            if bundle.transcription_loss is None:
                return None
            return bundle.transcription_loss(enhanced_tokens, batch)

        if train:
            bundle.optimizer.zero_grad(set_to_none=True)
        loss, report, _ = gate_training_loss(
            bundle.gate,
            mixed,
            pca,
            clean,
            token_mask=mask,
            transcription_loss=text_loss if bundle.transcription_loss is not None else None,
            lambda_token=lambda_token,
            lambda_text=lambda_text,
        )
        if train:
            loss.backward()
            bundle.optimizer.step()
        batch_size = int(mixed.shape[0])
        examples += batch_size
        total += report.total * batch_size
        token_mse += report.token_mse * batch_size
        if report.transcription is not None:
            transcription_sum += report.transcription * batch_size
            transcription_count += batch_size
    if examples == 0:
        raise ValueError("Gate training epoch received no batches.")
    return GateLoss(
        total=total / examples,
        token_mse=token_mse / examples,
        transcription=(transcription_sum / transcription_count) if transcription_count else None,
    )
