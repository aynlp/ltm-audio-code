from __future__ import annotations

import pytest

from ltm_ae.adapters.injection import replace_masked_embeddings
from ltm_ae.gate.token_gate import TokenGate, freeze_module, gate_training_loss

torch = pytest.importorskip("torch")


def test_embedding_replacement_preserves_non_audio_positions_and_gradients() -> None:
    embeddings = torch.zeros((1, 5, 3), requires_grad=True)
    mask = torch.tensor([[False, True, True, False, False]])
    tokens = torch.tensor([[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]], requires_grad=True)
    replaced = replace_masked_embeddings(embeddings, mask, tokens)
    assert torch.equal(replaced[0, 1:3], tokens[0])
    assert torch.equal(replaced[0, 0], torch.zeros(3))
    replaced.sum().backward()
    assert tokens.grad is not None


def test_token_gate_range_and_frozen_decoder_loss() -> None:
    gate = TokenGate(hidden_size=4, bottleneck=2)
    frozen = torch.nn.Linear(4, 2)
    freeze_module(frozen)
    mixed = torch.randn(2, 3, 4)
    pca = torch.randn(2, 3, 4)
    clean = torch.randn(2, 3, 4)
    total, report, values = gate_training_loss(
        gate,
        mixed,
        pca,
        clean,
        transcription_loss=lambda enhanced: frozen(enhanced).square().mean(),
    )
    total.backward()
    assert 0.0 <= float(values.min()) <= float(values.max()) <= 1.0
    assert report.transcription is not None
    assert all(parameter.grad is None for parameter in frozen.parameters())
