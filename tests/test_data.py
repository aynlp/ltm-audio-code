from __future__ import annotations

import json

import numpy as np
import pytest

from ltm_ae.data.audio import construct_mixture, rms
from ltm_ae.data.tokens import group_reference_tokens, load_validation_pairs


def test_mixture_matches_requested_combined_ratio() -> None:
    generator = np.random.default_rng(11)
    target = generator.normal(size=16_000).astype(np.float32)
    interferers = [generator.normal(size=16_000).astype(np.float32) for _ in range(3)]
    _, combined = construct_mixture(target, interferers, target_to_interference_db=-10.0)
    ratio = 20 * np.log10(rms(target) / rms(combined))
    assert ratio == pytest.approx(-10.0, abs=1e-4)


def test_reference_and_validation_indices_are_loaded(tmp_path) -> None:
    token_a = np.ones((2, 3), dtype=np.float32)
    token_b = np.full((3, 3), 2.0, dtype=np.float32)
    clean = np.zeros((2, 3), dtype=np.float32)
    np.save(tmp_path / "a.npy", token_a)
    np.save(tmp_path / "b.npy", token_b)
    np.save(tmp_path / "mixed.npy", token_a)
    np.save(tmp_path / "clean.npy", clean)
    reference_index = tmp_path / "references.jsonl"
    reference_index.write_text(
        "\n".join(
            [
                json.dumps({"category": "x", "source_group": "r1", "tokens_path": "a.npy"}),
                json.dumps({"category": "x", "source_group": "r2", "tokens_path": "b.npy"}),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    validation_index = tmp_path / "validation.jsonl"
    validation_index.write_text(
        json.dumps(
            {
                "category": "x",
                "source_group": "v1",
                "mixed_tokens_path": "mixed.npy",
                "clean_tokens_path": "clean.npy",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    assert group_reference_tokens(reference_index)["x"].shape == (5, 3)
    assert group_reference_tokens(reference_index, clips_per_category=1)["x"].shape == (2, 3)
    assert len(load_validation_pairs(validation_index)["x"]) == 1


def test_reference_index_requires_source_group(tmp_path) -> None:
    np.save(tmp_path / "a.npy", np.ones((2, 3), dtype=np.float32))
    index = tmp_path / "references.jsonl"
    index.write_text(json.dumps({"category": "x", "tokens_path": "a.npy"}) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="source_group"):
        group_reference_tokens(index)
