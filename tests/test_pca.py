from __future__ import annotations

import numpy as np
import pytest

from ltm_ae.memory.archive import (
    CalibratedMemory,
    load_calibrated_memory,
    load_memory,
    save_calibrated_memory,
    save_memory,
)
from ltm_ae.memory.pca import calibrate_memory, fit_pca


def test_alpha_zero_is_identity_and_alpha_one_is_reconstruction() -> None:
    generator = np.random.default_rng(3)
    references = generator.normal(size=(40, 6)).astype(np.float32)
    tokens = generator.normal(size=(7, 6)).astype(np.float32)
    memory = fit_pca(references, max_components=4, solver="full")

    assert np.allclose(memory.enhance(tokens, rank=3, alpha=0.0), tokens)
    assert np.allclose(memory.enhance(tokens, rank=3, alpha=1.0), memory.reconstruct(tokens, rank=3))


def test_full_rank_projection_preserves_reference_space() -> None:
    references = np.array([[-1.0, 0.0], [0.0, 1.0], [1.0, 0.0], [0.0, -1.0]], dtype=np.float32)
    memory = fit_pca(references, max_components=2, solver="full")
    assert np.allclose(memory.reconstruct(references, rank=2), references, atol=1e-5)


def test_calibration_selects_exact_target_projection() -> None:
    generator = np.random.default_rng(4)
    references = generator.normal(size=(80, 5)).astype(np.float32)
    memory = fit_pca(references, max_components=5, solver="full")
    mixed = generator.normal(size=(10, 5)).astype(np.float32)
    clean = memory.reconstruct(mixed, rank=2)
    selected = calibrate_memory(memory, [(mixed, clean)], ranks=[1, 2, 3], alphas=[0.0, 0.5, 1.0])
    assert selected.rank == 2
    assert selected.alpha == 1.0
    assert selected.mse == pytest.approx(0.0, abs=1e-10)


def test_memory_archive_round_trip(tmp_path) -> None:
    generator = np.random.default_rng(5)
    memory = fit_pca(generator.normal(size=(32, 4)).astype(np.float32), max_components=3, solver="full")
    raw_path = tmp_path / "raw.npz"
    calibrated_path = tmp_path / "calibrated.npz"
    save_memory(raw_path, memory)
    raw = load_memory(raw_path)
    assert np.allclose(raw.mean, memory.mean)
    selection = calibrate_memory(raw, [(np.ones((2, 4), np.float32), np.ones((2, 4), np.float32))], [1], [0.0])
    save_calibrated_memory(calibrated_path, CalibratedMemory(raw, selection))
    restored = load_calibrated_memory(calibrated_path)
    assert restored.calibration == selection
    assert np.allclose(restored.memory.components, memory.components)


def test_randomized_solver_matches_full_projection_on_low_rank_data() -> None:
    generator = np.random.default_rng(6)
    latent = generator.normal(size=(120, 3))
    mixing = generator.normal(size=(3, 16))
    references = (latent @ mixing).astype(np.float32)
    probe = (generator.normal(size=(9, 3)) @ mixing).astype(np.float32)
    full = fit_pca(references, max_components=3, solver="full")
    randomized = fit_pca(
        references,
        max_components=3,
        solver="randomized",
        random_state=9,
        n_power_iterations=5,
    )
    assert np.allclose(
        full.reconstruct(probe, rank=3),
        randomized.reconstruct(probe, rank=3),
        atol=2e-4,
    )
