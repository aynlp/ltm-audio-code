#!/usr/bin/env python3
"""Run a no-data smoke test of PCA memory, calibration, retrieval, and WER."""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
from _bootstrap import bootstrap

bootstrap()

from ltm_ae.evaluation.retrieval import predict_category
from ltm_ae.evaluation.wer import word_error_rate
from ltm_ae.memory.archive import CalibratedMemory, load_calibrated_memory, save_calibrated_memory
from ltm_ae.memory.pca import calibrate_memory, fit_pca


def main() -> None:
    generator = np.random.default_rng(7)
    references = generator.normal(size=(128, 12)).astype(np.float32)
    memory = fit_pca(references, max_components=8, solver="full", metadata={"model_id": "smoke"})
    incoming = generator.normal(size=(9, 12)).astype(np.float32)
    assert np.allclose(memory.enhance(incoming, rank=4, alpha=0.0), incoming)
    reconstruction = memory.reconstruct(incoming, rank=4)
    assert np.allclose(memory.enhance(incoming, rank=4, alpha=1.0), reconstruction)

    clean = memory.reconstruct(incoming, rank=2)
    selection = calibrate_memory(memory, [(incoming, clean)], ranks=[1, 2, 4], alphas=[0.0, 0.5, 1.0])
    assert selection.rank == 2 and selection.alpha == 1.0
    with tempfile.TemporaryDirectory() as temporary:
        archive_path = Path(temporary) / "memory.npz"
        save_calibrated_memory(archive_path, CalibratedMemory(memory=memory, calibration=selection))
        restored = load_calibrated_memory(archive_path)
        assert restored.calibration == selection

    gallery = {
        "left": np.tile(np.array([1.0, 0.0]), (5, 1)),
        "right": np.tile(np.array([0.0, 1.0]), (5, 1)),
    }
    assert predict_category(np.array([[0.8, 0.1]], dtype=np.float32), gallery).category == "left"
    assert word_error_rate("one two", "one three").rate == 0.5
    print("Smoke test passed.")


if __name__ == "__main__":
    main()
