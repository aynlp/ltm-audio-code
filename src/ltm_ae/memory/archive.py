"""Portable, data-free serialization for fitted PCA memories."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ltm_ae.memory.pca import CalibrationResult, PCAMemory


@dataclass(frozen=True)
class CalibratedMemory:
    """A PCA memory with the rank and interpolation weight selected for it."""

    memory: PCAMemory
    calibration: CalibrationResult


def save_memory(path: str | Path, memory: PCAMemory) -> None:
    """Save an uncalibrated PCA memory before validation selection."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        destination,
        mean=memory.mean,
        components=memory.components,
        payload=json.dumps({"metadata": memory.metadata}, sort_keys=True),
    )


def load_memory(path: str | Path) -> PCAMemory:
    """Load an uncalibrated archive made by ``save_memory``."""

    source = Path(path)
    with np.load(source, allow_pickle=False) as archive:
        payload = json.loads(str(archive["payload"].item()))
        return PCAMemory(
            mean=archive["mean"],
            components=archive["components"],
            metadata=dict(payload.get("metadata", {})),
        )


def save_calibrated_memory(path: str | Path, item: CalibratedMemory) -> None:
    """Save one deployable memory archive as a compressed NumPy file."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "metadata": item.memory.metadata,
        "rank": item.calibration.rank,
        "alpha": item.calibration.alpha,
        "validation_mse": item.calibration.mse,
        "evaluated_candidates": item.calibration.evaluated_candidates,
    }
    np.savez_compressed(
        destination,
        mean=item.memory.mean,
        components=item.memory.components,
        payload=json.dumps(payload, sort_keys=True),
    )


def load_calibrated_memory(path: str | Path) -> CalibratedMemory:
    """Load and validate a memory archive produced by ``save_calibrated_memory``."""

    source = Path(path)
    with np.load(source, allow_pickle=False) as archive:
        payload = json.loads(str(archive["payload"].item()))
        memory = PCAMemory(
            mean=archive["mean"],
            components=archive["components"],
            metadata=dict(payload.get("metadata", {})),
        )
        calibration = CalibrationResult(
            rank=int(payload["rank"]),
            alpha=float(payload["alpha"]),
            mse=float(payload["validation_mse"]),
            evaluated_candidates=int(payload["evaluated_candidates"]),
        )
    memory._validate_rank(calibration.rank)
    if not 0.0 <= calibration.alpha <= 1.0:
        raise ValueError("Stored calibration alpha is outside [0, 1].")
    return CalibratedMemory(memory=memory, calibration=calibration)
