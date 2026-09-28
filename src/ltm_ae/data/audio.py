"""Deterministic audio standardization and mixture construction."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

import numpy as np
import soundfile as sf

Array = np.ndarray


def _resample(waveform: Array, source_rate: int, target_rate: int) -> Array:
    if source_rate == target_rate:
        return waveform
    try:
        from scipy.signal import resample_poly
    except ImportError as error:  # pragma: no cover - dependency error is environment-specific
        raise ImportError("Resampling requires scipy. Install with `pip install scipy`.") from error

    divisor = int(np.gcd(source_rate, target_rate))
    return resample_poly(waveform, target_rate // divisor, source_rate // divisor).astype(np.float32)


def load_mono_audio(
    path: str | Path,
    *,
    sample_rate: int = 16_000,
    duration_seconds: float = 3.0,
) -> Array:
    """Read, downmix, resample, and zero-pad/trim an audio clip deterministically."""

    waveform, source_rate = sf.read(Path(path), always_2d=True, dtype="float32")
    waveform = waveform.mean(axis=1)
    waveform = _resample(waveform, int(source_rate), sample_rate)
    expected_samples = int(round(sample_rate * duration_seconds))
    if expected_samples <= 0:
        raise ValueError("duration_seconds must be positive.")
    if waveform.shape[0] >= expected_samples:
        return waveform[:expected_samples].astype(np.float32, copy=False)
    return np.pad(waveform, (0, expected_samples - waveform.shape[0])).astype(np.float32, copy=False)


def rms(waveform: Array, *, epsilon: float = 1e-8) -> float:
    """Compute root-mean-square amplitude with a silence guard."""

    value = float(np.sqrt(np.mean(np.square(np.asarray(waveform, dtype=np.float64)))))
    if value < epsilon:
        raise ValueError("Cannot RMS-match a silent waveform.")
    return value


def scale_to_rms(waveform: Array, target_rms: float) -> Array:
    """Scale a non-silent waveform to ``target_rms`` without clipping it."""

    if target_rms <= 0:
        raise ValueError("target_rms must be positive.")
    return np.asarray(waveform, dtype=np.float32) * (target_rms / rms(waveform))


def construct_mixture(
    target: Array,
    interferers: Iterable[Array],
    *,
    target_to_interference_db: float = -10.0,
) -> tuple[Array, Array]:
    """Mix one target with multiple interferers at a combined RMS ratio.

    Each interferer is first RMS-matched to the target. Their sum is then
    scaled so ``20 log10(rms(target) / rms(interference))`` equals the requested
    target-to-combined-interference ratio. A value of `-10` therefore makes the
    combined interferer 10 dB stronger than the target.
    """

    target = np.asarray(target, dtype=np.float32)
    if target.ndim != 1 or target.size == 0:
        raise ValueError("target must be a non-empty mono waveform.")
    target_level = rms(target)
    interferer_list = [np.asarray(item, dtype=np.float32) for item in interferers]
    if not interferer_list:
        raise ValueError("At least one interferer is required.")
    if any(item.shape != target.shape for item in interferer_list):
        raise ValueError("All interferers must have the same shape as target.")

    matched = [scale_to_rms(item, target_level) for item in interferer_list]
    combined = np.sum(matched, axis=0, dtype=np.float64)
    desired_interference_rms = target_level / (10.0 ** (target_to_interference_db / 20.0))
    combined = scale_to_rms(combined, desired_interference_rms)
    mixture = target.astype(np.float64) + combined.astype(np.float64)
    return mixture.astype(np.float32), combined.astype(np.float32)


def write_audio(path: str | Path, waveform: Array, *, sample_rate: int = 16_000) -> None:
    """Write a mono waveform; the caller owns the destination and metadata."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    sf.write(destination, np.asarray(waveform, dtype=np.float32), sample_rate, subtype="FLOAT")
