"""PCA long-term memories and validation-based calibration.

The implementation follows the token-level rule used by LTM-AE.  A memory is
the mean of clean reference tokens plus orthonormal PCA directions.  For an
incoming token matrix ``Z``, enhancement is

``Z' = Z + alpha * (mu + P_k (Z - mu) - Z)``.

Rows are tokens and columns are embedding coordinates throughout this module.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Literal

import numpy as np

Array = np.ndarray
Solver = Literal["full", "randomized", "auto"]


def _as_float_matrix(values: Array, name: str) -> Array:
    matrix = np.asarray(values)
    if matrix.ndim != 2:
        raise ValueError(f"{name} must have shape [tokens, dimensions], got {matrix.shape}.")
    if not np.issubdtype(matrix.dtype, np.floating):
        raise TypeError(f"{name} must have a floating dtype, got {matrix.dtype}.")
    if matrix.shape[0] == 0 or matrix.shape[1] == 0:
        raise ValueError(f"{name} must be non-empty, got {matrix.shape}.")
    if not np.isfinite(matrix).all():
        raise ValueError(f"{name} contains non-finite values.")
    return matrix


def _randomized_components(
    centered: Array,
    n_components: int,
    random_state: int,
    n_oversamples: int,
    n_power_iterations: int,
) -> Array:
    """Compute leading right singular vectors with a randomized range finder."""

    n_tokens, dimension = centered.shape
    target_rank = min(n_components + n_oversamples, n_tokens, dimension)
    generator = np.random.default_rng(random_state)
    test_matrix = generator.standard_normal((dimension, target_rank))
    sample = centered @ test_matrix
    basis, _ = np.linalg.qr(sample, mode="reduced")

    for _ in range(n_power_iterations):
        basis, _ = np.linalg.qr(centered @ (centered.T @ basis), mode="reduced")

    compressed = basis.T @ centered
    _, _, right_vectors = np.linalg.svd(compressed, full_matrices=False)
    return right_vectors[:n_components]


@dataclass(frozen=True)
class PCAMemory:
    """Mean and ordered PCA directions for one model/category memory."""

    mean: Array
    components: Array
    metadata: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        mean = np.asarray(self.mean)
        components = np.asarray(self.components)
        if mean.ndim != 1:
            raise ValueError(f"mean must have shape [dimensions], got {mean.shape}.")
        if components.ndim != 2:
            raise ValueError(f"components must have shape [components, dimensions], got {components.shape}.")
        if components.shape[1] != mean.shape[0]:
            raise ValueError("mean and components have incompatible embedding dimensions.")
        if components.shape[0] == 0:
            raise ValueError("At least one PCA direction is required.")
        if not np.isfinite(mean).all() or not np.isfinite(components).all():
            raise ValueError("PCAMemory contains non-finite values.")

        gram = components @ components.T
        if not np.allclose(gram, np.eye(components.shape[0]), atol=2e-4, rtol=2e-4):
            raise ValueError("components must be orthonormal row vectors.")

    @property
    def dimension(self) -> int:
        return int(self.mean.shape[0])

    @property
    def max_rank(self) -> int:
        return int(self.components.shape[0])

    def _validate_rank(self, rank: int) -> int:
        if not isinstance(rank, (int, np.integer)):
            raise TypeError("rank must be an integer.")
        if rank < 1 or rank > self.max_rank:
            raise ValueError(f"rank must be in [1, {self.max_rank}], got {rank}.")
        return int(rank)

    def reconstruct(self, tokens: Array, rank: int) -> Array:
        """Project tokens onto the selected memory subspace, preserving its mean."""

        rank = self._validate_rank(rank)
        tokens = _as_float_matrix(tokens, "tokens")
        if tokens.shape[1] != self.dimension:
            raise ValueError(f"tokens have dimension {tokens.shape[1]}, expected {self.dimension}.")
        work_dtype = np.result_type(tokens.dtype, self.mean.dtype, np.float64)
        centered = tokens.astype(work_dtype, copy=False) - self.mean.astype(work_dtype, copy=False)
        basis = self.components[:rank].astype(work_dtype, copy=False)
        reconstruction = self.mean.astype(work_dtype, copy=False) + (centered @ basis.T) @ basis
        return reconstruction.astype(np.result_type(tokens.dtype, np.float32), copy=False)

    def enhance(self, tokens: Array, rank: int, alpha: float | Array) -> Array:
        """Interpolate each token with its rank-``rank`` reconstruction.

        ``alpha`` may be a scalar or a tensor broadcastable to all leading token
        dimensions. Values are constrained to `[0, 1]` because the method uses
        interpolation rather than extrapolation.
        """

        tokens = _as_float_matrix(tokens, "tokens")
        alpha_array = np.asarray(alpha, dtype=np.float64)
        if not np.isfinite(alpha_array).all() or np.any(alpha_array < 0.0) or np.any(alpha_array > 1.0):
            raise ValueError("alpha must contain finite values in [0, 1].")
        if alpha_array.ndim == tokens.ndim - 1:
            alpha_array = alpha_array[..., None]
        try:
            alpha_array = np.broadcast_to(alpha_array, tokens.shape)
        except ValueError as error:
            raise ValueError(
                f"alpha with shape {np.asarray(alpha).shape} cannot broadcast to {tokens.shape}."
            ) from error

        reconstruction = self.reconstruct(tokens, rank)
        enhanced = tokens + alpha_array * (reconstruction - tokens)
        return enhanced.astype(np.result_type(tokens.dtype, np.float32), copy=False)


@dataclass(frozen=True)
class CalibrationResult:
    """Selected hyperparameters and validation objective for one memory."""

    rank: int
    alpha: float
    mse: float
    evaluated_candidates: int


def fit_pca(
    reference_tokens: Array,
    max_components: int,
    *,
    solver: Solver = "auto",
    random_state: int = 0,
    n_oversamples: int = 10,
    n_power_iterations: int = 5,
    metadata: dict[str, object] | None = None,
) -> PCAMemory:
    """Fit a PCA memory from all clean-reference tokens.

    ``reference_tokens`` is the vertical concatenation of token sequences from
    the clean reference clips.  Full SVD is exact; randomized PCA uses a
    deterministic random seed and is preferable for large matrices.
    """

    reference_tokens = _as_float_matrix(reference_tokens, "reference_tokens")
    allowed_rank = min(reference_tokens.shape)
    if not isinstance(max_components, (int, np.integer)) or max_components < 1:
        raise ValueError("max_components must be a positive integer.")
    n_components = min(int(max_components), allowed_rank)

    mean = reference_tokens.mean(axis=0, dtype=np.float64)
    centered = reference_tokens.astype(np.float64, copy=False) - mean
    if solver == "auto":
        solver = "full" if max(centered.shape) <= 4096 else "randomized"
    if solver == "full":
        _, _, right_vectors = np.linalg.svd(centered, full_matrices=False)
        components = right_vectors[:n_components]
    elif solver == "randomized":
        components = _randomized_components(
            centered,
            n_components=n_components,
            random_state=random_state,
            n_oversamples=n_oversamples,
            n_power_iterations=n_power_iterations,
        )
    else:
        raise ValueError(f"Unsupported solver: {solver!r}.")

    stored_metadata: dict[str, object] = {
        "n_reference_tokens": int(reference_tokens.shape[0]),
        "dimension": int(reference_tokens.shape[1]),
        "max_components": n_components,
        "solver": solver,
        "random_state": int(random_state),
    }
    if metadata:
        stored_metadata.update(metadata)
    return PCAMemory(
        mean=mean.astype(np.float32),
        components=components.astype(np.float32),
        metadata=stored_metadata,
    )


def calibrate_memory(
    memory: PCAMemory,
    validation_pairs: Iterable[tuple[Array, Array]],
    ranks: Iterable[int],
    alphas: Iterable[float],
) -> CalibrationResult:
    """Select `(rank, alpha)` by mean token MSE on aligned validation pairs.

    The iterable contains `(mixed_tokens, clean_target_tokens)` pairs of equal
    shape. Candidate order is made deterministic: lower rank, then lower alpha,
    wins an exact tie. Evaluation examples are intentionally not accepted here.
    """

    validated_pairs: list[tuple[Array, Array]] = []
    for pair_index, (mixed, clean) in enumerate(validation_pairs):
        mixed = _as_float_matrix(mixed, f"validation_pairs[{pair_index}].mixed")
        clean = _as_float_matrix(clean, f"validation_pairs[{pair_index}].clean")
        if mixed.shape != clean.shape:
            raise ValueError(
                f"validation pair {pair_index} has mixed shape {mixed.shape} and clean shape {clean.shape}."
            )
        if mixed.shape[1] != memory.dimension:
            raise ValueError("Validation embedding dimension does not match memory dimension.")
        validated_pairs.append((mixed, clean))
    if not validated_pairs:
        raise ValueError("At least one validation pair is required for calibration.")

    candidate_ranks = sorted({memory._validate_rank(int(rank)) for rank in ranks})
    candidate_alphas = sorted({float(alpha) for alpha in alphas})
    if not candidate_ranks or not candidate_alphas:
        raise ValueError("At least one rank and one alpha candidate are required.")
    if any(not np.isfinite(alpha) or alpha < 0.0 or alpha > 1.0 for alpha in candidate_alphas):
        raise ValueError("All alpha candidates must be finite values in [0, 1].")

    best: CalibrationResult | None = None
    candidate_count = 0
    for rank in candidate_ranks:
        for alpha in candidate_alphas:
            squared_error = 0.0
            coordinate_count = 0
            for mixed, clean in validated_pairs:
                enhanced = memory.enhance(mixed, rank=rank, alpha=alpha)
                difference = enhanced.astype(np.float64) - clean.astype(np.float64)
                squared_error += float(np.square(difference).sum())
                coordinate_count += int(difference.size)
            result = CalibrationResult(
                rank=rank,
                alpha=alpha,
                mse=squared_error / coordinate_count,
                evaluated_candidates=0,
            )
            candidate_count += 1
            if best is None or result.mse < best.mse:
                best = result

    assert best is not None
    return CalibrationResult(
        rank=best.rank,
        alpha=best.alpha,
        mse=best.mse,
        evaluated_candidates=candidate_count,
    )
