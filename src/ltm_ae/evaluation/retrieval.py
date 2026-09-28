"""Prompt-free retrieval directly in an audio-token representation space."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np

Array = np.ndarray


def temporal_mean(tokens: Array) -> Array:
    """Average a non-empty `[time, dimension]` token sequence over time."""

    tokens = np.asarray(tokens)
    if tokens.ndim != 2 or tokens.shape[0] == 0:
        raise ValueError(f"Expected non-empty [time, dimension] tokens, got {tokens.shape}.")
    if not np.issubdtype(tokens.dtype, np.floating) or not np.isfinite(tokens).all():
        raise ValueError("Tokens must be finite floating-point values.")
    return tokens.mean(axis=0, dtype=np.float64)


def l2_normalize(vector: Array, *, epsilon: float = 1e-12) -> Array:
    """Normalize a vector and reject zero vectors that have undefined cosine similarity."""

    vector = np.asarray(vector, dtype=np.float64)
    if vector.ndim != 1:
        raise ValueError(f"Expected one-dimensional vector, got {vector.shape}.")
    norm = float(np.linalg.norm(vector))
    if not np.isfinite(norm) or norm < epsilon:
        raise ValueError("Cannot normalize a zero or non-finite vector.")
    return vector / norm


@dataclass(frozen=True)
class RetrievalPrediction:
    """Predicted category and per-category top-k cosine scores."""

    category: str
    scores: dict[str, float]


def predict_category(
    query_tokens: Array,
    gallery: Mapping[str, Array],
    *,
    top_k: int = 5,
) -> RetrievalPrediction:
    """Classify a query by the average of its largest category similarities.

    Gallery values have shape `[references, dimension]`, usually obtained by
    temporal averaging separately held-out clean reference clips. The caller
    must ensure that the gallery does not overlap memory-construction clips.
    """

    if top_k < 1:
        raise ValueError("top_k must be positive.")
    query = l2_normalize(temporal_mean(query_tokens))
    scores: dict[str, float] = {}
    for category, reference_vectors in sorted(gallery.items()):
        reference_vectors = np.asarray(reference_vectors, dtype=np.float64)
        if reference_vectors.ndim != 2 or reference_vectors.shape[0] == 0:
            raise ValueError(f"Gallery category {category!r} must have shape [references, dimension].")
        if reference_vectors.shape[1] != query.shape[0]:
            raise ValueError(f"Gallery category {category!r} has an incompatible embedding dimension.")
        norms = np.linalg.norm(reference_vectors, axis=1)
        if np.any(~np.isfinite(norms)) or np.any(norms == 0.0):
            raise ValueError(f"Gallery category {category!r} contains zero or non-finite vectors.")
        similarities = (reference_vectors / norms[:, None]) @ query
        use_k = min(top_k, similarities.size)
        top_values = np.partition(similarities, similarities.size - use_k)[-use_k:]
        scores[str(category)] = float(top_values.mean())
    if not scores:
        raise ValueError("Gallery is empty.")
    predicted_category = max(scores, key=lambda name: (scores[name], name))
    return RetrievalPrediction(category=predicted_category, scores=scores)


def macro_accuracy(predictions: Sequence[str], targets: Sequence[str]) -> float:
    """Compute category-balanced accuracy for a closed set of target labels."""

    if len(predictions) != len(targets) or not targets:
        raise ValueError("predictions and non-empty targets must have equal length.")
    per_class: dict[str, list[bool]] = {}
    for predicted, target in zip(predictions, targets, strict=True):
        per_class.setdefault(str(target), []).append(str(predicted) == str(target))
    return float(np.mean([np.mean(values) for values in per_class.values()]))
