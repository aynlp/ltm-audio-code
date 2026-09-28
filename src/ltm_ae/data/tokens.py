"""Loading clean and paired audio-token arrays from caller-provided manifests."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable
from pathlib import Path

import numpy as np

from ltm_ae.data.manifests import read_jsonl, require_fields, resolve_manifest_path

Array = np.ndarray


def load_token_array(path: str | Path) -> Array:
    """Load a `[time, dimension]` floating array saved with ``numpy.save``."""

    source = Path(path)
    values = np.load(source, allow_pickle=False)
    if values.ndim != 2:
        raise ValueError(f"Token array {source} must be 2D, got {values.shape}.")
    if not np.issubdtype(values.dtype, np.floating):
        raise TypeError(f"Token array {source} has non-floating dtype {values.dtype}.")
    if values.shape[0] == 0 or values.shape[1] == 0:
        raise ValueError(f"Token array {source} is empty.")
    if not np.isfinite(values).all():
        raise ValueError(f"Token array {source} contains non-finite values.")
    return values


def concatenate_token_arrays(paths: Iterable[str | Path]) -> Array:
    """Concatenate token sequences while requiring a common embedding dimension."""

    sequences = [load_token_array(path) for path in paths]
    if not sequences:
        raise ValueError("At least one token array is required.")
    dimensions = {sequence.shape[1] for sequence in sequences}
    if len(dimensions) != 1:
        raise ValueError(f"Token arrays have inconsistent dimensions: {sorted(dimensions)}.")
    return np.concatenate(sequences, axis=0)


def group_reference_tokens(
    index_path: str | Path,
    *,
    clips_per_category: int | None = None,
) -> dict[str, Array]:
    """Return vertically concatenated clean-reference tokens grouped by category.

    Expected JSONL fields: ``category``, ``source_group``, and ``tokens_path``.
    Paths may be relative to the manifest file. The function deliberately does
    not infer categories from directory names.
    """

    if clips_per_category is not None and clips_per_category < 1:
        raise ValueError("clips_per_category must be positive when specified.")
    grouped_paths: dict[str, list[Path]] = defaultdict(list)
    for row_number, row in enumerate(read_jsonl(index_path), start=1):
        require_fields(
            row,
            ["category", "source_group", "tokens_path"],
            context=f"reference row {row_number}",
        )
        category = str(row["category"])
        if clips_per_category is None or len(grouped_paths[category]) < clips_per_category:
            grouped_paths[category].append(resolve_manifest_path(str(row["tokens_path"]), index_path))
    if not grouped_paths:
        raise ValueError("Reference index is empty.")
    if clips_per_category is not None:
        insufficient = {
            category: len(paths) for category, paths in grouped_paths.items() if len(paths) != clips_per_category
        }
        if insufficient:
            details = ", ".join(f"{category}={count}" for category, count in sorted(insufficient.items()))
            raise ValueError(f"Reference index does not provide {clips_per_category} clips per category ({details}).")
    return {category: concatenate_token_arrays(paths) for category, paths in sorted(grouped_paths.items())}


def load_validation_pairs(index_path: str | Path) -> dict[str, list[tuple[Array, Array]]]:
    """Load aligned mixed/clean token pairs grouped by target category.

    Expected JSONL fields: ``category``, ``source_group``,
    ``mixed_tokens_path``, and ``clean_tokens_path``. Alignment is checked at
    token-array resolution.
    """

    pairs: dict[str, list[tuple[Array, Array]]] = defaultdict(list)
    for row_number, row in enumerate(read_jsonl(index_path), start=1):
        require_fields(
            row,
            ["category", "source_group", "mixed_tokens_path", "clean_tokens_path"],
            context=f"validation row {row_number}",
        )
        mixed = load_token_array(resolve_manifest_path(str(row["mixed_tokens_path"]), index_path))
        clean = load_token_array(resolve_manifest_path(str(row["clean_tokens_path"]), index_path))
        if mixed.shape != clean.shape:
            raise ValueError(
                f"Validation row {row_number} has mixed shape {mixed.shape} and clean shape {clean.shape}."
            )
        pairs[str(row["category"])].append((mixed, clean))
    if not pairs:
        raise ValueError("Validation index is empty.")
    return dict(pairs)
