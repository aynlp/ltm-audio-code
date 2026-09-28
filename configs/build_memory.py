#!/usr/bin/env python3
"""Build one clean-reference PCA memory per target category."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from _bootstrap import bootstrap

bootstrap()

from ltm_ae.data.tokens import group_reference_tokens
from ltm_ae.memory.archive import save_memory
from ltm_ae.memory.pca import fit_pca
from ltm_ae.utils.paths import slugify


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-index", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model-id", required=True, help="Stable identifier for this model interface.")
    parser.add_argument("--max-components", type=int, default=128)
    parser.add_argument(
        "--reference-clips-per-category",
        type=int,
        default=None,
        help="Use the first N pre-selected rows per category in manifest order.",
    )
    parser.add_argument("--solver", choices=("auto", "full", "randomized"), default="randomized")
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    references = group_reference_tokens(
        args.reference_index,
        clips_per_category=args.reference_clips_per_category,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary: list[dict[str, object]] = []
    for category, tokens in references.items():
        memory = fit_pca(
            tokens,
            max_components=args.max_components,
            solver=args.solver,
            random_state=args.seed,
            metadata={"model_id": args.model_id, "category": category},
        )
        output_path = args.output_dir / f"{slugify(category)}.npz"
        save_memory(output_path, memory)
        summary.append(
            {
                "category": category,
                "memory_path": output_path.name,
                "reference_tokens": memory.metadata["n_reference_tokens"],
                "dimension": memory.dimension,
                "max_rank": memory.max_rank,
            }
        )
        print(f"Built {category!r}: {memory.metadata['n_reference_tokens']} tokens, rank {memory.max_rank}.")
    (args.output_dir / "build_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
