#!/usr/bin/env python3
"""Evaluate prompt-free category retrieval from extracted audio-token arrays."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from _bootstrap import bootstrap

bootstrap()

from ltm_ae.data.manifests import read_jsonl, require_fields, resolve_manifest_path, write_jsonl
from ltm_ae.data.tokens import load_token_array
from ltm_ae.evaluation.retrieval import macro_accuracy, predict_category, temporal_mean
from ltm_ae.memory.archive import load_calibrated_memory
from ltm_ae.utils.paths import slugify


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gallery-index", type=Path, required=True)
    parser.add_argument("--query-index", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--memory-dir", type=Path, default=None)
    parser.add_argument("--top-k", type=int, default=5)
    return parser.parse_args()


def _load_gallery(index_path: Path) -> dict[str, np.ndarray]:
    vectors: dict[str, list[np.ndarray]] = defaultdict(list)
    for row_number, row in enumerate(read_jsonl(index_path), start=1):
        require_fields(
            row,
            ["category", "source_group", "tokens_path"],
            context=f"gallery row {row_number}",
        )
        vectors[str(row["category"])].append(
            temporal_mean(load_token_array(resolve_manifest_path(str(row["tokens_path"]), index_path)))
        )
    if not vectors:
        raise ValueError("Gallery index is empty.")
    return {category: np.stack(items) for category, items in vectors.items()}


def main() -> None:
    args = parse_args()
    gallery = _load_gallery(args.gallery_index)
    rows: list[dict[str, object]] = []
    targets: list[str] = []
    predictions: list[str] = []

    for row_number, row in enumerate(read_jsonl(args.query_index), start=1):
        require_fields(
            row,
            ["example_id", "category", "source_group", "tokens_path"],
            context=f"query row {row_number}",
        )
        category = str(row["category"])
        tokens = load_token_array(resolve_manifest_path(str(row["tokens_path"]), args.query_index))
        if args.memory_dir is not None:
            memory_path = args.memory_dir / f"{slugify(category)}.npz"
            memory = load_calibrated_memory(memory_path)
            tokens = memory.memory.enhance(tokens, memory.calibration.rank, memory.calibration.alpha)
        prediction = predict_category(tokens, gallery, top_k=args.top_k)
        targets.append(category)
        predictions.append(prediction.category)
        rows.append(
            {
                "example_id": str(row["example_id"]),
                "target_category": category,
                "predicted_category": prediction.category,
                "correct": prediction.category == category,
                "scores": prediction.scores,
            }
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.output, rows)
    summary = {"examples": len(rows), "macro_accuracy": macro_accuracy(predictions, targets)}
    args.output.with_suffix(".summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
