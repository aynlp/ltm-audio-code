#!/usr/bin/env python3
"""Calibrate PCA rank and interpolation weight on aligned validation tokens."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from _bootstrap import bootstrap

bootstrap()

from ltm_ae.data.tokens import load_validation_pairs
from ltm_ae.memory.archive import CalibratedMemory, load_memory, save_calibrated_memory
from ltm_ae.memory.pca import calibrate_memory
from ltm_ae.utils.paths import slugify


def _comma_list(value: str, cast):
    try:
        values = [cast(item.strip()) for item in value.split(",") if item.strip()]
    except ValueError as error:
        raise argparse.ArgumentTypeError(f"Could not parse {value!r}.") from error
    if not values:
        raise argparse.ArgumentTypeError("Candidate list cannot be empty.")
    return values


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--memory-dir", type=Path, required=True)
    parser.add_argument("--validation-index", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--ranks", default="8,16,32,64,128")
    parser.add_argument("--alphas", default="0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ranks = _comma_list(args.ranks, int)
    alphas = _comma_list(args.alphas, float)
    validation_pairs = load_validation_pairs(args.validation_index)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary: list[dict[str, object]] = []

    for category, pairs in sorted(validation_pairs.items()):
        raw_memory_path = args.memory_dir / f"{slugify(category)}.npz"
        if not raw_memory_path.is_file():
            raise FileNotFoundError(f"No raw memory for category {category!r}: {raw_memory_path}")
        memory = load_memory(raw_memory_path)
        selected = calibrate_memory(memory, pairs, ranks=ranks, alphas=alphas)
        output_path = args.output_dir / f"{slugify(category)}.npz"
        save_calibrated_memory(output_path, CalibratedMemory(memory=memory, calibration=selected))
        summary.append(
            {
                "category": category,
                "memory_path": output_path.name,
                "rank": selected.rank,
                "alpha": selected.alpha,
                "validation_mse": selected.mse,
                "evaluated_candidates": selected.evaluated_candidates,
            }
        )
        print(f"Calibrated {category!r}: rank={selected.rank}, alpha={selected.alpha:.3f}, MSE={selected.mse:.6g}.")
    (args.output_dir / "calibration_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
