#!/usr/bin/env python3
"""Apply a calibrated PCA memory to one extracted audio-token sequence."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from _bootstrap import bootstrap

bootstrap()

from ltm_ae.data.tokens import load_token_array
from ltm_ae.memory.archive import load_calibrated_memory


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokens", type=Path, required=True)
    parser.add_argument("--memory", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--alpha-scale", type=float, default=1.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    item = load_calibrated_memory(args.memory)
    tokens = load_token_array(args.tokens)
    if not 0.0 <= args.alpha_scale <= 1.0:
        raise ValueError("alpha-scale must be in [0, 1].")
    enhanced = item.memory.enhance(
        tokens,
        rank=item.calibration.rank,
        alpha=item.calibration.alpha * args.alpha_scale,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.save(args.output, enhanced)
    print(f"Wrote {enhanced.shape[0]} enhanced tokens with dimension {enhanced.shape[1]}.")


if __name__ == "__main__":
    main()
