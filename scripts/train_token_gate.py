#!/usr/bin/env python3
"""Train a token gate using a caller-provided frozen-model training bundle.

The bundle factory is deliberately external: it owns data, model weights,
teacher-forcing inputs, and model-specific decoding while this repository owns
the gate architecture and optimization logic.
"""

from __future__ import annotations

import argparse
import importlib
import json
from pathlib import Path

from _bootstrap import bootstrap

bootstrap()

from ltm_ae.gate.training import GateTrainingBundle, run_epoch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle-factory", required=True, help="module.path:callable returning GateTrainingBundle")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--lambda-token", type=float, default=1.0)
    parser.add_argument("--lambda-text", type=float, default=1.0)
    return parser.parse_args()


def load_bundle(factory_spec: str) -> GateTrainingBundle:
    if ":" not in factory_spec:
        raise ValueError("bundle-factory must use module.path:callable syntax.")
    module_name, callable_name = factory_spec.split(":", maxsplit=1)
    factory = getattr(importlib.import_module(module_name), callable_name, None)
    if factory is None or not callable(factory):
        raise ValueError(f"Cannot load callable {factory_spec!r}.")
    bundle = factory()
    if not isinstance(bundle, GateTrainingBundle):
        raise TypeError("bundle-factory must return ltm_ae.gate.training.GateTrainingBundle.")
    return bundle


def main() -> None:
    args = parse_args()
    if args.epochs < 1:
        raise ValueError("epochs must be positive.")
    bundle = load_bundle(args.bundle_factory)
    history: list[dict[str, object]] = []
    for epoch in range(1, args.epochs + 1):
        train = run_epoch(
            bundle,
            bundle.train_batches(),
            lambda_token=args.lambda_token,
            lambda_text=args.lambda_text,
            train=True,
        )
        record: dict[str, object] = {
            "epoch": epoch,
            "train_total": train.total,
            "train_token_mse": train.token_mse,
            "train_transcription": train.transcription,
        }
        if bundle.validation_batches is not None:
            validation = run_epoch(
                bundle,
                bundle.validation_batches(),
                lambda_token=args.lambda_token,
                lambda_text=args.lambda_text,
                train=False,
            )
            record.update(
                {
                    "validation_total": validation.total,
                    "validation_token_mse": validation.token_mse,
                    "validation_transcription": validation.transcription,
                }
            )
        history.append(record)
        print(json.dumps(record, sort_keys=True))

    try:
        import torch
    except ImportError as error:  # pragma: no cover - optional environment failure
        raise ImportError("Saving a token gate requires the optional torch dependency.") from error
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"gate_state_dict": bundle.gate.state_dict(), "history": history}, args.output)


if __name__ == "__main__":
    main()
