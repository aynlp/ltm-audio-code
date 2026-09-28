#!/usr/bin/env python3
"""Score pre-generated constrained or free-form classification outputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml
from _bootstrap import bootstrap

bootstrap()

from ltm_ae.data.manifests import read_jsonl, require_fields, write_jsonl
from ltm_ae.evaluation.classification import normalize_free_form_response, parse_constrained_response
from ltm_ae.evaluation.retrieval import macro_accuracy


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--mapping", type=Path, required=True)
    parser.add_argument("--mode", choices=("constrained", "free-form"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    mapping = yaml.safe_load(args.mapping.read_text(encoding="utf-8"))
    if not isinstance(mapping, dict):
        raise ValueError("Mapping file must contain a YAML object.")
    normalized_rows: list[dict[str, object]] = []
    expected: list[str] = []
    predicted: list[str] = []
    for row_number, row in enumerate(read_jsonl(args.predictions), start=1):
        require_fields(row, ["example_id", "category", "response"], context=f"prediction row {row_number}")
        response = str(row["response"])
        if args.mode == "constrained":
            result = parse_constrained_response(response, mapping)
        else:
            result = normalize_free_form_response(response, mapping)
        normalized_rows.append(
            {
                "example_id": str(row["example_id"]),
                "target_category": str(row["category"]),
                "prediction": result,
                "correct": result == str(row["category"]),
            }
        )
        if result is not None:
            expected.append(str(row["category"]))
            predicted.append(result)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.output, normalized_rows)
    summary = {
        "examples": len(normalized_rows),
        "valid_predictions": len(predicted),
        "macro_accuracy_valid_only": macro_accuracy(predicted, expected) if predicted else None,
    }
    args.output.with_suffix(".summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
