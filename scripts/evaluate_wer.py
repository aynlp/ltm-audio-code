#!/usr/bin/env python3
"""Compute WER from a JSONL file containing reference and hypothesis text."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from _bootstrap import bootstrap

bootstrap()

from ltm_ae.data.manifests import read_jsonl, require_fields, write_jsonl
from ltm_ae.evaluation.wer import word_error_rate


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--transcripts", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows: list[dict[str, object]] = []
    substitutions = deletions = insertions = references = 0
    for row_number, row in enumerate(read_jsonl(args.transcripts), start=1):
        require_fields(row, ["example_id", "reference", "hypothesis"], context=f"transcript row {row_number}")
        result = word_error_rate(str(row["reference"]), str(row["hypothesis"]))
        substitutions += result.substitutions
        deletions += result.deletions
        insertions += result.insertions
        references += result.reference_words
        rows.append(
            {
                "example_id": str(row["example_id"]),
                "substitutions": result.substitutions,
                "deletions": result.deletions,
                "insertions": result.insertions,
                "reference_words": result.reference_words,
                "wer": result.rate,
            }
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.output, rows)
    summary = {
        "substitutions": substitutions,
        "deletions": deletions,
        "insertions": insertions,
        "reference_words": references,
        "wer": (substitutions + deletions + insertions) / references if references else None,
    }
    args.output.with_suffix(".summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
