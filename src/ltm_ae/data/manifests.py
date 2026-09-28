"""Small, explicit JSONL-manifest helpers.

The repository ships schemas and examples only.  Real manifests are generated
locally by the caller and should remain ignored by version control.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Iterator, Mapping
from pathlib import Path
from typing import Any


def read_jsonl(path: str | Path) -> Iterator[dict[str, Any]]:
    """Yield JSON objects from a JSONL file and identify malformed line numbers."""

    source = Path(path)
    with source.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            text = line.strip()
            if not text:
                continue
            try:
                item = json.loads(text)
            except json.JSONDecodeError as error:
                raise ValueError(f"Invalid JSON in {source} at line {line_number}.") from error
            if not isinstance(item, dict):
                raise ValueError(f"Expected object in {source} at line {line_number}.")
            yield item


def write_jsonl(path: str | Path, rows: Iterable[Mapping[str, Any]]) -> None:
    """Write JSONL deterministically enough for result and manifest inspection."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), sort_keys=True, ensure_ascii=False))
            handle.write("\n")


def append_jsonl(path: str | Path, row: Mapping[str, Any]) -> None:
    """Append one JSON object for resumable, single-process result writing."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(dict(row), sort_keys=True, ensure_ascii=False))
        handle.write("\n")


def require_fields(row: Mapping[str, Any], fields: Iterable[str], *, context: str) -> None:
    """Raise a clear error if a manifest row lacks a required field."""

    missing = [field for field in fields if field not in row or row[field] in (None, "")]
    if missing:
        joined = ", ".join(missing)
        raise ValueError(f"{context} is missing required field(s): {joined}.")


def resolve_manifest_path(value: str, manifest_path: str | Path) -> Path:
    """Resolve a path relative to the directory containing its manifest."""

    candidate = Path(value).expanduser()
    if candidate.is_absolute():
        return candidate
    return Path(manifest_path).resolve().parent / candidate
