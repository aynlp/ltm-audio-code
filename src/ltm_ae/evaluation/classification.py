"""Readout normalization for constrained and free-form category classification."""

from __future__ import annotations

import re
from collections.abc import Mapping

_SINGLE_LABEL = re.compile(r"\b([A-Z])\b")


def parse_constrained_response(response: str, label_to_category: Mapping[str, str]) -> str | None:
    """Extract a unique valid single-letter answer from a constrained response.

    Invalid or ambiguous output returns ``None`` rather than silently guessing.
    """

    candidates = [label for label in _SINGLE_LABEL.findall(response.strip()) if label in label_to_category]
    unique = list(dict.fromkeys(candidates))
    if len(unique) != 1:
        return None
    return str(label_to_category[unique[0]])


def normalize_free_form_response(response: str, aliases: Mapping[str, str]) -> str | None:
    """Map a free-form response to one category with caller-specified aliases.

    Longer aliases are tested first so a detailed category phrase wins over a
    shorter substring. Inputs are normalized conservatively to alphanumerics
    and spaces, and ambiguous matches return ``None``.
    """

    normalized = re.sub(r"[^a-z0-9 ]+", " ", response.lower())
    normalized = re.sub(r"\s+", " ", normalized).strip()
    matches: list[str] = []
    for alias, category in sorted(aliases.items(), key=lambda item: (-len(item[0]), item[0])):
        normalized_alias = re.sub(r"[^a-z0-9 ]+", " ", alias.lower())
        normalized_alias = re.sub(r"\s+", " ", normalized_alias).strip()
        if normalized_alias and re.search(rf"\b{re.escape(normalized_alias)}\b", normalized):
            matches.append(str(category))
    unique = list(dict.fromkeys(matches))
    return unique[0] if len(unique) == 1 else None
