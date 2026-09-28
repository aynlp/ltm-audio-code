"""Safe, stable filenames for caller-supplied category identifiers."""

from __future__ import annotations

import re


def slugify(value: str) -> str:
    """Turn a human-readable category into a conservative filename component."""

    slug = re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")
    if not slug:
        raise ValueError(f"Cannot create a filename from {value!r}.")
    return slug
