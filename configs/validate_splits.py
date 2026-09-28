#!/usr/bin/env python3
"""Validate source-group disjointness across experiment split manifests."""

from __future__ import annotations

import argparse
from pathlib import Path

from _bootstrap import bootstrap

bootstrap()

from ltm_ae.data.manifests import read_jsonl, require_fields


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--validation", type=Path, required=True)
    parser.add_argument("--gallery", type=Path, required=True)
    parser.add_argument("--evaluation", type=Path, required=True)
    return parser.parse_args()


def read_groups(path: Path, role: str) -> set[str]:
    groups: set[str] = set()
    for row_number, row in enumerate(read_jsonl(path), start=1):
        require_fields(row, ["source_group"], context=f"{role} row {row_number}")
        groups.add(str(row["source_group"]))
    if not groups:
        raise ValueError(f"{role} manifest is empty.")
    return groups


def main() -> None:
    args = parse_args()
    roles = {
        "reference": read_groups(args.reference, "reference"),
        "validation": read_groups(args.validation, "validation"),
        "gallery": read_groups(args.gallery, "gallery"),
        "evaluation": read_groups(args.evaluation, "evaluation"),
    }
    names = sorted(roles)
    overlaps: list[str] = []
    for index, left_name in enumerate(names):
        for right_name in names[index + 1 :]:
            shared = sorted(roles[left_name] & roles[right_name])
            if shared:
                preview = ", ".join(shared[:10])
                suffix = " ..." if len(shared) > 10 else ""
                overlaps.append(f"{left_name}/{right_name}: {preview}{suffix}")
    if overlaps:
        raise ValueError("Source groups overlap across split roles:\n" + "\n".join(overlaps))
    print("Split validation passed: all source groups are role-disjoint.")


if __name__ == "__main__":
    main()
