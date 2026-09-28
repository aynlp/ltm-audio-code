#!/usr/bin/env python3
"""Check a release tree for common anonymity and accidental-artifact risks."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

TEXT_SUFFIXES = {
    ".cfg",
    ".ini",
    ".json",
    ".md",
    ".py",
    ".rst",
    ".sh",
    ".toml",
    ".txt",
    ".yaml",
    ".yml",
}
FORBIDDEN_SUFFIXES = {
    ".bin",
    ".ckpt",
    ".csv",
    ".flac",
    ".m4a",
    ".mp3",
    ".npy",
    ".npz",
    ".ogg",
    ".pt",
    ".pth",
    ".safetensors",
    ".wav",
    ".jsonl",
}
FORBIDDEN_TOP_LEVEL = {"artifacts", "cache", "checkpoints", "data", "logs", "models", "outputs", "runs"}
PATTERNS = {
    "absolute-user-path": re.compile(r"/(?:Users|home|root)/"),
    "machine-hostname": re.compile(r"\b(?:autodl|localhost)\b", re.IGNORECASE),
    "email-address": re.compile(r"\b[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}\b"),
    "credential-assignment": re.compile(
        r"\b(?:password|passwd|api[_-]?key|access[_-]?token|secret)\b\s*[:=]",
        re.IGNORECASE,
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--max-file-bytes", type=int, default=2_000_000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    findings: list[str] = []
    for top_level in FORBIDDEN_TOP_LEVEL:
        if (root / top_level).exists():
            findings.append(f"forbidden top-level artifact directory: {top_level}/")

    for path in sorted(root.rglob("*")):
        relative = path.relative_to(root)
        if any(part in {".git", "__pycache__", ".venv"} for part in relative.parts):
            continue
        if relative == Path("scripts/audit_release.py"):
            # The detector definitions intentionally contain the strings it detects.
            continue
        if path.is_dir():
            continue
        if path.suffix.lower() in FORBIDDEN_SUFFIXES:
            findings.append(f"forbidden artifact suffix: {relative}")
            continue
        if path.stat().st_size > args.max_file_bytes:
            findings.append(f"unexpectedly large file ({path.stat().st_size} bytes): {relative}")
        if path.suffix.lower() not in TEXT_SUFFIXES:
            continue
        try:
            lines = path.read_text(encoding="utf-8").splitlines()
        except UnicodeDecodeError:
            findings.append(f"non-UTF-8 text candidate: {relative}")
            continue
        for line_number, line in enumerate(lines, start=1):
            for name, pattern in PATTERNS.items():
                if pattern.search(line):
                    # Do not print the matching text: an audit must not echo a secret.
                    findings.append(f"{name}: {relative}:{line_number}")

    if findings:
        print("Release audit failed:")
        for finding in findings:
            print(f"- {finding}")
        raise SystemExit(1)
    print(f"Release audit passed: {root}")


if __name__ == "__main__":
    main()
