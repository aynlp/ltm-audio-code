"""A dependency-free word-error-rate implementation."""

from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class WordErrorRate:
    substitutions: int
    deletions: int
    insertions: int
    reference_words: int

    @property
    def rate(self) -> float:
        if self.reference_words == 0:
            raise ZeroDivisionError("WER is undefined for an empty reference transcript.")
        return (self.substitutions + self.deletions + self.insertions) / self.reference_words


def normalize_transcript(text: str) -> list[str]:
    """Normalize an English transcript consistently before WER calculation."""

    return re.findall(r"[a-z0-9]+(?:'[a-z0-9]+)?", text.lower())


def word_error_rate(reference: str, hypothesis: str) -> WordErrorRate:
    """Return substitutions, deletions, insertions, and the resulting WER."""

    reference_words = normalize_transcript(reference)
    hypothesis_words = normalize_transcript(hypothesis)
    if not reference_words:
        raise ValueError("Reference transcript is empty after normalization.")

    # Each cell stores cost and the full error breakdown; the tuple ordering
    # gives deterministic handling to otherwise equivalent edit paths.
    table: list[list[tuple[int, int, int, int]]] = [
        [(0, 0, 0, 0) for _ in range(len(hypothesis_words) + 1)] for _ in range(len(reference_words) + 1)
    ]
    for row in range(1, len(reference_words) + 1):
        table[row][0] = (row, 0, row, 0)
    for column in range(1, len(hypothesis_words) + 1):
        table[0][column] = (column, 0, 0, column)

    for row, reference_word in enumerate(reference_words, start=1):
        for column, hypothesis_word in enumerate(hypothesis_words, start=1):
            if reference_word == hypothesis_word:
                table[row][column] = table[row - 1][column - 1]
                continue
            substitute = table[row - 1][column - 1]
            delete = table[row - 1][column]
            insert = table[row][column - 1]
            candidates = [
                (substitute[0] + 1, substitute[1] + 1, substitute[2], substitute[3]),
                (delete[0] + 1, delete[1], delete[2] + 1, delete[3]),
                (insert[0] + 1, insert[1], insert[2], insert[3] + 1),
            ]
            table[row][column] = min(candidates)

    _, substitutions, deletions, insertions = table[-1][-1]
    return WordErrorRate(
        substitutions=substitutions,
        deletions=deletions,
        insertions=insertions,
        reference_words=len(reference_words),
    )
