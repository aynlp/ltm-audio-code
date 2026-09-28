from __future__ import annotations

import numpy as np
import pytest

from ltm_ae.evaluation.classification import normalize_free_form_response, parse_constrained_response
from ltm_ae.evaluation.retrieval import macro_accuracy, predict_category
from ltm_ae.evaluation.wer import word_error_rate


def test_constrained_and_free_form_normalization() -> None:
    assert parse_constrained_response("Answer: B", {"A": "piano", "B": "dog"}) == "dog"
    assert parse_constrained_response("A or B", {"A": "piano", "B": "dog"}) is None
    assert normalize_free_form_response("A dog barking loudly", {"dog barking": "dog"}) == "dog"
    assert normalize_free_form_response("cat and dog", {"cat": "cat", "dog": "dog"}) is None


def test_prompt_free_retrieval_uses_top_k() -> None:
    gallery = {
        "x": np.array([[1.0, 0.0], [1.0, 0.0], [1.0, 0.0]], dtype=np.float32),
        "y": np.array([[0.0, 1.0], [0.0, 1.0], [0.0, 1.0]], dtype=np.float32),
    }
    prediction = predict_category(np.array([[0.9, 0.1]], dtype=np.float32), gallery, top_k=5)
    assert prediction.category == "x"
    assert macro_accuracy(["x", "y", "x"], ["x", "y", "y"]) == pytest.approx(0.75)


def test_word_error_rate_reports_all_error_types() -> None:
    result = word_error_rate("one two three", "one four")
    assert result.substitutions == 1
    assert result.deletions == 1
    assert result.insertions == 0
    assert result.rate == pytest.approx(2 / 3)
