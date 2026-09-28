"""Selection and reporting for absent-category LTM negative controls.

This analysis asks whether injecting a category memory changes free-form model
answers when that category, and optionally its broader source family, is absent
from every component of a mixture. It is an analysis of directional bias, not a
replacement for target-perception accuracy.
"""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

from ltm_ae.evaluation.classification import normalize_free_form_response


def canonical_category(label: str) -> str:
    """Normalize category labels for equality checks without changing report text."""

    return re.sub(r"[^a-z0-9]+", " ", label.lower()).strip()


@dataclass(frozen=True)
class ManifestFieldMap:
    """Field names used by a caller-owned mixture manifest."""

    example_id: str = "mixture_id"
    audio_path: str = "mixture_audio"
    split: str = "split"
    target_label: str = "target_label"
    target_source_group: str = "target_source_group_id"
    noise_labels: tuple[str, ...] = (
        "noise_1_label",
        "noise_2_label",
        "noise_3_label",
    )
    noise_source_groups: tuple[str, ...] = (
        "noise_1_source_group_id",
        "noise_2_source_group_id",
        "noise_3_source_group_id",
    )
    sample_rate: str = "sample_rate_hz"
    samples: str = "samples"
    headroom_gain: str = "common_peak_headroom_gain"
    target_rms: str = "target_rms"

    def __post_init__(self) -> None:
        if len(self.noise_labels) != len(self.noise_source_groups):
            raise ValueError("noise_labels and noise_source_groups must have equal length.")
        if not self.noise_labels:
            raise ValueError("At least one interferer field is required.")


@dataclass(frozen=True)
class CounterfactualCriteria:
    """Constraints for one absent-category counterfactual condition."""

    injected_category: str
    excluded_categories: tuple[str, ...]
    source_target_categories: tuple[str, ...]
    examples_per_source_category: int | None = None
    source_target_counts: tuple[tuple[str, int], ...] | None = None
    evaluation_split: str = "mixture_eval"
    expected_sample_rate: int | None = 16_000
    expected_samples: int | None = 48_000

    def __post_init__(self) -> None:
        if not self.source_target_categories:
            raise ValueError("source_target_categories cannot be empty.")
        if (self.examples_per_source_category is None) == (self.source_target_counts is None):
            raise ValueError(
                "Specify exactly one of examples_per_source_category or source_target_counts."
            )
        if self.examples_per_source_category is not None and self.examples_per_source_category < 1:
            raise ValueError("examples_per_source_category must be positive.")
        injected = canonical_category(self.injected_category)
        excluded = {canonical_category(category) for category in self.excluded_categories}
        if injected not in excluded:
            raise ValueError("excluded_categories must contain injected_category.")
        sources = [canonical_category(category) for category in self.source_target_categories]
        if len(sources) != len(set(sources)):
            raise ValueError("source_target_categories contains duplicates.")
        overlap = excluded & set(sources)
        if overlap:
            raise ValueError(
                "source_target_categories must be outside the excluded family: " + ", ".join(sorted(overlap))
            )
        if self.source_target_counts is not None:
            normalized_counts: dict[str, int] = {}
            for category, count in self.source_target_counts:
                normalized_category = canonical_category(category)
                if normalized_category in normalized_counts:
                    raise ValueError("source_target_counts contains duplicate categories.")
                if count < 1:
                    raise ValueError("Every source_target_counts value must be positive.")
                normalized_counts[normalized_category] = count
            if set(normalized_counts) != set(sources):
                raise ValueError(
                    "source_target_counts must define exactly the configured source_target_categories."
                )

    def requested_count(self, source_category: str) -> int:
        """Return the deterministic sample quota for one source target category."""

        if self.examples_per_source_category is not None:
            return self.examples_per_source_category
        assert self.source_target_counts is not None
        counts = {canonical_category(category): count for category, count in self.source_target_counts}
        return counts[canonical_category(source_category)]

    @property
    def total_examples(self) -> int:
        """Return the expected total sample count for this control."""

        return sum(self.requested_count(category) for category in self.source_target_categories)


@dataclass(frozen=True)
class CounterfactualExample:
    """A selected mixture with no identifying source metadata beyond local paths."""

    example_id: str
    audio_path: str
    source_target_category: str
    component_categories: tuple[str, ...]
    source_groups: tuple[str, ...]
    headroom_gain: float | None
    target_rms: float | None


def _required(row: Mapping[str, Any], field: str, *, context: str) -> Any:
    value = row.get(field)
    if value is None or value == "":
        raise ValueError(f"{context} is missing required field {field!r}.")
    return value


def _optional_float(row: Mapping[str, Any], field: str) -> float | None:
    value = row.get(field)
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"Field {field!r} must be numeric when present.") from error


def _candidate_from_row(
    row: Mapping[str, Any],
    *,
    source_category: str,
    criteria: CounterfactualCriteria,
    fields: ManifestFieldMap,
    row_number: int,
) -> CounterfactualExample | None:
    context = f"mixture manifest row {row_number}"
    if str(row.get(fields.split, "")) != criteria.evaluation_split:
        return None
    target_category = str(_required(row, fields.target_label, context=context))
    if canonical_category(target_category) != canonical_category(source_category):
        return None

    noise_categories = tuple(str(_required(row, field, context=context)) for field in fields.noise_labels)
    component_categories = (target_category, *noise_categories)
    normalized_components = tuple(canonical_category(label) for label in component_categories)
    if len(normalized_components) != len(set(normalized_components)):
        return None
    excluded = {canonical_category(category) for category in criteria.excluded_categories}
    if excluded & set(normalized_components):
        return None

    source_groups = (
        str(_required(row, fields.target_source_group, context=context)),
        *(str(_required(row, field, context=context)) for field in fields.noise_source_groups),
    )
    if len(source_groups) != len(set(source_groups)):
        return None
    if criteria.expected_sample_rate is not None:
        if int(_required(row, fields.sample_rate, context=context)) != criteria.expected_sample_rate:
            return None
    if criteria.expected_samples is not None:
        if int(_required(row, fields.samples, context=context)) != criteria.expected_samples:
            return None

    return CounterfactualExample(
        example_id=str(_required(row, fields.example_id, context=context)),
        audio_path=str(_required(row, fields.audio_path, context=context)),
        source_target_category=target_category,
        component_categories=component_categories,
        source_groups=source_groups,
        headroom_gain=_optional_float(row, fields.headroom_gain),
        target_rms=_optional_float(row, fields.target_rms),
    )


def select_absent_category_examples(
    rows: Iterable[Mapping[str, Any]],
    *,
    criteria: CounterfactualCriteria,
    fields: ManifestFieldMap = ManifestFieldMap(),
) -> list[CounterfactualExample]:
    """Select balanced, family-absent mixtures from a held-out manifest.

    Every selected mixture excludes the injected category *and all categories
    listed in ``excluded_categories``* from its target plus interferers. This
    is essential when an injected memory could otherwise move answers toward a
    broader family, such as a generic music description.
    """

    materialized_rows = list(rows)
    selected: list[CounterfactualExample] = []
    seen_ids: set[str] = set()
    for source_category in criteria.source_target_categories:
        requested_count = criteria.requested_count(source_category)
        candidates = [
            candidate
            for row_number, row in enumerate(materialized_rows, start=1)
            if (
                candidate := _candidate_from_row(
                    row,
                    source_category=source_category,
                    criteria=criteria,
                    fields=fields,
                    row_number=row_number,
                )
            )
            is not None
        ]
        # Higher retained headroom gain means less peak reduction. Prefer it,
        # then prefer stronger clean targets. The final ID tie-break is stable.
        candidates.sort(
            key=lambda item: (
                -(item.headroom_gain if item.headroom_gain is not None else float("-inf")),
                -(item.target_rms if item.target_rms is not None else float("-inf")),
                item.example_id,
            )
        )
        if len(candidates) < requested_count:
            raise ValueError(
                f"Only {len(candidates)} valid family-absent examples for {source_category!r}; "
                f"need {requested_count}."
            )
        for candidate in candidates[:requested_count]:
            if candidate.example_id in seen_ids:
                raise ValueError(f"Duplicate selected example_id {candidate.example_id!r}.")
            seen_ids.add(candidate.example_id)
            selected.append(candidate)

    expected = criteria.total_examples
    if len(selected) != expected:
        raise AssertionError(f"Selected {len(selected)} examples, expected {expected}.")
    return selected


def selection_design(
    criteria: CounterfactualCriteria,
    fields: ManifestFieldMap,
    *,
    prompt: str,
) -> dict[str, object]:
    """Return non-result metadata to save alongside a caller-owned run."""

    return {
        "analysis_type": "absent-category LTM negative control",
        "prompt": prompt,
        "criteria": asdict(criteria),
        "manifest_fields": asdict(fields),
        "interpretation": (
            "An increased target- or family-mention rate is evidence of directional bias; "
            "it is not a target-perception accuracy improvement."
        ),
    }


def _paired_binary_metrics(
    baseline: Sequence[bool],
    intervention: Sequence[bool],
    *,
    bootstrap_samples: int,
    bootstrap_seed: int,
) -> dict[str, object]:
    """Report paired binary shifts, including a paired bootstrap interval."""

    if len(baseline) != len(intervention) or not baseline:
        raise ValueError("Paired binary metrics require matching, non-empty inputs.")
    baseline_array = np.asarray(baseline, dtype=np.float64)
    intervention_array = np.asarray(intervention, dtype=np.float64)
    counts = Counter(zip(baseline, intervention, strict=True))
    delta = float(intervention_array.mean() - baseline_array.mean())
    result: dict[str, object] = {
        "baseline_positive_count": int(baseline_array.sum()),
        "baseline_positive_rate": float(baseline_array.mean()),
        "ltm_positive_count": int(intervention_array.sum()),
        "ltm_positive_rate": float(intervention_array.mean()),
        "paired_transitions": {
            "n00": counts[(False, False)],
            "n01": counts[(False, True)],
            "n10": counts[(True, False)],
            "n11": counts[(True, True)],
        },
        "paired_rate_difference": delta,
    }
    if bootstrap_samples > 0:
        generator = np.random.default_rng(bootstrap_seed)
        indices = generator.integers(0, len(baseline_array), size=(bootstrap_samples, len(baseline_array)))
        samples = intervention_array[indices].mean(axis=1) - baseline_array[indices].mean(axis=1)
        lower, upper = np.quantile(samples, [0.025, 0.975])
        result["paired_bootstrap_95_ci"] = [float(lower), float(upper)]
        result["bootstrap_samples"] = bootstrap_samples
        result["bootstrap_seed"] = bootstrap_seed
    return result


def _pattern_metrics(
    rows: Sequence[Mapping[str, Any]],
    pattern: re.Pattern[str],
    *,
    bootstrap_samples: int,
    bootstrap_seed: int,
) -> dict[str, object]:
    baseline = [bool(pattern.search(str(row["baseline_response"]))) for row in rows]
    intervention = [bool(pattern.search(str(row["ltm_response"]))) for row in rows]
    return _paired_binary_metrics(
        baseline,
        intervention,
        bootstrap_samples=bootstrap_samples,
        bootstrap_seed=bootstrap_seed,
    )


def _mapped_label_metrics(
    rows: Sequence[Mapping[str, Any]],
    *,
    aliases: Mapping[str, str],
    injected_output_label: str,
    bootstrap_samples: int,
    bootstrap_seed: int,
) -> dict[str, object]:
    baseline = [
        normalize_free_form_response(str(row["baseline_response"]), aliases) == injected_output_label
        for row in rows
    ]
    intervention = [
        normalize_free_form_response(str(row["ltm_response"]), aliases) == injected_output_label
        for row in rows
    ]
    return _paired_binary_metrics(
        baseline,
        intervention,
        bootstrap_samples=bootstrap_samples,
        bootstrap_seed=bootstrap_seed,
    )


def summarize_counterfactual(
    response_rows: Sequence[Mapping[str, Any]],
    *,
    target_pattern: str | None = None,
    family_pattern: str | None = None,
    aliases: Mapping[str, str] | None = None,
    injected_output_label: str | None = None,
    bootstrap_samples: int = 10_000,
    bootstrap_seed: int = 20260919,
) -> dict[str, object]:
    """Summarize target-specific and optional family-level response shifts.

    Response rows need the string fields ``baseline_response`` and
    ``ltm_response``. The primary endpoint is a pre-specified category mapping
    when aliases and an injected output label are supplied. Exact surface terms
    and broader-family terms are reported separately so generic semantic drift
    is not mistaken for a target-specific effect.
    """

    if not response_rows:
        raise ValueError("Cannot summarize an empty counterfactual response set.")
    for row_number, row in enumerate(response_rows, start=1):
        _required(row, "baseline_response", context=f"response row {row_number}")
        _required(row, "ltm_response", context=f"response row {row_number}")
    if bootstrap_samples < 0:
        raise ValueError("bootstrap_samples cannot be negative.")
    if (aliases is None) != (injected_output_label is None):
        raise ValueError("aliases and injected_output_label must be supplied together.")
    summary: dict[str, object] = {"examples": len(response_rows)}
    if aliases is not None and injected_output_label is not None:
        summary["injected_label_mapping"] = _mapped_label_metrics(
            response_rows,
            aliases=aliases,
            injected_output_label=injected_output_label,
            bootstrap_samples=bootstrap_samples,
            bootstrap_seed=bootstrap_seed,
        )
    if target_pattern is not None:
        summary["target_surface_mentions"] = _pattern_metrics(
            response_rows,
            re.compile(target_pattern, flags=re.IGNORECASE),
            bootstrap_samples=bootstrap_samples,
            bootstrap_seed=bootstrap_seed,
        )
    if family_pattern is not None:
        summary["family_level"] = _pattern_metrics(
            response_rows,
            re.compile(family_pattern, flags=re.IGNORECASE),
            bootstrap_samples=bootstrap_samples,
            bootstrap_seed=bootstrap_seed,
        )
    return summary
