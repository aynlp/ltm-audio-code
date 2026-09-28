#!/usr/bin/env python3
"""Run a free-form absent-category LTM negative-control analysis.

The selected mixtures must not contain the injected category or any category in
its excluded source family. For every example, the script uses the same prompt
and decoding settings for the baseline and LTM-injected condition.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from _bootstrap import bootstrap

bootstrap()

from ltm_ae.adapters.loading import load_adapter
from ltm_ae.analysis.absent_category import (
    CounterfactualCriteria,
    ManifestFieldMap,
    select_absent_category_examples,
    selection_design,
    summarize_counterfactual,
)
from ltm_ae.data.manifests import append_jsonl, read_jsonl, resolve_manifest_path, write_jsonl
from ltm_ae.memory.archive import CalibratedMemory, load_calibrated_memory


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--mixture-manifest", type=Path, required=True)
    parser.add_argument("--memory", type=Path, required=True)
    parser.add_argument("--adapter-factory", required=True, help="module.path:callable")
    parser.add_argument(
        "--adapter-kwargs-json",
        type=Path,
        default=None,
        help="Local JSON object passed to the external adapter factory; do not commit it.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument(
        "--verify-native-baseline",
        action="store_true",
        help="Require the adapter's optional native-path equivalence preflight before the paired run.",
    )
    parser.add_argument("--native-baseline-examples", type=int, default=8)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--summarize-only", action="store_true")
    parser.add_argument("--skip-audio-validation", action="store_true")
    return parser.parse_args()


def _load_config(path: Path) -> dict[str, Any]:
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError("Counterfactual config must contain a YAML object.")
    return config


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _criteria_from_config(config: dict[str, Any]) -> tuple[CounterfactualCriteria, ManifestFieldMap, str]:
    selection = config.get("selection")
    if not isinstance(selection, dict):
        raise ValueError("Config must contain a selection object.")
    fields_config = config.get("manifest_fields", {})
    if not isinstance(fields_config, dict):
        raise ValueError("manifest_fields must be an object when present.")
    fields = ManifestFieldMap(
        **{
            key: tuple(value) if key in {"noise_labels", "noise_source_groups"} else value
            for key, value in fields_config.items()
        }
    )
    criteria = CounterfactualCriteria(
        injected_category=str(selection["injected_category"]),
        excluded_categories=tuple(str(item) for item in selection["excluded_categories"]),
        source_target_categories=tuple(str(item) for item in selection["source_target_categories"]),
        examples_per_source_category=(
            int(selection["examples_per_source_category"])
            if selection.get("examples_per_source_category") is not None
            else None
        ),
        source_target_counts=(
            tuple((str(category), int(count)) for category, count in selection["source_target_counts"].items())
            if isinstance(selection.get("source_target_counts"), dict)
            else None
        ),
        evaluation_split=str(selection.get("evaluation_split", "mixture_eval")),
        expected_sample_rate=(
            int(selection["expected_sample_rate"])
            if selection.get("expected_sample_rate") is not None
            else None
        ),
        expected_samples=(
            int(selection["expected_samples"])
            if selection.get("expected_samples") is not None
            else None
        ),
    )
    prompt = str(config.get("prompt", "What is the main sound in this audio?"))
    return criteria, fields, prompt


def _load_adapter_kwargs(path: Path | None) -> dict[str, Any]:
    if path is None:
        return {}
    values = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(values, dict):
        raise ValueError("adapter-kwargs-json must contain a JSON object.")
    return values


def _load_response_metrics(config: dict[str, Any], config_path: Path) -> dict[str, Any]:
    metrics = config.get("response_metrics")
    if not isinstance(metrics, dict):
        raise ValueError("Config must contain a response_metrics object.")
    alias_reference = metrics.get("aliases")
    aliases: dict[str, str] | None = None
    aliases_sha256: str | None = None
    if alias_reference is not None:
        aliases_path = Path(str(alias_reference))
        if not aliases_path.is_absolute():
            aliases_path = config_path.parent / aliases_path
        aliases_path = aliases_path.resolve()
        raw_aliases = yaml.safe_load(aliases_path.read_text(encoding="utf-8"))
        if not isinstance(raw_aliases, dict) or not all(
            isinstance(alias, str) and isinstance(category, str) for alias, category in raw_aliases.items()
        ):
            raise ValueError("response_metrics aliases must resolve to a YAML string-to-string mapping.")
        aliases = {str(alias): str(category) for alias, category in raw_aliases.items()}
        aliases_sha256 = _sha256_file(aliases_path)
    injected_output_label = metrics.get("injected_output_label")
    if (aliases is None) != (injected_output_label is None):
        raise ValueError("response_metrics aliases and injected_output_label must be configured together.")
    return {
        "aliases": aliases,
        "aliases_sha256": aliases_sha256,
        "injected_output_label": str(injected_output_label) if injected_output_label is not None else None,
        "target_pattern": str(metrics["target_pattern"]) if metrics.get("target_pattern") else None,
        "family_pattern": str(metrics["family_pattern"]) if metrics.get("family_pattern") else None,
        "bootstrap_samples": int(metrics.get("bootstrap_samples", 10_000)),
        "bootstrap_seed": int(metrics.get("bootstrap_seed", 20260919)),
    }


def _validate_audio(path: Path, *, expected_rate: int | None, expected_samples: int | None) -> None:
    try:
        import soundfile as sf
    except ImportError as error:  # pragma: no cover - environment specific
        raise ImportError("Audio validation requires soundfile.") from error
    info = sf.info(path)
    if expected_rate is not None and info.samplerate != expected_rate:
        raise ValueError(f"Unexpected sample rate in {path}: {info.samplerate}.")
    if expected_samples is not None and info.frames != expected_samples:
        raise ValueError(f"Unexpected sample count in {path}: {info.frames}.")
    waveform, _ = sf.read(path, dtype="float32", always_2d=False)
    if not np.isfinite(waveform).all():
        raise ValueError(f"Non-finite audio values in {path}.")
    peak = float(np.abs(waveform).max(initial=0.0))
    rms = float(np.sqrt(np.mean(np.square(waveform, dtype=np.float64))))
    if rms <= 1e-5 or peak > 1.0001:
        raise ValueError(f"Low-quality or clipped audio in {path}: peak={peak}, rms={rms}.")


def _enhance_adapter_tokens(tokens: Any, item: CalibratedMemory) -> Any:
    if isinstance(tokens, np.ndarray):
        if tokens.ndim != 3:
            raise ValueError("Adapter tokens must have shape [batch, time, dimension].")
        return np.stack(
            [
                item.memory.enhance(sequence, item.calibration.rank, item.calibration.alpha)
                for sequence in tokens
            ]
        )
    if not hasattr(tokens, "detach"):
        raise TypeError("Adapter tokens must be a NumPy array or a torch tensor.")
    values = tokens.detach()
    if values.ndim != 3:
        raise ValueError("Adapter tokens must have shape [batch, time, dimension].")
    enhanced = np.stack(
        [
            item.memory.enhance(sequence.cpu().numpy(), item.calibration.rank, item.calibration.alpha)
            for sequence in values
        ]
    )
    return values.new_tensor(enhanced)


def _existing_results(path: Path, *, run_id: str) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    rows = list(read_jsonl(path))
    indexed = {str(row["example_id"]): row for row in rows}
    if len(indexed) != len(rows):
        raise ValueError(f"Duplicate example_id in existing result file: {path}")
    if any(str(row.get("run_id", "")) != run_id for row in rows):
        raise ValueError("Existing responses belong to a different run fingerprint; use a new output directory.")
    return indexed


def _verify_or_write_design(path: Path, design: dict[str, object], *, has_results: bool) -> None:
    """Prevent a resumed run from mixing results from two different designs."""

    if path.exists() and has_results:
        existing = json.loads(path.read_text(encoding="utf-8"))
        if existing != design:
            raise ValueError(
                "Existing responses belong to a different counterfactual design; use a new output directory."
            )
        return
    path.write_text(json.dumps(design, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _verify_or_write_selected(
    path: Path,
    selected_rows: list[dict[str, object]],
    *,
    has_results: bool,
) -> None:
    """Keep a resumed run tied to the exact selected mixture list."""

    if path.exists() and has_results:
        existing = list(read_jsonl(path))
        if existing != selected_rows:
            raise ValueError(
                "Existing responses do not match this selected mixture list; use a new output directory."
            )
        return
    write_jsonl(path, selected_rows)


def _run_design(
    *,
    args: argparse.Namespace,
    criteria: CounterfactualCriteria,
    fields: ManifestFieldMap,
    prompt: str,
    response_metrics: dict[str, Any],
) -> dict[str, object]:
    """Fingerprint inputs that can invalidate an interrupted paired run."""

    design: dict[str, object] = {
        "schema_version": 1,
        "selection": selection_design(criteria, fields, prompt=prompt),
        "max_new_tokens": args.max_new_tokens,
        "adapter_factory": args.adapter_factory,
        "input_fingerprints": {
            "config_sha256": _sha256_file(args.config),
            "manifest_sha256": _sha256_file(args.mixture_manifest),
            "memory_sha256": _sha256_file(args.memory),
            "response_aliases_sha256": response_metrics["aliases_sha256"],
        },
    }
    payload = json.dumps(design, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    design["run_id"] = hashlib.sha256(payload).hexdigest()
    return design


def _summarize(rows: list[dict[str, Any]], response_metrics: dict[str, Any]) -> dict[str, object]:
    return summarize_counterfactual(
        rows,
        target_pattern=response_metrics["target_pattern"],
        family_pattern=response_metrics["family_pattern"],
        aliases=response_metrics["aliases"],
        injected_output_label=response_metrics["injected_output_label"],
        bootstrap_samples=response_metrics["bootstrap_samples"],
        bootstrap_seed=response_metrics["bootstrap_seed"],
    )


def _validate_completed_rows(
    rows: list[dict[str, Any]],
    *,
    expected_ids: set[str],
    run_id: str,
) -> None:
    observed_ids = [str(row["example_id"]) for row in rows]
    if len(observed_ids) != len(set(observed_ids)):
        raise ValueError("Response file contains duplicate example IDs.")
    if set(observed_ids) != expected_ids:
        raise ValueError("Response file does not match the selected counterfactual examples.")
    if any(str(row.get("run_id", "")) != run_id for row in rows):
        raise ValueError("Response file belongs to a different run fingerprint.")


def _verify_native_baseline(
    *,
    adapter: Any,
    examples: list[Any],
    audio_paths: list[Path],
    prompt: str,
    max_new_tokens: int,
    count: int,
    output_path: Path,
    run_id: str,
) -> None:
    """Check optional native decoding against unmodified injected-token decoding."""

    if count < 1:
        raise ValueError("native-baseline-examples must be positive.")
    rows: list[dict[str, object]] = []
    for example, audio_path in list(zip(examples, audio_paths, strict=True))[:count]:
        native = adapter.generate_native([audio_path], [prompt], max_new_tokens=max_new_tokens)
        if native is None:
            raise ValueError("This adapter does not implement generate_native for the requested preflight.")
        batch = adapter.extract([audio_path], [prompt])
        injected = adapter.generate_from_tokens(
            batch,
            batch.tokens,
            [prompt],
            max_new_tokens=max_new_tokens,
        )
        if len(native) != 1 or len(injected) != 1:
            raise ValueError("Native baseline preflight expects one response per one-example batch.")
        rows.append(
            {
                "run_id": run_id,
                "example_id": example.example_id,
                "native_response": native[0],
                "injected_baseline_response": injected[0],
                "exact_match": native[0] == injected[0],
            }
        )
    write_jsonl(output_path, rows)
    mismatches = sum(not bool(row["exact_match"]) for row in rows)
    if mismatches:
        raise ValueError(f"Native baseline preflight found {mismatches}/{len(rows)} decoding mismatches.")


def main() -> None:
    args = parse_args()
    if args.max_new_tokens < 1:
        raise ValueError("max-new-tokens must be positive.")
    if args.native_baseline_examples < 1:
        raise ValueError("native-baseline-examples must be positive.")
    config = _load_config(args.config)
    criteria, fields, prompt = _criteria_from_config(config)
    response_metrics = _load_response_metrics(config, args.config)
    examples = select_absent_category_examples(
        read_jsonl(args.mixture_manifest),
        criteria=criteria,
        fields=fields,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    result_path = args.output_dir / "responses.jsonl"
    design = _run_design(
        args=args,
        criteria=criteria,
        fields=fields,
        prompt=prompt,
        response_metrics=response_metrics,
    )
    _verify_or_write_design(args.output_dir / "design.json", design, has_results=result_path.exists())
    selected_rows = [
        {
            "example_id": example.example_id,
            "audio_path": example.audio_path,
            "source_target_category": example.source_target_category,
            "component_categories": list(example.component_categories),
            "source_groups": list(example.source_groups),
        }
        for example in examples
    ]
    _verify_or_write_selected(
        args.output_dir / "selected_examples.jsonl",
        selected_rows,
        has_results=result_path.exists(),
    )
    resolved_audio_paths = [resolve_manifest_path(example.audio_path, args.mixture_manifest) for example in examples]
    if not args.skip_audio_validation:
        for path in resolved_audio_paths:
            _validate_audio(
                path,
                expected_rate=criteria.expected_sample_rate,
                expected_samples=criteria.expected_samples,
            )
    if args.dry_run:
        print(f"Validated {len(examples)} family-absent counterfactual mixtures; no model loaded.")
        return

    if args.summarize_only:
        rows = list(read_jsonl(result_path))
        expected_ids = {example.example_id for example in examples}
        _validate_completed_rows(rows, expected_ids=expected_ids, run_id=str(design["run_id"]))
        report = _summarize(rows, response_metrics)
        (args.output_dir / "summary.json").write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        print(f"Summarized {len(rows)} paired responses.")
        return

    adapter = load_adapter(args.adapter_factory, **_load_adapter_kwargs(args.adapter_kwargs_json))
    item = load_calibrated_memory(args.memory)
    adapter.validate_memory_metadata(item.memory.metadata)
    expected_ids = {example.example_id for example in examples}
    completed = _existing_results(result_path, run_id=str(design["run_id"]))
    unexpected_ids = set(completed) - expected_ids
    if unexpected_ids:
        raise ValueError("Existing response file contains examples outside this selected design.")
    if args.verify_native_baseline:
        _verify_native_baseline(
            adapter=adapter,
            examples=examples,
            audio_paths=resolved_audio_paths,
            prompt=prompt,
            max_new_tokens=args.max_new_tokens,
            count=args.native_baseline_examples,
            output_path=args.output_dir / "native_baseline_check.jsonl",
            run_id=str(design["run_id"]),
        )
    for index, (example, audio_path) in enumerate(zip(examples, resolved_audio_paths, strict=True), start=1):
        if example.example_id in completed:
            continue
        batch = adapter.extract([audio_path], [prompt])
        baseline = adapter.generate_from_tokens(
            batch,
            batch.tokens,
            [prompt],
            max_new_tokens=args.max_new_tokens,
        )[0]
        intervention = adapter.generate_from_tokens(
            batch,
            _enhance_adapter_tokens(batch.tokens, item),
            [prompt],
            max_new_tokens=args.max_new_tokens,
        )[0]
        append_jsonl(
            result_path,
            {
                "run_id": design["run_id"],
                "example_id": example.example_id,
                "source_target_category": example.source_target_category,
                "component_categories": list(example.component_categories),
                "model_id": adapter.model_id,
                "baseline_response": baseline,
                "ltm_response": intervention,
            },
        )
        print(f"Completed {index}/{len(examples)}: {example.example_id}", flush=True)

    rows = list(read_jsonl(result_path))
    if len(rows) != len(examples):
        raise ValueError(f"Expected {len(examples)} responses, found {len(rows)}.")
    _validate_completed_rows(rows, expected_ids=expected_ids, run_id=str(design["run_id"]))
    report = _summarize(rows, response_metrics)
    (args.output_dir / "summary.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"Completed {len(rows)} paired responses.")


if __name__ == "__main__":
    main()
