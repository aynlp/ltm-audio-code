#!/usr/bin/env python3
"""Construct fixed-duration target-plus-interferer mixtures from a JSONL recipe."""

from __future__ import annotations

import argparse
from pathlib import Path

from _bootstrap import bootstrap

bootstrap()

from ltm_ae.data.audio import construct_mixture, load_mono_audio, write_audio
from ltm_ae.data.manifests import read_jsonl, require_fields, resolve_manifest_path, write_jsonl
from ltm_ae.utils.paths import slugify


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recipe", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--target-to-interference-db", type=float, default=-10.0)
    parser.add_argument("--sample-rate", type=int, default=16_000)
    parser.add_argument("--duration-seconds", type=float, default=3.0)
    parser.add_argument("--num-interferers", type=int, default=3)
    return parser.parse_args()


def _validate_recipe_row(row: dict[str, object], row_number: int, expected_interferers: int) -> list[dict[str, object]]:
    require_fields(
        row,
        ["mixture_id", "target_category", "target_audio_path", "target_source_group", "interferers"],
        context=f"mixture recipe row {row_number}",
    )
    interferers = row["interferers"]
    if not isinstance(interferers, list) or len(interferers) != expected_interferers:
        raise ValueError(f"Recipe row {row_number} must contain exactly {expected_interferers} interferers.")
    checked: list[dict[str, object]] = []
    seen_categories = {str(row["target_category"])}
    seen_groups = {str(row["target_source_group"])}
    for interferer_index, item in enumerate(interferers, start=1):
        if not isinstance(item, dict):
            raise ValueError(f"Recipe row {row_number}, interferer {interferer_index} is not an object.")
        require_fields(
            item,
            ["category", "audio_path", "source_group"],
            context=f"recipe row {row_number}, interferer {interferer_index}",
        )
        category = str(item["category"])
        group = str(item["source_group"])
        if category in seen_categories:
            raise ValueError(f"Recipe row {row_number} reuses target or interferer category {category!r}.")
        if group in seen_groups:
            raise ValueError(f"Recipe row {row_number} reuses source group {group!r}.")
        seen_categories.add(category)
        seen_groups.add(group)
        checked.append(item)
    return checked


def main() -> None:
    args = parse_args()
    if args.num_interferers < 1:
        raise ValueError("num-interferers must be positive.")
    output_rows: list[dict[str, object]] = []
    seen_ids: set[str] = set()

    for row_number, row in enumerate(read_jsonl(args.recipe), start=1):
        interferers = _validate_recipe_row(row, row_number, args.num_interferers)
        mixture_id = str(row["mixture_id"])
        if mixture_id in seen_ids:
            raise ValueError(f"Duplicate mixture_id in recipe: {mixture_id!r}.")
        seen_ids.add(mixture_id)
        target = load_mono_audio(
            resolve_manifest_path(str(row["target_audio_path"]), args.recipe),
            sample_rate=args.sample_rate,
            duration_seconds=args.duration_seconds,
        )
        noise = [
            load_mono_audio(
                resolve_manifest_path(str(item["audio_path"]), args.recipe),
                sample_rate=args.sample_rate,
                duration_seconds=args.duration_seconds,
            )
            for item in interferers
        ]
        mixture, _ = construct_mixture(
            target,
            noise,
            target_to_interference_db=args.target_to_interference_db,
        )
        relative_path = Path("audio") / f"{slugify(mixture_id)}.wav"
        write_audio(args.output_dir / relative_path, mixture, sample_rate=args.sample_rate)
        output_rows.append(
            {
                "mixture_id": mixture_id,
                "target_category": str(row["target_category"]),
                "mixture_audio_path": relative_path.as_posix(),
                "sample_rate": args.sample_rate,
                "duration_seconds": args.duration_seconds,
                "target_to_interference_db": args.target_to_interference_db,
            }
        )
        print(f"Built mixture {row_number}: {mixture_id}")

    write_jsonl(args.output_dir / "mixtures.jsonl", output_rows)
    print(f"Wrote {len(output_rows)} mixtures to {args.output_dir}.")


if __name__ == "__main__":
    main()
