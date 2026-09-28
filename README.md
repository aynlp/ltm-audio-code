# LTM-AE: Anonymous Code Release

This repository contains a clean, data-free implementation of long-term
memory-guided audio-token enhancement. It is prepared for anonymous peer
review: it contains source code, configuration templates, tests, and
reproduction instructions only.

The method builds a separate PCA memory for every `(model, target category)`
pair from clean-reference audio tokens. For an incoming mixture token `z`, it
projects the centered token onto a selected PCA subspace and interpolates the
projection with the original token. The language backbone remains frozen.

```text
clean reference tokens -> PCA memory -> validation calibration
incoming mixture tokens -> reconstruction + interpolation -> frozen decoder
```

## What is included

- Data-independent PCA-memory construction and calibrated token enhancement.
- Deterministic audio-mixture construction utilities.
- Prompt-free retrieval, constrained-label parsing, and WER utilities.
- A token-level gate implementation for the speech-transcription extension.
- An analysis-only absent-category negative-control runner with strict
  source-family exclusion, paired free-form decoding, and resumable outputs.
- Adapter contracts and model-specific integration helpers for Qwen2-Audio,
  Kimi-Audio, and Step-Audio-2.
- Synthetic tests for the mathematical invariants used by the method.

## What is deliberately excluded

No audio, manifests containing real paths, model weights, precomputed token
arrays, PCA memories, checkpoints, logs, predictions, figures, tables, or
Git history are included. Third-party model weights and datasets must be
obtained independently under their respective licenses.

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e '.[dev]'
pytest -q
```

The synthetic smoke test has no external model or dataset dependency:

```bash
python scripts/smoke_test.py
```

## Reproduction workflow

1. Prepare clean-reference, validation, and evaluation manifests from data to
   which you have access. Templates are in `configs/data/`.
2. Use the corresponding external model adapter to extract audio tokens. One
   memory is constructed per model and target category; do not share memories
   across models.
3. Build a memory archive and calibrate its retained rank and interpolation
   weight on the held-out validation split.
4. Apply the calibrated memory to mixture tokens before the frozen language
   backbone, then run the desired readout.
5. For the speech extension, train only the token gate; keep the memory and
   the audio-language model frozen.
6. Optionally run the absent-category control to measure category-directional
   bias. This is an analysis result, not target-perception accuracy.

Concrete commands and the manifest schema are documented in
[`docs/reproduction.md`](docs/reproduction.md). Every path supplied to a
script is local to the caller; no machine-specific paths are encoded in this
repository.

## Repository checks before release

```bash
python scripts/audit_release.py --root .
pytest -q
```

The audit reports common anonymity and accidental-artifact risks. It is a
guardrail rather than a replacement for a manual review before publishing.

## License

Released under the [MIT License](LICENSE).
