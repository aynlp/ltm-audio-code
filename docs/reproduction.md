# Reproduction Protocol

This repository intentionally starts from caller-owned audio and caller-owned
model installations. It does not contain audio, source identifiers, model
weights, precomputed features, or results. Use a fresh output directory that
is ignored by Git.

## 1. Environment

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e '.[dev,torch]'
```

Install the dependencies required by the specific external audio-language
model separately. Pin its code revision and model revision in your local run
record. The code here does not download or select a model revision on your
behalf.

## 2. Prepare split-disjoint manifests

Create JSONL files according to the schemas in `configs/data/`:

- clean references for memory construction;
- paired mixed/clean examples for rank and interpolation calibration;
- a clean retrieval gallery;
- final mixture queries; and
- optionally, a speech-transcription development and held-out split.

Keep source groups disjoint across reference, validation, gallery, and final
evaluation roles. Do not choose ranks, interpolation weights, or gate loss
weights from the final test split.

Validate this before token extraction or evaluation:

```bash
python scripts/validate_splits.py \
  --reference local_data/reference_tokens.jsonl \
  --validation local_data/validation_pairs.jsonl \
  --gallery local_data/retrieval_gallery.jsonl \
  --evaluation local_data/retrieval_queries.jsonl
```

For the main protocol, standardize clips to mono 16 kHz, three seconds, then
construct one target plus three category-distinct interferers at a
target-to-combined-interference ratio of -10 dB:

```bash
python scripts/build_mixtures.py \
  --recipe local_data/mixture_recipe.jsonl \
  --output-dir local_data/mixtures
```

The mixture script rejects duplicate source groups and categories inside a
recipe row. It emits a minimal output manifest without copying source paths or
source identifiers into result rows.

## 3. Extract model-specific token arrays

Implement an external `AudioTokenAdapter` following
[`adapter_contract.md`](adapter_contract.md). For each model:

1. Extract continuous audio tokens at the exact interface immediately before
   the language backbone.
2. Save each token array as a floating `[time, dimension]` NumPy array.
3. Make clean and mixed pairs use the same model revision and preprocessing.
4. Do not share PCA memories between models, even when category labels match.

The Qwen2-Audio helper applies at the post-multimodal-projector interface.
The Kimi-Audio and Step-Audio-2 helpers apply at their post-audio-adaptor
interfaces. In every case, token count/order and all text embeddings remain
unchanged.

## 4. Build and calibrate memories

```bash
python scripts/build_memory.py \
  --reference-index local_data/reference_tokens.jsonl \
  --output-dir local_data/raw_memories/qwen2-audio \
  --model-id qwen2-audio \
  --reference-clips-per-category 20 \
  --max-components 128 --solver randomized

python scripts/calibrate_memory.py \
  --memory-dir local_data/raw_memories/qwen2-audio \
  --validation-index local_data/validation_pairs.jsonl \
  --output-dir local_data/calibrated_memories/qwen2-audio \
  --ranks 8,16,32,64,128 \
  --alphas 0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0
```

For each category, calibration minimizes token-level mean squared error between
enhanced mixture tokens and aligned clean target tokens. The resulting archive
contains only the mean, PCA basis, selected rank, selected interpolation
weight, and non-identifying construction metadata.

## 5. Run the readouts

Use the same target category as a listening cue to choose the corresponding
calibrated memory. Apply the enhancement before the frozen language decoder,
then generate output with either prompt template in `configs/prompts/`.

Score saved outputs with:

```bash
python scripts/score_readouts.py --mode constrained \
  --predictions local_data/constrained_outputs.jsonl \
  --mapping configs/prompts/constrained_labels.yaml \
  --output local_data/constrained_scores.jsonl

python scripts/evaluate_retrieval.py \
  --gallery-index local_data/retrieval_gallery.jsonl \
  --query-index local_data/retrieval_queries.jsonl \
  --memory-dir local_data/calibrated_memories/qwen2-audio \
  --output local_data/retrieval_scores.jsonl
```

Prompt-free retrieval temporally averages each query and gallery token sequence,
compares cosine similarities, and predicts the category with the highest mean
of its five best clean-gallery matches.

## 6. Absent-category negative control

`configs/experiments/absent_category_control.yaml` specifies an analysis-only
counterfactual: inject a category memory into mixtures for which that category
and its entire source family are absent from all four components. The default
configuration injects the piano memory while excluding every music class. It
uses 100 held-out, all-non-music mixtures with fixed per-target quotas.

First validate the deterministic selection without loading a model:

```bash
python scripts/run_absent_category_control.py \
  --config configs/experiments/absent_category_control.yaml \
  --mixture-manifest local_data/mixture_eval.jsonl \
  --memory local_data/calibrated_memories/qwen2-audio/piano.npz \
  --adapter-factory local_adapters.qwen2_audio:build_adapter \
  --adapter-kwargs-json local_data/qwen2_adapter.json \
  --output-dir runs/absent_piano_nonmusic/qwen2-audio \
  --dry-run
```

The runner extracts each audio input once, then decodes the unmodified and
enhanced token sequences under identical prompt and decoding settings.
Adapters that implement `generate_native` can additionally validate their
untouched native path before the paired run:

```bash
python scripts/run_absent_category_control.py \
  --config configs/experiments/absent_category_control.yaml \
  --mixture-manifest local_data/mixture_eval.jsonl \
  --memory local_data/calibrated_memories/qwen2-audio/piano.npz \
  --adapter-factory local_adapters.qwen2_audio:build_adapter \
  --adapter-kwargs-json local_data/qwen2_adapter.json \
  --output-dir runs/absent_piano_nonmusic/qwen2-audio \
  --verify-native-baseline
```

The output directory records a run fingerprint over the configuration,
manifest, memory archive, free-form alias mapping, and decoding setup. A
resumed run with any mismatch is rejected. Do not commit `design.json`,
selected-example lists, responses, summaries, or native-path checks: they may
contain local paths, source metadata, or generated model text.

Interpret this control as directional-bias analysis only. Its primary endpoint
is the pre-specified free-form mapping to the injected category, with paired
`n01`/`n10` transitions and a paired bootstrap interval. Surface mentions of
the injected category and generic music wording are secondary diagnostics; a
generic music shift does not establish piano-specific perception.

Further details are in [`absent_category_control.md`](absent_category_control.md).

## 7. Speech token gate

For the speech extension, use a separate speech memory and a development split
with 80 training and 20 selection examples. Sweep the token reconstruction
weight from 0.0 through 1.0 in steps of 0.1; break an exact WER tie by choosing
the smaller reconstruction weight. Retrain the selected setup on all 100
development examples before the held-out evaluation.

`scripts/train_token_gate.py` accepts a caller-owned factory that returns a
`GateTrainingBundle`. That factory loads the external frozen decoder, provides
teacher-forcing transcription loss, and constructs batches. The package checks
that external modules are frozen and updates only the gate.

## 8. Verification

```bash
pytest -q
python scripts/smoke_test.py
python scripts/audit_release.py --root .
```

Before publishing, delete generated outputs and re-run the audit from a clean
checkout. Review every tracked file manually; automated checks cannot prove
that an identifier is non-sensitive.
