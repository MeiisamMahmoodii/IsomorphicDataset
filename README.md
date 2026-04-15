# IsomorphicDataset

IsomorphicDataset is a research pipeline for constructing and validating a **surface-diverse but representation-consistent** text dataset across multiple LLMs. It operationalizes *latent space isomorphism* by:

- generating **per-example constraints** (banned words + target length bins),
- rewriting each example with a **fleet of models (one model at a time)** under those constraints (with retries),
- extracting **last-layer** representations (mean pooling, last token, attention-weighted),
- learning **orthogonal Procrustes alignments** across models and pruning misaligned items,
- applying a final **reference-model gate** (Gemma-4-31B) using Wasserstein-style distance + cosine similarity,
- exporting a final dataset with full provenance and run artifacts for publication-grade reproducibility.

---

## Project objective (start → finish)

### Phase A — Reference constraint table

Using a large **reference model** (default: `ModelRewriter.MODELS[7]`, Gemma-4-31B), the pipeline reads a raw dataset and produces a constraint table where each row contains:

- **seed sentence**
- **banned words** (to force surface-level divergence)
- **length bins**: **5–10 words** and **15–20 words**
- **rewrite specs / prompts** (persisted for reproducibility)

Implementation: [`isomorphic/datasets/reference_constraints.py`](isomorphic/datasets/reference_constraints.py), persisted as `phase_a/constraint_table.jsonl`.

### Phase B — Per-model constrained rewriting (with retries)

For each model in your experiment config, the pipeline:

- loads **one model at a time**,
- rewrites each seed for each length bin,
- validates constraints (word count + banned words),
- retries up to **5 attempts** per (seed, length bin, model),
- logs all attempts.

Implementation: [`isomorphic/generation/rewriter.py`](isomorphic/generation/rewriter.py), validation helpers in [`isomorphic/generation/constraint_utils.py`](isomorphic/generation/constraint_utils.py).

### Phase C — Embedding extraction (last layer)

For each accepted rewrite, extract **three last-layer views**:

- `mean_pooling`
- `last_token`
- `attention_weighted`

Implementation: [`isomorphic/extractors/base_extractor.py`](isomorphic/extractors/base_extractor.py).

### Phase D — Cross-model alignment + pruning (pooling ablation)

Compute alignments with **orthogonal Procrustes (SVD)** and prune misaligned rows. The current implementation uses a **hub** strategy:

- choose the **first configured model** as the hub,
- align hub → each other model,
- intersect keep-masks,
- repeat for each pooling method, pick the best pooling method by mean alignment quality.

Implementation: [`isomorphic/alignment/procrustes.py`](isomorphic/alignment/procrustes.py), helpers in [`isomorphic/alignment/evaluation.py`](isomorphic/alignment/evaluation.py).

### Phase E — Reference gate (Gemma-4-31B)

Run a final semantic consistency gate using the reference model:

- compute **reference cosine** and a **Wasserstein-style** distance in the *single* reference latent space,
- drop entries that fail configured thresholds.

Implementation: [`isomorphic/validators/semantic_judge.py`](isomorphic/validators/semantic_judge.py).

### Phase F — Artifacts, report, final dataset

Each run writes:

- `metrics.json`
- `config.json`
- `EXPERIMENT_REPORT.md`
- `final_dataset.jsonl`
- Phase-A constraint table: `phase_a/constraint_table.jsonl`

Implementation: [`isomorphic/pipeline.py`](isomorphic/pipeline.py), reporter: [`isomorphic/utils/reporting.py`](isomorphic/utils/reporting.py).

---

## Repository structure (high-signal)

- [`main.py`](main.py): CLI entrypoint (loads config and runs pipeline)
- [`config/`](config/): YAML configs (default + smoke test + multiconcept baseline)
- [`isomorphic/pipeline.py`](isomorphic/pipeline.py): end-to-end orchestration
- [`isomorphic/datasets/`](isomorphic/datasets/): dataset loaders + constraint builder
- [`isomorphic/generation/`](isomorphic/generation/): rewriting + constraint checks
- [`isomorphic/extractors/`](isomorphic/extractors/): embedding extraction
- [`isomorphic/alignment/`](isomorphic/alignment/): Procrustes + evaluation helpers
- [`isomorphic/validators/`](isomorphic/validators/): reference judge
- [`scripts/eval_funnel.py`](scripts/eval_funnel.py): print useful run summaries from `metrics.json`
- [`docs/HF_MODEL_PINNING.md`](docs/HF_MODEL_PINNING.md): reproducibility guidance

---

## Setup

### Dependencies

This project uses PyTorch + Transformers. GPU is strongly recommended (VRAM requirements depend on chosen models).

- Install via `pip install -r requirements.txt`, or
- If you use `uv`, run `uv sync`.

From [`requirements.txt`](requirements.txt), PyTorch is pinned to CUDA 11.8 wheels.

### Environment (recommended)

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

---

## How to run

### 1) Smoke test (tiny dataset)

```bash
python3 main.py --config config/smoke_test.yaml
```

Note: hub alignment requires **≥ 2 models** in the config; if your smoke config only has one model, alignment is skipped.

### 2) Default run

```bash
python3 main.py --config config/default.yaml
```

### 3) Full fleet run (Trevor reference config)

```bash
python3 main.py --config config/full_fleet_trevor_reference.yaml
```

This config runs the full Huihui/Llama rewrite fleet and is intended for production-scale experiments.

### 4) Custom output directory

```bash
python3 main.py --config config/default.yaml --output experiments/results
```

---

## Outputs

Runs are stored under:

```
experiments/results/<experiment_name>_<timestamp>/
```

Key artifacts:

- **`phase_a/constraint_table.jsonl`**: seeds + banned words + length bins + persisted rewrite prompts
- **`metrics.json`**: funnel + alignment + reference-gate metrics
- **`EXPERIMENT_REPORT.md`**: report generated from metrics
- **`final_dataset.jsonl`**: the final accepted dataset rows (seed + all rewrites + metadata)

To print a quick summary from a run:

```bash
python3 scripts/eval_funnel.py experiments/results/<run>/metrics.json
```

---

## Configuration

Configs are YAML files under [`config/`](config/). The main knobs you’ll care about:

- **Models**: `experiment.models[]`
- **Datasets**: `experiment.datasets[]`
- **Reference model (config intent)**: `experiment.reference_model`
- **Alignment thresholds** (Phase D/E):
  - `experiment.alignment.alignment_quality_threshold`
  - `experiment.alignment.wasserstein_threshold`
  - `experiment.alignment.reference_cosine_min`

Examples:

- [`config/default.yaml`](config/default.yaml): baseline two-model run
- [`config/smoke_test.yaml`](config/smoke_test.yaml): tiny sanity run
- [`config/full_fleet_trevor_reference.yaml`](config/full_fleet_trevor_reference.yaml): full production fleet
- [`config/multiconcept_constraint_baseline.yaml`](config/multiconcept_constraint_baseline.yaml): multiconcept baseline

Note: the current pipeline code uses an internal default reference model index for Phase A/Phase E. If you want `experiment.reference_model` to be the active runtime source-of-truth, wire that field in `isomorphic/pipeline.py`.

---

## Tests

Lightweight unit tests (no external deps like `pytest`) are under [`tests/`](tests/).

```bash
python3 -m unittest tests.test_constraint_utils -v
```

---

## Reproducibility checklist (NeurIPS-ready)

Recommended for every experiment run:

- **Pin Hub revisions** for every model used (see [`docs/HF_MODEL_PINNING.md`](docs/HF_MODEL_PINNING.md)).
- Save `config.json`, `metrics.json`, and the Phase A constraint table.
- Report:
  - constraint success rates and retry histogram,
  - pooling-method ablation (which representation aligns best),
  - alignment/pruning yield (funnel),
  - reference-gate distributions (Wasserstein + cosine),
  - compute budget (GPU hours / tokens / model list).

---

## License / notes

Model and dataset licenses vary by source (Hugging Face datasets/models). For publication, ensure your chosen datasets/models are compatible with redistribution and your target track’s dataset release policies.

