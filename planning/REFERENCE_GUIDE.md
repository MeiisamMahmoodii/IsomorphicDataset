# Project Reference Guide: Quick Decisions & Architecture
## IsomorphicDataSet - NeurIPS Ready Implementation

---

## 1. PROJECT OVERVIEW DIAGRAM

```
RESEARCH QUESTION:
Do different LLMs encode the same concepts in isomorphic latent spaces?

         ┌─────────────────────────────────────────────────────┐
         │  Input: Seed sentences (from 5 datasets)            │
         └────────────────┬────────────────────────────────────┘
                          │
         ┌───────────────┴──────────────────┐
         ▼                                  │
    PREPROCESSING                          │
    ├─ Load dataset                        │
    ├─ Extract forbidden words             │
    ├─ Standardize format                  │
    └─ Split: short/medium/long            │
                          │                │
         ┌───────────────┴──────────────────┐
         │  Pipeline.py Orchestrates All    │
         └───────────────┬──────────────────┘
                          │
         ┌────────────────┴──────────────────────┐
         ▼                                       │
    VARIATION GENERATION                        │
    ├─ Model: Llama, Mistral                    │
    ├─ Constraint: forbidden words              │
    ├─ Length: 5-10, 15-20, 20-30 words        │
    └─ Output: 3-5 variations per seed         │
                          │                      │
         ┌────────────────┴──────────────────────┐
         ▼                                       │
    VECTOR EXTRACTION                          │
    ├─ Model: Llama 3.1-8B, Mistral 7B        │
    ├─ Method: Mean Pooling / Last Token / Hybrid
    │          / Attention-Weighted             │
    ├─ Output: [d=4096] vectors                │
    └─ Cache: data/vectors/*.npy                │
                          │                     │
         ┌────────────────┴──────────────────────┐
         ▼                                       │
    ANCHOR ALIGNMENT                           │
    ├─ Anchors: 100 neutral reference sentences│
    ├─ Extract: Anchor vectors for each model  │
    ├─ Solver: SVD-based Procrustes            │
    │  Find Q^opt: argmin ||Y - X·Q||_F       │
    └─ Output: Rotation matrix Q [d, d]        │
                          │                     │
         ┌────────────────┴──────────────────────┐
         ▼                                       │
    ALIGNMENT EVALUATION                        │
    ├─ Pre-alignment similarity: cos(X, Y)     │
    ├─ Post-alignment: cos(X·Q, Y)             │
    ├─ Improvement %: (before - after) / before│
    └─ Significance: t-tests, ANOVA, p-values  │
                          │                     │
         └────────────────┴──────────────────────┘
                          │
         ┌────────────────▼─────────────────┐
         │  FINDING: Cross-model alignment  │
         │  proves latent space isomorphism │
         └────────────────┬─────────────────┘
                          │
         ┌────────────────▼─────────────────┐
         │  PUBLICATION: NeurIPS Paper +    │
         │  GitHub + DOI + Reproducible     │
         └─────────────────────────────────┘
```

---

## 2. DATA PIPELINE DECISION TREE

```
START: What do I want to do?
│
├─── "I want to load data"
│    │
│    └─ Use: DatasetLoader.load('toxigen', config)
│         └─ Returns: Dataset object with seeds loaded
│         └─ Methods: .load(), .preprocess(), .statistics()
│
├─── "I want to generate variations"
│    │
│    ├─ Pick method:
│    │  ├─ Fast: Use cached variations (if available)
│    │  ├─ Accurate: Call generator.get_validated_variation()
│    │  └─ Batch: generator.generate_batch(seeds, n_variations=3)
│    │
│    └─ Config: ForbiddenWordsConstraint + LengthConstraint
│
├─── "I want to extract vectors"
│    │
│    ├─ Pick extraction method:
│    │  ├─ Mean Pooling: Fast, robust (RECOMMENDED)
│    │  ├─ Last Token: Fast, simple
│    │  ├─ Hybrid: Slower, better quality (RECOMMENDED)
│    │  └─ Attention-Weighted: Slowest, most principled
│    │
│    ├─ Pick model:
│    │  ├─ Llama 3.1-8B: Better semantic understanding
│    │  └─ Mistral 7B: Faster inference
│    │
│    └─ Config: device='cuda', batch_size=8
│
├─── "I want to align latent spaces"
│    │
│    ├─ Prepare anchors:
│    │  ├─ Sentence anchors (100 reference texts) ← USE THIS
│    │  ├─ Word anchors (87 neutral words)
│    │  └─ Concept anchors (15 abstract concepts)
│    │
│    ├─ Extract anchor vectors:
│    │  ├─ For source model (e.g., Mistral)
│    │  └─ For target model (e.g., Llama)
│    │
│    ├─ Run Procrustes solver:
│    │  └─ Q, error = ProcrustesAligner.align(X, Y)
│    │
│    └─ Verify: Q^T·Q ≈ I, det(Q) ≈ 1
│
├─── "I want to run an experiment"
│    │
│    └─ See EXECUTION_PLAN.md for Exp 1-8
│        ├─ Exp 1: Anchor strategy comparison
│        ├─ Exp 2: Extraction methods
│        ├─ Exp 3: Cross-dataset generalization
│        ├─ Exp 4: Model pairs
│        ├─ Exp 5: Alignment stability
│        ├─ Exp 6: Scalability analysis
│        ├─ Exp 7: Semantic quality
│        └─ Exp 8: Statistical significance
│
└─── "I want to publish findings"
     │
     ├─ Run all 8 experiments
     ├─ Collect results → experiments/results/
     ├─ Generate figures
     ├─ Write paper
     └─ Release code on GitHub + Zenodo
```

---

## 3. TESTING PYRAMID - WHAT TO TEST WHEN

```
                    ▲
                    │        E2E Tests (5)
                ┌───┴───┐    - Full pipeline
                │  E2E  │    - Reproducibility
                └───┬───┘    - Multi-dataset
                    │
                  ◀─┼─▶
            ┌───────┴───────┐
            │  Integration  │  Integration Tests (15)
            │   Tests       │  - Component interaction
            ├───────┬───────┤  - Pipeline stages
            │       │       │  - Error handling
            └───────┼───────┘
                    │
              ◀─────┼──────▶
    ┌─────────┬─────┴─────┬─────────┐
    │  Unit  │   Unit    │  Unit   │   Unit Tests (80)
    │ Tests  │   Tests   │  Tests  │   - Extractors (15)
    │ (20)   │   (35)    │  (25)   │   - Anchors (12)
    │        │           │         │   - Aligners (18)
    └────────┴───────────┴─────────┘   - Generators (20)
                                        - Validators (15)

Build continuously: 10→20→30→50→75→90→100+ tests
```

---

## 4. ARCHITECTURE DECISION MATRIX

| Decision | Option A | Option B | Option C | **CHOSEN** |
|----------|----------|----------|----------|-----------|
| **Dataset Support** | ToxiGen only | 3 datasets | **5 datasets** | 5 datasets ✅ |
| **Anchor Strategy** | Word anchors | **Sentence anchors** | Concept anchors | Sentence (100) ✅ |
| **Vector Extraction** | Mean pooling only | **4 methods** (mean, last, hybrid, attn-weighted) | Single best | 4 methods ✅ |
| **Alignment Method** | Neural network | **Procrustes SVD** | Linear regression | Procrustes ✅ |
| **Model Pairs** | Llama only | **2+ models** (Llama, Mistral, GPT) | Single pair | 3+ pairs ✅ |
| **Evaluation** | Alignment quality only | **Quality + Stability + Scalability + Stats** | Single metric | Comprehensive ✅ |
| **Reproducibility** | No seed fixing | **Fixed seeds + versioned deps** | Best effort | Exact reproduction ✅ |
| **Publication Target** | ML workshop | **NeurIPS/ICML** | Journal | NeurIPS ✅ |

---

## 5. QUICK REFERENCE: FILE LOCATIONS

### Code Structure
```
isomorphic/
├── config.py                    # Configuration loading
├── loader.py                    # Multi-dataset factory
│
├── datasets/                    # Dataset implementations
│   ├── base_dataset.py
│   ├── toxigen_dataset.py
│   ├── jigsaw_dataset.py
│   ├── hatexplain_dataset.py
│   ├── sbic_dataset.py
│   ├── ethos_dataset.py
│   └── custom_dataset.py
│
├── extractors/                  # Vector extraction
│   ├── base_extractor.py
│   ├── mean_pooling.py
│   ├── last_token.py
│   ├── hybrid_extraction.py
│   └── attention_weighted.py
│
├── anchors/                     # Anchor management
│   ├── sentence_anchors.py      # 100 reference sentences
│   ├── word_anchors.py
│   ├── concept_anchors.py
│   └── anchor_selection.py
│
├── alignment_utils.py           # Procrustes solver
├── pipeline.py                  # End-to-end orchestration
└── utils/
    ├── logger.py
    ├── cache.py
    └── device_utils.py
```

### Data Locations
```
data/
├── raw/                         # Downloaded datasets
│   ├── toxigen/
│   ├── jigsaw/
│   ├── hatexplain/
│   ├── sbic/
│   └── ethos/
│
├── processed/                   # Processed JSON
│   ├── toxigen_processed.json
│   ├── jigsaw_processed.json
│   ├── ...
│
├── vectors/                     # Cached embeddings (GITIGNORE)
│   ├── llama_sentence_anchors.npy
│   ├── mistral_sentence_anchors.npy
│   └── ...
│
├── alignments/                  # Computed Q matrices (GITIGNORE)
│   ├── llama_mistral_q.pkl
│   └── ...
│
└── metadata/
    ├── anchor_definitions.json
    └── dataset_statistics.json
```

### Testing
```
tests/
├── test_unit/                   # Unit tests
│   ├── test_extractors.py
│   ├── test_anchors.py
│   ├── test_aligners.py
│   ├── test_generators.py
│   └── test_validators.py
│
├── test_integration/            # Integration tests
│   ├── test_multi_dataset_loader.py
│   ├── test_end_to_end_pipeline.py
│   └── test_cross_model.py
│
├── test_performance/            # Performance tests
│   ├── test_memory_usage.py
│   └── test_computation_time.py
│
├── test_statistics/             # Statistical tests
│   ├── test_statistical_significance.py
│   └── test_confidence_intervals.py
│
├── conftest.py                  # Pytest fixtures
└── fixtures/
    ├── sample_seeds.json
    └── mock_embeddings.npy
```

### Experiments
```
experiments/
├── exp_001_anchor_strategy_comparison.py
├── exp_002_extraction_methods.py
├── exp_003_cross_dataset_alignment.py
├── exp_004_model_pairs.py
├── exp_005_alignment_stability.py
├── exp_006_scalability.py
├── exp_007_semantic_preservation.py
├── exp_008_statistical_significance.py
│
├── results/
│   ├── exp_00X_results.json     # Raw results
│   ├── plots/                   # PDF figures
│   │   ├── anchor_comparison.pdf
│   │   ├── extraction_methods.pdf
│   │   └── ...
│   ├── tables/                  # CSV tables
│   │   └── *.csv
│   └── metadata.json            # Experiment metadata
│
└── analysis/
    ├── analyze_results.py       # Post-processing
    ├── generate_figures.py      # Plot generation
    └── statistical_summary.py   # Stats summary
```

---

## 6. DECISION: WHICH DATASET TO START WITH?

```
START WITH TOXIGEN because:
✅ Smallest dataset (274K)
✅ Well-organized on HuggingFace
✅ Clear category structure
✅ Good documentation
✅ Used in prior work

THEN ADD:
1. SBIC (150K) - Simpler, good quality
2. HateXplain (20K) - Rationale-based, novel
3. Jigsaw (2M) - Large-scale, challenging
4. ETHOS (998) - Smallest, multilingual

RESULT: 5-domain diversity ✅
```

---

## 7. DECISION: WHICH EXTRACTION METHOD TO PICK?

```
Evaluation Criteria:

Method              Speed    Quality   Stability   Compute
─────────────────────────────────────────────────────────
Mean Pooling        ⭐⭐⭐⭐⭐  ⭐⭐⭐⭐   ⭐⭐⭐     Low
Last Token          ⭐⭐⭐⭐⭐  ⭐⭐⭐    ⭐⭐⭐     Low
Hybrid              ⭐⭐⭐⭐   ⭐⭐⭐⭐   ⭐⭐⭐⭐   Medium
Attention-Weighted  ⭐⭐⭐    ⭐⭐⭐⭐⭐  ⭐⭐⭐⭐⭐  High

RECOMMENDATION:
Phase 1: Use Mean Pooling (fast iteration)
Phase 2: Compare all 4 in Exp 2
→ Use winner for Exp 3-8
```

---

## 8. DECISION: WHICH MODELS TO USE?

### Embedding Model (Vector Extraction - Fixed)
```
PRIMARY: Qwen2.5-7B-Instruct-Abliterated
Link: huihui-ai/Qwen2.5-7B-Instruct-abliterated-v2

Why: 
✅ Already available in your environment
✅ Excellent performance on semantic tasks
✅ Consistent embeddings across datasets
✅ No generation needed - pure extraction
```

### Rewriting Models (Variation Generation - Multiple for comparison)

**Recommended Primary Models** (Exp 1-7):
```
1️⃣ Llama-3.1-8B-Instruct-Abliterated
   Link: mlabonne/Meta-Llama-3.1-8B-Instruct-abliterated
   - General-purpose local use
   - Consumer GPU friendly (8GB VRAM)
   
2️⃣ Qwen2.5-7B-Instruct-Abliterated  
   Link: huihui-ai/Qwen2.5-7B-Instruct-abliterated-v2
   - Same size as embedding model (consistency)
   - Excellent performance
   
3️⃣ Qwen3-32B-Instruct-Abliterated
   Link: huihui-ai/Qwen3-32B-abliterated
   - Advanced coding/logic (for variant quality)
   - Requires higher VRAM (24GB+)
```

**Extended Comparison Models** (Exp 4: Model Pairs):
```
🎯 Primary Pairs (aligned by size/architecture):
- Llama-3.1-8B ↔ Qwen2.5-7B        (8B class, base models)
- Llama-3.1-8B ↔ Qwen3-32B          (8B vs 32B, reasoning variant)
- Qwen2.5-7B ↔ Mistral-Nemo-12B    (cross-family: 7B vs 12B)

🔬 Optional Advanced Models (if budget allows):
- Llama-3.2-3B-Instruct-Abliterated      (lightweight: 3B)
- Mistral-Small-24B-Instruct-Abliterated (mid-range: 24B)
- DeepSeek-R1-Distill-Qwen-32B           (reasoning: 32B)
```

**Model Selection By Experiment:**

```
Exp 1 (Anchor Strategy):
  Embedding: Qwen2.5-7B
  Rewriting: Llama-3.1-8B + Qwen2.5-7B

Exp 2 (Extraction Methods):
  Embedding: Qwen2.5-7B (all 4 extraction methods)
  Rewriting: Llama-3.1-8B for initial variations

Exp 3 (Cross-Dataset):
  Embedding: Qwen2.5-7B (fixed)
  Rewriting: Llama-3.1-8B ↔ Qwen2.5-7B

Exp 4 (Model Pairs):
  Embedding: Qwen2.5-7B (fixed)
  Rewriting: 3 primary pairs listed above

Exp 5 (Stability):
  Embedding: Qwen2.5-7B (with noise injection)
  Rewriting: Llama-3.1-8B

Exp 6 (Scalability):
  Embedding: Qwen2.5-7B
  Rewriting: Llama-3.1-8B (largest bottleneck)

Exp 7 (Semantic Quality):
  Embedding: Qwen2.5-7B (for similarity measurement)
  Rewriting: All 3 primary models

Exp 8 (Statistical Significance):
  Embedding: Qwen2.5-7B
  Rewriting: All pairs from Exp 4
```

### Why This Approach?

✅ **Qwen2.5-7B (Embedding only)**: 
- Already available
- Consistent semantic space
- Fast vector extraction
- No API calls, fully local

✅ **Llama-3.1-8B (Rewriting)**:
- General purpose quality
- Widely used baseline
- Good instruction following

✅ **Qwen3-32B & Others (Comparison)**:
- Test if larger models generate better variations
- Cross-family semantic agreement
- Statistical comparison basis
```

---

## 9. DECISION: STATISTICAL TESTING STRATEGY

```
For each experiment, run:

Descriptive Stats:
├─ Mean ± SE (standard error)
├─ 95% CI (confidence interval)
├─ Median
├─ Std Dev
└─ N (sample size)

Inferential Tests:
├─ Normality test (Shapiro-Wilk)
├─ Homogeneity of variance (Levene's)
├─ ANOVA or Kruskal-Wallis (comparing 3+ groups)
├─ t-test or Mann-Whitney U (comparing 2 groups)
├─ Post-hoc test (if ANOVA significant)
│  └─ Bonferroni correction (multiple comparisons)
│
└─ Effect Size:
   ├─ Cohen's d (for t-tests)
   ├─ Hedges' g (for unequal samples)
   └─ Eta-squared (for ANOVA)

Significance Level: α = 0.05
Multiple Comparisons: Bonferroni-corrected

OUTPUT: Results table with
✓ Mean ± SE
✓ 95% CI [lower, upper]
✓ Test statistic (F, t, U, etc.)
✓ p-value
✓ Effect size
✓ Interpretation (sig? trivial? large effect?)
```

---

## 10. QUICK COMMAND REFERENCE

```bash
# Setup
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev,ml]"

# Load data
python -c "
from isomorphic.loader import DatasetLoader
ds = DatasetLoader.load('toxigen', {'limit': 100})
print(ds.statistics())
"

# Extract vectors
python -c "
from isomorphic.extractors.mean_pooling import MeanPoolingExtractor
extractor = MeanPoolingExtractor('meta-llama/Meta-Llama-3-8B-Instruct')
vec = extractor.extract('Hello world')
print(vec.shape)  # Should be (4096,)
"

# Compute alignment
python -c "
from isomorphic.alignment_utils import ProcrustesAligner
import numpy as np
X = np.random.randn(100, 4096)
Y = np.random.randn(100, 4096)
Q, error = ProcrustesAligner.align(X, Y)
print(f'Alignment error: {error:.4f}')
"

# Run tests
pytest tests/ -v --cov=isomorphic --cov-report=html

# Run single experiment
python experiments/exp_001_anchor_strategy_comparison.py

# Generate figures
python experiments/analysis/generate_figures.py

# Full pipeline (if implemented)
python scripts/run_pipeline.py --config config/default.yaml
```

---

## 11. KEY METRICS CHECKLIST

### Every Experiment Should Report:

```
Results Table Format:
┌─────────────┬────────────┬────────────┬──────────┬───────────┬─────────────┐
│ Method      │ Mean ± SE  │ 95% CI     │ N        │ p-value   │ Effect Size │
├─────────────┼────────────┼────────────┼──────────┼───────────┼─────────────┤
│ Approach 1  │ 0.85±0.02  │ [0.81-0.89]│ 1000     │ <0.001*** │ d=1.2 large │
│ Approach 2  │ 0.78±0.03  │ [0.72-0.84]│ 1000     │           │ (baseline)  │
│ Approach 3  │ 0.92±0.01  │ [0.90-0.94]│ 1000     │ <0.001*** │ d=1.8 large │
└─────────────┴────────────┴────────────┴──────────┴───────────┴─────────────┘

Legend:
*** p < 0.001 (highly significant)
**  p < 0.01  (very significant)
*   p < 0.05  (significant)
ns  p ≥ 0.05  (not significant)

Effect Size Interpretation:
d < 0.2    = trivial
0.2-0.5    = small
0.5-0.8    = medium
d > 0.8    = large ← TARGET FOR PUBLICATION
```

---

## 12. GO/NO-GO MILESTONES

| Week | Milestone | Go/No-Go |
|------|-----------|----------|
| 2 | Config + extractors working, 20 tests | Go if: ✓ All tests pass |
| 4 | All 5 datasets loading, 50 tests | Go if: ✓ 1st 100 samples load |
| 6 | Anchor alignment complete, 90 tests | Go if: ✓ Frobenius error < threshold |
| 8 | Exp 1-3 complete with figures | Go if: ✓ p-values significant |
| 10 | Exp 4-8 complete + stats reported | Go if: ✓ Main findings are novel |
| 12 | Paper + code released | Go if: ✓ Reproducible + 100+ tests |

---

## 13. FAILURE SCENARIOS & MITIGATIONS

```
Scenario 1: Alignment quality is poor (Frobenius > 10%)
├─ Possible causes:
│  ├─ Anchor vectors are wrong (reload/verify)
│  ├─ Models encode very differently (not isomorphic)
│  └─ Numerical issues in SVD (check condition number)
├─ Mitigation:
│  ├─ Plot anchor vectors in PCA space
│  ├─ Try different anchor strategies
│  └─ Add numerical stability checks

Scenario 2: Can't reach p < 0.05 significance
├─ Possible causes:
│  ├─ Sample size too small (n < 500)
│  ├─ High variance in measurements
│  └─ Effect size too small
├─ Mitigation:
│  ├─ Increase sample size (bigger datasets)
│  ├─ Reduce measurement noise (better anchors)
│  └─ If still fails: report exact p-value + context

Scenario 3: Results not reproducible
├─ Possible causes:
│  ├─ Random seeds not fixed
│  ├─ Non-deterministic GPU operations
│  └─ Data loading order varies
├─ Mitigation:
│  ├─ Fix seeds: np.seed(42), torch.manual_seed(42)
│  ├─ Use deterministic flag: CUDA_LAUNCH_BLOCKING=1
│  └─ Use sorted file lists

Scenario 4: Computational time too long
├─ Possible causes:
│  ├─ Inefficient batch processing
│  ├─ Extracting vectors is slow
│  └─ Models too large
├─ Mitigation:
│  ├─ Profile code: use torch.profiler
│  ├─ Increase batch size
│  └─ Use smaller models (7B instead of 70B)
```

---

## FINAL: PUBLICATION CHECKLIST

✅ **Scientific**
- [ ] Research question clearly stated
- [ ] Hypothesis testable and novel
- [ ] 8 experiments run with statistical rigor
- [ ] Results statistically significant (p < 0.05)
- [ ] Effect sizes practically meaningful (d > 0.8)
- [ ] Limitations discussed
- [ ] Broader impact statement included

✅ **Engineering**
- [ ] 100+ tests with >80% coverage
- [ ] Code on GitHub public
- [ ] DOI via Zenodo
- [ ] requirements.txt + environment.yml
- [ ] Reproducibility: 3 runs = identical results
- [ ] API documented (docstrings + README)

✅ **Presentation**
- [ ] Paper: clear intro + rigorous methodology
- [ ] Figures: publication-quality (300 dpi)
- [ ] Tables: statistical annotations (p-values, CIs)
- [ ] Results: honest reporting of limitations
- [ ] Writing: proofread, no grammatical errors

✅ **Compliance**
- [ ] All datasets properly cited + licensed
- [ ] No identifying information (anonymous)
- [ ] Supplementary material complete
- [ ] Meets venue requirements (length, format)

---

**Questions? Check PROJECT_PLAN_NEURIPS.md, IMPLEMENTATION_GUIDE.md, or EXECUTION_PLAN.md**

