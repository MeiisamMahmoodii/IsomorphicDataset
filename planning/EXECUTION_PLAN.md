# Project Execution Plan: Week-by-Week Roadmap
## IsomorphicDataSet - NeurIPS Ready Implementation

---

## PHASE 1: FOUNDATION (Weeks 1-2)
### **Goal**: Core infrastructure ready for data loading

#### Week 1: Setup & Architecture
**Mon-Tue: Project Setup**
- [ ] Initialize git repo + GitHub setup
- [ ] Create directory structure (copy from PROJECT_PLAN_NEURIPS.md)
- [ ] Set up development environment (.venv, requirements)
- [ ] Create pyproject.toml with all dependencies
- [ ] Initialize pre-commit hooks (black, isort, flake8, mypy)

**Wed: Configuration System & Reporting Infrastructure**
- [ ] Implement `isomorphic/config.py` (Config dataclasses)
- [ ] Create default YAML configs in `config/` (models.yaml, datasets.yaml, etc.)
- [ ] Implement ConfigManager class
- [ ] **NEW**: Set up results reporting infrastructure:
  - [ ] Create `experiments/results/` directory structure
  - [ ] Create `experiments/reports/` for CSV/markdown reports
  - [ ] Implement ResultsLogger class (save JSON + CSV per experiment)
  - [ ] Create experiment registry system (metadata.json)
- [ ] Write 5 unit tests for config loading + 3 for results logging

**Thu: Logging & Utils**
- [ ] Set up structured logging (isomorphic/utils/logger.py)
- [ ] Implement device utils (GPU/CPU detection)
- [ ] Add result caching decorator
- [ ] **NEW**: Pre-download model checkpoints (avoid repeated downloads)
- [ ] Create utility tests

**Fri: Documentation Setup**
- [ ] Write GETTING_STARTED.md (includes results/reporting section)
- [ ] Create API documentation template
- [ ] Set up contributing guidelines
- [ ] **NEW**: Document proven prompts (copy from source/generator.py)
- [ ] Review and fix any issues

**Deliverable**: 
- Empty project structure + config system working
- Results reporting infrastructure ready
- Proven prompts documented
**Tests**: 10+ passing tests

---

#### Week 2: Core Framework + Model Setup
**Mon-Tue: Base Extractor Class**
- [ ] Implement `isomorphic/extractors/base_extractor.py`
- [ ] Design batch processing interface
- [ ] Add type hints throughout
- [ ] Write abstract method tests

**Wed: Vector Extraction Implementations**
- [ ] Implement MeanPoolingExtractor
- [ ] Implement LastTokenExtractor
- [ ] Implement HybridExtractor
- [ ] Add numerical stability checks

**Thu: Alignment Core + Banned Words Extractor**
- [ ] Implement ProcrustesAligner (SVD-based)
- [ ] Add orthogonality verification + stability analysis
- [ ] **NEW**: Implement `isomorphic/banned_words_extractor.py`
  - Load Qwen2.5-72B model (high-quality semantic extraction)
  - Use proven extraction prompt (see PROJECT_PLAN_NEURIPS.md Part 3.5)
  - Extract 5-7 semantic forbidden words per seed
  - Validate extraction quality (no common words, >4 chars, semantic value)
- [ ] Implement metrics computation

**Fri: Testing Suite Scaffold**
- [ ] Create conftest.py with fixtures
- [ ] Create mock data (sample embeddings, texts)
- [ ] Write 20 basic unit tests
- [ ] **NEW**: Test banned word extraction accuracy
  - Test that extraction skips common words
  - Test that extracted words are from original text
  - Test extraction consistency
- [ ] Set up continuous testing workflow

**Deliverable**: 
- Functional extractors + aligner + banned words extractor
- Models downloaded: Qwen2.5-7B (embedding) + Qwen2.5-72B (banned words) + Llama-3.1-8B
- 25 passing tests
**Code Quality**: mypy passes, no style violations

---

## PHASE 2: MULTI-DATASET (Weeks 3-4)
### **Goal**: Support 5 datasets, all preprocessing pipeline working

#### Week 3: Dataset Framework + Preprocessing
**Mon: Dataset Base Class + Preprocessing Pipeline**
- [ ] Implement `isomorphic/datasets/base_dataset.py`
- [ ] Define standard format (seed, forbidden_words, semantic_intent, etc.)
- [ ] Implement validation interface
- [ ] **NEW**: Add preprocessing with 70B banned word extraction:
  - [ ] Load Qwen2.5-72B model at startup
  - [ ] For each seed: extract banned words using proven prompt
  - [ ] Cache banned words (don't recompute)
  - [ ] Log extraction stats to reports
- [ ] Write 8 tests for abstract class

**Tue-Wed: ToxiGen Implementation + Results Reporting**
- [ ] Implement ToxiGenDataset class
- [ ] Test download from HuggingFace
- [ ] Implement preprocessing (USE 70B banned word extraction)
- [ ] Create 100 reference sentences for anchors
- [ ] **NEW**: Log preprocessing results to `experiments/results/reports/preprocessing_report.csv`:
  - Seeds loaded, validated, dropped
  - Banned words extraction time
  - Average banned words per seed
- [ ] Write 12 tests specific to ToxiGen

**Thu-Fri: Additional Dataset Stubs + Reporting**
- [ ] Create JigsawDataset skeleton (with comments on download process)
- [ ] Create HateXplainDataset skeleton
- [ ] Create SBICDataset skeleton
- [ ] Create ETHOSDataset skeleton
- [ ] Implement DatasetLoader factory class
- [ ] **NEW**: Generate preprocessing report showing stats for each dataset

**Deliverable**: 
- ToxiGen fully working with 70B banned words extraction
- 4 others stubbed
- Preprocessing reports generated
- 30+ unit tests passing

**Reports Generated**:
- `experiments/results/reports/preprocessing_report.csv` (stats per dataset)

---

#### Week 4: Complete Dataset Pipeline + Results Reporting
**Mon-Tue: Jigsaw Implementation + Results Tracking**
- [ ] Implement full JigsawDataset (data loading from Kaggle)
- [ ] Handle large dataset efficiently (~2M samples)
- [ ] Use 70B banned words extraction for subset (first 1000)
- [ ] Implement category-based preprocessing
- [ ] **NEW**: Log results to `experiments/results/reports/preprocessing_report.csv`
- [ ] Write 12 tests

**Wed: HateXplain + SBIC + Results Aggregation**
- [ ] Implement HateXplainDataset (70B banned words extraction)
- [ ] Implement SBICDataset (70B banned words extraction)
- [ ] Handle rationale extraction (HateXplain-specific)
- [ ] Handle identity annotations (SBIC-specific)
- [ ] **NEW**: Aggregate preprocessing stats across 5 datasets

**Thu: Testing + Integration + Report Generation**
- [ ] Integration test: load all 5 datasets
- [ ] Integration test: preprocess pipeline with 70B extraction
- [ ] Performance test: load time per dataset
- [ ] **NEW**: Generate comprehensive preprocessing report:
  - [ ] `preprocessing_report.csv` with stats per dataset
  - [ ] `preprocessing_summary.md` with aggregate findings
  - [ ] Banned words extractio quality metrics
  - [ ] Time/resource usage per dataset
- [ ] Create dataset statistics generation script

**Fri: Documentation + Weekly Report**
- [ ] Document each dataset: license, URL, size, preprocessing
- [ ] Create DATASETS.md guide
- [ ] Add dataset download instructions
- [ ] **NEW**: Create Week 4 Results Report
  - Datasets loaded: 5/5 ✅
  - Prohibited words extracted: 5000 seeds ✅
  - Average time per seed: X seconds
  - Report files: Link to all CSVs and markdown
- [ ] Create example notebook (01_exploratory_analysis.ipynb)

**Deliverable**: 
- All 5 datasets loading + preprocessing with 70B extraction
- 15+ integration tests passing
- Comprehensive results reporting setup
- First week of detailed reports generated

**Reports Generated**:
- `experiments/results/reports/preprocessing_report.csv` (ALL datasets)
- `experiments/results/reports/preprocessing_summary.md`
- `experiments/results/reports/week_4_summary.md` (status report)

---

## PHASE 3: VECTOR EXTRACTION (Week 5)
### **Goal**: Extract vectors from all datasets, validate quality

#### Week 5: Complete Extraction Pipeline
**Mon-Tue: Extraction System + Results Logging**
- [ ] Implement ExtractorFactory class
- [ ] Add batch processing with memory optimization
- [ ] Implement caching for extracted vectors
- [ ] Add GPU memory profiling
- [ ] **NEW**: Implement extraction results logger
  - Log total vectors extracted
  - Log time per vector (mean/std)
  - Log memory peak usage
  - Save to `experiments/results/reports/extraction_report.csv`

**Wed: Testing Suite for Extractors + Reports**
- [ ] Test mean pooling correctness
- [ ] Test last token extraction
- [ ] Test hybrid method
- [ ] Test batch vs sequential equivalence
- [ ] Test gradient flow (if using for training later)
- [ ] Test numerical stability edge cases
- [ ] **NEW**: Write test results to reports

**Thu: Anchor Extraction + Results Reporting**
- [ ] Load 100 reference sentence anchors
- [ ] Extract vectors for all anchors (Qwen2.5-7B embedding)
- [ ] Cache anchor vectors to disk
- [ ] Validate anchor vector quality
- [ ] **NEW**: Log anchor extraction metrics:
  - Extraction time
  - Vector statistics (mean, std, min, max)
  - Save to `experiments/results/reports/anchor_extraction.csv`

**Fri: Validation Experiments + Weekly Report**
- [ ] Compare extraction methods on sample seeds
- [ ] Compute cosine similarity between methods
- [ ] Create extraction method comparison visualization
- [ ] **NEW**: Generate Week 5 Results Report
  - Total vectors extracted: X
  - Extraction success rate: Y%
  - Average time per vector: Z seconds
  - Memory usage: W GB
  - Report files: Link to CSVs

**Deliverable**: Anchor vectors cached + extraction tested
**Tests**: 25+ extraction tests passing

**Reports Generated**:
- `experiments/results/reports/extraction_report.csv`
- `experiments/results/reports/anchor_extraction.csv`
- `experiments/results/reports/week_5_summary.md`

**Outputs**: 
- `data/vectors/qwen_sentence_anchors.npy`

---

## PHASE 4: ALIGNMENT (Week 6)
### **Goal**: Compute cross-model alignments, establish baseline metrics

#### Week 6: Procrustes Alignment + Results Reporting
**Mon: Core Alignment + Results Logging**
- [ ] Extract vectors from 100 ToxiGen seeds (Llama-3.1-8B + Qwen2.5-7B rewriting)
- [ ] Embed with Qwen2.5-7B-Instruct-Abliterated (fixed model for all extraction)
- [ ] Compute Procrustes alignment Q^opt (Llama embeddings → Qwen embeddings)
- [ ] Verify orthogonality constraints (Q^T·Q = I)
- [ ] **NEW**: Implement alignment results logger
  - Log Frobenius error per model pair
  - Log orthogonality error (||Q^T Q - I||_F)
  - Log alignment time per pair
  - Save to `experiments/results/reports/alignment_report.csv`

**Tue: Alignment Quality Metrics + Reporting**
- [ ] Pre-alignment vs Post-alignment cosine similarity
- [ ] Cross-model agreement quantification (Llama vs Qwen)
- [ ] Semantic clustering analysis
- [ ] Statistical significance (p-value)
- [ ] **NEW**: Log detailed alignment metrics:
  - Mean Frobenius error: X
  - Orthogonality validation: PASS/FAIL
  - Pre-alignment similarity: A
  - Post-alignment similarity: B
  - Save to `experiments/results/reports/alignment_detailed.csv`

**Wed: Stability Analysis**
- [ ] Add noise to Qwen anchor embeddings (5%, 10%, 15%)
- [ ] Recompute alignments with perturbations
- [ ] Measure Q matrix stability: ||Q_perturbed - Q_clean||_F
- [ ] Document robustness findings
- [ ] **NEW**: Log stability metrics to `experiments/results/reports/alignment_stability.csv`

**Thu: Multi-Model Pairs + Cross-Dataset**
- [ ] Align Llama-3.1-8B → Qwen2.5-7B (base)
- [ ] Align Qwen3-32B → Qwen2.5-7B (reasoning variant)
- [ ] Align Mistral-Nemo-12B → Qwen2.5-7B (cross-family)
- [ ] Compare alignment quality across models
- [ ] **NEW**: Log cross-model alignment:
  - Per-model Frobenius error
  - Per-model orthogonality
  - Save to `experiments/results/reports/alignment_cross_model.csv`

**Fri: Experiment 1 + Weekly Report**
- [ ] Document "Anchor Strategy Comparison" findings (with actual models used)
- [ ] Create comparison table (sentence vs word vs concept anchors)
- [ ] Model pair in results: Llama-3.1-8B ↔ Qwen2.5-7B
- [ ] Generate publication-quality figure
- [ ] **NEW**: Generate Week 6 Results Report
  - Model pairs aligned: X
  - Mean Frobenius error: Y
  - Stability robustness: Z%
  - Report file links

**Deliverable**: First alignment experiments complete (with real models)
**Tests**: 15+ alignment tests passing
**Reports Generated**:
- `experiments/results/reports/alignment_report.csv`
- `experiments/results/reports/alignment_detailed.csv`
- `experiments/results/reports/alignment_stability.csv`
- `experiments/results/reports/alignment_cross_model.csv`
- `experiments/results/reports/week_6_summary.md`

**Outputs**:
- `experiments/results/exp_001_anchor_strategy_comparison.json` (includes model IDs)
- Figures: `experiments/results/plots/anchor_comparison.pdf`

---

## PHASE 5: EXPERIMENTATION (Weeks 7-10)
### **Goal**: Run 8 experiments, collect statistical evidence, write findings

#### Week 7: Extraction Method Comparison (Exp 2) + Results Reporting
**Goal**: {Mean Pooling, Last Token, Hybrid, Attention-Weighted} × {All 5 Datasets}

- [ ] Extract vectors using all 4 methods on 500 seeds from each dataset
- [ ] All extraction uses Qwen2.5-7B-Instruct-Abliterated (fixed/controlled)
- [ ] Variations generated by: Llama-3.1-8B, Qwen2.5-7B, Qwen3-32B
- [ ] Compute alignment quality for each method
- [ ] Create comparison table (Method × Dataset × Model)
- [ ] Statistical test: ANOVA on alignment quality across methods
- [ ] Find: **Which extraction method is best?**
- [ ] **NEW**: Log extraction method results:
  - Per-method Frobenius error (X ± σ)
  - ANOVA test p-value
  - Post-hoc pairwise comparisons
  - Save to `experiments/results/reports/exp_002_extraction_methods_results.csv`
  - Generate summary: `experiments/results/reports/exp_002_summary.md`
- [ ] Output: `experiments/results/exp_002_extraction_methods.json`

---

#### Week 8: Cross-Dataset Generalization (Exp 3) + Results Reporting
**Goal**: Train alignment on Dataset A, test on Datasets B-E

- [ ] Generate variations using Llama-3.1-8B on ToxiGen (1000 samples)
- [ ] Extract embeddings using Qwen2.5-7B-Instruct-Abliterated (training set)
- [ ] Compute Q^opt alignment (Llama → Qwen embeddings)
- [ ] Generate test set variations from Jigsaw, HateXplain, SBIC, ETHOS (500 each)
- [ ] Apply same Q matrix to test set embeddings
- [ ] Measure alignment quality degradation across domains
- [ ] Find: **Does alignment generalize across domains?**
- [ ] **NEW**: Log generalization results:
  - Train Frobenius error (ToxiGen): A
  - Test Frobenius errors (Jigsaw, HateXplain, SBIC, ETHOS): [B1, B2, B3, B4]
  - Degradation ratio: max(Bi) / A
  - Save to `experiments/results/reports/exp_003_generalization_results.csv`
  - Generate summary: `experiments/results/reports/exp_003_summary.md`
- [ ] Output: `experiments/results/exp_003_cross_dataset_alignment.json`

---

#### Week 9: Model Pair Analysis + Scalability (Exp 4 + 6) + Results Reporting
**Exp 4: Model Pairs** {Llama-3.1-8B ↔ Qwen2.5-7B, Llama-3.1-8B ↔ Qwen3-32B, Qwen2.5-7B ↔ Mistral-Nemo-12B}
- [ ] Generate variations with each rewriting model
- [ ] Extract embeddings with Qwen2.5-7B-Instruct-Abliterated (fixed)
- [ ] Compute 3 pairwise alignments
- [ ] Compare alignment quality across model pairs
- [ ] Measure semantic agreement baseline
- [ ] Statistical test: ANOVA on model pair alignment quality
- [ ] Find: **Which model pair aligns best?**
- [ ] **NEW**: Log model pair results:
  - Per-pair Frobenius error with 95% CI
  - ANOVA p-value
  - Best/worst performing pair
  - Save to `experiments/results/reports/exp_004_model_pairs_results.csv`
  - Generate summary: `experiments/results/reports/exp_004_summary.md`
- [ ] Output: `experiments/results/exp_004_model_pairs.json`

**Exp 6: Scalability Analysis** {100, 500, 1K, 5K sample sizes}
- [ ] Use Llama-3.1-8B + Qwen2.5-7B (fixed)
- [ ] Time & memory profiling for each scale
- [ ] Alignment quality trend with sample size
- [ ] Scaling plot (O(n), O(n log n)?)
- [ ] Identify bottlenecks: variation generation vs embedding vs alignment
- [ ] Find: **O(n) time complexity maintained?**
- [ ] **NEW**: Log scalability results:
  - Per-scale: runtime (mean/std), memory peak, Frobenius error
  - Time complexity order estimate
  - Bottleneck identification
  - Save to `experiments/results/reports/exp_006_scalability_results.csv`
  - Generate summary: `experiments/results/reports/exp_006_summary.md`
- [ ] Output: `experiments/results/exp_006_scalability.json` + scaling curves

**Reports Generated**:
- `experiments/results/reports/exp_004_model_pairs_results.csv`
- `experiments/results/reports/exp_004_summary.md`
- `experiments/results/reports/exp_006_scalability_results.csv`
- `experiments/results/reports/exp_006_summary.md`
- `experiments/results/reports/week_9_summary.md`

**Plots**: 
- `experiments/results/plots/scaling_curves.pdf`
- `experiments/results/plots/model_pair_comparison.pdf`

---

#### Week 10: Semantic Quality + Statistical Validation (Exp 5 + 7 + 8) + Results Reporting
**Exp 5: Stability Analysis**
- [ ] Models: Llama-3.1-8B (rewriting) + Qwen2.5-7B (embedding)
- [ ] Add noise to Qwen embeddings: 5%, 10%, 15%, 20%
- [ ] Recompute Q for each noise level
- [ ] Measure ||Q_perturbed - Q_clean||_F
- [ ] Find: Robustness threshold
- [ ] **NEW**: Log stability results:
  - Per-noise-level: Q stability (||Q_pert - Q_clean||_F)
  - Alignment quality degradation
  - Robustness threshold (noise level where >5% error)
  - Save to `experiments/results/reports/exp_005_stability_results.csv`
  - Generate summary: `experiments/results/reports/exp_005_summary.md`

**Exp 7: Semantic Preservation**
- [ ] Generate variations with: Llama-3.1-8B, Qwen2.5-7B, Qwen3-32B
- [ ] 100 ToxiGen seeds × 3 variations × 3 models = 900 variations
- [ ] Extract embeddings with Qwen2.5-7B-Instruct-Abliterated (fixed)
- [ ] Measure cosine similarity within clusters
- [ ] Verify 95%+ meet threshold (0.85)
- [ ] Find: Variation quality metrics by model
- [ ] **NEW**: Log semantic preservation results:
  - Per-model: mean cosine sim (A ± σ), % ≥0.85
  - Outlier analysis (semantic drift seeds)
  - Save to `experiments/results/reports/exp_007_semantic_results.csv`
  - Generate summary: `experiments/results/reports/exp_007_summary.md`

**Exp 8: Statistical Significance**
- [ ] Aggregate all experiments (Exp 1-7) results
- [ ] Models involved: Llama-3.1-8B, Qwen2.5-7B, Qwen3-32B, Mistral-Nemo-12B
- [ ] Compute across all dataset types (5000+ total samples)
- [ ] Run:
  - ANOVA (model effect on alignment quality)
  - t-tests (pairwise model comparisons)
  - Permutation tests (null distribution)
  - Effect sizes: Cohen's d, Hedges' g
  - 95% confidence intervals
- [ ] Multiple comparison correction: Bonferroni
- [ ] Sample size power analysis (n=5000 adequate?)
- [ ] Create statistical results table with all annotations
- [ ] **NEW**: Log aggregated statistics:
  - ANOVA F-statistic, p-value
  - Pairwise t-test results (all model pairs)
  - Effect sizes (Cohen's d for each pair)
  - Power analysis summary
  - Save to `experiments/results/reports/exp_008_statistical_results.csv`
  - Generate comprehensive summary: `experiments/results/reports/exp_008_summary.md`

**Reports Generated**:
- `experiments/results/reports/exp_005_stability_results.csv`
- `experiments/results/reports/exp_005_summary.md`
- `experiments/results/reports/exp_007_semantic_results.csv`
- `experiments/results/reports/exp_007_summary.md`
- `experiments/results/reports/exp_008_statistical_results.csv`
- `experiments/results/reports/exp_008_summary.md`
- `experiments/results/reports/week_10_summary.md`

**Outputs**:
- `experiments/results/exp_005_stability.json`
- `experiments/results/exp_007_semantic_preservation.json`
- `experiments/results/exp_008_statistical_significance.json`
- `experiments/analysis/statistical_summary.md` (comprehensive stats + table)

---

## PHASE 6: PUBLICATION READINESS (Weeks 11-12)
### **Goal**: Paper draft + reproducibility verification + release

#### Week 11: Paper Writing
**Main Paper Structure (15-20 pages)**:

1. **Abstract** (150 words)
   - Problem: Different LLMs encode concepts differently
   - Solution: Prove latent space isomorphism via Procrustes alignment
   - Impact: Enables zero-shot knowledge transfer

2. **Introduction**
   - Why LLM alignment matters
   - Gap in prior work (limitations of word-level anchors)
   - **Contributions**:
     - Set-ConCA theoretical framework
     - Sentence-level anchor strategy (100 semantic anchors)
     - Multi-dataset experimental validation
     - Statistical evidence of isomorphism

3. **Related Work** (2 pages)
   - Embedding alignment (prior work)
   - Cross-model knowledge transfer
   - Dataset selection for semantic research

4. **Methodology** (3 pages per experiment)
   - Formal problem statement
   - Data pipeline (preprocessing, anchor selection)
   - Vector extraction methods (4 approaches)
   - Procrustes alignment algorithm (mathematical formulation)
   - Metrics definition

5. **Experiments** (8 experiments)
   - Exp 1-2: Methods comparison
   - Exp 3: Generalization
   - Exp 4: Model pairs
   - Exp 5: Stability
   - Exp 6: Scalability
   - Exp 7: Semantic quality
   - Exp 8: Statistical validation
   - Each with: hypothesis, method, results table, figure, p-value

6. **Results** (with figures & tables)
   - Table 1: Anchor strategy comparison + statistical tests
   - Figure 1: Extraction method performance (+error bars)
   - Table 2: Cross-dataset generalization matrix
   - Figure 2: Alignment quality by model pair
   - Table 3: Scalability analysis + time/memory breakdown
   - Figure 3: Stability under perturbations (robustness curves)
   - Figure 4: Semantic preservation histogram
   - Table 4: Complete statistical summary (effect sizes, CIs, p-values)

7. **Discussion**
   - Summary of findings
   - Theoretical implications (what does isomorphism mean?)
   - Practical applications (zero-shot transfer, multi-model collaboration)
   - Limitations (model architecture constraints, anchor selection bias)
   - Future work (longer texts, image+text, supervised alignment)

8. **Conclusion** (1 page)
   - Restate main finding
   - Impact for MLL community
   - Call for reproducible research

9. **References**

**Supplementary Material** (10+ pages):
- Mathematical proofs (SVD correctness, orthogonality preservation)
- Extended experimental results
- Additional dataset details
- Ablation studies
- Failure case analysis

**Tasks for Week 11**:
- [ ] **NEW**: Compile all experiment results from JSON/CSV
  - Aggregate exp_001 through exp_008 results
  - Extract key metrics for paper tables
  - Verify all statistics (means, stds, p-values, CIs)
  - Cross-check CSVs match JSON outputs
  - Verify sample sizes and metadata

- [ ] **NEW**: Generate publication tables from results reports
  - Table 1: Anchor strategy comparison (from exp_001, exp_002 results CSVs)
  - Table 2: Cross-dataset generalization (from exp_003 CSV)
  - Table 3: Model pair performance (from exp_004 CSV)
  - Table 4: Scalability metrics (from exp_006 CSV)
  - Table 5: Statistical summary (from exp_008 CSV) + all effect sizes

- [ ] Complete sections 1-4 (Intro + Methodology)
- [ ] Create all figures (polished, publication-ready)
  - Figure 1: Extraction method performance (from exp_002 results)
  - Figure 2: Alignment quality by model pair (from exp_004 results)
  - Figure 3: Scalability curves (from exp_006 results)
  - Figure 4: Stability under perturbations (from exp_005 results)
  - Figure 5: Semantic preservation histogram (from exp_007 results)
  
- [ ] Write sections 5-8 (Results + Discussion)
  - Reference Exp 1-8 detailed summaries in `reports/` folder
  - Link statistical findings to p-values in CSV reports
  - Include confidence intervals from aggregated results
  
- [ ] Generate citations (BibTeX)
- [ ] Peer review draft (internal)

**Result Integration Checklist**:
- [ ] All 8 experiment JSONs parsed and validated
- [ ] CSV reports cross-checked against JSON results
- [ ] Statistical significance confirmed (all p-values recorded)
- [ ] Effect sizes computed and reported (Cohen's d, Hedges' g)
- [ ] Confidence intervals verified (95% CIs on all means)
- [ ] Sample sizes recorded (n values per experiment)
- [ ] Outliers identified and justified
- [ ] Reproducibility note: All results from seeded runs (seed=42)

**Deliverable**: Paper draft complete (all sections + figures) with verified results from all experiments

---

#### Week 12: Reproducibility + Results Archival + Release
**Results Archival & Verification**:
- [ ] **NEW**: Verify all results reports exist and are valid
  - Check all experiment JSONs: exp_001 through exp_008 ✓
  - Check all results CSVs:
    - exp_001_anchor_strategy_results.csv
    - exp_002_extraction_methods_results.csv
    - exp_003_generalization_results.csv
    - exp_004_model_pairs_results.csv
    - exp_005_stability_results.csv
    - exp_006_scalability_results.csv
    - exp_007_semantic_results.csv
    - exp_008_statistical_results.csv
  - Check all summary markdowns: exp_001_summary.md through exp_008_summary.md
  - Check all weekly summaries: week_1_summary.md through week_10_summary.md
  - Total: 8 JSONs + 8 CSVs + 8 Summaries + 10 Weekly = 34 report files

- [ ] **NEW**: Archive results for reproducibility
  - Create: `experiments/results/data_archives/final_results_TIMESTAMP.tar.gz`
  - Include: All JSONs, CSVs, summaries, plots
  - Generate: Manifest with file checksums (SHA256)
  - Upload to Zenodo with paper (for preservation)
  - Document: Archive location in REPRODUCIBILITY.md

- [ ] **NEW**: Generate Results Appendix
  - Create: `RESULTS_APPENDIX.md` summarizing all experiments
  - Include: Links to individual experiment summaries
  - Include: Master table aggregating key metrics across all 8 experiments
  - Include: Instructions for regenerating any individual result

**Code Finalization**:
- [ ] Code review: ensure all functions have docstrings
- [ ] Add 100+ tests (coverage >80%)
- [ ] Run full test suite: `pytest --cov`
- [ ] **NEW**: Add tests for ResultsLogger class
  - Test JSON output format matches schema
  - Test CSV generation correctness
  - Test markdown summary generation
- [ ] Format with black, isort, check with mypy
- [ ] Generate code quality report (pylint)

**Reproducibility Verification**:
- [ ] Fix random seeds: `np.random.seed(42)`, `torch.manual_seed(42)`
- [ ] **NEW**: Run full pipeline 3 times: should get identical results
  - Verify exp_001 through exp_008 produce identical JSON outputs
  - Verify statistical metrics match (to floating point precision)
  - Document reproducibility validation in REPRODUCIBILITY.md
- [ ] Document exact versions in requirements.txt + environment.yml
- [ ] Create reproducibility checklist
- [ ] Test on fresh Python environment (from requirements)
- [ ] **NEW**: Provide script to regenerate all results: `scripts/regenerate_all_results.py`

**Documentation**:
- [ ] Write README.md (5-min overview + results summary)
- [ ] Write INSTALL.md (step-by-step setup)
- [ ] Write REPRODUCIBILITY.md (exact commands to reproduce + results verification)
- [ ] Create API documentation (sphinx)
- [ ] Create example notebooks (6 notebooks for different use cases)
- [ ] **NEW**: Add section to README linking to results reports folder

**Release**:
- [ ] Final GitHub commit + push (with "Archive results" message)
- [ ] Create Release on GitHub (v0.1.0)
- [ ] Generate DOI on Zenodo (for citation + include results archive)
- [ ] Create CITATION.cff file
- [ ] Update paper with code/data/results URLs
- [ ] Create preprint (arXiv) if applicable

**Final Checklist before Submission**:
- [ ] All tests passing (CI/CD green)
- [ ] Code coverage >80%
- [ ] Code quality A+ (no pylint warnings)
- [ ] Paper anonymized (no author identifying info)
- [ ] All figures high-resolution (300+ dpi) with results from reports
- [ ] All tables properly formatted (LaTeX) pulling from result CSVs
- [ ] Statistical results properly reported (mean ± SE, p-values) from CSV
- [ ] Limitations discussed
- [ ] Broader impact statement (ethics of toxicity datasets)
- [ ] Supplementary material zipped + uploaded
- [ ] **NEW**: Results appendix linked in paper
- [ ] **NEW**: Results archive DOI provided in paper for reproducibility

**Deliverable**: 
- Production-ready code on GitHub + Zenodo
- Paper camera-ready (PDF) with verified results
- Results archive (tar.gz) with all 34 report files
- Reproducible end-to-end workflow (scripts + documentation)

---

## TESTING ROADMAP

### Unit Tests (Build Continuously)
```
Week 1: 10 tests (config system)
Week 2: 20 tests (extractors + aligner)
Week 3: 30 tests (base dataset class)
Week 4: 50 tests (all dataset implementations)
Week 5: 75 tests (extraction pipeline)
Week 6: 90 tests (alignment experiments)
Week 10: 100+ tests (complete suite)
```

### Integration Tests (After Week 4)
```
Week 4+: Multi-dataset loader pipeline
Week 5+: End-to-end extraction
Week 6+: Full pipeline (seed → alignment)
```

### Performance Tests (After Week 5)
```
Week 6: Extract vectors (1000 samples in time X?)
Week 7: Compute alignment (< Y seconds?)
Week 8: Full pipeline memory usage (< Z GB?)
```

### Continuous Integration (Weeks 1+)
```
GitHub Actions workflows:
- tests.yml: Run all tests on push
- lint.yml: black, isort, flake8 checks
- coverage.yml: Report test coverage
- reproducibility.yml: Verify results don't change
```

---

## RESOURCE ALLOCATION

### Team Roles (assuming team of 3-4)

**Lead Researcher** (50% time):
- Design experiments (Weeks 7-10)
- Write paper + findings (Weeks 11-12)
- Oversee statistical analysis

**Software Engineer** (100% time):
- Implement all code (Weeks 1-6)
- Build testing infrastructure
- Maintain code quality

**Data Engineer** (50% time):
- Set up dataset pipelines (Weeks 3-4)
- Handle large-scale data processing
- Manage caching/storage

**Junior Researcher** (50% time):
- Run experiments (Weeks 7-10)
- Collect results + metrics
- Help with paper writing

---

## SUCCESS METRICS

✅ **Week 1-2**: Config + base extractors working
✅ **Week 4**: All 5 datasets loading
✅ **Week 6**: First anchor comparison experiment complete
✅ **Week 7**: 2+ experiments with published-quality figures
✅ **Week 10**: 8 experiments complete + statistical validation
✅ **Week 12**: Paper submitted + code on GitHub

---

## FAILURE POINTS & CONTINGENCY

| Risk | Impact | Mitigation |
|------|--------|-----------|
| Llama/Mistral APIs down | Blocking extraction | Use cached embeddings or alternative models |
| Jigsaw/Kaggle access issues | Dataset loading fails | Prioritize ToxiGen + public datasets first |
| Computation too slow | Can't complete experiments | Optimize batch size + use distributed processing |
| Statistical tests don't reach p<0.05 | Findings weak | Increase sample size or investigate effect size |
| Reproducibility issues | Can't verify results | Fix random seeds + use Docker containers |

---

## SIGN-OFF

- [ ] Planning complete & reviewed
- [ ] Resources allocated
- [ ] Week 1 kickoff scheduled
- [ ] All team members trained on tooling

**Estimated Timeline**: 12 weeks (3 calendar months)
**Estimated Compute**: 100-200 GPU hours
**Estimated Team**: 2-3 FTE

