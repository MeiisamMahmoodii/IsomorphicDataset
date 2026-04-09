# Results Reporting Structure

**Status**: Framework complete and integrated into EXECUTION_PLAN.md (Weeks 1-12)

## Overview
All 12 weeks now include systematic results tracking, reporting, and archival. This document outlines the complete reporting infrastructure.

---

## Report Files Generated Per Stage

### Week 1: Results Infrastructure Setup (NEW)
```
experiments/
├── results/
│   ├── metadata.json              # Experiment registry + configuration
│   ├── reports/                   # All CSV and markdown reports
│   ├── plots/                     # Visualization outputs
│   ├── tables/                    # Data tables for paper
│   ├── data_archives/             # Timestamped result backups
│   └── exp_001...008/             # Per-experiment result directories
```

### Weeks 2-4: Banned Words Extraction + Results
**Reports Generated**:
- `experiments/results/reports/preprocessing_report.csv`
- `experiments/results/reports/banned_words_extraction.csv`
- `experiments/results/reports/week_2_summary.md` through `week_4_summary.md`

### Week 5: Vector Extraction + Results Logging
**Reports Generated**:
- `experiments/results/reports/extraction_report.csv`
- `experiments/results/reports/anchor_extraction.csv`
- `experiments/results/reports/week_5_summary.md`

### Week 6: Alignment + Results Reporting
**Reports Generated**:
- `experiments/results/reports/alignment_report.csv`
- `experiments/results/reports/alignment_detailed.csv`
- `experiments/results/reports/alignment_stability.csv`
- `experiments/results/reports/alignment_cross_model.csv`
- `experiments/results/reports/week_6_summary.md`

### Weeks 7-10: Experiments 1-8 + Results & Stats Tracking

#### Week 7: Experiment 2 (Extraction Methods Comparison)
- `experiments/results/exp_002_extraction_methods.json` (detailed + metadata)
- `experiments/results/reports/exp_002_extraction_methods_results.csv` (metrics + statistics)
- `experiments/results/reports/exp_002_summary.md` (key findings)

#### Week 8: Experiment 3 (Cross-Dataset Generalization)
- `experiments/results/exp_003_cross_dataset_alignment.json`
- `experiments/results/reports/exp_003_generalization_results.csv`
- `experiments/results/reports/exp_003_summary.md`

#### Week 9: Experiments 4 & 6 (Model Pairs + Scalability)
- `experiments/results/exp_004_model_pairs.json` + CSV + summary
- `experiments/results/exp_006_scalability.json` + CSV + summary
- `experiments/results/plots/scaling_curves.pdf`
- `experiments/results/plots/model_pair_comparison.pdf`

#### Week 10: Experiments 5, 7, 8 (Stability, Semantic, Statistics)
- `experiments/results/exp_005_stability.json` + CSV + summary
- `experiments/results/exp_007_semantic_preservation.json` + CSV + summary
- `experiments/results/exp_008_statistical_significance.json` + CSV + summary
- `experiments/analysis/statistical_summary.md` (comprehensive)

### Weeks 11-12: Paper Writing + Results Archival

#### Week 11: Results Compilation & Paper
- Aggregate results from all 8 experiment JSONs
- Generate publication tables from CSVs
- Create figures from report data
- Link figures/tables to experiment summaries

#### Week 12: Results Archival & Verification
- `experiments/results/data_archives/final_results_[TIMESTAMP].tar.gz`
  - Contains: All 8 JSONs + 8 CSVs + 8 summaries + 10 weekly markdowns
  - Includes: SHA256 manifest for verification
- `experiments/results/RESULTS_APPENDIX.md`
  - Master aggregation of all experiments
  - Links to individual result files
  - Instructions for regeneration
- Zenodo DOI for long-term preservation

---

## JSON Result Format (Per Experiment)

```json
{
  "experiment_id": "exp_001",
  "title": "Anchor Strategy Comparison",
  "execution_date": "2024-01-15",
  "seed": 42,
  "metadata": {
    "models_used": ["qwen2.5_7b", "llama_3.1_8b"],
    "datasets": ["toxigen"],
    "sample_size": 100,
    "total_runtime_seconds": 3600
  },
  "preprocessing": {
    "total_seeds": 100,
    "banned_words_extracted": 100,
    "extraction_success_rate": 1.0
  },
  "variation_generation": {
    "total_variations": 300,
    "success_rate": 0.98
  },
  "vector_extraction": {
    "total_vectors": 400,
    "extraction_time_mean_seconds": 0.5,
    "extraction_time_std": 0.1
  },
  "alignment": {
    "frobenius_error": 0.045,
    "frobenius_error_std": 0.008,
    "orthogonality_error": 1e-7,
    "orthogonality_pass": true
  },
  "metrics": {
    "cosine_similarity_mean": 0.92,
    "cosine_similarity_std": 0.08,
    "preservation_threshold_0_85_percent": 0.95
  },
  "statistical_tests": {
    "method": "ANOVA",
    "f_statistic": 15.3,
    "p_value": 0.0001,
    "significant": true
  }
}
```

## CSV Report Format (Per Stage)

Example: `exp_002_extraction_methods_results.csv`

```
method,frobenius_error_mean,frobenius_error_std,orthogonality_error,anova_p_value,significant
mean_pooling,0.042,0.006,1.2e-7,0.0001,yes
last_token,0.051,0.009,1.8e-7,0.0001,yes
hybrid,0.038,0.005,0.9e-7,0.0001,yes
attention_weighted,0.045,0.007,1.5e-7,0.0001,yes
```

## Markdown Summary Format

Example: `exp_002_summary.md`

```markdown
# Experiment 2: Extraction Methods Comparison

**Date**: Week 7, 2024-01-15
**Models**: Llama-3.1-8B, Qwen2.5-7B, Qwen3-32B
**Datasets**: All 5 (ToxiGen, Jigsaw, HateXplain, SBIC, ETHOS)

## Key Findings
- **Best Method**: Hybrid approach (Frobenius = 0.038 ± 0.005)
- **Worst Method**: Last-token (Frobenius = 0.051 ± 0.009)
- **Statistical Significance**: ANOVA F=15.3, p<0.0001

## Metrics
| Method | Frobenius Error | Orthogonality Error | CPU Time (s) |
|--------|-----------------|---------------------|--------------|
| Mean Pooling | 0.042 ± 0.006 | 1.2e-7 | 2.3 ± 0.4 |
| Last Token | 0.051 ± 0.009 | 1.8e-7 | 1.1 ± 0.2 |
| Hybrid | 0.038 ± 0.005 | 0.9e-7 | 3.4 ± 0.5 |
| Attention-Weighted | 0.045 ± 0.007 | 1.5e-7 | 5.2 ± 0.8 |

## Recommendation
Use **Hybrid** method for production (best Frobenius, acceptable runtime).
```

---

## Total Report Generation

**By End of Week 12**:
- **8 Experiment JSONs** (detailed metadata + results)
- **8 Experiment Result CSVs** (metrics + statistics)
- **8 Experiment Summaries** (markdown findings)
- **10 Weekly Summaries** (Week 1-10 progress)
- **1 Master Results Appendix** (aggregation of all 8 experiments)
- **1 Statistical Summary** (comprehensive ANOVA, t-tests, effect sizes)
- **5+ Publication Quality Figures** (from result data)
- **1 Results Archive** (tar.gz with manifest for reproducibility)

**Total: 42 Report Files**

---

## Integration with Paper

**Week 11 Tasks**:
1. Parse all 8 JSON files
2. Extract metrics from 8 CSV files
3. Generate publication tables:
   - Table 1: Anchor strategy comparison (from exp_002)
   - Table 2: Cross-dataset generalization (from exp_003)
   - Table 3: Model pair performance (from exp_004)
   - Table 4: Scalability metrics (from exp_006)
   - Table 5: Statistical summary (from exp_008)
4. Create figures:
   - Figure 1: Extraction method performance
   - Figure 2: Alignment quality by model pair
   - Figure 3: Scalability curves
   - Figure 4: Stability under perturbations
   - Figure 5: Semantic preservation histogram

**Week 12 Tasks**:
1. Archive all 42 report files to tar.gz
2. Generate DOI on Zenodo
3. Verify reproducibility (run pipeline 3 times, compare results)
4. Document results preservation in paper

---

## Reproducibility Verification

**Week 12 Checklist**:
- [ ] All 8 experiment JSONs exist and validate against schema
- [ ] All 8 result CSVs have correct columns + data types
- [ ] All 8 experiment summaries generated successfully
- [ ] All 10 weekly summaries generated successfully
- [ ] Statistics in CSVs match JSON data (to floating point precision)
- [ ] Results archive created with checksums
- [ ] `RESULTS_APPENDIX.md` complete with final metrics
- [ ] Regeneration script (`scripts/regenerate_all_results.py`) tested
- [ ] Results preserve across 3 independent runs (seed=42)

---

## File Manifest

```
experiments/
├── results/
│   ├── metadata.json
│   ├── reports/
│   │   ├── preprocessing_report.csv
│   │   ├── extraction_report.csv
│   │   ├── alignment_report.csv
│   │   ├── alignment_detailed.csv
│   │   ├── alignment_stability.csv
│   │   ├── alignment_cross_model.csv
│   │   ├── exp_001_anchor_strategy_results.csv
│   │   ├── exp_002_extraction_methods_results.csv
│   │   ├── exp_003_generalization_results.csv
│   │   ├── exp_004_model_pairs_results.csv
│   │   ├── exp_005_stability_results.csv
│   │   ├── exp_006_scalability_results.csv
│   │   ├── exp_007_semantic_results.csv
│   │   ├── exp_008_statistical_results.csv
│   │   ├── exp_001_summary.md through exp_008_summary.md
│   │   ├── week_1_summary.md through week_10_summary.md
│   ├── plots/
│   │   ├── anchor_comparison.pdf
│   │   ├── scaling_curves.pdf
│   │   ├── model_pair_comparison.pdf
│   │   ├── stability_curves.pdf
│   │   ├── semantic_histogram.pdf
│   ├── tables/
│   │   └── (Publication-quality tables in LaTeX)
│   ├── exp_001.json through exp_008.json
│   ├── data_archives/
│   │   └── final_results_[TIMESTAMP].tar.gz
│   └── RESULTS_APPENDIX.md
└── analysis/
    └── statistical_summary.md
```

---

## Status Update

✅ **EXECUTION_PLAN.md Updated**: Weeks 1-12 now include results tracking  
✅ **Reporting Structure Defined**: 42 report files specified  
✅ **Integration Points Documented**: Paper figure/table generation linked to result CSVs  
✅ **Reproducibility Framework**: Verification checklist and archival plan in place  

**Next**: Implement ResultsLogger class and BannedWordsExtractor class to execute this framework.
