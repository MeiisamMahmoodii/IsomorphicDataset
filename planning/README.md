# Planning & Strategy Documents

This folder contains the complete strategic and operational planning for the IsomorphicDataSet project for NeurIPS publication.

## Documents

### Core Planning
- **PROJECT_PLAN_NEURIPS.md** - Main strategic plan with full directory structure, 8 experiments, 100+ tests
  - Problem statement and theoretical framework
  - Complete project structure (~80 files)
  - 5-dataset support (ToxiGen, Jigsaw, HateXplain, SBIC, ETHOS)
  - 8 rigorously designed experiments
  - Proven prompts & strict constraints (locked from source/generator.py)
  - Results schema & reporting structures

### Implementation
- **IMPLEMENTATION_GUIDE.md** - Code structure and templates
  - Base classes architecture (Extractors, Datasets, Aligners)
  - Python templates with concrete examples
  - Pydantic config system
  - Vector extraction methods
  - Procrustes alignment implementation

- **MODEL_IMPLEMENTATION_GUIDE.md** - Complete model usage guide
  - Model assignment matrix (Experiments × Models)
  - Implementation checklist with download timeline
  - Code examples for dynamic model loading
  - VRAM management strategy
  - Reproducibility checklist

- **config/models.yaml** - Model specifications and assignments
  - Embedding model: Qwen2.5-7B (fixed, no generation)
  - Rewriting models: 5 primary variants (Llama-8B, Qwen-7B, Qwen-32B, Mistral-12B)
  - Banned word extraction: Qwen2.5-72B (40GB, batch_size=1)
  - Experiment assignments per model
  - Hardware requirements
  - Generation parameters

### Execution & Workflow
- **EXECUTION_PLAN.md** - Week-by-week detailed breakdown (12 weeks)
  - Week 1-2: Foundation + model setup + results infrastructure
  - Week 3-4: Multi-dataset framework with 70B banned word extraction
  - Week 5-6: Vector extraction + alignment with tracking
  - Week 7-10: 8 Experiments with comprehensive results logging
  - Week 11-12: Paper writing + results archival + release

### Reference & Documentation
- **REFERENCE_GUIDE.md** - Quick decision trees and matrices
  - Model selection framework
  - Experiment design patterns
  - Common pitfalls & solutions
  - Performance optimization strategies

- **REPORTING_STRUCTURE.md** - Complete results framework
  - 42 total report files specification
  - JSON, CSV, and Markdown formats
  - Integration with paper generation
  - Reproducibility verification checklist

---

## Key Characteristics

### 100% Specification
- ✅ Exact model IDs (HuggingFace model names)
- ✅ Specific dataset choices (5 for multi-domain comparison)
- ✅ Complete experiment design (8 experiments with hypotheses)
- ✅ Proven prompts locked (from source/generator.py)

### Model Configuration
- **Primary Embedding**: Qwen2.5-7B-Instruct-Abliterated (fixed across all)
- **Rewriting Variants**: 5 models for comparison
- **Quality Extraction**: Qwen2.5-72B for banned word extraction

### Quality Assurance
- 100+ automated tests specified
- Results tracking at every pipeline stage
- Statistical validation (ANOVA, t-tests, effect sizes)
- Reproducibility verification (3 runs, seeded, identical results)

### Publication-Ready
- 42 report files for comprehensive metrics
- Results archival with DOI
- Zenodo registration for data/code preservation
- All statistics with 95% confidence intervals

---

## Quick Navigation

**Starting Point**: Read PROJECT_PLAN_NEURIPS.md Part 1 (Overview)

**For Implementation**: IMPLEMENTATION_GUIDE.md → CODE TEMPLATES

**For Models**: MODEL_IMPLEMENTATION_GUIDE.md → QUICK SUMMARY

**For Execution**: EXECUTION_PLAN.md → Week 1 Tasks

**For Results**: REPORTING_STRUCTURE.md → Report File Manifest

**For Decisions**: REFERENCE_GUIDE.md → Decision Matrices

---

## Status

✅ All planning documents completed and organized
✅ Integration with existing codebase documented
✅ Proven prompts extracted and locked
✅ Results reporting framework designed (42 report files)
✅ Model configuration finalized

**Next Steps**: 
1. Implement ResultsLogger class
2. Implement BannedWordsExtractor with Qwen2.5-72B
3. Execute EXECUTION_PLAN weeks 1-12
