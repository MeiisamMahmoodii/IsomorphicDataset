# IsomorphicDataSet - Production Framework

**A rigorous, NeurIPS-ready framework for proving latent space isomorphism between Large Language Models**

---

## 📖 Overview

IsomorphicDataSet is a production-grade research framework that:

1. **Proves latent space isomorphism** between different LLMs through semantic-preserving concept variations
2. **Aligns latent spaces** using Procrustes analysis with SVD-based rotation computation
3. **Generates publication-ready results** for NeurIPS and other top-tier venues
4. **Supports multiple datasets** (ToxiGen, Jigsaw, HateXplain, SBIC, ETHOS)
5. **Provides comprehensive metrics** including alignment quality, orthogonality verification, and statistical analysis

### Key Innovation: Set-ConCA Theoretical Perspective

We extract mathematical "evidence" (latent vectors) from model outputs and use **Procrustes Analysis** to find the optimal orthogonal rotation that aligns different models' latent spaces. This mathematically proves that concept variations are truly **isomorphic** across models.

---

## 🚀 Quick Start

### Installation

```bash
# Clone repository
git clone https://github.com/yourname/IsomorphicDataset.git
cd IsomorphicDataset/production

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Run Pipeline

```bash
# Default run with sample data
python main.py

# With custom configuration
python main.py --config config/neurips_submission.yaml

# Custom parameters
python main.py --dataset toxigen_sample --samples 500 --output results/custom/
```

---

## 📁 Project Structure

```
production/
├── isomorphic/                          # Core framework
│   ├── __init__.py
│   ├── config.py                        # Configuration management
│   ├── pipeline.py                      # Main pipeline orchestrator
│   │
│   ├── datasets/
│   │   ├── __init__.py
│   │   └── base_dataset.py              # Dataset classes (ToxiGen, Jigsaw, etc.)
│   │
│   ├── extractors/
│   │   ├── __init__.py
│   │   └── base_extractor.py            # Vector extraction methods
│   │                                     # - Mean Pooling
│   │                                     # - Last Token
│   │                                     # - Hybrid
│   │                                     # - Attention-Weighted
│   │
│   ├── alignment/
│   │   ├── __init__.py
│   │   └── procrustes.py                # Procrustes SVD solver
│   │
│   ├── validators/                      # Validation modules
│   ├── utils/                           # Utility functions
│   └── anchors/                         # Anchor word strategies
│
├── config/                              # Configuration files
│   ├── default.yaml                     # Default configuration
│   └── neurips_submission.yaml          # NeurIPS submission config
│
├── papers/                              # Research documentation
│   ├── MAIN_PAPER.md                    # Main research paper
│   ├── METHODOLOGY.md                   # Detailed methodology
│   └── FINDINGS.md                      # Experimental findings
│
├── experiments/                         # Results and logs
│   ├── results/                         # Output results
│   └── logs/                            # Experiment logs
│
├── notebooks/                           # Jupyter notebooks for analysis
├── data/                                # Data storage
│   ├── raw/                             # Raw datasets
│   └── processed/                       # Processed datasets
│
├── main.py                              # Entry point
├── requirements.txt                     # Dependencies
└── README.md                            # This file

```

---

## 📊 Pipeline Overview

### Step 1: Data Loading & Preprocessing
- Load multiple datasets (ToxiGen, Jigsaw, HateXplain, SBIC)
- Standardize to common format: (seed, forbidden_words, semantic_intent, variations)
- Extract semantic forbidden words using 70B language model
- Validate data quality

### Step 2: Vector Extraction
- Support multiple extraction methods:
  - **Mean Pooling** (attention-masked)
  - **Last Token**
  - **Hybrid** (concatenation)
  - **Attention-Weighted**
- Batch processing with GPU acceleration
- Optional normalization

### Step 3: Latent Space Alignment
- Extract anchor word vectors (77 canonical English words)
- Compute Procrustes rotation using SVD decomposition
- Verify orthogonality constraints
- Measure alignment quality using cosine similarity

### Step 4: Metrics & Analysis
- **Alignment Quality**: Mean cosine similarity after rotation
- **Orthogonality**: Verify Q @ Q.T ≈ I
- **Variance Analysis**: Source vs. target variance retention
- **Statistical Significance**: p-values, confidence intervals

### Step 5: Report Generation
- Comprehensive experiment report (Markdown)
- Metrics in JSON format
- Visualization plots (if matplotlib enabled)
- Configuration snapshot for reproducibility

---

## 🔧 Configuration

### Default Configuration

Edit `config/default.yaml`:

```yaml
experiment:
  name: isomorphic_baseline
  description: "Baseline alignment between Llama and Mistral"
  
  models:
    - name: Llama-3-8B
      model_id: failspy/Meta-Llama-3-8B-Instruct-abliterated-v3
      batch_size: 16
    
    - name: Mistral-7B
      model_id: evolveon/Mistral-7B-Instruct-v0.3-abliterated
      batch_size: 16
  
  datasets:
    - name: toxigen_sample
      dataset_type: toxigen
      max_samples: 100
      use_banned_word_extraction: true
  
  extraction:
    method: mean_pooling
    normalize: true
  
  alignment:
    method: procrustes_svd
    num_anchors: 77
    wasserstein_threshold: 0.5

logging_level: INFO
seed: 42
device: cuda
mixed_precision: true
```

---

## 📈 Understanding the Results

### Alignment Quality Metric
- **Range**: 0 to 1 (higher is better)
- **Interpretation**:
  - > 0.95: Excellent alignment (isomorphism confirmed)
  - 0.85-0.95: Good alignment (significant isomorphism)
  - < 0.85: Weak alignment (models diverge)

### Orthogonality Error
- **Target**: Close to 0 (ideally < 1e-4)
- **Meaning**: How much Q violates orthogonality constraint
- **Interpretation**:
  - < 1e-4: Perfect Procrustes solution
  - 1e-4 - 1e-3: Numerical precision acceptable
  - > 1e-3: Solution may be suboptimal

### Key Finding
If alignment quality is high (> 0.95) with low orthogonality error, this mathematically proves that:
- Different LLMs encode similar concepts in aligned coordinate systems
- Concept variations preserve semantic meaning across model boundaries
- Latent space structure is fundamentally **isomorphic**

---

## 🧪 Supported Datasets

### ToxiGen
- Public, HuggingFace hosted
- Toxic comment variations
- ~100k seeds

### Jigsaw Unintended Bias
- Kaggle dataset (requires manual download)
- Bias annotations
- ~2M samples

### HateXplain
- Hate speech detection with rationales
- Explanation annotations
- ~20k samples

### SBIC (Social Bias Inference Corpus)
- Social bias language
- Identity annotations
- ~150k samples

---

## 💻 Supported Models

The framework has been tested with:

- **Llama 3.1** (8B, 70B)
- **Mistral 7B**
- **Qwen 2.5** (7B, 72B)
- **Any HuggingFace transformer model**

---

## 📊 Output Files

After running the pipeline, find:

```
experiments/results/isomorphic_baseline_YYYYMMDD_HHMMSS/
├── RESULTS_REPORT.md          # Main results summary
├── metrics.json               # Quantitative metrics
├── config.json                # Configuration snapshot
├── experiment_log.txt         # Detailed execution log
└── visualizations/            # (Optional) Plots and charts
    ├── alignment_quality.png
    ├── orthogonality_analysis.png
    └── variance_comparison.png
```

---

## 📚 Citation

If you use this framework, please cite:

```bibtex
@article{isomorphicdataset2025,
  title={Latent Space Isomorphism in Large Language Models: Mathematical Evidence},
  author={... (your details)},
  journal={NeurIPS},
  year={2025}
}
```

---

## 🤝 Contributing

Contributions welcome! Please ensure:
- Code follows PEP 8 style guide
- All tests pass: `pytest tests/`
- Type hints included: `mypy --strict`
- Documentation updated

---

## 📝 License

MIT License - see LICENSE file for details

---

## ❓ FAQ

**Q: How many GPU memory do I need?**
A: Minimum 8GB for single model. Recommended 16GB+ with `device_map="auto"` for multi-GPU distribution.

**Q: Can I use CPU only?**
A: Yes, set `device: cpu` in config, but expect 10-100x slower execution.

**Q: What's the difference between extraction methods?**
A: Mean Pooling is most robust, Last Token is fastest, Hybrid captures both. See papers/ for comparison.

**Q: How do I interpret negative orthogonality errors?**
A: Shouldn't occur. Errors > 0 indicate numerical issues. Re-run with `mixed_precision: false`.

**Q: Can I add custom datasets?**
A: Yes! Extend `BaseDataset` class in `isomorphic/datasets/base_dataset.py`.

---

## 📞 Support

- Issues: GitHub Issues
- Discussions: GitHub Discussions  
- Email: (your email)

---

**Last Updated**: 2025-04-09
**Status**: Production Ready ✓
