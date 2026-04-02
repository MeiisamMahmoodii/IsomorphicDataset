# IsomorphicDataSet

A cutting-edge framework for proving **Latent Space Isomorphism** between different Large Language Models (LLMs). This project demonstrates that different models, despite their architectural differences, encode the same semantic concepts in mathematically aligned coordinate systems.

## 🎯 Core Innovation

**Set-ConCA Theoretical Perspective**: We extract mathematical "evidence" (latent vectors) from model outputs and use **Procrustes Analysis** to find the optimal orthogonal rotation that aligns different models' latent spaces. This proves that concept variations are truly isomorphic across models.

## ✨ Key Features

### 1. **Concept Variation Generation**
- Generate isomorphic text variations using validated rewriting
- Support for both Llama and Mistral models
- Forbidden word constraints to control semantics
- Perspective injection for forced semantic consistency

### 2. **Multiple Vector Extraction Methods**
- **Mean Pooling**: Attention-masked averaging of all tokens
- **Last Token**: Final token representation only
- **Hybrid**: Concatenation of both methods for richer representation

### 3. **Latent Space Alignment**
- **Procrustes Analysis**: SVD-based orthogonal rotation computation
- **Anchor Word Strategy**: Use 84+ neutral reference words for alignment
- **Data Centering**: Numerically stable transformation
- **Cross-model verification**: Measure alignment quality

### 4. **Validation & Metrics**
- Intra-model concept stability (short vs. long variations)
- Cross-model similarity (before/after alignment)
- Procrustes rotation matrix properties
- Frobenius norm error measurement

## 📦 Project Structure

```
IsomorphicDataSet/
├── main.py                          # Main alignment workflow
├── example_cross_model_alignment.py # Step-by-step example
├── pyproject.toml                   # Project dependencies
├── README.md                        # This file
└── source/
    ├── generator.py                 # ConceptGenerator class
    ├── alignment.py                 # Anchor-based alignment
    └── alignment_utils.py           # Procrustes SVD solver
```

## 🚀 Quick Start

### Installation

```bash
# Clone repository
git clone https://github.com/yourusername/IsomorphicDataSet.git
cd IsomorphicDataSet

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
# Or with uv:
uv sync
```

### Basic Usage

```python
from source.generator import ConceptGenerator

# Initialize model
model_name = "meta-llama/Meta-Llama-3-8B-Instruct"
gen = ConceptGenerator(model_name)

# Generate isomorphic variation
seed = "your concept here"
forbidden = ["word1", "word2"]
variation = gen.get_validated_variation(seed, forbidden, min_words=5, max_words=10)

# Extract latent vector
vector = gen.get_hybrid_vector(variation)

# Get latent vector with mean pooling
vector = gen.get_latent_vector(variation)

# Get latent vector from last token
vector = gen.get_last_token_vector(variation)
```

### Full Workflow

Run the main alignment pipeline:
```bash
uv run main.py
```

Or the detailed step-by-step example:
```bash
uv run example_cross_model_alignment.py
```

## 🔬 Theoretical Background

### Procrustes Problem
Given matrices X (source) and Y (target), find orthogonal rotation Q that minimizes:
```
||Y - X·Q||_F
```

**Solution**: Q = U·V^T where U, S, V^T = SVD(Y^T·X)

### Perspective Injection
Forces models to maintain consistent viewpoint by explicit prompt engineering:
```
"You MUST maintain the original perspective and supportive stance.
Do not offer alternatives or counterarguments."
```

### Anchor Word Strategy
- 84+ neutral reference words across 10 categories
- Compute rotation from anchor word alignments
- Verify rotation on concept vectors

## 📊 Output Example

```
Before Alignment:     0.0234 (raw cross-model similarity)
After Alignment:      0.5678 (post-Procrustes)
Improvement:          +0.5444

Anchor Alignment Quality: 0.8524
Rotation Matrix Properties:
  - Is orthogonal: True ✓
  - Determinant: 1.0 (true rotation)
```

## 🛠️ Core Components

### ConceptGenerator Class
```python
class ConceptGenerator:
    def __init__(self, model_name)
    def get_validated_variation(seed, forbidden, min_words, max_words, maintain_perspective=False)
    def get_latent_vector(text)              # Mean pooling
    def get_last_token_vector(text)          # Last token
    def get_hybrid_vector(text)              # Both combined
```

### Alignment Functions
```python
# alignment_utils.py
calculate_procrustes_rotation(source_vectors, target_vectors)
apply_alignment(vector, Q, source_mean, target_mean)
apply_alignment_batch(vectors, Q, source_mean, target_mean)
compute_alignment_quality(source, target, Q, src_mean, tgt_mean)

# alignment.py
extract_anchor_vectors(generator, method="mean|last|hybrid")
align_latent_spaces(anchors_source, anchors_target)
get_all_anchor_words()
```

## 🎓 Research Applications

This framework enables:
1. **Cross-model concept alignment** for federated learning
2. **Model interoperability studies** for semantic equivalence
3. **Transfer learning** in latent space
4. **Concept equivalence verification** across architectures
5. **Bias analysis** through isomorphic mapping

## 📝 Citation

If you use this framework in research, please cite:

```bibtex
@software{isomorphic_dataset_2026,
  title={IsomorphicDataSet: Latent Space Alignment through Procrustes Analysis},
  author={Your Name},
  year={2026},
  url={https://github.com/yourusername/IsomorphicDataSet}
}
```

## 🔗 References

- **Procrustes Analysis**: Gower, J. C. (1975). "Generalized procrustes analysis." Psychometrika.
- **SVD-based Orthogonal Procrustes**: "Orthogonal Procrustes Problem" (classical method)
- **Set-ConCA**: Conceptual alignment through constrained concept analysis

## ⚠️ Important Notes

- This project works with models available via Hugging Face
- Requires GPU for efficient inference (recommended: NVIDIA GPU with 8GB+ VRAM)
- Large models require substantial memory (Llama-3-8B: ~16GB)
- Perspective injection requires careful prompt engineering

## 📋 Dependencies

- `torch >= 2.0`
- `transformers >= 4.30`
- `numpy >= 1.24`

See `pyproject.toml` for complete dependencies.

## 🤝 Contributing

Contributions are welcome! Please feel free to:
1. Fork the repository
2. Create a feature branch
3. Submit a pull request

## 📄 License

MIT License - see LICENSE file for details.

## 🙏 Acknowledgments

- Hugging Face for model hosting and transformers library
- PyTorch team for deep learning framework
- Original Procrustes analysis research community

---

**Created**: April 2026  
**Status**: Active Development  
**Python Version**: 3.8+
