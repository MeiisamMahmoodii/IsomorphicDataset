# Implementation Guide: Building IsomorphicDataSet from Ground Up

---

## PHASE 1: FOUNDATION ARCHITECTURE (Week 1-2)

### 1.1 Project Initialization

```bash
# Create project structure
mkdir isomorphic-dataset && cd isomorphic-dataset
python -m venv .venv
source .venv/bin/activate  # or .venv\Scripts\activate on Windows

# Initialize git
git init
git config user.email "research@neurips.ai"
git config user.name "IsomorphicDataSet"

# Create directory tree
mkdir -p isomorphic/{datasets,extractors,anchors,utils}
mkdir -p tests/{test_unit,test_integration,test_performance,test_statistics,fixtures}
mkdir -p experiments/{results/plots,results/tables,analysis}
mkdir -p data/{raw,processed,vectors,alignments,metadata}
mkdir -p notebooks config scripts docs/papers papers/supplementary
```

### 1.2 Core Dependencies (pyproject.toml)

```toml
[build-system]
requires = ["setuptools>=65.0", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "isomorphic-dataset"
version = "0.1.0"
description = "Proving latent space isomorphism between LLMs"
requires-python = ">=3.10"
license = {text = "MIT"}
authors = [{name = "Research Team", email = "research@example.com"}]

keywords = ["nlp", "llm", "latent-space", "isomorphism", "alignment"]
classifiers = [
    "Development Status :: 3 - Alpha",
    "Intended Audience :: Science/Research",
    "License :: OSI Approved :: MIT License",
    "Programming Language :: Python :: 3.10",
    "Programming Language :: Python :: 3.11",
    "Topic :: Scientific/Engineering :: Artificial Intelligence",
]

dependencies = [
    "torch>=2.0.0",
    "transformers>=4.35.0",
    "numpy>=1.24.0",
    "scipy>=1.11.0",
    "pandas>=2.0.0",
    "scikit-learn>=1.3.0",
    "datasets>=2.14.0",
    "pydantic>=2.0.0",
    "pyyaml>=6.0",
    "tqdm>=4.66.0",
    "accelerate>=0.24.0",
]

[project.optional-dependencies]
dev = [
    "pytest>=7.4.0",
    "pytest-cov>=4.1.0",
    "pytest-xdist>=3.3.0",
    "black>=23.10.0",
    "isort>=5.12.0",
    "flake8>=6.1.0",
    "mypy>=1.6.0",
    "pylint>=3.0.0",
]

ml = [
    "matplotlib>=3.8.0",
    "seaborn>=0.13.0",
    "plotly>=5.17.0",
    "jupyter>=1.0.0",
    "ipywidgets>=8.1.0",
]

docs = [
    "sphinx>=7.2.0",
    "sphinx-rtd-theme>=1.3.0",
    "sphinx-autodoc-typehints>=1.24.0",
]

[project.urls]
"Homepage" = "https://github.com/yourorg/isomorphic-dataset"
"Documentation" = "https://isomorphic-dataset.readthedocs.io"
"Repository" = "https://github.com/yourorg/isomorphic-dataset"
"Bug Tracker" = "https://github.com/yourorg/isomorphic-dataset/issues"

[tool.setuptools.packages.find]
where = ["."]

[tool.black]
line-length = 100
target-version = ['py310', 'py311']

[tool.isort]
profile = "black"
line_length = 100

[tool.mypy]
python_version = "3.10"
check_untyped_defs = true
ignore_missing_imports = true

[tool.pytest.ini_options]
testpaths = ["tests"]
python_files = ["test_*.py"]
addopts = "-v --cov=isomorphic --cov-report=html"
```

### 1.3 Configuration System (isomorphic/config.py)

```python
from dataclasses import dataclass, field
from typing import Dict, List, Optional
from pathlib import Path
import yaml

@dataclass
class ModelConfig:
    """LLM model specifications"""
    name: str
    model_id: str
    purpose: str          # "embedding" or "rewriting"
    device: str = "cuda"
    dtype: str = "bfloat16"  # bfloat16, float32, float16
    max_length: int = 2048
    batch_size: int = 8
    load_in_8bit: bool = False  # For large models

@dataclass
class DatasetConfig:
    """Dataset specifications"""
    name: str
    source: str  # toxigen, jigsaw, hatexplain, sbic, ethos, custom
    path: Optional[str] = None
    download_url: Optional[str] = None
    split: str = "train"
    limit: Optional[int] = None  # Limit samples for testing

@dataclass
class ExtractionConfig:
    """Vector extraction settings"""
    method: str  # mean_pooling, last_token, hybrid, attention_weighted
    embedding_dim: int = 4096
    use_attention_mask: bool = True
    normalize: bool = True

@dataclass
class AnchorConfig:
    """Anchor strategy settings"""
    strategy: str  # sentence, word, concept
    num_anchors: int = 100
    anchor_file: Optional[str] = None
    auto_select: bool = False

@dataclass
class AlignmentConfig:
    """Procrustes alignment settings"""
    method: str = "procrustes_svd"
    center_data: bool = True
    add_centering_column: bool = False
    numerical_stability_threshold: float = 1e-10
    compute_confidence_intervals: bool = True

@dataclass
class ExperimentConfig:
    """Experiment settings"""
    name: str
    models: List[ModelConfig]
    datasets: List[DatasetConfig]
    extraction: ExtractionConfig
    anchors: AnchorConfig
    alignment: AlignmentConfig
    random_seed: int = 42
    num_runs: int = 3  # For statistical testing
    output_dir: Path = field(default_factory=lambda: Path("experiments/results"))

class ConfigManager:
    """Load/save configurations"""
    
    @staticmethod
    def load_yaml(path: Path) -> Dict:
        with open(path) as f:
            return yaml.safe_load(f)
    
    @staticmethod
    def save_yaml(config: Dict, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)
```

---

## PHASE 2: MULTI-DATASET FRAMEWORK (Week 3-4)

### 2.1 Dataset Base Class (isomorphic/datasets/base_dataset.py)

```python
from abc import ABC, abstractmethod
from typing import List, Dict
import logging

logger = logging.getLogger(__name__)

class Dataset(ABC):
    """Abstract base class for all datasets"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.name = config.get("name")
        self.seeds = []
        self.metadata = {}
    
    @abstractmethod
    def download(self) -> None:
        """Download dataset if not present"""
        pass
    
    @abstractmethod
    def load(self) -> List[str]:
        """Load raw data into self.seeds"""
        pass
    
    @abstractmethod
    def validate(self) -> bool:
        """Check data integrity"""
        pass
    
    @abstractmethod
    def preprocess(self) -> List[Dict]:
        """Convert to standard format:
        {
            'seed': str,
            'forbidden_words': List[str],
            'semantic_intent': str,
            'length_range': (int, int),
            'dataset_source': str,
            'original_id': str
        }
        """
        pass
    
    def statistics(self) -> Dict:
        """Return dataset statistics"""
        if not self.seeds:
            self.load()
        
        lengths = [len(s.split()) for s in self.seeds]
        return {
            'total_seeds': len(self.seeds),
            'avg_length': sum(lengths) / len(lengths) if lengths else 0,
            'min_length': min(lengths) if lengths else 0,
            'max_length': max(lengths) if lengths else 0,
            'license': self.metadata.get('license'),
            'source_url': self.metadata.get('source_url')
        }
```

### 2.2 ToxiGen Dataset Implementation

```python
# isomorphic/datasets/toxigen_dataset.py
from datasets import load_dataset
from .base_dataset import Dataset
import json

class ToxiGenDataset(Dataset):
    """ToxiGen toxicity dataset
    
    https://huggingface.co/datasets/toxigen/toxigen/
    License: CC-BY-4.0
    Size: ~274K examples
    """
    
    HF_DATASET = "toxigen/toxigen"
    FORBIDDEN_WORDS_FILE = "data/metadata/toxigen_forbidden_words.json"
    
    def download(self) -> None:
        """Verify dataset can be loaded (HF handles download)"""
        try:
            load_dataset(self.HF_DATASET, split="train", split=1)
        except Exception as e:
            logger.error(f"Failed to access ToxiGen: {e}")
            raise
    
    def load(self) -> List[str]:
        """Load from HuggingFace"""
        limit = self.config.get("limit")
        split = self.config.get("split", "train")
        
        ds = load_dataset(self.HF_DATASET, split=split, cache_dir="data/raw/toxigen")
        
        if limit:
            ds = ds.select(range(min(limit, len(ds))))
        
        self.seeds = ds["text"]
        self.metadata = {
            "total": len(self.seeds),
            "license": "CC-BY-4.0",
            "source_url": "https://huggingface.co/datasets/toxigen/toxigen/",
            "categories": set(ds.get("category", []))
        }
        return self.seeds
    
    def validate(self) -> bool:
        """Check for obvious issues"""
        if not self.seeds:
            return False
        
        # Check no empty strings
        if any(not s.strip() for s in self.seeds):
            logger.warning("Found empty strings")
            self.seeds = [s for s in self.seeds if s.strip()]
        
        return len(self.seeds) > 0
    
    def preprocess(self) -> List[Dict]:
        """Convert to standard format"""
        if not self.seeds:
            self.load()
        
        # Load forbidden words pool
        forbidden_pool = self._load_forbidden_words()
        
        processed = []
        for idx, seed in enumerate(self.seeds):
            # Extract 5 key concepts from seed
            forbidden = self._extract_forbidden_words(seed, forbidden_pool, k=5)
            
            processed.append({
                'seed': seed,
                'forbidden_words': forbidden,
                'semantic_intent': 'toxic perspective analysis',
                'length_range': (5, 10),  # Short variations
                'dataset_source': 'toxigen',
                'original_id': f'toxigen_{idx}'
            })
        
        return processed
    
    @staticmethod
    def _load_forbidden_words() -> Dict[str, List[str]]:
        """Load predefined forbidden word sets by category"""
        # Could load from JSON or define inline
        return {
            'exclusion': ['exclude', 'ban', 'remove', 'eliminate', ...],
            'dehumanization': ['animal', 'beast', 'monster', ...],
            # ... more categories
        }
    
    @staticmethod
    def _extract_forbidden_words(seed: str, pool: Dict, k: int = 5) -> List[str]:
        """Extract k important words from seed"""
        # Could use TF-IDF or keyword extraction
        pass
```

### 2.3 Jigsaw Dataset Implementation

```python
# isomorphic/datasets/jigsaw_dataset.py
from .base_dataset import Dataset
import pandas as pd

class JigsawDataset(Dataset):
    """Jigsaw Unintended Bias in Toxicity Classification
    
    https://www.kaggle.com/c/jigsaw-unintended-bias-in-toxicity-classification
    License: CC-BY-SA-3.0
    Size: ~2M comments
    """
    
    def download(self) -> None:
        """Download from Kaggle (requires kaggle API credentials)"""
        import subprocess
        subprocess.run([
            "kaggle", "competitions", "download",
            "-c", "jigsaw-unintended-bias-in-toxicity-classification",
            "-p", "data/raw/jigsaw"
        ])
    
    def load(self) -> List[str]:
        """Load CSV data"""
        df = pd.read_csv("data/raw/jigsaw/train.csv")
        
        limit = self.config.get("limit")
        if limit:
            df = df.head(limit)
        
        # Filter for toxic comments (toxicity > 0.5)
        toxic = df[df['target'] > 0.5]
        self.seeds = toxic['comment_text'].tolist()
        
        self.metadata = {
            "total": len(self.seeds),
            "license": "CC-BY-SA-3.0",
            "toxic_examples": len(toxic),
            "bias_labels": list(df.columns[6:])  # identity_attack, insult, etc.
        }
        return self.seeds
    
    def validate(self) -> bool:
        if not self.seeds:
            return False
        self.seeds = [s for s in self.seeds if isinstance(s, str) and len(s) > 5]
        return len(self.seeds) > 0
    
    def preprocess(self) -> List[Dict]:
        """Convert to standard format"""
        if not self.seeds:
            self.load()
        
        processed = []
        for idx, seed in enumerate(self.seeds):
            processed.append({
                'seed': seed,
                'forbidden_words': self._extract_key_terms(seed),
                'semantic_intent': 'toxicity analysis with bias awareness',
                'length_range': (10, 20),
                'dataset_source': 'jigsaw',
                'original_id': f'jigsaw_{idx}'
            })
        
        return processed
```

### 2.4 Dataset Loader Factory (isomorphic/loader.py)

```python
# isomorphic/loader.py
from typing import Dict, List
from .datasets import (
    ToxiGenDataset, JigsawDataset, HateXplainDataset,
    SBICDataset, ETHOSDataset, CustomDataset
)

class DatasetLoader:
    """Factory for loading any supported dataset"""
    
    REGISTRY = {
        'toxigen': ToxiGenDataset,
        'jigsaw': JigsawDataset,
        'hatexplain': HateXplainDataset,
        'sbic': SBICDataset,
        'ethos': ETHOSDataset,
        'custom': CustomDataset,
    }
    
    @classmethod
    def load(cls, dataset_name: str, config: Dict):
        """Load dataset by name"""
        if dataset_name not in cls.REGISTRY:
            raise ValueError(f"Unknown dataset: {dataset_name}")
        
        dataset_class = cls.REGISTRY[dataset_name]
        dataset = dataset_class(config)
        
        dataset.download()
        dataset.load()
        dataset.validate()
        
        return dataset
    
    @classmethod
    def load_multiple(cls, configs: List[Dict]):
        """Load multiple datasets"""
        datasets = {}
        for config in configs:
            name = config['name']
            datasets[name] = cls.load(name, config)
        
        return datasets
```

---

## PHASE 3: VECTOR EXTRACTION (Week 3)

### 3.1 Extraction Base Class

```python
# isomorphic/extractors/base_extractor.py
from abc import ABC, abstractmethod
from typing import Tuple
import torch

class VectorExtractor(ABC):
    """Base class for extracting latent vectors"""
    
    def __init__(self, model_name: str, device: str = "cuda"):
        self.model_name = model_name
        self.device = device
        self._load_model()
    
    @abstractmethod
    def _load_model(self) -> None:
        """Load model and tokenizer"""
        pass
    
    @abstractmethod
    def extract(self, text: str) -> torch.Tensor:
        """Extract vector from single text"""
        pass
    
    def extract_batch(self, texts: List[str], batch_size: int = 8) -> torch.Tensor:
        """Extract vectors from multiple texts"""
        vectors = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            batch_vectors = self._extract_batch_internal(batch)
            vectors.append(batch_vectors)
        
        return torch.cat(vectors, dim=0)
    
    @abstractmethod
    def _extract_batch_internal(self, texts: List[str]) -> torch.Tensor:
        """Efficient batch extraction"""
        pass


# isomorphic/extractors/mean_pooling.py
class MeanPoolingExtractor(VectorExtractor):
    """Mean pooling of token embeddings"""
    
    def _load_model(self) -> None:
        from transformers import AutoTokenizer, AutoModel
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModel.from_pretrained(
            self.model_name,
            device_map=self.device,
            torch_dtype=torch.bfloat16
        )
        self.model.eval()
    
    def extract(self, text: str) -> torch.Tensor:
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
        inputs = inputs.to(self.device)
        
        with torch.no_grad():
            outputs = self.model(**inputs, output_hidden_states=True)
            embeddings = outputs.last_hidden_state  # [1, seq_len, dim]
            
            # Mean pooling with attention mask
            attention_mask = inputs['attention_mask']
            mask_expanded = attention_mask.unsqueeze(-1).expand(embeddings.size()).float()
            sum_embeddings = (embeddings * mask_expanded).sum(1)
            sum_mask = mask_expanded.sum(1)
            mean_embeddings = sum_embeddings / sum_mask.clamp(min=1e-9)
        
        return mean_embeddings.squeeze(0)  # [dim]
    
    def _extract_batch_internal(self, texts: List[str]) -> torch.Tensor:
        inputs = self.tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512
        )
        inputs = inputs.to(self.device)
        
        with torch.no_grad():
            outputs = self.model(**inputs, output_hidden_states=True)
            embeddings = outputs.last_hidden_state  # [batch, seq_len, dim]
            
            attention_mask = inputs['attention_mask']
            mask_expanded = attention_mask.unsqueeze(-1).expand(embeddings.size()).float()
            sum_embeddings = (embeddings * mask_expanded).sum(1)
            sum_mask = mask_expanded.sum(1)
            mean_embeddings = sum_embeddings / sum_mask.clamp(min=1e-9)
        
        return mean_embeddings  # [batch, dim]
```

---

## PHASE 4: PROCRUSTES ALIGNMENT (Week 4)

### 4.1 Alignment Solver

```python
# isomorphic/alignment_utils.py
import numpy as np
from scipy.linalg import svd
from typing import Tuple, Dict

class ProcrustesAligner:
    """SVD-based Procrustes solver"""
    
    @staticmethod
    def align(X: np.ndarray, Y: np.ndarray, center: bool = True) -> Tuple[np.ndarray, float]:
        """
        Find optimal orthogonal rotation Q such that ||Y - X·Q||_F is minimized
        
        Args:
            X: Source manifold [n_anchors, d]
            Y: Target manifold [n_anchors, d]
            center: Whether to center data
        
        Returns:
            Q: Orthogonal rotation matrix [d, d]
            error: Frobenius norm after alignment
        """
        assert X.shape == Y.shape, f"Shape mismatch: {X.shape} vs {Y.shape}"
        
        if center:
            X_mean = X.mean(axis=0)
            Y_mean = Y.mean(axis=0)
            X_centered = X - X_mean
            Y_centered = Y - Y_mean
        else:
            X_centered = X
            Y_centered = Y
        
        # SVD decomposition: Y^T · X = U · S · V^T
        U, S, Vt = svd(Y_centered.T @ X_centered, full_matrices=False)
        
        # Optimal rotation: Q = U · V^T
        Q = U @ Vt
        
        # Ensure proper rotation (det(Q) = 1, not reflection)
        if np.linalg.det(Q) < 0:
            # Flip sign of last column of U
            U[:, -1] *= -1
            Q = U @ Vt
        
        # Compute error after alignment
        aligned_X = X_centered @ Q
        error = np.linalg.norm(Y_centered - aligned_X, 'fro')
        
        return Q, error.item()
    
    @staticmethod
    def verify_orthogonality(Q: np.ndarray, tol: float = 1e-6) -> Dict[str, float]:
        """Verify Q is orthogonal"""
        should_be_identity = Q.T @ Q
        identity_error = np.linalg.norm(should_be_identity - np.eye(Q.shape[0]), 'fro')
        
        det_Q = np.linalg.det(Q)
        
        return {
            'orthogonality_error': identity_error,
            'determinant': det_Q,
            'is_proper_rotation': abs(det_Q - 1.0) < tol and identity_error < tol
        }
    
    @staticmethod
    def align_and_evaluate(X: np.ndarray, Y: np.ndarray) -> Dict:
        """Full alignment pipeline"""
        Q, error = ProcrustesAligner.align(X, Y)
        verification = ProcrustesAligner.verify_orthogonality(Q)
        
        return {
            'Q': Q,
            'frobenius_error': error,
            'verification': verification,
            'aligned_X': X @ Q
        }
```

---

## SUMMARY: CRITICAL IMPLEMENTATION POINTS

### Must-Have Components:

1. **Config System**: YAML-based, all hyperparameters externalized
2. **Multi-Dataset Support**: Abstract base class + 5 implementations
3. **Vector Extraction**: 4+ methods, batch processing, device management
4. **Procrustes Solver**: SVD-based, stability checks, verification
5. **Testing**: 100+ tests covering unit/integration/statistical levels
6. **Experiment Framework**: Reproducible experiment runs with result logging
7. **Documentation**: Docstrings, README, API docs, paper draft

### Design Principles:

- **Modularity**: Each component independently testable
- **Extensibility**: Easy to add new datasets, extractors, experiments
- **Reproducibility**: Fixed seeds, deterministic operations, result caching
- **Production Quality**: Type hints, logging, error handling, profiling

### Quality Metrics:

- Code coverage: >80%
- Documentation: >50% comments-to-code ratio
- Tests: 100+, all passing
- Performance: <1 min per 1000 samples (end-to-end)

---

**Continue with Phase 5 (Experimentation) and Phase 6 (Publication) after foundation is solid.**
