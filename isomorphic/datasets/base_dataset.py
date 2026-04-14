"""
Dataset Framework - Abstract base classes and utilities

Provides unified interface for loading, preprocessing, and managing diverse datasets.
All datasets follow the standard format of (seed, forbidden_words, semantic_intent, variations).
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple
import pandas as pd
from pathlib import Path


@dataclass
class LengthConstraint:
    """Length constraint for sentence generation."""
    min_words: int
    max_words: int


@dataclass
class DatasetEntry:
    """Standard dataset entry format - enhanced for Set-ConCA."""
    seed_id: str
    seed_text: str
    semantic_intent: str
    original_category: str
    forbidden_words: List[str]  # Banned words that MUST NOT be used in variations
    length_constraints: List[LengthConstraint]  # e.g., [5-10, 15-20]
    variations: Dict[str, str]  # Keyed by model_id + constraint_desc
    metadata: Dict[str, any]


class BaseDataset(ABC):
    """Abstract base class for all datasets."""
    
    def __init__(self, name: str, source: str, max_samples: Optional[int] = None):
        self.name = name
        self.source = source
        self.max_samples = max_samples
        self._data: List[DatasetEntry] = []
        self._loaded = False
    
    @abstractmethod
    def load(self) -> None:
        """Load dataset from source."""
        pass
    
    @abstractmethod
    def preprocess(self) -> None:
        """Preprocess and standardize dataset."""
        pass
    
    def __len__(self) -> int:
        return len(self._data)
    
    def __getitem__(self, idx: int) -> DatasetEntry:
        if not self._loaded:
            self.load()
            self.preprocess()
            self._loaded = True
        return self._data[idx]
    
    def to_dataframe(self) -> pd.DataFrame:
        """Convert to pandas DataFrame for analysis."""
        records = []
        for entry in self._data:
            records.append({
                "seed_id": entry.seed_id,
                "seed_text": entry.seed_text,
                "semantic_intent": entry.semantic_intent,
                "category": entry.original_category,
                "forbidden_words": "|".join(entry.forbidden_words),
                "num_variations": len(entry.variations),
                **{f"var_{k}": v for k, v in entry.variations.items()}
            })
        return pd.DataFrame(records)
    
    def save_to_csv(self, output_path: Path) -> None:
        """Save dataset to CSV."""
        self.to_dataframe().to_csv(output_path, index=False)
        print(f"[DONE] Saved {len(self)} entries to {output_path}")
    
    def get_statistics(self) -> Dict[str, any]:
        """Compute dataset statistics."""
        if not self._data:
            return {}
        
        seed_lengths = [len(e.seed_text.split()) for e in self._data]
        forbidden_counts = [len(e.forbidden_words) for e in self._data]
        var_counts = [len(e.variations) for e in self._data]
        
        return {
            "total_entries": len(self._data),
            "categories": len(set(e.original_category for e in self._data)),
            "avg_seed_length": sum(seed_lengths) / len(seed_lengths),
            "avg_forbidden_words": sum(forbidden_counts) / len(forbidden_counts),
            "avg_variations": sum(var_counts) / len(var_counts),
            "unique_intents": len(set(e.semantic_intent for e in self._data)),
        }


class ToxiGenDataset(BaseDataset):
    """ToxiGen Dataset - Toxic Comment Variations."""
    
    def __init__(self, max_samples: Optional[int] = None):
        super().__init__(name="toxigen", source="huggingface", max_samples=max_samples)
        self.ANCHOR_WORDS = [
            "mountain", "keyboard", "chair", "water", "tree", "rock", "door", "window",
            "light", "sound", "wind", "rain", "snow", "fire", "earth", "sky",
            "science", "philosophy", "mathematics", "nature", "universe", "concept", "idea", "theory",
            "book", "table", "pen", "paper", "bread", "fruit", "flower", "animal",
            "number", "energy", "force", "motion", "atom", "cell", "element", "compound",
            "time", "space", "distance", "moment", "direction", "location", "dimension", "point",
            "color", "shape", "size", "weight", "texture", "bright", "dark", "smooth",
            "language", "word", "sentence", "meaning", "symbol", "letter", "sound", "voice",
            "one", "two", "three", "zero", "infinite"
        ]
    
    def load(self) -> None:
        """Load ToxiGen from HuggingFace."""
        try:
            from datasets import load_dataset
            dataset = load_dataset("toxigen/toxigen-data", split="train")
            
            # Limit to max_samples if specified
            if self.max_samples:
                dataset = dataset.select(range(min(self.max_samples, len(dataset))))
            
            for idx, item in enumerate(dataset):
                entry = DatasetEntry(
                    seed_id=f"toxigen_{idx}",
                    seed_text=item.get("prompt", ""),
                    semantic_intent="toxic_comment_generation",
                    original_category=item.get("attribute", "unknown"),
                    forbidden_words=[],
                    length_constraints=[],
                    variations={},
                    metadata=item
                )
                self._data.append(entry)
            
            print(f"[DONE] Loaded {len(self._data)} samples from ToxiGen")
        except ImportError:
            raise ImportError("Please install: pip install datasets")
    
    def preprocess(self) -> None:
        """Preprocess ToxiGen data."""
        print(f"[DONE] Preprocessing {len(self._data)} ToxiGen entries...")
        # Standard preprocessing: clean text, validate, etc.
        pass


class JigsawDataset(BaseDataset):
    """Jigsaw Unintended Bias Dataset."""
    
    def __init__(self, max_samples: Optional[int] = None):
        super().__init__(name="jigsaw", source="kaggle", max_samples=max_samples)
    
    def load(self) -> None:
        """Load Jigsaw Unintended Bias dataset."""
        print("⚠️  Jigsaw dataset requires manual download from Kaggle")
        print("Install: pip install kaggle")
        print("Setup: kaggle datasets download -d jigsaw-unintended-bias-in-toxicity-classification")
        pass
    
    def preprocess(self) -> None:
        """Preprocess Jigsaw data."""
        pass


class HateXplainDataset(BaseDataset):
    """HateXplain Dataset - Hate Speech Detection with Explanations."""
    
    def __init__(self, max_samples: Optional[int] = None):
        super().__init__(name="hatexplain", source="huggingface", max_samples=max_samples)
    
    def load(self) -> None:
        """Load HateXplain from HuggingFace."""
        try:
            from datasets import load_dataset
            dataset = load_dataset("hatexplain", split="train")
            
            if self.max_samples:
                dataset = dataset.select(range(min(self.max_samples, len(dataset))))
            
            for idx, item in enumerate(dataset):
                entry = DatasetEntry(
                    seed_id=f"hatexplain_{idx}",
                    seed_text=item.get("post_tokens", []),
                    semantic_intent="hate_speech_detection",
                    original_category=["normal", "offensive", "hate"][item.get("label", 0)],
                    forbidden_words=[],
                    length_constraints=[],
                    variations={},
                    metadata=item
                )
                self._data.append(entry)
            
            print(f"[DONE] Loaded {len(self._data)} samples from HateXplain")
        except ImportError:
            raise ImportError("Please install: pip install datasets")
    
    def preprocess(self) -> None:
        """Preprocess HateXplain data."""
        pass


class SBICDataset(BaseDataset):
    """Social Bias Inference Corpus Dataset."""
    
    def __init__(self, max_samples: Optional[int] = None):
        super().__init__(name="sbic", source="huggingface", max_samples=max_samples)
    
    def load(self) -> None:
        """Load SBIC from HuggingFace or local cache."""
        print("ℹ️  SBIC dataset loader - contact authors for access")
        pass
    
    def preprocess(self) -> None:
        """Preprocess SBIC data."""
        pass


class DatasetFactory:
    """Factory for creating and managing datasets."""
    
    AVAILABLE_DATASETS = {
        "toxigen": ToxiGenDataset,
        "jigsaw": JigsawDataset,
        "hatexplain": HateXplainDataset,
        "sbic": SBICDataset,
    }
    
    @classmethod
    def create(cls, dataset_name: str, **kwargs) -> BaseDataset:
        """Create dataset instance."""
        if dataset_name not in cls.AVAILABLE_DATASETS:
            raise ValueError(f"Unknown dataset: {dataset_name}")
        
        return cls.AVAILABLE_DATASETS[dataset_name](**kwargs)
    
    @classmethod
    def list_datasets(cls) -> List[str]:
        """List available datasets."""
        return list(cls.AVAILABLE_DATASETS.keys())
