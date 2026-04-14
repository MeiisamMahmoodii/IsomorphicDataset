"""
Configuration Management System for IsomorphicDataSet

Handles all configurable parameters, model settings, dataset configurations,
and experiment tracking metadata.
"""

from dataclasses import dataclass, field, asdict
from typing import Dict, List, Any, Optional
import json
import yaml
from pathlib import Path


@dataclass
class ModelConfig:
    """Configuration for LLM models."""
    name: str
    model_id: str
    device_map: str = "auto"
    torch_dtype: str = "float16"
    max_length: int = 128
    batch_size: int = 32


@dataclass
class DatasetConfig:
    """Configuration for datasets."""
    name: str
    dataset_type: str
    source: str
    subset: Optional[str] = None
    max_samples: Optional[int] = None
    use_banned_word_extraction: bool = True


@dataclass
class AlignmentConfig:
    """Configuration for Procrustes alignment."""
    method: str = "procrustes_svd"
    num_anchors: int = 77
    anchor_type: str = "words"
    normalize_embeddings: bool = True
    wasserstein_threshold: float = 0.5


@dataclass
class ExtractionConfig:
    """Configuration for vector extraction methods."""
    method: str = "mean_pooling"
    normalize: bool = True
    batch_size: int = 32


@dataclass
class ExperimentConfig:
    """Configuration for experiment execution."""
    name: str
    description: str = ""
    models: List[ModelConfig] = field(default_factory=list)
    datasets: List[DatasetConfig] = field(default_factory=list)
    extraction: ExtractionConfig = field(default_factory=ExtractionConfig)
    alignment: AlignmentConfig = field(default_factory=AlignmentConfig)
    output_dir: Path = field(default_factory=lambda: Path("experiments/results"))
    save_embeddings: bool = True
    save_rotation_matrix: bool = True
    compute_statistics: bool = True


@dataclass
class Config:
    """Main configuration container."""
    experiment: ExperimentConfig
    logging_level: str = "INFO"
    seed: int = 42
    device: str = "cuda"
    mixed_precision: bool = True


class ConfigManager:
    """Manages loading, saving, and updating configuration."""
    
    def __init__(self, config_dir: Path = Path("config")):
        self.config_dir = Path(config_dir)
        self.config_dir.mkdir(parents=True, exist_ok=True)
    
    def load_config(self, config_path: Path) -> Config:
        """Load configuration from YAML or JSON file."""
        config_path = Path(config_path)
        
        if config_path.suffix == ".yaml" or config_path.suffix == ".yml":
            with open(config_path) as f:
                data = yaml.safe_load(f)
        elif config_path.suffix == ".json":
            with open(config_path) as f:
                data = json.load(f)
        else:
            raise ValueError(f"Unsupported config format: {config_path.suffix}")
        
        return self._dict_to_config(data)
    
    def save_config(self, config: Config, output_path: Path, format: str = "yaml"):
        """Save configuration to file."""
        data = self._config_to_dict(config)
        output_path = Path(output_path)
        
        if format == "yaml":
            with open(output_path, "w") as f:
                yaml.dump(data, f, default_flow_style=False)
        elif format == "json":
            with open(output_path, "w") as f:
                json.dump(data, f, indent=2)
        else:
            raise ValueError(f"Unsupported format: {format}")
    
    def _dict_to_config(self, data: Dict[str, Any]) -> Config:
        """Convert dictionary to Config object."""
        exp_data = data.get("experiment", {})
        
        models = [ModelConfig(**m) for m in exp_data.get("models", [])]
        datasets = [DatasetConfig(**d) for d in exp_data.get("datasets", [])]
        extraction = ExtractionConfig(**exp_data.get("extraction", {}))
        alignment = AlignmentConfig(**exp_data.get("alignment", {}))
        
        experiment = ExperimentConfig(
            name=exp_data.get("name", "default"),
            description=exp_data.get("description", ""),
            models=models,
            datasets=datasets,
            extraction=extraction,
            alignment=alignment,
        )
        
        return Config(
            experiment=experiment,
            logging_level=data.get("logging_level", "INFO"),
            seed=data.get("seed", 42),
            device=data.get("device", "cuda"),
            mixed_precision=data.get("mixed_precision", True),
        )
    
    def _config_to_dict(self, config: Config) -> Dict[str, Any]:
        """Convert Config object to dictionary."""
        return asdict(config)
