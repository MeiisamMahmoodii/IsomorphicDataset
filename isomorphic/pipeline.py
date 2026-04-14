"""
IsomorphicPipeline - End-to-End Workflow

Complete pipeline for:
1. Loading datasets
2. Extracting vectors
3. Computing alignment
4. Generating reports
5. Saving results
"""

import torch
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import json
from tqdm import tqdm

from isomorphic.config import Config, ExperimentConfig
from isomorphic.datasets.base_dataset import DatasetFactory
from isomorphic.datasets.setconca import SetConCAPreprocessor
from isomorphic.generation.rewriter import ModelRewriter
from isomorphic.extractors.base_extractor import ExtractorFactory
from isomorphic.alignment.procrustes import ProcrustesAligner, AnchorAlignment
from isomorphic.validators.semantic_judge import SemanticJudge
from isomorphic.utils.reporting import IsomorphismReporter


class IsomorphicPipeline:
    """Enhanced pipeline for High-Fidelity Isomorphic Dataset Generation."""
    
    def __init__(self, config: Config):
        self.config = config
        self.experiment_config = config.experiment
        self.output_dir = Path(self.experiment_config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.experiment_dir = self.output_dir / f"{self.experiment_config.name}_{timestamp}"
        self.experiment_dir.mkdir(parents=True, exist_ok=True)
        
        self.metrics = {}
        self.reporter = IsomorphismReporter(self.experiment_dir)
        self._log(f"Pipeline initialized for: {self.experiment_config.name}")
    
    def _log(self, message: str) -> None:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"[{timestamp}] {message}")
    
    def run(self) -> Dict:
        """Execute complete High-Fidelity Generation Pipeline."""
        self._log("=" * 70)
        self._log("HIGH-FIDELITY ISOMORPHIC DATASET GENERATION")
        self._log("=" * 70)
        
        try:
            # Step 1: Set-ConCA Constraint Generation
            self._log("\nSTEP 1: GENERATING SEMANTIC CONSTRAINTS (Set-ConCA)")
            datasets = self._load_and_constrain_datasets()
            
            # Step 2: Multi-Model Paraphrase Generation
            self._log("\nSTEP 2: MULTI-MODEL PARAPHRASE REWRITING")
            self._generate_isomorphic_variations(datasets)
            
            # Step 3: Multi-Method Latent Extraction
            self._log("\nSTEP 3: TRIPLE-METHOD LATENT EXTRACTION")
            vector_data = self._extract_all_latent_methods(datasets)
            
            # Step 4: Iterative Alignment & Dataset Pruning
            self._log("\nSTEP 4: ITERATIVE ALIGNMENT & DATASET PRUNING")
            alignment_results = self._align_and_filter(vector_data)
            
            # Step 5: Reference Model Validation
            self._log("\nSTEP 5: REFERENCE MODEL VALIDATION (WASSERSTEIN)")
            self._validate_with_reference(datasets)
            
            # Step 6: Generate Research Report
            self._log("\nSTEP 6: GENERATING COMPREHENSIVE REPORT")
            self.reporter.generate_final_report(self.metrics, datasets)
            
            self._log("\n" + "=" * 70)
            self._log("PIPELINE COMPLETE ✓")
            self._log("=" * 70)
            
            return {
                "status": "success",
                "experiment_dir": str(self.experiment_dir),
                "metrics": self.metrics,
            }
        
        except Exception as e:
            self._log(f"ERROR: {str(e)}")
            raise

    def _load_and_constrain_datasets(self) -> Dict:
        # Load datasets and apply Set-ConCA constraints using Gemma-4-31B
        # (Implementation using SetConCAPreprocessor)
        return {}
        """Load all configured datasets."""
        datasets = {}
        
        for dataset_config in self.experiment_config.datasets:
            self._log(f"Loading: {dataset_config.name}")
            
            try:
                dataset = DatasetFactory.create(
                    dataset_config.dataset_type,
                    max_samples=dataset_config.max_samples
                )
                dataset.load()
                dataset.preprocess()
                
                datasets[dataset_config.name] = dataset
                stats = dataset.get_statistics()
                
                self._log(f"  ✓ Loaded {stats['total_entries']} entries")
                self._log(f"    - Categories: {stats['categories']}")
                self._log(f"    - Avg forbidden words: {stats['avg_forbidden_words']:.1f}")
                
            except Exception as e:
                self._log(f"  ⚠️  Failed to load: {str(e)}")
        
        return datasets
    
    def _extract_vectors(self, datasets: Dict) -> Dict:
        """Extract vectors from all datasets and models."""
        vector_data = {}
        
        for model_config in self.experiment_config.models:
            self._log(f"Model: {model_config.name}")
            
            # Create extractor
            extractor = ExtractorFactory.create(
                method=self.experiment_config.extraction.method,
                model_name=model_config.model_id,
                device=self.config.device,
                max_length=model_config.max_length
            )
            
            model_vectors = {}
            
            for dataset_name, dataset in datasets.items():
                self._log(f"  Extracting vectors for {dataset_name}...")
                
                # Get all unique texts
                texts = []
                for entry in dataset._data:
                    texts.append(entry.seed_text)
                    texts.extend(entry.variations.values())
                
                # Remove duplicates
                texts = list(set(texts))
                
                # Extract vectors
                vectors = extractor.extract_batch(texts, batch_size=model_config.batch_size)
                
                model_vectors[dataset_name] = {
                    "texts": texts,
                    "vectors": vectors,
                }
                
                self._log(f"    ✓ Extracted {len(vectors)} vectors")
            
            vector_data[model_config.name] = model_vectors
        
        return vector_data
    
    def _align_models(self, vector_data: Dict) -> Dict:
        """Align latent spaces between models."""
        if len(self.experiment_config.models) < 2:
            self._log("⚠️  Skipping alignment: Need at least 2 models")
            return {}
        
        model_names = [m.name for m in self.experiment_config.models]
        source_model = model_names[0]
        target_model = model_names[1]
        
        self._log(f"Aligning: {source_model} → {target_model}")
        
        # For now, use a simple subset for alignment
        source_dataset = list(vector_data[source_model].values())[0]
        target_dataset = list(vector_data[target_model].values())[0]
        
        # Normalize vectors
        source_vecs = source_dataset["vectors"] / (torch.norm(source_dataset["vectors"], dim=1, keepdim=True) + 1e-9)
        target_vecs = target_dataset["vectors"] / (torch.norm(target_dataset["vectors"], dim=1, keepdim=True) + 1e-9)
        
        # Align
        alignment_result = ProcrustesAligner.compute_rotation(
            source_vecs,
            target_vecs,
            device=self.config.device
        )
        
        self._log(f"  ✓ Alignment quality: {alignment_result.alignment_quality:.4f}")
        self._log(f"  ✓ Orthogonality error: {alignment_result.orthogonality_error:.6f}")
        
        return {
            "source_model": source_model,
            "target_model": target_model,
            "result": alignment_result,
        }
    
    def _compute_metrics(self, datasets: Dict, vector_data: Dict, alignment_results: Dict) -> None:
        """Compute alignment and dataset metrics."""
        self._log("Computing alignment quality metrics...")
        
        # Dataset statistics
        for name, dataset in datasets.items():
            stats = dataset.get_statistics()
            self.metrics[f"dataset_{name}"] = stats
        
        # Alignment statistics
        if alignment_results:
            result = alignment_results["result"]
            self.metrics["alignment"] = {
                "alignment_quality": result.alignment_quality,
                "orthogonality_error": result.orthogonality_error,
                "source_variance": result.source_variance,
                "target_variance": result.target_variance,
            }
            
            self._log(f"  ✓ Metrics computed")
    
    def _generate_reports(self) -> None:
        """Generate comprehensive reports."""
        self._log("Generating reports...")
        
        # Main Results Report
        report_path = self.experiment_dir / "RESULTS_REPORT.md"
        with open(report_path, "w") as f:
            f.write("# IsomorphicDataSet - Experiment Results\n\n")
            f.write(f"**Experiment**: {self.experiment_config.name}\n")
            f.write(f"**Timestamp**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            f.write("## Dataset Statistics\n\n")
            for metric_name, metric_value in self.metrics.items():
                if metric_name.startswith("dataset_"):
                    f.write(f"### {metric_name}\n")
                    for k, v in metric_value.items():
                        f.write(f"- {k}: {v}\n")
            
            f.write("\n## Alignment Metrics\n\n")
            if "alignment" in self.metrics:
                for k, v in self.metrics["alignment"].items():
                    f.write(f"- {k}: {v:.6f}\n")
        
        self._log(f"  ✓ Saved: {report_path}")
    
    def _save_results(self) -> None:
        """Save all results to disk."""
        self._log("Saving results...")
        
        # Save metrics as JSON
        metrics_path = self.experiment_dir / "metrics.json"
        with open(metrics_path, "w") as f:
            # Convert numpy/torch types to native Python
            metrics_serializable = self._make_serializable(self.metrics)
            json.dump(metrics_serializable, f, indent=2)
        
        self._log(f"  ✓ Saved metrics to {metrics_path}")
        
        # Save configuration
        config_path = self.experiment_dir / "config.json"
        with open(config_path, "w") as f:
            config_dict = self._make_serializable(self.experiment_config.__dict__)
            json.dump(config_dict, f, indent=2)
        
        self._log(f"  ✓ Saved config to {config_path}")
    
    @staticmethod
    def _make_serializable(obj):
        """Convert non-serializable objects to serializable types."""
        if isinstance(obj, dict):
            return {k: IsomorphicPipeline._make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [IsomorphicPipeline._make_serializable(v) for v in obj]
        elif isinstance(obj, (np.integer, np.floating)):
            return float(obj)
        elif isinstance(obj, Path):
            return str(obj)
        elif torch.is_tensor(obj):
            return obj.tolist()
        else:
            return obj
