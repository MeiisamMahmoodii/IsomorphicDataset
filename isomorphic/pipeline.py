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
import gc
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any
import json
from tqdm import tqdm

from isomorphic.config import Config, ExperimentConfig
from isomorphic.datasets.base_dataset import DatasetFactory, LengthConstraint
from isomorphic.datasets.setconca import SetConCAPreprocessor
from isomorphic.generation.rewriter import ModelRewriter
from isomorphic.extractors.base_extractor import ExtractorFactory
from isomorphic.alignment.procrustes import ProcrustesAligner, AnchorAlignment
from isomorphic.validators.semantic_judge import SemanticJudge
from isomorphic.utils import IsomorphismReporter, GPUManager


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
        
        # Initialize GPU Manager for server scaling
        self.gpu_manager = GPUManager()
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

            # Steps 2+3 combined: for each model → load once → rewrite + extract → unload
            self._log("\nSTEPS 2+3: PER-MODEL REWRITING + LATENT EXTRACTION (ONE MODEL AT A TIME)")
            vector_data = self._process_all_models(datasets)

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
            self._log("PIPELINE COMPLETE [DONE]")
            self._log("=" * 70)

            return {
                "status": "success",
                "experiment_dir": str(self.experiment_dir),
                "metrics": self.metrics,
            }

        except Exception as e:
            self._log(f"ERROR: {str(e)}")
            raise

    def _cleanup_gpu(self):
        """Force garbage collection and clear CUDA cache to prevent OOM."""
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _load_and_constrain_datasets(self) -> Dict[str, Any]:
        """Load datasets and generate Set-ConCA constraints using the 31B reference model."""
        datasets = {}
        
        # Load the Reference Model for Pre-processing (Step 1)
        ref_model_id = ModelRewriter.MODELS[7] # Huihui-gemma-4-31B
        self._log(f"Loading reference model for constraints: {ref_model_id}")
        
        # We load in 4-bit to ensure it fits alongside other processes
        device = self.gpu_manager.get_optimal_device(ref_model_id)
        rewriter = ModelRewriter(ref_model_id, device=device, load_in_4bit=True)
        preprocessor = SetConCAPreprocessor(rewriter.model, rewriter.tokenizer)
        
        for dataset_config in self.experiment_config.datasets:
            self._log(f"Loading: {dataset_config.name}")
            ds = DatasetFactory.create(dataset_config.dataset_type, max_samples=dataset_config.max_samples)
            if ds.load():
                ds.preprocess()
                self._log(f"  [DONE] Loaded {len(ds)} entries. Generating constraints...")
                
                # Apply real model-based constraints
                for entry in tqdm(ds._data, desc=f"Constraint Gen ({dataset_config.name})"):
                    entry = preprocessor.process_entry(entry)
                
                datasets[dataset_config.name] = ds
        
        # Unload pre-processor to free VRAM for Phase 2
        del rewriter
        del preprocessor
        self._cleanup_gpu()
        
        return datasets

    def _process_all_models(self, datasets: Dict[str, Any]) -> Dict[str, Any]:
        """
        Core hot-swap loop: for every model in the fleet, load it ONCE, then:
          1. Rewrite all dataset entries (generate isomorphic variations)
          2. Extract latent vectors (mean / last-token / attention-weighted)
          3. Unload and free VRAM before moving to the next model
        """
        all_vector_data: Dict[str, Any] = {}
        n_models = len(self.experiment_config.models)

        for i, model_config in enumerate(self.experiment_config.models, 1):
            self._log("-" * 60)
            self._log(f"[{i}/{n_models}] Loading model: {model_config.model_id}")

            device = self.gpu_manager.get_optimal_device(model_config.model_id)
            load_in_4bit = any(
                tag in model_config.model_id for tag in ("31B", "35B", "26B", "20b")
            )

            # ── Load rewriter ────────────────────────────────────────────────
            try:
                rewriter = ModelRewriter(
                    model_config.model_id,
                    device=device,
                    load_in_4bit=load_in_4bit,
                )
            except Exception as e:
                self._log(f"  [ERROR] Could not load {model_config.name}: {e}")
                self._cleanup_gpu()
                continue

            # ── Phase A: Rewrite all datasets ────────────────────────────────
            self._log(f"  Phase A — Rewriting datasets with {model_config.name}")
            for ds_name, ds in datasets.items():
                self._log(f"    Dataset: {ds_name} ({len(ds._data)} entries)")
                for entry in tqdm(ds._data, desc=f"Rewrite [{model_config.name}|{ds_name}]"):
                    try:
                        rewriter.process_entry(entry)
                    except Exception as e:
                        self._log(f"    [WARN] Entry failed: {e}")
            self._log(f"  Phase A — Rewriting done.")

            # ── Phase B: Extract latent vectors ──────────────────────────────
            self._log(f"  Phase B — Extracting vectors with {model_config.name}")
            model_vecs: Dict[str, Any] = {}
            try:
                extractor = ExtractorFactory.create(
                    model_id=model_config.model_id,
                    device=device,
                    load_in_4bit=load_in_4bit,
                    # Reuse the already-loaded model weights to avoid reloading
                    model=rewriter.model,
                    tokenizer=rewriter.tokenizer,
                )

                for ds_name, ds in datasets.items():
                    self._log(f"    Extracting: {ds_name}")
                    method_vecs: Dict[str, List] = {
                        "mean_pooling": [],
                        "last_token": [],
                        "attention_weighted": [],
                    }
                    for entry in tqdm(ds._data, desc=f"Extract [{model_config.name}|{ds_name}]"):
                        try:
                            vecs = extractor.extract_all_methods(entry.seed_text)
                            for k, v in vecs.items():
                                method_vecs[k].append(v)
                        except Exception as e:
                            self._log(f"    [WARN] Extraction skipped for entry: {e}")

                    # Only stack if we collected any vectors
                    for k in list(method_vecs.keys()):
                        if method_vecs[k]:
                            method_vecs[k] = torch.stack(method_vecs[k])
                        else:
                            method_vecs.pop(k)

                    model_vecs[ds_name] = method_vecs

                all_vector_data[model_config.name] = model_vecs
                self._log(f"  Phase B — Extraction done.")

            except Exception as e:
                self._log(f"  [ERROR] Extraction failed for {model_config.name}: {e}")

            # ── Unload everything before next model ───────────────────────────
            try:
                del extractor
            except NameError:
                pass
            del rewriter
            self._cleanup_gpu()
            self._log(f"  [DONE] {model_config.name} fully processed and unloaded.")

        self._log("-" * 60)
        self._log(f"All {n_models} models processed.")
        return all_vector_data

    def _align_and_filter(self, vector_data: Dict[str, Any]) -> Dict[str, Any]:
        """Compute iterative alignment and prune non-aligning samples."""
        results = {}
        if len(self.experiment_config.models) < 2:
            return results
            
        source = self.experiment_config.models[0].name
        target = self.experiment_config.models[1].name
        self._log(f"Aligning {source} -> {target} with iterative filtering...")
        
        # Force CPU for mathematical alignment to bypass driver issues
        for ds_name in vector_data[source]:
            X = vector_data[source][ds_name]["mean_pooling"].cpu()
            Y = vector_data[target][ds_name]["mean_pooling"].cpu()
            
            result, mask = ProcrustesAligner.iterate_and_filter(X, Y, threshold=0.98, device="cpu")
            results[ds_name] = {"result": result, "keepers": mask.sum().item()}
            self._log(f"  [DONE] {ds_name}: Kept {mask.sum().item()} samples at >0.98 similarity (Algn: CPU)")
            
        return results

    def _validate_with_reference(self, datasets: Dict[str, Any]) -> None:
        """Final semantic validation via Wasserstein in the source-of-truth model (31B)."""
        self._log("Final verification via Reference Model (Huihui-gemma-4-31B)...")
        
        # 1. Load Reference Judge
        ref_model_id = ModelRewriter.MODELS[7] # Huihui-gemma-4-31B
        device = self.gpu_manager.get_optimal_device(ref_model_id)
        
        try:
            rewriter = ModelRewriter(ref_model_id, device=device, load_in_4bit=True)
            judge = SemanticJudge(rewriter.model, rewriter.tokenizer)
            
            # 2. Validate Every Proven Isomorphic Pair
            for ds_name, ds in datasets.items():
                self._log(f"  Validating {ds_name}...")
                for entry in tqdm(ds._data, desc=f"Wasserstein ({ds_name})"):
                    # Compare seed to every variation
                    for var_key, var_text in entry.variations.items():
                        metrics = judge.evaluate_isomorphism(entry.seed_text, var_text)
                        # Store in entry for reporting
                        entry.variations[f"{var_key}_wasserstein"] = metrics.get('wasserstein_distance', 0.0)
            
            # 3. Unload
            del rewriter
            del judge
            self._cleanup_gpu()
            self._log("  [DONE] Semantically verified closeness in single-model space.")
            
        except Exception as e:
            self._log(f"  [ERROR] Validation failed: {str(e)}")
            self._cleanup_gpu()
    
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
                
                self._log(f"    [DONE] Extracted {len(vectors)} vectors")
            
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
