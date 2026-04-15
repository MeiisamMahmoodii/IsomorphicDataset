"""
IsomorphicPipeline - End-to-End Workflow

Constraint table (reference model) -> per-model rewrites + retries -> last-layer vectors
-> hub Procrustes (pooling ablation) -> reference Wasserstein gate -> artifacts.
"""

from __future__ import annotations

import gc
import json
import torch
import numpy as np
from pathlib import Path
from datetime import datetime
from collections import defaultdict
from typing import Any, Dict, List, Optional

from tqdm import tqdm

from isomorphic.config import Config, ExperimentConfig
from isomorphic.datasets.base_dataset import DatasetFactory
from isomorphic.datasets.reference_constraints import ReferenceConstraintBuilder
from isomorphic.generation.rewriter import ModelRewriter
from isomorphic.generation.constraint_utils import WRITER_MAX_ATTEMPTS
from isomorphic.extractors.base_extractor import ExtractorFactory
from isomorphic.alignment.evaluation import (
    POOLING_METHODS,
    hub_intersection_mask,
    pick_best_pooling_method,
)
from isomorphic.validators.semantic_judge import SemanticJudge
from isomorphic.utils import IsomorphismReporter, GPUManager


class IsomorphicPipeline:
    """High-fidelity isomorphic dataset generation pipeline."""

    def __init__(self, config: Config):
        self.config = config
        self.experiment_config = config.experiment
        self.output_dir = Path(self.experiment_config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.experiment_dir = self.output_dir / f"{self.experiment_config.name}_{timestamp}"
        self.experiment_dir.mkdir(parents=True, exist_ok=True)

        self.metrics: Dict[str, Any] = {
            "funnel": {},
            "alignment": {},
            "reference_gate": {},
            "rewrite_stats": {},
        }
        self.reporter = IsomorphismReporter(self.experiment_dir)

        self.gpu_manager = GPUManager()
        self._log(f"Pipeline initialized for: {self.experiment_config.name}")

    def _log(self, message: str) -> None:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"[{timestamp}] {message}")

    def run(self) -> Dict:
        """Execute full pipeline."""
        self._log("=" * 70)
        self._log("HIGH-FIDELITY ISOMORPHIC DATASET GENERATION")
        self._log("=" * 70)

        try:
            self._log("\nSTEP 1: REFERENCE CONSTRAINT GENERATION")
            datasets = self._load_and_constrain_datasets()
            self._persist_phase_a_table(datasets)
            self.metrics["funnel"]["after_phase_a_entries"] = sum(len(ds._data) for ds in datasets.values())

            self._log("\nSTEPS 2–3: PER-MODEL REWRITES + LATENT EXTRACTION (ONE MODEL AT A TIME)")
            vector_data = self._process_all_models(datasets)
            self.metrics["funnel"]["models_vectorized"] = list(vector_data.keys())

            self._log("\nSTEP 4: HUB ALIGNMENT AND POOLING COMPARISON")
            align_cfg = self.experiment_config.alignment
            alignment_results = self._align_and_filter(
                vector_data,
                datasets,
                threshold=align_cfg.alignment_quality_threshold,
            )

            self._log("\nSTEP 5: REFERENCE MODEL VALIDATION (WASSERSTEIN + COSINE GATE)")
            self._validate_with_reference(
                datasets,
                wasserstein_max=align_cfg.wasserstein_threshold,
                cosine_min=align_cfg.reference_cosine_min,
            )

            self._log("\nSTEP 6: REPORTS AND ARTIFACTS")
            self._save_config_and_metrics()
            self.reporter.generate_final_report(self.metrics, datasets)
            self._export_final_dataset_jsonl(datasets)

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

    def _cleanup_gpu(self) -> None:
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _load_and_constrain_datasets(self) -> Dict[str, Any]:
        """Load raw datasets and run reference constraint builder."""
        datasets: Dict[str, Any] = {}
        ref_model_id = (
            self.experiment_config.reference_model.model_id
            if getattr(self.experiment_config, "reference_model", None) is not None
            else ModelRewriter.MODELS[7]
        )
        self._log(f"Loading reference model for constraints: {ref_model_id}")

        # Reference model can be large; allow sharding if multiple GPUs are available.
        device_map = "auto" if torch.cuda.device_count() > 1 else self.gpu_manager.get_optimal_device(ref_model_id)
        rewriter = ModelRewriter(ref_model_id, device=str(device_map), device_map=device_map, load_in_4bit=True)
        builder = ReferenceConstraintBuilder(rewriter.model, rewriter.tokenizer)

        for dataset_config in self.experiment_config.datasets:
            self._log(f"Loading: {dataset_config.name}")
            ds = DatasetFactory.create(dataset_config.dataset_type, max_samples=dataset_config.max_samples)
            if not ds.load():
                self._log(f"  [SKIP] Dataset failed to load: {dataset_config.name}")
                continue
            ds.preprocess()
            self._log(f"  [DONE] Loaded {len(ds)} entries. Building constraints...")

            for entry in tqdm(ds._data, desc=f"Constraints ({dataset_config.name})"):
                builder.process_entry(entry)

            datasets[dataset_config.name] = ds

        del rewriter
        del builder
        self._cleanup_gpu()
        return datasets

    def _persist_phase_a_table(self, datasets: Dict[str, Any]) -> None:
        phase_dir = self.experiment_dir / "phase_a"
        phase_dir.mkdir(parents=True, exist_ok=True)
        path = phase_dir / "constraint_table.jsonl"
        schema_version = 1
        with open(path, "w", encoding="utf-8") as f:
            for ds_name, ds in datasets.items():
                for entry in ds._data:
                    row = {
                        "schema_version": schema_version,
                        "dataset": ds_name,
                        "seed_id": entry.seed_id,
                        "seed_text": entry.seed_text,
                        "forbidden_words": entry.forbidden_words,
                        "length_bins": [
                            f"{c.min_words}-{c.max_words}" for c in entry.length_constraints
                        ],
                        "rewrite_specs": entry.rewrite_specs,
                    }
                    f.write(json.dumps(row, ensure_ascii=False) + "\n")
        self._log(f"  [DONE] Phase A table: {path}")

    def _process_all_models(self, datasets: Dict[str, Any]) -> Dict[str, Any]:
        """Rewrite with each model, then extract last-layer vectors for each variation row."""
        all_vector_data: Dict[str, Any] = {}
        n_models = len(self.experiment_config.models)
        rewrite_stats: Dict[str, Any] = {}

        for i, model_config in enumerate(self.experiment_config.models, 1):
            self._log("-" * 60)
            self._log(f"[{i}/{n_models}] Loading model: {model_config.model_id}")

            # If config requests sharding, respect it; otherwise place on best single GPU.
            if getattr(model_config, "device_map", None) in (None, "", "none"):
                requested_map = None
            else:
                requested_map = model_config.device_map
            if isinstance(requested_map, str) and requested_map.strip().lower() == "auto":
                device_map = "auto"
                device = "auto"
            else:
                device = self.gpu_manager.get_optimal_device(model_config.model_id)
                device_map = requested_map or device
            load_in_4bit = any(
                tag in model_config.model_id for tag in ("31B", "35B", "26B", "20b")
            )

            try:
                rewriter = ModelRewriter(
                    model_config.model_id,
                    device=device,
                    device_map=device_map,
                    load_in_4bit=load_in_4bit,
                )
            except Exception as e:
                self._log(f"  [ERROR] Could not load {model_config.name}: {e}")
                self._cleanup_gpu()
                continue

            self._log(f"  Rewriting with {model_config.name}")
            attempts_total = 0
            successes = 0
            for ds_name, ds in datasets.items():
                for entry in tqdm(ds._data, desc=f"Rewrite [{model_config.name}|{ds_name}]"):
                    try:
                        rewriter.process_entry(entry)
                        prefix = f"{rewriter.model_id}::"
                        for rk, log in entry.rewrite_logs.items():
                            if not rk.startswith(prefix):
                                continue
                            attempts_total += len(log.get("attempts", []))
                            if log.get("success"):
                                successes += 1
                    except Exception as e:
                        self._log(f"    [WARN] Entry failed: {e}")

            rewrite_stats[model_config.name] = {
                "attempts_total": attempts_total,
                "constraint_passes_recorded": successes,
                "max_attempts_per_slot": WRITER_MAX_ATTEMPTS,
            }

            self._log(f"  Extracting vectors with {model_config.name}")
            model_vecs: Dict[str, Any] = {}
            try:
                extractor = ExtractorFactory.create_shared_encoder(
                    model_name=model_config.model_id,
                    device=device,
                    load_in_4bit=load_in_4bit,
                    model=rewriter.model,
                    tokenizer=rewriter.tokenizer,
                    max_length=model_config.max_length,
                )

                for ds_name, ds in datasets.items():
                    method_vecs: Dict[str, List] = {m: [] for m in POOLING_METHODS}
                    for entry in tqdm(ds._data, desc=f"Extract [{model_config.name}|{ds_name}]"):
                        for constraint in entry.length_constraints:
                            desc = f"{constraint.min_words}-{constraint.max_words}"
                            key = f"{model_config.model_id}_{desc}"
                            text = entry.variations.get(key, "")
                            if not isinstance(text, str):
                                text = ""
                            tex = text if (text and str(text).strip()) else entry.seed_text
                            try:
                                vecs = extractor.extract_all_methods(tex)
                                for k in POOLING_METHODS:
                                    if k in vecs:
                                        method_vecs[k].append(vecs[k])
                            except Exception as e:
                                self._log(f"    [WARN] Extraction: {e}")
                                vecs = extractor.extract_all_methods(entry.seed_text)
                                for k in POOLING_METHODS:
                                    if k in vecs:
                                        method_vecs[k].append(vecs[k])

                    for k in list(method_vecs.keys()):
                        if method_vecs[k]:
                            method_vecs[k] = torch.stack(method_vecs[k])
                        else:
                            method_vecs.pop(k, None)

                    model_vecs[ds_name] = method_vecs

                all_vector_data[model_config.name] = model_vecs

            except Exception as e:
                self._log(f"  [ERROR] Extraction failed for {model_config.name}: {e}")

            del extractor
            del rewriter
            self._cleanup_gpu()
            self._log(f"  [DONE] {model_config.name} fully processed and unloaded.")

        self.metrics["rewrite_stats"] = rewrite_stats
        self._log("-" * 60)
        self._log(f"All fleet models processed ({len(all_vector_data)} loaded).")
        return all_vector_data

    def _num_bins(self, datasets: Dict[str, Any]) -> int:
        for ds in datasets.values():
            if ds._data:
                return len(ds._data[0].length_constraints)
        return 1

    def _align_and_filter(
        self,
        vector_data: Dict[str, Any],
        datasets: Dict[str, Any],
        threshold: float,
    ) -> Dict[str, Any]:
        """Hub = first configured model; intersect masks per pooling method; apply best."""
        results: Dict[str, Any] = {}
        model_cfgs = self.experiment_config.models
        if len(model_cfgs) < 2:
            self._log("  [SKIP] Alignment needs at least 2 models.")
            return results

        hub_name = model_cfgs[0].name
        model_names = [m.name for m in model_cfgs]
        if hub_name not in vector_data:
            self._log(f"  [SKIP] Hub model {hub_name} not in vector data.")
            return results

        method_scores_avg: Dict[str, List[float]] = defaultdict(list)
        best_masks: Dict[str, torch.Tensor] = {}

        for ds_name in datasets:
            per_ds: Dict[str, Any] = {}
            for method in POOLING_METHODS:
                mask, quality = hub_intersection_mask(
                    vector_data,
                    hub_name,
                    model_names,
                    ds_name,
                    method,
                    threshold=threshold,
                    device="cpu",
                )
                if mask is not None:
                    method_scores_avg[method].append(float(quality or 0.0))
                    best_masks[f"{ds_name}::{method}"] = mask
                    per_ds[method] = {
                        "keepers": int(mask.sum().item()),
                        "total": int(mask.numel()),
                        "alignment_quality": quality,
                    }
            results[ds_name] = per_ds

        averaged = {
            m: sum(vals) / len(vals) for m, vals in method_scores_avg.items() if vals
        }
        best_method_global = pick_best_pooling_method(averaged)
        if best_method_global is None and POOLING_METHODS:
            best_method_global = POOLING_METHODS[0]

        self.metrics["alignment"]["hub"] = hub_name
        self.metrics["alignment"]["per_dataset_method"] = results
        self.metrics["alignment"]["method_scores_avg"] = averaged
        self.metrics["alignment"]["best_pooling_method"] = best_method_global

        # Apply mask from best method to entries (row-major: entry_i * num_bins + bin_j)
        num_bins = self._num_bins(datasets)
        for ds_name, ds in datasets.items():
            key_mask = f"{ds_name}::{best_method_global}"
            mask_t = best_masks.get(key_mask)
            if mask_t is None:
                continue
            mask = mask_t.cpu()
            for ei, entry in enumerate(ds._data):
                for bi in range(num_bins):
                    ri = ei * num_bins + bi
                    if ri >= mask.shape[0]:
                        continue
                    if not bool(mask[ri].item()):
                        entry.accepted = False
                        entry.drop_reason = (entry.drop_reason or "") + ";alignment"

        self._log(f"  [DONE] Alignment best pooling (heuristic): {best_method_global}")
        return results

    def _validate_with_reference(
        self,
        datasets: Dict[str, Any],
        wasserstein_max: float,
        cosine_min: float,
    ) -> None:
        """Gemma-31B reference: Wasserstein + cosine gate; drops failing entries."""
        ref_model_id = (
            self.experiment_config.reference_model.model_id
            if getattr(self.experiment_config, "reference_model", None) is not None
            else ModelRewriter.MODELS[7]
        )
        self._log(f"Reference gate: {ref_model_id}")

        device = self.gpu_manager.get_optimal_device(ref_model_id)
        dropped = 0
        judged = 0

        try:
            rewriter = ModelRewriter(ref_model_id, device=device, load_in_4bit=True)
            judge = SemanticJudge(rewriter.model, rewriter.tokenizer)

            for ds_name, ds in datasets.items():
                for entry in tqdm(ds._data, desc=f"Reference gate ({ds_name})"):
                    if not entry.accepted:
                        continue
                    gate_detail: Dict[str, Any] = {}
                    ok_entry = True
                    for var_key, var_text in list(entry.variations.items()):
                        if not isinstance(var_text, str) or not var_text.strip():
                            continue
                        judged += 1
                        m = judge.evaluate_isomorphism(entry.seed_text, var_text)
                        w = m.get("wasserstein_distance", float("inf"))
                        cos = m.get("reference_cosine_similarity", m.get("cosine_similarity", 0.0))
                        gate_detail[var_key] = {"wasserstein_distance": w, "cosine": cos}
                        if w > wasserstein_max or cos < cosine_min:
                            ok_entry = False
                    entry.metadata["reference_gate"] = gate_detail
                    if not ok_entry:
                        entry.accepted = False
                        entry.drop_reason = (entry.drop_reason or "") + ";wasserstein_gate"
                        dropped += 1

            del rewriter
            del judge
            self._cleanup_gpu()
        except Exception as e:
            self._log(f"  [ERROR] Reference gate failed: {e}")
            self._cleanup_gpu()

        self.metrics["reference_gate"] = {
            "pairs_judged": judged,
            "entries_failed_gate": dropped,
            "wasserstein_max": wasserstein_max,
            "cosine_min": cosine_min,
        }
        self.metrics["funnel"]["final_accepted"] = sum(
            1 for ds in datasets.values() for e in ds._data if e.accepted
        )

    def _save_config_and_metrics(self) -> None:
        metrics_path = self.experiment_dir / "metrics.json"
        with open(metrics_path, "w", encoding="utf-8") as f:
            json.dump(self._make_serializable(self.metrics), f, indent=2)
        self._log(f"  [DONE] metrics.json")

        cfg_path = self.experiment_dir / "config.json"
        with open(cfg_path, "w", encoding="utf-8") as f:
            json.dump(self._make_serializable(self.experiment_config.__dict__), f, indent=2)
        self._log(f"  [DONE] config.json")

    def _export_final_dataset_jsonl(self, datasets: Dict[str, Any]) -> None:
        path = self.experiment_dir / "final_dataset.jsonl"
        n = 0
        with open(path, "w", encoding="utf-8") as f:
            for ds_name, ds in datasets.items():
                for entry in ds._data:
                    if not entry.accepted:
                        continue
                    row = {
                        "dataset": ds_name,
                        "seed_id": entry.seed_id,
                        "seed_text": entry.seed_text,
                        "forbidden_words": entry.forbidden_words,
                        "variations": {k: v for k, v in entry.variations.items() if isinstance(v, str)},
                        "metadata": entry.metadata,
                    }
                    f.write(json.dumps(row, ensure_ascii=False) + "\n")
                    n += 1
        self._log(f"  [DONE] final_dataset.jsonl ({n} rows)")

    @staticmethod
    def _make_serializable(obj: Any) -> Any:
        try:
            from dataclasses import is_dataclass, asdict
        except Exception:  # pragma: no cover
            is_dataclass = None
            asdict = None

        if is_dataclass is not None and is_dataclass(obj):
            return IsomorphicPipeline._make_serializable(asdict(obj))
        if isinstance(obj, dict):
            return {k: IsomorphicPipeline._make_serializable(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [IsomorphicPipeline._make_serializable(v) for v in obj]
        if isinstance(obj, (np.integer, np.floating)):
            return float(obj)
        if isinstance(obj, Path):
            return str(obj)
        if torch.is_tensor(obj):
            return obj.tolist()
        return obj
