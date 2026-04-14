"""
Server Run Orchestrator - 4x A100 Optimization

Main entry point for large-scale dataset generation across 14-model fleet.
Utilizes GPU placement strategy and mixed precision for maximum throughput.
"""

from pathlib import Path
from isomorphic.config import Config, ExperimentConfig, ModelConfig, DatasetConfig
from isomorphic.pipeline import IsomorphicPipeline
from isomorphic.generation.rewriter import ModelRewriter

def create_server_config() -> Config:
    """Configures the full 14-model fleet for server execution."""
    
    models = []
    for model_id in ModelRewriter.MODELS:
        name = model_id.split("/")[-1]
        # Larger models (31B, 35B) will be handled with 4-bit quantization
        # This occurs internally in the Rewriter/Extractor based on model_id
        models.append(ModelConfig(
            name=name,
            model_id=model_id,
            batch_size=32, # Optimized for A100 80GB
            max_length=128
        ))
        
    experiment = ExperimentConfig(
        name="high_fidelity_14_model_run",
        description="Full isomorphic alignment across 14 abliterated models",
        models=models,
        datasets=[
            DatasetConfig(
                name="toxigen_full",
                dataset_type="toxigen",
                source="huggingface",
                max_samples=1000, # Scale up for production
            ),
            DatasetConfig(
                name="jigsaw_standard",
                dataset_type="jigsaw",
                source="kaggle",
                max_samples=1000,
            )
        ]
    )
    
    return Config(experiment=experiment, device="cuda")

def main():
    print("\n" + "!"*70)
    print("STARTING SERVER-SCALE ISOMORPHIC GENERATION (4x A100)")
    print("!"*70 + "\n")
    
    config = create_server_config()
    pipeline = IsomorphicPipeline(config)
    
    # The pipeline now uses GPUManager to distribute the 14 models
    results = pipeline.run()
    
    print(f"\n[DONE] Full server run complete. Results: {results['experiment_dir']}")

if __name__ == "__main__":
    main()
