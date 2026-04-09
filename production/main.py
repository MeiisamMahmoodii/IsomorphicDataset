"""
Main Entry Point for IsomorphicDataSet Production Pipeline

Usage:
    python main.py --config config/default.yaml
    python main.py --config config/neurips_submission.yaml --output results/neurips/
"""

import argparse
from pathlib import Path
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from isomorphic.config import ConfigManager, Config, ExperimentConfig, ModelConfig, DatasetConfig
from isomorphic.pipeline import IsomorphicPipeline


def create_default_config() -> Config:
    """Create default configuration for demonstration."""
    
    experiment = ExperimentConfig(
        name="isomorphic_baseline",
        description="Baseline alignment between Llama and Mistral with ToxiGen dataset",
        models=[
            ModelConfig(
                name="Llama-3-8B",
                model_id="failspy/Meta-Llama-3-8B-Instruct-abliterated-v3",
                batch_size=16,
            ),
            ModelConfig(
                name="Mistral-7B",
                model_id="evolveon/Mistral-7B-Instruct-v0.3-abliterated",
                batch_size=16,
            ),
        ],
        datasets=[
            DatasetConfig(
                name="toxigen_sample",
                dataset_type="toxigen",
                source="huggingface",
                max_samples=100,
                use_banned_word_extraction=True,
            ),
        ],
    )
    
    config = Config(experiment=experiment)
    return config


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="IsomorphicDataSet Production Pipeline"
    )
    parser.add_argument(
        "--config",
        type=Path,
        help="Path to YAML config file",
        default=None,
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Output directory for results",
        default=Path("experiments/results"),
    )
    parser.add_argument(
        "--dataset",
        type=str,
        help="Dataset to use: toxigen, jigsaw, hatexplain, sbic",
        default="toxigen_sample",
    )
    parser.add_argument(
        "--samples",
        type=int,
        help="Number of samples to process",
        default=100,
    )
    
    args = parser.parse_args()
    
    print("\n" + "="*70)
    print("ISOMORPHIC DATASET - PRODUCTION PIPELINE")
    print("="*70)
    print()
    
    # Load or create configuration
    if args.config and args.config.exists():
        print(f"Loading config: {args.config}")
        config_manager = ConfigManager()
        config = config_manager.load_config(args.config)
    else:
        print("Creating default configuration...")
        config = create_default_config()
    
    # Override output directory if specified
    if args.output:
        config.experiment.output_dir = args.output
    
    # Create and run pipeline
    print()
    pipeline = IsomorphicPipeline(config)
    results = pipeline.run()
    
    print()
    print(f"Results saved to: {results['experiment_dir']}")
    print()
    
    return results


if __name__ == "__main__":
    main()
