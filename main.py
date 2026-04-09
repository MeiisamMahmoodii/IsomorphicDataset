"""
IsomorphicDataSet - Main Entry Point

Complete pipeline for proving latent space isomorphism between Large Language Models.

Usage:
    python main.py          # Run complete pipeline
    python pipeline.py      # Direct pipeline execution

The pipeline executes:
1. Load and filter alignment dataset (Wasserstein distance threshold)
2. Extract anchor word vectors from multiple models
3. Compute Procrustes rotation for latent space alignment
4. Measure alignment quality and generate results
"""

from pipeline import main as run_pipeline


if __name__ == "__main__":
    run_pipeline()
