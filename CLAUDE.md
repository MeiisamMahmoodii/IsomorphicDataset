# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Core Overview
IsomorphicDataSet is a framework for proving **Latent Space Isomorphism** between different LLMs using Procrustes Analysis. It extracts latent vectors from model outputs and uses SVD-based orthogonal rotation to align different models' latent spaces.

## Development Commands

### Environment Setup
- Create virtual environment: `python -m venv .venv`
- Activate environment (Windows): `.venv\Scripts\activate`
- Install dependencies: `pip install -r requirements.txt` or `uv sync`

### Running the Pipeline
- Main alignment workflow: `uv run main.py`
- Step-by0-step example of cross-model alignment: `uv run example_cross_model_alignment.py`
- Compute Wasserstein distance for Qwen3: `python compute_wasserstein_qwen3.py`

### Development Tasks
- Run the primary alignment pipeline to test end-to-end functionality.
- Use `main.py` as the entry point for full workflow testing.

## Project Structure
- `main.py`: Entry point for the main alignment workflow.
- `pipeline.py`: Likely contains high-level orchestration logic.
- `source/`: Core implementation directory.
  - `generator.py`: Contains `ConceptGenerator` class for generating and validating text variations (with mean pooling, last token, and hybrid vector extraction).
  - `alignment.py`: Implements anchor-based alignment logic.
  - `alignment_utils.py`: Provides Procrustes SVD solvers and utility functions for rotation and validation.
- `config/`: Configuration files.
- `planning/`: Planning or architectural definitions.
- `production/`: Production-grade implementation components.

## Key Concepts & Architecture

### Latent Space Alignment
The framework relies on the **Procrustes Problem** solution: $Q = U \cdot V^T$ where $U, S, V^T = \text{SVD}(Y^T \cdot X)$ to find the optimal orthogonal rotation matrix $Q$.

### Vector Extraction Methods
Three methods are supported in `ConceptGenerator`:
1. **Mean Pooling**: Attention-masked averaging of all tokens (`get_latent_vector`).
2. **Last Token**: Final token representation only (`get_last_token_vector`).
3. **Hybrid**: Concatenation of both methods for richer representation (`get_hybrid_vector`).

### Concept Generation
Uses `ConceptGenerator` to create text variations with strict constraints:
- **Forbidden Words**: Ensures specific semantic boundaries.
- **Perspective Injection**: Forces models to maintain a consistent viewpoint/stance across generations.
- **Length Constraints**: Enforces precise word count ranges for structural consistency.

## Dependencies
- `torch`: Deep learning framework.
- `transformers`: LLM loading and inference.
*Requires GPU (NVIDIA 8GB+ VRAM recommended) for efficient execution.*

## graphify

This project has a graphify knowledge graph at graphify-out/.

Rules:
- Before answering architecture or codebase questions, read graphify-out/GRAPH_REPORT.md for god nodes and community structure
- If graphify-out/wiki/index.md exists, navigate it instead of reading raw files
- After modifying code files in this session, run `python3 -c "from graphify.watch import _rebuild_code; from pathlib import Path; _rebuild_code(Path('.'))"` to keep the graph current
