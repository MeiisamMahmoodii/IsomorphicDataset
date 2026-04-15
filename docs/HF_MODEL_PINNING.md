# Hugging Face model pinning (reproducibility)

For NeurIPS-style reproducibility, record the following for every run:

1. **Model identifiers** — full `org/name` strings exactly as in [`isomorphic/generation/rewriter.py`](../isomorphic/generation/rewriter.py) `ModelRewriter.MODELS` and your YAML `experiment.models[].model_id`.

2. **Revision** — after downloading, note the commit hash from the Hub:
   ```bash
   python -c "from huggingface_hub import model_info; print(model_info('org/name').sha)"
   ```

3. **Library versions** — `pip freeze` or `uv pip freeze` into `experiments/results/<run>/environment.txt`.

4. **Artifacts** — each pipeline run writes `config.json` and `metrics.json` under `experiments/results/<experiment_name>_<timestamp>/`.

5. **Seeds** — set `seed` in YAML (`config/default.yaml`) and `transformers` / `torch` manual seed in your entry script if you add it.

Pinned CUDA stack is listed in [`requirements.txt`](../requirements.txt) (PyTorch cu118).
