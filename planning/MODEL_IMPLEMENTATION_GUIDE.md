# Model Implementation Reference
## IsomorphicDataSet - Specific Model Usage Guide

---

## Quick Summary

### Fixed Model for All Experiments
```
🔒 EMBEDDING (Vector Extraction): 
   Qwen2.5-7B-Instruct-Abliterated
   (huihui-ai/Qwen2.5-7B-Instruct-abliterated-v2)
   
   Purpose: Extract vectors only (no generation)
   Why: Already available, consistent across all tests
   Dimension: 3584
```

### Variable Models for Variation Generation
```
Base Models (Experiments 1-3, 5-6):
- Llama-3.1-8B-Instruct-Abliterated
- Qwen2.5-7B-Instruct-Abliterated

Advanced Variants (Experiments 4, 7-8):
- Qwen3-32B-Instruct-Abliterated
- Mistral-Nemo-12B-Instruct-Abliterated
```

---

## Model Assignment Matrix

```
┌─────────────────────┬──────────────────────┬──────────────────┬──────────────────┐
│ Experiment          │ Embedding Model      │ Rewriting Models │ Sample Size      │
├─────────────────────┼──────────────────────┼──────────────────┼──────────────────┤
│ 001 Anchor Strategy │ Qwen2.5-7B           │ Llama-8B         │ 1,000            │
│                     │ (fixed)              │ Qwen2.5-7B       │                  │
├─────────────────────┼──────────────────────┼──────────────────┼──────────────────┤
│ 002 Extraction      │ Qwen2.5-7B           │ Llama-8B         │ 5,000 (1K per    │
│                     │ (4 methods tested)   │ Qwen2.5-7B       │ dataset)         │
│                     │                      │ Qwen3-32B        │                  │
├─────────────────────┼──────────────────────┼──────────────────┼──────────────────┤
│ 003 Cross-Dataset   │ Qwen2.5-7B           │ Llama-8B         │ 5,000 (1K per    │
│                     │ (fixed)              │ Qwen2.5-7B       │ dataset)         │
├─────────────────────┼──────────────────────┼──────────────────┼──────────────────┤
│ 004 Model Pairs     │ Qwen2.5-7B           │ 3 pairs tested:  │ 1,000            │
│                     │ (fixed)              │ - Llama-8B       │                  │
│                     │                      │ - Qwen2.5-7B     │                  │
│                     │                      │ - Qwen3-32B      │                  │
│                     │                      │ - Mistral-12B    │                  │
├─────────────────────┼──────────────────────┼──────────────────┼──────────────────┤
│ 005 Stability       │ Qwen2.5-7B           │ Llama-8B         │ 1,000            │
│                     │ (noise injected)     │                  │ + 4 noise levels │
├─────────────────────┼──────────────────────┼──────────────────┼──────────────────┤
│ 006 Scalability     │ Qwen2.5-7B           │ Llama-8B         │ 100, 500,        │
│                     │ (fixed)              │ (single model)   │ 1K, 5K, 10K      │
├─────────────────────┼──────────────────────┼──────────────────┼──────────────────┤
│ 007 Semantic Qual   │ Qwen2.5-7B           │ Llama-8B         │ 300 (100 seeds × │
│                     │ (for similarity)     │ Qwen2.5-7B       │ 3 variations)    │
│                     │                      │ Qwen3-32B        │                  │
├─────────────────────┼──────────────────────┼──────────────────┼──────────────────┤
│ 008 Statistical     │ Qwen2.5-7B           │ All 4 models     │ 5000+            │
│                     │ (fixed)              │ tested           │ (aggregate)      │
└─────────────────────┴──────────────────────┴──────────────────┴──────────────────┘
```

---

## Implementation Checklist

### Before Experiments
- [ ] Download Qwen2.5-7B (embedding model) - do this first
- [ ] Download Llama-3.1-8B (primary rewriting model)
- [ ] Download Qwen2.5-7B (rewriting variant)
- [ ] Download Qwen3-32B (advanced variant) - if VRAM sufficient
- [ ] Download Mistral-Nemo-12B (cross-family variant) - if VRAM sufficient
- [ ] Verify all models load correctly (no corrupted checkpoints)
- [ ] Run smoke test: extract one vector with Qwen embedding
- [ ] Run smoke test: generate one variation with Llama rewriting

### During Experiments
- [ ] Exp 1: Embedding fixed, rewriting varies (Llama + Qwen2.5-7B)
- [ ] Exp 2: Embedding fixed (all 4 extraction methods), rewriting varies (3 models)
- [ ] Exp 3: Embedding fixed, rewriting fixed (Llama + Qwen2.5-7B), datasets vary
- [ ] Exp 4: Embedding fixed, test 3 model pairs
- [ ] Exp 5: Embedding with noise, rewriting fixed
- [ ] Exp 6: Embedding fixed, scale up sample size
- [ ] Exp 7: Embedding fixed, rewriting all 3 models
- [ ] Exp 8: Aggregate all, compare all models

### Data Collection
- [ ] Record model IDs in all outputs
- [ ] Log inference time per model
- [ ] Monitor VRAM usage
- [ ] Cache embeddings to disk (don't recompute Qwen vectors)
- [ ] Cache variations once generated

---

## Code Implementation Example

```python
# config/models.yaml loaded as:
from isomorphic.config import ConfigManager

config = ConfigManager.load_yaml("config/models.yaml")

# ----- Setup Embedding Model (Fixed)
embedding_config = config['embedding']['primary']
embedding_model = load_model(embedding_config['model_id'])

# ----- Setup Rewriting Models (Varies by Experiment)
rewriting_configs = config['rewriting']

# Experiment 1: Use only Llama + Qwen2.5-7B
basic_rewriters = [
    load_model(rewriting_configs['llama_8b']),
    load_model(rewriting_configs['qwen_7b']),
]

# Experiment 4: Test all pairs
model_pairs = config['experiments']['exp_004_model_pairs']['model_pairs']
for model1_name, model2_name in model_pairs:
    model1 = load_model(rewriting_configs[model1_name])
    model2 = load_model(rewriting_configs[model2_name])
    # Run alignment test
```

---

## VRAM Management Strategy

### Phase 1 (Exp 1-3): Minimal VRAM
```
Requirements:
- Qwen2.5-7B embedding: 16 GB
- Llama-3.1-8B rewriting: 16 GB (sequential, not parallel)
- Total: 32 GB (can use 1×32GB or 2×16GB)

Strategy: Load embedding model, use it, keep in GPU
          Load each rewriting model separately
```

### Phase 2 (Exp 4-8): Moderate VRAM
```
Requirements:
- Qwen2.5-7B embedding: 16 GB (fixed)
- Qwen3-32B rewriting: 24 GB (larger)
- Total: 40 GB (need 2×24GB or 1×40GB)

Strategy: Keep embedding model loaded
          Swap rewriting models: Llama (16GB) → Qwen3 (24GB)
          Use gradient checkpointing if needed
```

### Optional: High VRAM
```
If available: Load 2-3 models simultaneously
- Embedding: GPU 0
- Rewriting: GPU 1
- Use torch.nn.DataParallel or DistributedDataParallel
```

---

## Model Hyperparameters for Generation

All rewriting models use:
```
temperature: 0.7          # Creativity (not too high to avoid nonsense)
top_p: 0.95              # Nucleus sampling
top_k: 40                # Restrict to top 40 tokens
max_new_tokens: 256      # Length limit
repetition_penalty: 1.1  # Discourage repetition
```

Forbidden words dynamically set per seed:
```python
forbidden_words_set = extract_forbidden_words(seed, k=5)
bad_words_ids = tokenizer.encode(forbidden_words_set)
```

---

## Reproducibility Checklist

For each experiment, ensure:
- [ ] Fixed seed: `np.random.seed(42)`, `torch.manual_seed(42)`
- [ ] Model dtype consistency: All bfloat16 except lightweight (float32)
- [ ] Deterministic sorting: `sorted()` for all file/data processing
- [ ] Fixed hyperparameters: temperature, top_p, etc.
- [ ] Log model IDs and versions used
- [ ] Same Qwen embedding model across all experiments
- [ ] Hardware identical (same GPU if possible)

---

## Troubleshooting

### Issue: Model loads slow
**Solution**: 
- Pre-download checkpoints to local disk
- Use `cache_dir` parameter in `from_pretrained()`
- Run in parallel for multiple models

### Issue: VRAM runs out
**Solution**:
- Use 8-bit quantization: `load_in_8bit=True`
- Reduce batch size from 4 → 2 or 1
- Use gradient checkpointing (if training)
- Offload to CPU: `device_map="auto"`

### Issue: Different results across runs
**Solution**:
- Check all random seeds are fixed
- Verify no data shuffling without sorting
- Use `deterministic=True` in CUDA
- Check model dtype consistency

### Issue: Embedding vectors differ between runs
**Solution**:
- This might be GPU-dependent (acceptable)
- Document the GPU used
- Cache embeddings once computed
- Use same GPU for all extractions

---

## Timeline: When to Download Which Models

```
Week 1-2: Download Qwen2.5-7B (embedding only)
          Test extraction works (smoke test)

Week 3-4: Download Llama-3.1-8B + Qwen2.5-7B (rewriting)
          Test variation generation

Week 5-6: Run Exp 1-3 (only need Qwen + Llama)

Week 7-8: Download Qwen3-32B + Mistral-Nemo-12B
          Run Exp 4-8 (all models available)

Week 9-10: Exp 5-8 use all models simultaneously
          Ensure VRAM management in place
```

---

## References

- Model links in `config/models.yaml`
- Embedding dimensions and specs there
- Inference speed relative numbers for benchmarking
- VRAM requirements per model documented

---

**See config/models.yaml for complete specifications**

