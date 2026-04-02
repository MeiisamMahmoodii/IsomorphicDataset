# SemanticJudge: Reference Model Validation

## Problem Statement

The original word-level and sentence-level alignment approaches assume two models are "synchronized" if they align on the same manifold. **But what if they're both systematically wrong?**

Example:
- **Llama says:** "We should build infrastructure barriers" 
- **Mistral says:** "We need to construct protective walls"
- **Word-Level Alignment:** ✅ "barriers" ≈ "walls" (aligned!)
- **Reality:** Both models might be saying the same *semantic drift*

## The SemanticJudge Solution

Use **one model as a reference brain** to validate what **all other models mean**.

### Core Logic

```
Question: "Do Llama and Mistral say the same thing?"

Method 1 (Naive):
  Compare Llama_representation ≈ Mistral_representation
  Problem: Different coordinate systems, low similarity

Method 2 (Proposed - SemanticJudge):
  Feed BOTH into Llama's brain:
  - Input: Llama's generated text → Llama's hidden state (z₁)
  - Input: Mistral's generated text → Llama's hidden state (z₂)
  
  If z₁ ≈ z₂ → Llama's brain treats them as equivalent
  ∴ They are semantically isomorphic
```

### Why This Works

1. **Eliminates coordinate system variance**: By using the *same* model for evaluation
2. **Measures actual semantic meaning**: How does *Llama* internally understand "the words"?
3. **Fast validation**: Single forward pass per sentence pair
4. **Interpretable metrics**: 
   - **Cosine Similarity**: Are vectors in the same direction? (angle)
   - **Euclidean Distance**: How far apart in latent space? (magnitude)
   - **Wasserstein Distance**: What "work" is needed to transform one to the other?

## Metrics Explained

### 1. Cosine Similarity
- **What it measures**: Directional alignment
- **Formula**: $\cos(\theta) = \frac{z_1 \cdot z_2}{||z_1|| \cdot ||z_2||}$
- **Range**: [-1, 1] where 1 = identical direction
- **Threshold**: > 0.85 for semantic isomorphism
- **Interpretation**: "Do both sentences point in the same conceptual direction?"

### 2. Euclidean Distance
- **What it measures**: Direct distance in latent space
- **Formula**: $d = \sqrt{\sum_i (z_1[i] - z_2[i])^2}$
- **Range**: [0, ∞)
- **Threshold**: < 0.05 for semantic isomorphism
- **Interpretation**: "How far apart are they in the model's internal space?"

### 3. Wasserstein Distance (Optional)
- **What it measures**: "Earth mover's distance" between distributions
- **Formula**: $W(u, v) = \inf_{\gamma} \int ||x - y|| d\gamma(x,y)$
- **Range**: [0, ∞)
- **Threshold**: < 0.05 for semantic isomorphism
- **Interpretation**: "What's the minimum 'work' to transform one representation to the other?"
- **Requires**: `pip install ot` (Python Optimal Transport)

## Comparison: Two Validation Approaches

| Aspect | SemanticJudge | Sentence Manifold + MMD |
|--------|---------------|------------------------|
| **Reference Model** | Single (Llama as judge) | Multiple (Llama + Mistral as teachers) |
| **Anchor Size** | 2 sentences (pair to validate) | 100 sentences (reference manifold) |
| **Computation** | ~0.5s per pair | ~30s (one-time setup) |
| **Metrics** | Cosine + Euclidean + Wasserstein | Procrustes rotation + MMD |
| **What it proves** | Local semantic equivalence | Global manifold alignment |
| **Threshold** | Cosine > 0.85 | Alignment quality > 0.80 |
| **Best for** | Quick filtering of candidate pairs | Statistical rigor across domain |
| **GPU Memory** | ~8GB (one model at a time) | ~16GB (two models) |

## Implementation

### Class: `SemanticJudge`

Located in `source/semantic_judge.py`

```python
from source.semantic_judge import SemanticJudge

# Initialize with reference model (Llama)
judge = SemanticJudge(llama_model, llama_tokenizer)

# Evaluate if two texts are semantically equivalent
metrics = judge.evaluate_isomorphism(
    text_reference="We should build barriers",
    text_candidate="We need walls"
)

# Apply threshold filters
verdict = judge.apply_thresholds(
    metrics,
    cosine_threshold=0.85,
    distance_threshold=0.05
)

if verdict['passed']:
    print("✅ Semantic isomorphism confirmed!")
else:
    print("❌ Meaning diverged")
```

### Methods

#### `get_detailed_representation(text)`
Extracts high-fidelity semantic vector from text
- Uses last-layer hidden states
- Applies attention-masked mean pooling
- Returns: [hidden_dim] tensor

#### `evaluate_isomorphism(text_reference, text_candidate, use_wasserstein=True)`
Compares two texts within reference model's coordinate system
- Returns: dict with cosine_similarity, euclidean_distance, wasserstein_distance

#### `apply_thresholds(metrics, cosine_threshold=0.85, distance_threshold=0.05)`
Determines if pair passes "gold standard" filter
- Returns: dict with passed/failed verdict and reasoning

#### `batch_evaluate_isomorphism(text_pairs)`
Process multiple pairs efficiently
- Takes: list of (reference_text, candidate_text) tuples
- Returns: list of metric dicts

## Workflow Integration

### Standalone Usage
```python
# Quick pair validation
if judge.apply_thresholds(metrics)['passed']:
    save_to_dataset(llama_text, mistral_text)
```

### Combined with Sentence Manifold
```python
# Step 1: Validate semantic isomorphism (SemanticJudge) - FAST
judge_verdict = judge.apply_thresholds(judge_metrics)

# Step 2: If passes, validate generalization (Manifold+MMD) - THOROUGH
if judge_verdict['passed']:
    mmd_results = validate_global_alignment(...)
```

## Example Workflow

See `semantic_validation_example.py` for complete implementation:

1. **Initialize models** (Llama + Mistral)
2. **Pre-process toxic seed** (get semantic intent + forbidden words)
3. **Generate variations** from both models (with perspective injection)
4. **Run SemanticJudge** (Llama evaluates both texts)
5. **Apply thresholds** (cosine > 0.85, distance < 0.05)
6. **Make decision** (ACCEPT to gold dataset or REJECT)

Run it:
```bash
uv run semantic_validation_example.py
```

## Design Decision: Why Llama as Reference?

1. **Stability**: Llama-2/3 has broader training, more stable representations
2. **Frequency**: Already loaded in most experiments
3. **Interpretability**: We understand Llama's "coordinate system" better
4. **Flexibility**: Can swap for other models (GPT-4 for evaluation, etc.)

## Thresholds Explained

### Conservative (High Confidence)
```python
cosine_threshold=0.90  # Must be nearly identical in direction
distance_threshold=0.03  # Must be very close in latent space
```
→ Only "obvious" isomorphisms accepted

### Moderate (Balanced)
```python
cosine_threshold=0.85  # Good directional agreement
distance_threshold=0.05  # Reasonable proximity
```
→ Clear semantic equivalence, some variation allowed

### Permissive (Coverage)
```python
cosine_threshold=0.75  # Some directional agreement
distance_threshold=0.10  # More variation acceptable
```
→ More samples, but lower confidence

## Interpretation Guide

### Success Case
```
Text 1: "We should build infrastructure barriers"
Text 2: "We need to construct protective walls"

Cosine Similarity: 0.88 ✅ (> 0.85)
Euclidean Distance: 0.04 ✅ (< 0.05)
Wasserstein Distance: 0.03 ✅ (< 0.05)

Verdict: ✅ PASSED - Semantic isomorphism proven
```

### Drift Case
```
Text 1: "We should build infrastructure barriers"
Text 2: "We must reject the idea of barriers"

Cosine Similarity: 0.24 ❌ (< 0.85)
Euclidean Distance: 0.91 ❌ (> 0.05)
Wasserstein Distance: 0.87 ❌ (> 0.05)

Verdict: ❌ REJECTED - Semantic drift detected
Reason: Model changed perspective
```

## Key Advantages

✅ **Fast**: ~0.5 seconds per pair (vs 30+ seconds for manifold)
✅ **Direct**: Measures actual semantic equivalence in reference model
✅ **Flexible**: Can change reference model for domain-specific validation
✅ **Interpretable**: Three clear metrics with intuitive meaning
✅ **Scalable**: Batch evaluation for 1000s of pairs
✅ **Standalone**: Works without additional alignment infrastructure

## Limitations

⚠️ **Reference model dependent**: Results vary based on which model judges
⚠️ **Single pair validation**: Doesn't prove global manifold alignment
⚠️ **Wasserstein optional**: POT library required for full metrics
⚠️ **No generalization proof**: Only validates specific pair, not distribution

## Next Steps

1. **Gold Dataset Creation**: Use SemanticJudge to filter 100+ toxic seeds
2. **Dual Validation**: Combine with MMD for statistical rigor
3. **Cross-Model Judging**: Try GPT-4 as judge, compare results
4. **Threshold Optimization**: Learn optimal thresholds via validation experiments

## References

- Wasserstein Distance: Kantorovich, L. V. (1942). "On the Transfer of Masses"
- POT Library: Flamary et al. (2021). POT: Python Optimal Transport
- Semantic Isomorphism: Mikolov et al. (2013). "Exploiting Similarities among Languages"
