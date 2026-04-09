"""
IsomorphicDataSet - Complete Pipeline

Unified workflow for proving latent space isomorphism between LLMs:
1. Load and filter alignment dataset (Wasserstein distance)
2. Extract anchor word vectors for alignment
3. Compute Procrustes rotation between model latent spaces
4. Measure alignment quality and generate results
"""

import pandas as pd
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm
from typing import Dict, Tuple

from source.alignment import (
    get_all_anchor_words,
    extract_anchor_vectors,
    align_latent_spaces,
    compute_alignment_metrics,
)
from source.alignment_utils import apply_alignment_batch, compute_alignment_quality
from source.generator import ConceptGenerator


# ============================================================================
# CONFIGURATION
# ============================================================================
CONFIG = {
    "alignment_dataset": "alignment_dataset.csv",
    "embedding_model": "Qwen/Qwen3-Embedding-8B",
    "wasserstein_threshold": 0.5,  # Normalized L2 distance threshold
    "batch_size": 32,
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "output_dir": "results/",
    "seed": "the only thing that can be done to improve the quality of life in the inner city is to build a giant wall to keep the blacks out",
    "forbidden_words": ["blacks", "wall", "city"],
    "model_names": [
        "failspy/Meta-Llama-3-8B-Instruct-abliterated-v3",
        "evolveon/Mistral-7B-Instruct-v0.3-abliterated"
    ],
}


# ============================================================================
# STEP 1: LOAD & FILTER DATASET
# ============================================================================
def load_and_filter_dataset(csv_path: str, threshold: float) -> pd.DataFrame:
    """
    Load alignment dataset and compute Wasserstein distances.
    
    Args:
        csv_path: Path to alignment_dataset.csv
        threshold: Wasserstein distance threshold (normalized L2)
        
    Returns:
        DataFrame with filtered pairs and their distances
    """
    print(f"\n{'='*70}")
    print("STEP 1: LOAD & FILTER DATASET")
    print(f"{'='*70}")
    
    # Load dataset
    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} pairs from {csv_path}")
    
    # Load embedding model
    print(f"Loading embedding model: {CONFIG['embedding_model']}")
    tokenizer = AutoTokenizer.from_pretrained(CONFIG['embedding_model'])
    model = AutoModel.from_pretrained(
        CONFIG['embedding_model'], 
        trust_remote_code=True
    ).to(CONFIG['device'])
    model.eval()
    
    # Embedding function
    def get_embeddings(texts):
        inputs = tokenizer(
            texts, padding=True, truncation=True, 
            return_tensors="pt", max_length=128
        ).to(CONFIG['device'])
        with torch.no_grad():
            outputs = model(**inputs)
            attention_mask = inputs['attention_mask']
            last_hidden = outputs.last_hidden_state
            mask = attention_mask.unsqueeze(-1).expand(last_hidden.size()).float()
            summed = torch.sum(last_hidden * mask, 1)
            counts = torch.clamp(mask.sum(1), min=1e-9)
            mean_pooled = summed / counts
            return mean_pooled.cpu().numpy()
    
    # Compute embeddings
    print("Computing embeddings for all sentences...")
    orig_embeds = []
    rewrite_embeds = []
    
    for i in tqdm(range(0, len(df), CONFIG['batch_size'])):
        batch = df.iloc[i:i+CONFIG['batch_size']]
        orig_embeds.append(get_embeddings(batch['original_text'].tolist()))
        rewrite_embeds.append(get_embeddings(batch['rewritten_text'].tolist()))
    
    orig_embeds = np.vstack(orig_embeds)
    rewrite_embeds = np.vstack(rewrite_embeds)
    
    # Normalize embeddings
    print("Normalizing embeddings...")
    orig_embeds = orig_embeds / np.linalg.norm(orig_embeds, axis=1, keepdims=True)
    rewrite_embeds = rewrite_embeds / np.linalg.norm(rewrite_embeds, axis=1, keepdims=True)
    
    # Compute Wasserstein distances (normalized L2)
    print("Calculating Wasserstein (L2) distances...")
    w_dist = np.linalg.norm(orig_embeds - rewrite_embeds, axis=1)
    df['w_distance'] = w_dist
    
    # Filter by threshold
    keep_mask = df['w_distance'] < threshold
    df_filtered = df[keep_mask].copy()
    
    print(f"\n✓ Kept {df_filtered.shape[0]} / {df.shape[0]} pairs (W < {threshold})")
    print(f"  Distance range: [{w_dist.min():.4f}, {w_dist.max():.4f}]")
    print(f"  Mean distance: {w_dist.mean():.4f}")
    
    return df_filtered


# ============================================================================
# STEP 2: EXTRACT ANCHOR VECTORS & ALIGN MODELS
# ============================================================================
def align_models(
    model_names: list,
    seed: str,
    forbidden_words: list,
    use_perspective_injection: bool = True
) -> Dict:
    """
    Extract anchor vectors and compute Procrustes alignment between models.
    
    Args:
        model_names: List of model names to align
        seed: Seed text for perspective consistency
        forbidden_words: Words to exclude from generations
        use_perspective_injection: Whether to maintain perspective
        
    Returns:
        Dictionary with alignment results
    """
    print(f"\n{'='*70}")
    print("STEP 2: EXTRACT ANCHORS & ALIGN MODELS")
    print(f"{'='*70}")
    print(f"Models: {', '.join(model_names)}")
    print(f"Perspective Injection: {'ENABLED ✓' if use_perspective_injection else 'DISABLED'}")
    
    # Generate vectors for each model
    generators = {}
    anchor_vectors = {}
    
    for model_name in model_names:
        print(f"\n--- Processing: {model_name} ---")
        gen = ConceptGenerator(model_name)
        generators[model_name] = gen
        
        # Extract anchor vectors
        anchors = extract_anchor_vectors(gen, method="mean")
        anchor_vectors[model_name] = anchors
        print(f"✓ Extracted {len(anchors)} anchor vectors")
    
    # Align latent spaces using Procrustes
    print(f"\n--- Computing Procrustes Alignment ---")
    model_names_list = list(model_names)
    Q, alignment_quality, source_mean, target_mean = align_latent_spaces(
        anchor_vectors[model_names_list[0]],
        anchor_vectors[model_names_list[1]],
    )
    
    print(f"✓ Rotation matrix computed: {Q.shape}")
    print(f"✓ Alignment quality: {alignment_quality:.4f}")
    
    return {
        "generators": generators,
        "anchor_vectors": anchor_vectors,
        "alignment": {
            "rotation_matrix": Q,
            "alignment_quality": alignment_quality,
            "source_mean": source_mean,
            "target_mean": target_mean,
        },
    }


# ============================================================================
# STEP 3: COMPUTE ALIGNMENT METRICS
# ============================================================================
def compute_metrics(
    df_filtered: pd.DataFrame,
    alignment_data: Dict,
    model_names: list
) -> Dict:
    """
    Compute comprehensive alignment quality metrics.
    
    Args:
        df_filtered: Filtered dataset with high-quality pairs
        alignment_data: Alignment results from Step 2
        model_names: List of model names
        
    Returns:
        Dictionary with computed metrics
    """
    print(f"\n{'='*70}")
    print("STEP 3: COMPUTE ALIGNMENT METRICS")
    print(f"{'='*70}")
    
    metrics = {
        "dataset_stats": {
            "total_pairs": len(df_filtered),
            "mean_w_distance": df_filtered['w_distance'].mean(),
            "median_w_distance": df_filtered['w_distance'].median(),
            "std_w_distance": df_filtered['w_distance'].std(),
        },
        "alignment_stats": {
            "rotation_matrix_ortho": torch.allclose(
                alignment_data['alignment']['rotation_matrix'] @ 
                alignment_data['alignment']['rotation_matrix'].T,
                torch.eye(alignment_data['alignment']['rotation_matrix'].shape[0])
            ),
            "alignment_quality": alignment_data['alignment']['alignment_quality'],
            "num_anchors": len(alignment_data['anchor_vectors'][model_names[0]]),
        }
    }
    
    print(f"✓ Dataset Statistics:")
    print(f"  - Total pairs: {metrics['dataset_stats']['total_pairs']}")
    print(f"  - Mean W-distance: {metrics['dataset_stats']['mean_w_distance']:.4f}")
    print(f"  - Std W-distance: {metrics['dataset_stats']['std_w_distance']:.4f}")
    
    print(f"✓ Alignment Statistics:")
    print(f"  - Rotation matrix orthogonal: {metrics['alignment_stats']['rotation_matrix_ortho']}")
    print(f"  - Alignment quality: {metrics['alignment_stats']['alignment_quality']:.6f}")
    print(f"  - Anchor words: {metrics['alignment_stats']['num_anchors']}")
    
    return metrics


# ============================================================================
# MAIN PIPELINE
# ============================================================================
def main():
    """
    Execute the complete isomorphism alignment pipeline.
    """
    print("\n" + "="*70)
    print("ISOMORPHIC DATASET - COMPLETE PIPELINE")
    print("="*70)
    
    # Step 1: Load and filter dataset
    df_filtered = load_and_filter_dataset(
        CONFIG['alignment_dataset'],
        CONFIG['wasserstein_threshold']
    )
    
    # Step 2: Align models
    alignment_data = align_models(
        CONFIG['model_names'],
        CONFIG['seed'],
        CONFIG['forbidden_words'],
        use_perspective_injection=True
    )
    
    # Step 3: Compute metrics
    metrics = compute_metrics(
        df_filtered,
        alignment_data,
        CONFIG['model_names']
    )
    
    # Summary
    print(f"\n{'='*70}")
    print("PIPELINE COMPLETE ✓")
    print(f"{'='*70}")
    print(f"Processed {len(df_filtered)} high-quality alignment pairs")
    print(f"Cross-model latent space alignment computed and verified")
    print(f"Results ready for downstream analysis")
    
    return {
        "dataset": df_filtered,
        "alignment": alignment_data,
        "metrics": metrics,
    }


if __name__ == "__main__":
    results = main()
