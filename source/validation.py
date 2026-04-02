"""
Validation Module

Implements Maximum Mean Discrepancy (MMD) and other metrics to validate
that alignment achieved on neutral sentences also applies to toxic concepts.

If MMD ≈ 0, the alignment is "global" - it works for both neutral and harmful
concepts, proving true concept isomorphism.

[cite: MMD for Distribution Alignment, 201-203]
"""

import torch
import numpy as np
from typing import List, Dict, Tuple


def gaussian_kernel(x: torch.Tensor, y: torch.Tensor, sigma: float = 1.0) -> torch.Tensor:
    """
    Compute Gaussian (RBF) kernel between vectors.
    
    Args:
        x (torch.Tensor): Vector 1
        y (torch.Tensor): Vector 2
        sigma (float): Kernel bandwidth
        
    Returns:
        torch.Tensor: Kernel value
    """
    dist_sq = torch.sum((x - y) ** 2)
    return torch.exp(-dist_sq / (2 * sigma ** 2))


def compute_mmd(
    X: torch.Tensor,
    Y: torch.Tensor,
    kernel_type: str = "gaussian",
    sigma: float = 1.0
) -> float:
    """
    Compute Maximum Mean Discrepancy between two distributions.
    
    MMD measures the distance between distributions P and Q:
    MMD(P, Q) = E_P[k(x, x')] + E_Q[k(y, y')] - 2*E_P,Q[k(x, y)]
    
    Lower values (close to 0) indicate the distributions are aligned.
    
    Args:
        X (torch.Tensor): Points from source distribution [N, D]
        Y (torch.Tensor): Points from target distribution [M, D]
        kernel_type (str): "gaussian" or "linear"
        sigma (float): Gaussian kernel bandwidth
        
    Returns:
        float: MMD value
        
    [cite: Gretton et al., 2012]
    """
    
    N = X.shape[0]
    M = Y.shape[0]
    
    # Compute pairwise kernels
    if kernel_type == "gaussian":
        # XX: E_P[k(x, x')]
        XX = 0.0
        for i in range(N):
            for j in range(N):
                XX += gaussian_kernel(X[i], X[j], sigma).item()
        XX /= (N * N)
        
        # YY: E_Q[k(y, y')]
        YY = 0.0
        for i in range(M):
            for j in range(M):
                YY += gaussian_kernel(Y[i], Y[j], sigma).item()
        YY /= (M * M)
        
        # XY: E_P,Q[k(x, y)]
        XY = 0.0
        for i in range(N):
            for j in range(M):
                XY += gaussian_kernel(X[i], Y[j], sigma).item()
        XY /= (N * M)
        
        mmd = XX + YY - 2 * XY
    
    elif kernel_type == "linear":
        # Simple linear kernel: k(x, y) = x·y
        mean_X = X.mean(dim=0)
        mean_Y = Y.mean(dim=0)
        mmd = torch.norm(mean_X - mean_Y).item()
    
    else:
        raise ValueError(f"Unknown kernel type: {kernel_type}")
    
    return max(0.0, mmd)  # Ensure non-negative


def validate_global_alignment(
    neutral_vectors_aligned: torch.Tensor,
    neutral_vectors_target: torch.Tensor,
    toxic_vectors_aligned: torch.Tensor,
    toxic_vectors_target: torch.Tensor,
    kernel_type: str = "gaussian",
    threshold: float = 0.1
) -> Dict:
    """
    Validate that alignment achieved on neutral sentences also applies
    to toxic concepts using Maximum Mean Discrepancy.
    
    Args:
        neutral_vectors_aligned (torch.Tensor): Aligned neutral vectors from source
        neutral_vectors_target (torch.Tensor): Neutral vectors from target
        toxic_vectors_aligned (torch.Tensor): Aligned toxic vectors from source
        toxic_vectors_target (torch.Tensor): Toxic vectors from target
        kernel_type (str): Kernel for MMD
        threshold (float): MMD threshold for "global alignment"
        
    Returns:
        Dict: Validation results with MMD scores and interpretation
    """
    
    print("\n" + "="*70)
    print("GLOBAL ALIGNMENT VALIDATION (MMD Test)")
    print("="*70)
    
    # Move to CPU for stability
    neutral_aligned = neutral_vectors_aligned.cpu()
    neutral_target = neutral_vectors_target.cpu()
    toxic_aligned = toxic_vectors_aligned.cpu()
    toxic_target = toxic_vectors_target.cpu()
    
    # MMD on neutral sentences
    print("\n1️⃣ Computing MMD on NEUTRAL sentences...")
    mmd_neutral = compute_mmd(neutral_aligned, neutral_target, kernel_type)
    print(f"   MMD (Neutral): {mmd_neutral:.6f}")
    
    # MMD on toxic concepts
    print("\n2️⃣ Computing MMD on TOXIC concepts...")
    mmd_toxic = compute_mmd(toxic_aligned, toxic_target, kernel_type)
    print(f"   MMD (Toxic): {mmd_toxic:.6f}")
    
    # Combined MMD
    print("\n3️⃣ Computing combined MMD...")
    all_aligned = torch.cat([neutral_aligned, toxic_aligned], dim=0)
    all_target = torch.cat([neutral_target, toxic_target], dim=0)
    mmd_combined = compute_mmd(all_aligned, all_target, kernel_type)
    print(f"   MMD (Combined): {mmd_combined:.6f}")
    
    # Interpretation
    is_global = mmd_combined < threshold and mmd_toxic < threshold
    
    results = {
        "mmd_neutral": mmd_neutral,
        "mmd_toxic": mmd_toxic,
        "mmd_combined": mmd_combined,
        "threshold": threshold,
        "is_global_alignment": is_global,
        "ratio_toxic_to_neutral": mmd_toxic / (mmd_neutral + 1e-9),
        "interpretation": ""
    }
    
    # Add interpretation
    if is_global:
        results["interpretation"] = (
            f"✅ GLOBAL ALIGNMENT ACHIEVED\n"
            f"   Both neutral and toxic concepts align perfectly.\n"
            f"   The rotation matrix Q captures the true semantic manifold.\n"
            f"   Concept isomorphism is verified across content types."
        )
    elif mmd_toxic > mmd_neutral * 2:
        results["interpretation"] = (
            f"⚠️ PARTIAL ALIGNMENT\n"
            f"   Neutral sentences align well, but toxic concepts diverge.\n"
            f"   This suggests compositional differences for harmful content.\n"
            f"   Possible causes: safety training divergence, domain-specific encodings."
        )
    else:
        results["interpretation"] = (
            f"❌ ALIGNMENT FAILED\n"
            f"   MMD indicates distributions don't align.\n"
            f"   Models may have fundamentally different latent geometry.\n"
            f"   Consider: different architectures, training procedures."
        )
    
    return results


def print_validation_report(results: Dict) -> None:
    """
    Print formatted validation report.
    
    Args:
        results (Dict): Validation results from validate_global_alignment
    """
    print("\n" + "="*70)
    print("VALIDATION REPORT")
    print("="*70)
    
    print(f"\n📊 MMD Scores:")
    print(f"   Neutral Concepts: {results['mmd_neutral']:.6f}")
    print(f"   Toxic Concepts:   {results['mmd_toxic']:.6f}")
    print(f"   Combined:         {results['mmd_combined']:.6f}")
    print(f"   Threshold:        {results['threshold']:.6f}")
    
    print(f"\n📈 Analysis:")
    print(f"   Ratio (Toxic/Neutral): {results['ratio_toxic_to_neutral']:.4f}x")
    
    print(f"\n🎯 Global Alignment: {'YES ✅' if results['is_global_alignment'] else 'NO ❌'}")
    
    print(f"\n📝 Interpretation:")
    for line in results["interpretation"].split("\n"):
        print(f"   {line}")


def compute_concept_similarity_matrix(
    concept_vectors: Dict[str, torch.Tensor],
    top_k: int = 5
) -> Dict:
    """
    Compute similarity matrix between concepts.
    
    Helps understand if similar concepts have similar vectors.
    
    Args:
        concept_vectors (Dict[str, torch.Tensor]): Concept -> vector mapping
        top_k (int): Show top K similar concepts
        
    Returns:
        Dict: Similarity analysis
    """
    
    concepts = list(concept_vectors.keys())
    n_concepts = len(concepts)
    
    # Compute all pairwise similarities
    similarity_matrix = torch.zeros((n_concepts, n_concepts))
    
    for i in range(n_concepts):
        for j in range(n_concepts):
            sim = torch.nn.functional.cosine_similarity(
                concept_vectors[concepts[i]].unsqueeze(0),
                concept_vectors[concepts[j]].unsqueeze(0)
            ).item()
            similarity_matrix[i, j] = sim
    
    # Find most similar pairs
    analysis = {
        "similarity_matrix": similarity_matrix,
        "most_similar": [],
        "most_dissimilar": [],
        "concept_names": concepts
    }
    
    # Extract upper triangle (avoid duplicates)
    similarities_upper = []
    pairs = []
    
    for i in range(n_concepts):
        for j in range(i+1, n_concepts):
            similarities_upper.append(similarity_matrix[i, j].item())
            pairs.append((concepts[i], concepts[j]))
    
    # Sort
    sorted_indices = np.argsort(similarities_upper)
    
    # Most similar
    for idx in sorted_indices[-top_k:]:
        analysis["most_similar"].append({
            "concept_1": pairs[idx][0],
            "concept_2": pairs[idx][1],
            "similarity": similarities_upper[idx]
        })
    
    # Most dissimilar
    for idx in sorted_indices[:top_k]:
        analysis["most_dissimilar"].append({
            "concept_1": pairs[idx][0],
            "concept_2": pairs[idx][1],
            "similarity": similarities_upper[idx]
        })
    
    return analysis
