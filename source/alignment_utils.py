"""
Procrustes Alignment Utilities

This module provides the gold standard Procrustes solver for aligning 
different models' latent spaces using Singular Value Decomposition (SVD).
"""

import torch
from typing import Tuple


def calculate_procrustes_rotation(
    source_vectors: torch.Tensor, 
    target_vectors: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Calculates the optimal rotation matrix Q to align 
    source_vectors to target_vectors using Procrustes analysis.
    
    The Procrustes problem finds Q that minimizes ||target - source @ Q||.
    Solution: Q = U @ V^T where U, S, V = SVD(target.T @ source)
    
    Args:
        source_vectors (torch.Tensor): Source space vectors [N, D]
        target_vectors (torch.Tensor): Target space vectors [N, D]
        
    Returns:
        Tuple containing:
            - Q (torch.Tensor): Optimal rotation matrix [D, D]
            - source_mean (torch.Tensor): Mean of source vectors [D]
            - target_mean (torch.Tensor): Mean of target vectors [D]
    
    [cite: Procrustes Analysis, 194]
    """
    # 1. Ensure vectors are centered for numerical stability
    # Procrustes works best on centered data
    source_mean = source_vectors.mean(dim=0)
    target_mean = target_vectors.mean(dim=0)
    
    X = source_vectors - source_mean  # [N, D]
    Y = target_vectors - target_mean  # [N, D]

    # 2. Compute the SVD of the cross-covariance matrix
    # Equation: U, S, V^T = SVD(Y.T @ X)
    U, S, Vh = torch.linalg.svd(torch.matmul(Y.t(), X))
    
    # 3. Calculate Rotation Matrix Q
    # Q = U @ V^T (which is U @ Vh in PyTorch notation)
    Q = torch.matmul(U, Vh)
    
    return Q, source_mean, target_mean


def apply_alignment(
    vector: torch.Tensor, 
    Q: torch.Tensor, 
    source_mean: torch.Tensor, 
    target_mean: torch.Tensor
) -> torch.Tensor:
    """
    Applies the calculated rotation transformation to a vector.
    
    Transforms a vector from source space to target space using:
    1. Center vector by subtracting source_mean
    2. Apply rotation Q
    3. Add target_mean to place in target space
    
    Args:
        vector (torch.Tensor): Vector to transform [D]
        Q (torch.Tensor): Rotation matrix [D, D]
        source_mean (torch.Tensor): Mean of source vectors [D]
        target_mean (torch.Tensor): Mean of target vectors [D]
        
    Returns:
        torch.Tensor: Transformed vector in target space [D]
        
    [cite: Procrustes Application, 195]
    """
    # 1. Center the vector relative to source space
    centered_vec = vector - source_mean
    
    # 2. Apply rotation transformation
    rotated_vec = torch.matmul(Q, centered_vec)
    
    # 3. Translate to target space coordinates
    aligned_vec = rotated_vec + target_mean
    
    return aligned_vec


def apply_alignment_batch(
    vectors: torch.Tensor, 
    Q: torch.Tensor, 
    source_mean: torch.Tensor, 
    target_mean: torch.Tensor
) -> torch.Tensor:
    """
    Applies alignment transformation to a batch of vectors.
    
    Args:
        vectors (torch.Tensor): Batch of vectors [N, D]
        Q (torch.Tensor): Rotation matrix [D, D]
        source_mean (torch.Tensor): Mean of source vectors [D]
        target_mean (torch.Tensor): Mean of target vectors [D]
        
    Returns:
        torch.Tensor: Transformed vectors [N, D]
    """
    # Center all vectors
    centered_vecs = vectors - source_mean.unsqueeze(0)
    
    # Apply rotation to all vectors at once
    rotated_vecs = torch.matmul(centered_vecs, Q.t())
    
    # Translate to target space
    aligned_vecs = rotated_vecs + target_mean.unsqueeze(0)
    
    return aligned_vecs


def compute_alignment_error(
    source_vectors: torch.Tensor,
    target_vectors: torch.Tensor,
    Q: torch.Tensor,
    source_mean: torch.Tensor,
    target_mean: torch.Tensor
) -> float:
    """
    Computes the Frobenius norm alignment error: ||target - source @ Q||_F
    Lower is better. Perfect alignment would be 0.
    
    Args:
        source_vectors (torch.Tensor): Source vectors [N, D]
        target_vectors (torch.Tensor): Target vectors [N, D]
        Q (torch.Tensor): Rotation matrix [D, D]
        source_mean (torch.Tensor): Mean of source
        target_mean (torch.Tensor): Mean of target
        
    Returns:
        float: Frobenius norm error
    """
    # Apply alignment to all source vectors
    aligned = apply_alignment_batch(source_vectors, Q, source_mean, target_mean)
    
    # Compute Frobenius norm of difference
    error = torch.norm(target_vectors - aligned, p='fro').item()
    
    return error


def compute_alignment_quality(
    source_vectors: torch.Tensor,
    target_vectors: torch.Tensor,
    Q: torch.Tensor,
    source_mean: torch.Tensor,
    target_mean: torch.Tensor
) -> float:
    """
    Computes mean cosine similarity between aligned source and target vectors.
    Higher is better (1.0 = perfect alignment).
    
    Args:
        source_vectors (torch.Tensor): Source vectors [N, D]
        target_vectors (torch.Tensor): Target vectors [N, D]
        Q (torch.Tensor): Rotation matrix [D, D]
        source_mean (torch.Tensor): Mean of source
        target_mean (torch.Tensor): Mean of target
        
    Returns:
        float: Mean cosine similarity (0 to 1)
    """
    # Apply alignment
    aligned = apply_alignment_batch(source_vectors, Q, source_mean, target_mean)
    
    # Compute cosine similarities
    similarities = torch.nn.functional.cosine_similarity(aligned, target_vectors, dim=1)
    
    return similarities.mean().item()
