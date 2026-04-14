"""
Procrustes Alignment - Mathematical Framework

Implements SVD-based Procrustes analysis for computing optimal rotation matrices
that align different LLM latent spaces. This is the mathematical core of the isomorphism proof.

Reference: [Procrustes Analysis, 1994]
"""

import torch
import numpy as np
from typing import Tuple, Dict, Optional, List
from dataclasses import dataclass


@dataclass
class AlignmentResult:
    """Results from Procrustes alignment."""
    rotation_matrix: torch.Tensor  # Q
    source_mean: torch.Tensor
    target_mean: torch.Tensor
    alignment_quality: float
    orthogonality_error: float
    source_variance: float
    target_variance: float


class ProcrustesAligner:
    """SVD-based Procrustes solver for optimal rotation computation."""
    
    @staticmethod
    def compute_rotation(
        source_vectors: torch.Tensor,
        target_vectors: torch.Tensor,
        device: str = "cuda"
    ) -> AlignmentResult:
        """
        Compute optimal rotation matrix Q to align source to target space.
        
        Solves: minimize ||Y - X @ Q.T||_F² subject to Q orthogonal
        
        Args:
            source_vectors: Source latent space vectors [N, D]
            target_vectors: Target latent space vectors [N, D]
            device: Computation device (defaults to 'cpu' for stability)
            
        Returns:
            AlignmentResult with rotation matrix and metrics
        """
        if device is None:
            device = "cpu"
        
        # Transfer to device for computation
        X = source_vectors.to(device).float()
        Y = target_vectors.to(device).float()
        
        # Step 1: Center data for numerical stability
        source_mean = X.mean(dim=0)
        target_mean = Y.mean(dim=0)
        
        X_centered = X - source_mean
        Y_centered = Y - target_mean
        
        # Step 2: Compute SVD of cross-covariance
        # U, S, V^T = SVD(Y.T @ X)
        U, S, Vh = torch.linalg.svd(torch.matmul(Y_centered.T, X_centered))
        
        # Step 3: Compute rotation matrix
        # Q = U @ V.T
        Q = torch.matmul(U, Vh)
        
        # Step 4: Verify orthogonality (Q @ Q.T should be I)
        QQt = torch.matmul(Q, Q.T)
        I = torch.eye(Q.shape[0], device=device, dtype=Q.dtype)
        orthogonality_error = torch.norm(QQt - I).item()
        
        # Step 5: Compute alignment quality
        X_aligned = torch.matmul(X_centered, Q.T)
        alignment_quality = (torch.nn.functional.cosine_similarity(X_aligned, Y_centered).mean()).item()
        
        # Step 6: Compute variance
        source_var = (X_centered ** 2).sum().item()
        target_var = (Y_centered ** 2).sum().item()
        
        return AlignmentResult(
            rotation_matrix=Q.cpu(),
            source_mean=source_mean.cpu(),
            target_mean=target_mean.cpu(),
            alignment_quality=alignment_quality,
            orthogonality_error=orthogonality_error,
            source_variance=source_var,
            target_variance=target_var,
        )
    
    @staticmethod
    def apply_transformation(
        vector: torch.Tensor,
        rotation_matrix: torch.Tensor,
        source_mean: torch.Tensor,
        target_mean: torch.Tensor
    ) -> torch.Tensor:
        """
        Apply computed Procrustes transformation to a vector.
        
        Transformation: v' = (v - source_mean) @ Q.T + target_mean
        """
        device = rotation_matrix.device
        v = vector.to(device).float()
        
        # Center in source space
        centered = v - source_mean.to(device)
        
        # Apply rotation
        rotated = torch.matmul(rotation_matrix.T.to(device), centered)
        
        # Translate to target space
        aligned = rotated + target_mean.to(device)
        
        return aligned.cpu()
    
    @staticmethod
    def apply_transformation_batch(
        vectors: torch.Tensor,
        rotation_matrix: torch.Tensor,
        source_mean: torch.Tensor,
        target_mean: torch.Tensor
    ) -> torch.Tensor:
        """Apply transformation to batch of vectors."""
        device = rotation_matrix.device
        V = vectors.to(device).float()
        
        # Center
        centered = V - source_mean.to(device).unsqueeze(0)
        
        # Rotate
        rotated = torch.matmul(centered, rotation_matrix.to(device))
        
        # Translate
        aligned = rotated + target_mean.to(device).unsqueeze(0)
        
        return aligned.cpu()


    @staticmethod
    def iterate_and_filter(
        source_vectors: torch.Tensor,
        target_vectors: torch.Tensor,
        threshold: float = 0.98,
        max_iterations: int = 10,
        device: str = "cpu"
    ) -> Tuple[AlignmentResult, torch.Tensor]:
        """
        Iteratively remove the worst-aligning samples until threshold is reached.
        
        Args:
            source_vectors: [N, D]
            target_vectors: [N, D]
            threshold: Target mean cosine similarity (e.g. 0.98)
            max_iterations: Max cleanup cycles
            
        Returns:
            Tuple of (final AlignmentResult, mask of kept indices)
        """
        N = source_vectors.shape[0]
        mask = torch.ones(N, dtype=torch.bool)
        
        current_result = None
        for i in range(max_iterations):
            X = source_vectors[mask]
            Y = target_vectors[mask]
            
            current_result = ProcrustesAligner.compute_rotation(X, Y, device)
            print(f"Iteration {i}: Quality={current_result.alignment_quality:.4f}, N={mask.sum().item()}")
            
            if current_result.alignment_quality >= threshold or mask.sum() < 5:
                break
                
            # Error per sample
            X_aligned = torch.matmul(X - current_result.source_mean.to(device), current_result.rotation_matrix.to(device))
            Y_centered = Y - current_result.target_mean.to(device)
            similarities = torch.nn.functional.cosine_similarity(X_aligned, Y_centered)
            
            # Remove worst 5% of CURRENT samples or those below mean similarity
            worst_threshold = torch.quantile(similarities, 0.05)
            batch_mask = similarities > worst_threshold
            
            # Map back to global mask
            indices = torch.where(mask)[0]
            mask[indices[~batch_mask]] = False
            
        return current_result, mask

    @staticmethod
    def prune_max_activations(vectors: torch.Tensor, top_k_dimensions: int = 5) -> torch.Tensor:
        """
        PruningMaxAct: Remove dimensions that exhibit the highest outlier activations.
        Often these dimensions represent model-specific biases ('noise') that hinder isomorphism.
        
        Args:
            vectors: [N, D]
            top_k_dimensions: Number of dimensions to zero out
            
        Returns:
            Pruned vectors [N, D]
        """
        # Find dimensions with highest max activations across the batch
        max_vals = torch.max(torch.abs(vectors), dim=0).values
        _, noisy_dims = torch.topk(max_vals, top_k_dimensions)
        
        pruned_vectors = vectors.clone()
        pruned_vectors[:, noisy_dims] = 0.0
        
        print(f"Pruning Max Activations: Zeroed out dims {noisy_dims.tolist()}")
        return pruned_vectors


class AnchorAlignment:
    """Alignment using anchor words as reference points."""
    
    @staticmethod
    def extract_anchor_vectors(
        extractor,
        anchor_words: list,
        batch_size: int = 16,
        methods: List[str] = ["mean_pooling"]
    ) -> Dict[str, Dict[str, torch.Tensor]]:
        """
        Extract vectors for anchor words using multiple methods.
        
        Returns:
            Nested dict: {method: {word: vector}}
        """
        results = {m: {} for m in methods}
        print(f"Extracting anchor vectors for {len(anchor_words)} words...")
        
        for word in anchor_words:
            multi_vectors = extractor.extract_all_methods(word)
            for m in methods:
                if m in multi_vectors:
                    results[m][word] = multi_vectors[m]
                    
        return results
    
    @staticmethod
    def align_via_anchors(
        source_anchors: Dict[str, torch.Tensor],
        target_anchors: Dict[str, torch.Tensor],
        device: str = "cuda",
        iterative_threshold: Optional[float] = None
    ) -> AlignmentResult:
        """
        Align using common anchor words.
        """
        common_words = list(set(source_anchors.keys()) & set(target_anchors.keys()))
        
        if len(common_words) < 5:
            raise ValueError(f"Not enough common anchors: {len(common_words)}")
        
        X = torch.stack([source_anchors[w] for w in common_words], dim=0)
        Y = torch.stack([target_anchors[w] for w in common_words], dim=0)
        
        if iterative_threshold:
            result, _ = ProcrustesAligner.iterate_and_filter(X, Y, threshold=iterative_threshold, device=device)
            return result
            
        return ProcrustesAligner.compute_rotation(X, Y, device)
