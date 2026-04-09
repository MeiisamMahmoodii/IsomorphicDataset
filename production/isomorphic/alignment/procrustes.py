"""
Procrustes Alignment - Mathematical Framework

Implements SVD-based Procrustes analysis for computing optimal rotation matrices
that align different LLM latent spaces. This is the mathematical core of the isomorphism proof.

Reference: [Procrustes Analysis, 1994]
"""

import torch
import numpy as np
from typing import Tuple, Dict, Optional
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
            device: Computation device
            
        Returns:
            AlignmentResult with rotation matrix and metrics
        """
        
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


class AnchorAlignment:
    """Alignment using anchor words as reference points."""
    
    @staticmethod
    def extract_anchor_vectors(
        extractor,
        anchor_words: list,
        batch_size: int = 16
    ) -> Dict[str, torch.Tensor]:
        """
        Extract vectors for anchor words.
        
        Args:
            extractor: Vector extractor instance
            anchor_words: List of anchor words
            batch_size: Batch size for extraction
            
        Returns:
            Dictionary mapping words to vectors
        """
        anchor_vectors = {}
        
        print(f"Extracting anchor vectors for {len(anchor_words)} words...")
        
        for i, word in enumerate(anchor_words):
            if i % batch_size == 0:
                print(f"  {i}/{len(anchor_words)} words processed")
            
            vector = extractor.extract_single(word)
            anchor_vectors[word] = vector
        
        print(f"✓ Extracted {len(anchor_vectors)} anchor vectors")
        return anchor_vectors
    
    @staticmethod
    def align_via_anchors(
        source_anchors: Dict[str, torch.Tensor],
        target_anchors: Dict[str, torch.Tensor],
        device: str = "cuda"
    ) -> AlignmentResult:
        """
        Align using common anchor words.
        
        Args:
            source_anchors: Source anchor vectors
            target_anchors: Target anchor vectors
            device: Computation device
            
        Returns:
            AlignmentResult
        """
        # Find common anchors
        common_words = set(source_anchors.keys()) & set(target_anchors.keys())
        
        if len(common_words) < 5:
            raise ValueError(f"Not enough common anchors: {len(common_words)}")
        
        print(f"Using {len(common_words)} common anchor words for alignment")
        
        # Stack vectors
        X_list = [source_anchors[w] for w in common_words]
        Y_list = [target_anchors[w] for w in common_words]
        
        X = torch.stack(X_list, dim=0)
        Y = torch.stack(Y_list, dim=0)
        
        # Compute alignment
        return ProcrustesAligner.compute_rotation(X, Y, device)
