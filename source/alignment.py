"""
Alignment module for cross-model latent space alignment using anchor words.
Implements Procrustes-style rotation to align different models' coordinate systems.
Uses the robust SVD-based Procrustes solver from alignment_utils.
"""

import torch
import numpy as np
from typing import Dict, List, Tuple
from .alignment_utils import (
    calculate_procrustes_rotation,
    apply_alignment,
    apply_alignment_batch,
    compute_alignment_error,
    compute_alignment_quality
)


# Comprehensive list of neutral anchor words across different categories
ANCHOR_WORDS = {
    # Physical Objects
    "objects": ["mountain", "keyboard", "chair", "water", "tree", "rock", "door", "window"],
    
    # Natural Phenomena
    "nature": ["light", "sound", "wind", "rain", "snow", "fire", "earth", "sky"],
    
    # Abstract Concepts
    "abstract": ["science", "philosophy", "mathematics", "nature", "universe", "concept", "idea", "theory"],
    
    # Common Items
    "common": ["book", "table", "pen", "paper", "bread", "fruit", "flower", "animal"],
    
    # Scientific Terms
    "scientific": ["number", "energy", "force", "motion", "atom", "cell", "element", "compound"],
    
    # Temporal/Spatial
    "temporal_spatial": ["time", "space", "distance", "moment", "direction", "location", "dimension", "point"],
    
    # Colors and Properties
    "properties": ["color", "shape", "size", "weight", "texture", "bright", "dark", "smooth"],
    
    # Language and Communication
    "language": ["language", "word", "sentence", "meaning", "symbol", "letter", "sound", "voice"],
    
    # Basic Verbs (nominalized)
    "actions": ["motion", "rotation", "movement", "creation", "destruction", "growth", "change", "transformation"],
    
    # Numbers and Quantities
    "quantities": ["one", "two", "three", "number", "amount", "quantity", "zero", "infinite"]
}


def get_all_anchor_words() -> List[str]:
    """
    Returns a flattened list of all anchor words from all categories.
    
    Returns:
        List[str]: All unique anchor words
    """
    all_words = []
    for category, words in ANCHOR_WORDS.items():
        all_words.extend(words)
    return list(set(all_words))  # Remove duplicates


def extract_anchor_vectors(generator, method: str = "mean") -> Dict[str, torch.Tensor]:
    """
    Extracts latent vectors for all anchor words using a specified method.
    
    Args:
        generator: ConceptGenerator instance
        method (str): One of "mean", "last", or "hybrid"
        
    Returns:
        Dict[str, torch.Tensor]: Mapping of anchor word -> latent vector
    """
    anchor_vectors = {}
    anchor_words = get_all_anchor_words()
    
    print(f"Extracting anchor vectors using {method} method...")
    
    for word in anchor_words:
        try:
            if method == "mean":
                vector = generator.get_latent_vector(word)
            elif method == "last":
                vector = generator.get_last_token_vector(word)
            elif method == "hybrid":
                vector = generator.get_hybrid_vector(word)
            else:
                raise ValueError(f"Unknown method: {method}")
            
            anchor_vectors[word] = vector
        except Exception as e:
            print(f"  ⚠️ Failed to extract vector for '{word}': {e}")
            continue
    
    print(f"✅ Successfully extracted {len(anchor_vectors)} anchor vectors")
    return anchor_vectors


def stack_anchor_vectors(anchor_vectors: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, List[str]]:
    """
    Stacks anchor vectors into a matrix for alignment computation.
    
    Args:
        anchor_vectors (Dict[str, torch.Tensor]): Anchor word -> vector mapping
        
    Returns:
        Tuple[torch.Tensor, List[str]]: (stacked matrix [N, D], word order list)
    """
    words = list(anchor_vectors.keys())
    vectors = [anchor_vectors[word] for word in words]
    stacked = torch.stack(vectors, dim=0)  # [N, hidden_dim]
    return stacked, words


def compute_procrustes_rotation(X: torch.Tensor, Y: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Wrapper around the robust SVD-based Procrustes solver.
    Computes the optimal orthogonal rotation matrix using Procrustes analysis.
    Finds Q such that ||Y - XQ|| is minimized.
    
    Args:
        X (torch.Tensor): Source space anchors [N, D]
        Y (torch.Tensor): Target space anchors [N, D]
        
    Returns:
        Tuple containing:
            - Q (torch.Tensor): Rotation matrix [D, D]
            - source_mean (torch.Tensor): Mean of source
            - target_mean (torch.Tensor): Mean of target
    """
    Q, source_mean, target_mean = calculate_procrustes_rotation(X, Y)
    return Q, source_mean, target_mean


def align_latent_spaces(
    anchors_source: Dict[str, torch.Tensor],
    anchors_target: Dict[str, torch.Tensor]
) -> Tuple[torch.Tensor, float, torch.Tensor, torch.Tensor]:
    """
    Computes alignment rotation from source to target latent space.
    Uses the robust Procrustes SVD solver with data centering.
    
    Args:
        anchors_source (Dict): Source model anchor vectors
        anchors_target (Dict): Target model anchor vectors
        
    Returns:
        Tuple containing:
            - Q (torch.Tensor): Rotation matrix
            - alignment_quality (float): Mean cosine similarity after alignment
            - source_mean (torch.Tensor): Mean for transforming new vectors
            - target_mean (torch.Tensor): Mean for transforming new vectors
    """
    # Get common anchor words
    common_words = set(anchors_source.keys()) & set(anchors_target.keys())
    
    if len(common_words) < 2:
        raise ValueError(f"Not enough common anchors: {len(common_words)}")
    
    print(f"Using {len(common_words)} common anchor words for alignment")
    
    # Stack vectors for common anchors
    X_list = [anchors_source[w] for w in common_words]
    Y_list = [anchors_target[w] for w in common_words]
    
    X = torch.stack(X_list, dim=0)
    Y = torch.stack(Y_list, dim=0)
    
    # Compute Procrustes rotation with centering
    Q, source_mean, target_mean = compute_procrustes_rotation(X, Y)
    
    # Compute alignment quality using the robust metric
    alignment_quality = compute_alignment_quality(X, Y, Q, source_mean, target_mean)
    
    print(f"✅ Rotation matrix computed. Alignment quality: {alignment_quality:.4f}")
    
    return Q, alignment_quality, source_mean, target_mean


def apply_rotation_to_vectors(
    vectors: Dict[str, torch.Tensor],
    Q: torch.Tensor,
    source_mean: torch.Tensor,
    target_mean: torch.Tensor
) -> Dict[str, torch.Tensor]:
    """
    Applies rotation matrix and centering transformation to vectors.
    Uses the Procrustes transformation: (vec - source_mean) @ Q + target_mean
    
    Args:
        vectors (Dict[str, torch.Tensor]): Vector mapping to transform
        Q (torch.Tensor): Rotation matrix [D, D]
        source_mean (torch.Tensor): Source space mean [D]
        target_mean (torch.Tensor): Target space mean [D]
        
    Returns:
        Dict[str, torch.Tensor]: Transformed vectors
    """
    transformed = {}
    for key, vector in vectors.items():
        transformed[key] = apply_alignment(vector, Q, source_mean, target_mean)
    
    return transformed


def compute_alignment_metrics(
    vectors_aligned: Dict[str, torch.Tensor],
    vectors_target: Dict[str, torch.Tensor]
) -> Dict[str, float]:
    """
    Computes metrics comparing aligned source vectors to target vectors.
    
    Args:
        vectors_aligned (Dict): Aligned source vectors
        vectors_target (Dict): Target vectors
        
    Returns:
        Dict[str, float]: Metrics including mean similarity and variance
    """
    common_keys = set(vectors_aligned.keys()) & set(vectors_target.keys())
    
    similarities = []
    for key in common_keys:
        sim = torch.nn.functional.cosine_similarity(
            vectors_aligned[key].unsqueeze(0),
            vectors_target[key].unsqueeze(0)
        ).item()
        similarities.append(sim)
    
    metrics = {
        "mean_similarity": np.mean(similarities),
        "std_similarity": np.std(similarities),
        "min_similarity": np.min(similarities),
        "max_similarity": np.max(similarities),
        "num_comparisons": len(similarities)
    }
    
    return metrics
