"""
Hub alignment helpers: combine pairwise Procrustes masks and compare pooling metrics.
"""

from __future__ import annotations

from typing import Dict, List, Tuple, Optional

import torch

from isomorphic.alignment.procrustes import ProcrustesAligner

POOLING_METHODS = ("mean_pooling", "last_token", "attention_weighted")


def hub_intersection_mask(
    vector_data: Dict[str, Dict[str, Dict[str, torch.Tensor]]],
    hub_name: str,
    model_names: List[str],
    dataset_name: str,
    method: str,
    threshold: float,
    device: str = "cpu",
) -> Tuple[Optional[torch.Tensor], Optional[float]]:
    """
    Intersect iterate_and_filter masks for hub vs every other model on the same rows.

    Returns (combined_bool_mask[N], last_alignment_quality) or (None, None) if skipped.
    """
    others = [m for m in model_names if m != hub_name]
    if not others:
        return None, None

    combined: Optional[torch.Tensor] = None
    last_quality: Optional[float] = None

    for other in others:
        if hub_name not in vector_data or other not in vector_data:
            continue
        if dataset_name not in vector_data[hub_name]:
            continue
        if dataset_name not in vector_data[other]:
            continue
        hub_ds = vector_data[hub_name][dataset_name]
        oth_ds = vector_data[other][dataset_name]
        if method not in hub_ds or method not in oth_ds:
            continue
        X = hub_ds[method].cpu()
        Y = oth_ds[method].cpu()
        if X.shape != Y.shape or X.shape[0] == 0:
            continue
        result, mask = ProcrustesAligner.iterate_and_filter(
            X, Y, threshold=threshold, device=device
        )
        last_quality = result.alignment_quality
        combined = mask if combined is None else (combined & mask)

    return combined, last_quality


def pick_best_pooling_method(
    method_scores: Dict[str, float]
) -> Optional[str]:
    """Higher alignment quality is better."""
    if not method_scores:
        return None
    return max(method_scores.items(), key=lambda kv: kv[1])[0]
