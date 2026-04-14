"""
SemanticJudge: Single-Model Reference Validation

Uses a reference model (e.g. Llama) as the "Standard Meter" to validate semantic isomorphism.
Instead of comparing across different model coordinate systems, we feed both sentences 
into the same model and measure whether the internal representations are equivalent.
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict


class SemanticJudge:
    """
    Validates semantic isomorphism using a single reference model.
    """
    
    def __init__(self, model, tokenizer):
        """
        Initialize the judge with a reference model.
        
        Args:
            model: Hugging Face language model
            tokenizer: Corresponding tokenizer
        """
        self.model = model
        self.tokenizer = tokenizer
        self.device = model.device
    
    def get_detailed_representation(self, text):
        """
        Extracts high-fidelity mean-pooled vector for a sentence.
        """
        inputs = self.tokenizer(text, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            outputs = self.model(**inputs, output_hidden_states=True)
        
        # Extract last layer hidden states: [batch_size=1, seq_len, hidden_dim]
        last_hidden = outputs.hidden_states[-1]
        
        # Create attention mask for pooling
        mask = inputs['attention_mask'].unsqueeze(-1).expand(last_hidden.size()).float()
        
        # Attention-Masked Mean Pooling
        sum_embeddings = torch.sum(last_hidden * mask, dim=1)
        sum_mask = torch.clamp(mask.sum(1), min=1e-9)
        mean_pooled = sum_embeddings / sum_mask
        
        return mean_pooled.squeeze()
    
    def evaluate_isomorphism(self, text_reference: str, text_candidate: str, use_wasserstein: bool = True) -> Dict[str, float]:
        """
        Validation in the Reference 'Universal Meter' Model.
        Calculates how similar two texts 'feel' to a single large brain.
        """
        z_reference = self.get_detailed_representation(text_reference)
        z_candidate = self.get_detailed_representation(text_candidate)
        
        metrics = {}
        
        # 1. Latent Cosine Similarity
        cos_sim = F.cosine_similarity(
            z_reference.unsqueeze(0), 
            z_candidate.unsqueeze(0)
        ).item()
        metrics['reference_cosine_similarity'] = cos_sim
        
        # 2. Wasserstein Distance
        # In a single-model latent space, Wasserstein measures the structural 
        # semantic shift between two concepts.
        if use_wasserstein:
            try:
                # 1D Wasserstein approximation: L1 distance of sorted feature distributions
                u_sorted = torch.sort(z_reference).values
                v_sorted = torch.sort(z_candidate).values
                w_dist = torch.mean(torch.abs(u_sorted - v_sorted)).item()
                metrics['wasserstein_distance'] = w_dist
            except Exception:
                metrics['wasserstein_distance'] = torch.norm(z_reference - z_candidate).item()
        
        return metrics
    
    def apply_thresholds(self, metrics, cosine_threshold=0.85, distance_threshold=0.05):
        """
        Determine if a sentence pair passes the isomorphism filter.
        """
        passed_cosine = metrics['cosine_similarity'] >= cosine_threshold
        passed_distance = metrics['euclidean_distance'] <= distance_threshold
        
        passed_wasserstein = True
        if 'wasserstein_distance' in metrics:
            passed_wasserstein = metrics['wasserstein_distance'] <= distance_threshold
        
        overall_passed = passed_cosine and passed_distance and passed_wasserstein
        
        verdict = {
            'passed': overall_passed,
            'cosine_passed': passed_cosine,
            'distance_passed': passed_distance,
            'wasserstein_passed': passed_wasserstein,
            'reason': []
        }
        
        if not overall_passed:
            if not passed_cosine:
                verdict['reason'].append(f"Cosine similarity {metrics['cosine_similarity']:.4f} < {cosine_threshold}")
            if not passed_distance:
                verdict['reason'].append(f"Euclidean distance {metrics['euclidean_distance']:.4f} > {distance_threshold}")
            if not passed_wasserstein and 'wasserstein_distance' in metrics:
                verdict['reason'].append(f"Wasserstein distance {metrics['wasserstein_distance']:.4f} > {distance_threshold}")
        else:
            verdict['reason'] = ["✅ PROVEN SEMANTIC ISOMORPHISM"]
        
        return verdict
