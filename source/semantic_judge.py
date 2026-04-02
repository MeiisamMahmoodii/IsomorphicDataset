"""
SemanticJudge: Single-Model Reference Validation

Uses Llama as the "Standard Meter" to validate semantic isomorphism.
Instead of comparing across different model coordinate systems,
we feed both sentences into the same "brain" (Llama) and measure
whether Llama's internal representation treats them as identical.

Metrics:
- Cosine Similarity: Directional alignment (angle between vectors)
- Wasserstein Distance: Distributional distance (physical "work" to transform)

Theory:
If two sentences generate similar hidden states in the same model,
they are semantically isomorphic (equivalent meaning, same coordinates).
"""

import torch
import torch.nn.functional as F
import numpy as np


class SemanticJudge:
    """
    Validates semantic isomorphism using a single reference model (Llama).
    
    The key insight: Even if Llama and Mistral generate different text,
    if Llama's internal representation treats both as identical,
    they are latent paraphrases.
    """
    
    def __init__(self, model, tokenizer):
        """
        Initialize the judge with a reference model.
        
        Args:
            model: Hugging Face language model (e.g., Llama-3)
            tokenizer: Corresponding tokenizer
        """
        self.model = model
        self.tokenizer = tokenizer
        self.device = model.device
    
    def get_detailed_representation(self, text):
        """
        Extracts high-fidelity mean-pooled vector for a sentence.
        
        This is the "semantic snapshot" of how the model internally
        represents the meaning of the sentence.
        
        Args:
            text (str): Input sentence
            
        Returns:
            torch.Tensor: [hidden_dim] mean-pooled vector
        """
        inputs = self.tokenizer(text, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            outputs = self.model(**inputs, output_hidden_states=True)
        
        # Extract last layer hidden states: [batch_size=1, seq_len, hidden_dim]
        last_hidden = outputs.hidden_states[-1]
        
        # Create attention mask for pooling: [batch_size=1, seq_len, 1]
        mask = inputs['attention_mask'].unsqueeze(-1).expand(last_hidden.size()).float()
        
        # Attention-Masked Mean Pooling
        # Sum all token embeddings, weighted by attention mask
        sum_embeddings = torch.sum(last_hidden * mask, dim=1)
        
        # Divide by number of valid tokens (attention mask sum)
        sum_mask = torch.clamp(mask.sum(1), min=1e-9)
        mean_pooled = sum_embeddings / sum_mask
        
        return mean_pooled.squeeze()
    
    def evaluate_isomorphism(self, text_reference, text_candidate, use_wasserstein=True):
        """
        Feeds both sentences into the reference model and measures
        if they generate similar semantic representations.
        
        Args:
            text_reference (str): Original sentence (e.g., Llama-generated)
            text_candidate (str): Candidate sentence (e.g., Mistral-generated)
            use_wasserstein (bool): Include Wasserstein distance metric
            
        Returns:
            dict: Metrics including cosine similarity and optional Wasserstein loss
        """
        # Get Llama's internal representation of both sentences
        z_reference = self.get_detailed_representation(text_reference)
        z_candidate = self.get_detailed_representation(text_candidate)
        
        metrics = {}
        
        # ====== METRIC 1: Cosine Similarity (Directional Check) ======
        # High cosine = same direction in latent space
        cos_sim = F.cosine_similarity(
            z_reference.unsqueeze(0), 
            z_candidate.unsqueeze(0)
        ).item()
        metrics['cosine_similarity'] = cos_sim
        
        # ====== METRIC 2: Euclidean Distance (Magnitude Check) ======
        # Low distance = close position in latent space
        euclidean_dist = torch.norm(z_reference - z_candidate).item()
        metrics['euclidean_distance'] = euclidean_dist
        
        # ====== METRIC 3: Wasserstein Distance (Optional) ======
        # Only compute if POT is available
        # Wasserstein measures "work" needed to transform one distribution to another
        if use_wasserstein:
            try:
                import ot
                
                # Convert to numpy for OT computation
                u = z_reference.cpu().float().numpy()
                v = z_candidate.cpu().float().numpy()
                
                # Compute 1D Wasserstein distance along each dimension
                # (simpler than full OT, but reveals magnitude mismatch)
                w_distances = []
                for i in range(len(u)):
                    # Sort each dimension and compute L1 distance
                    u_sorted = np.sort(u[i:i+1])
                    v_sorted = np.sort(v[i:i+1])
                    w_distances.append(np.abs(u_sorted - v_sorted).item())
                
                w_loss = np.mean(w_distances)
                metrics['wasserstein_distance'] = w_loss
                
            except ImportError:
                # POT not installed, use L2 distance as approximation
                metrics['wasserstein_distance'] = euclidean_dist
        
        return metrics
    
    def batch_evaluate_isomorphism(self, text_pairs, use_wasserstein=True):
        """
        Evaluate multiple sentence pairs at once.
        
        Args:
            text_pairs (list): List of (reference_text, candidate_text) tuples
            use_wasserstein (bool): Include Wasserstein metric
            
        Returns:
            list: List of metric dicts
        """
        results = []
        for ref_text, cand_text in text_pairs:
            metrics = self.evaluate_isomorphism(ref_text, cand_text, use_wasserstein)
            results.append(metrics)
        return results
    
    def apply_thresholds(self, metrics, cosine_threshold=0.85, distance_threshold=0.05):
        """
        Determine if a sentence pair passes the "gold standard" filter.
        
        Args:
            metrics (dict): Output from evaluate_isomorphism()
            cosine_threshold (float): Minimum acceptable cosine similarity
            distance_threshold (float): Maximum acceptable Euclidean distance
            
        Returns:
            dict: Verdict with pass/fail and reasoning
        """
        passed_cosine = metrics['cosine_similarity'] >= cosine_threshold
        passed_distance = metrics['euclidean_distance'] <= distance_threshold
        
        # If Wasserstein is available, also check it
        passed_wasserstein = True
        if 'wasserstein_distance' in metrics:
            passed_wasserstein = metrics['wasserstein_distance'] <= distance_threshold
        
        # All metrics must pass
        overall_passed = passed_cosine and passed_distance and passed_wasserstein
        
        verdict = {
            'passed': overall_passed,
            'cosine_passed': passed_cosine,
            'distance_passed': passed_distance,
            'wasserstein_passed': passed_wasserstein,
            'reason': []
        }
        
        if not passed_cosine:
            verdict['reason'].append(
                f"Cosine similarity {metrics['cosine_similarity']:.4f} < {cosine_threshold}"
            )
        if not passed_distance:
            verdict['reason'].append(
                f"Euclidean distance {metrics['euclidean_distance']:.4f} > {distance_threshold}"
            )
        if not passed_wasserstein and 'wasserstein_distance' in metrics:
            verdict['reason'].append(
                f"Wasserstein distance {metrics['wasserstein_distance']:.4f} > {distance_threshold}"
            )
        
        if overall_passed:
            verdict['reason'] = ["✅ PROVEN SEMANTIC ISOMORPHISM: Meaning preserved"]
        
        return verdict
    
    def print_verdict(self, metrics, verdict, text_reference, text_candidate):
        """
        Pretty-print the semantic validation results.
        
        Args:
            metrics (dict): Output from evaluate_isomorphism()
            verdict (dict): Output from apply_thresholds()
            text_reference (str): Original text
            text_candidate (str): Generated text
        """
        status = "✅ PASS" if verdict['passed'] else "❌ FAIL"
        
        print(f"\n{status} SEMANTIC JUDGE VERDICT")
        print("-" * 70)
        print(f"Reference:  {text_reference[:60]}...")
        print(f"Candidate:  {text_candidate[:60]}...")
        print("-" * 70)
        print(f"Cosine Similarity:      {metrics['cosine_similarity']:.4f}")
        print(f"Euclidean Distance:     {metrics['euclidean_distance']:.4f}")
        if 'wasserstein_distance' in metrics:
            print(f"Wasserstein Distance:   {metrics['wasserstein_distance']:.4f}")
        print("-" * 70)
        for reason in verdict['reason']:
            print(f"  {reason}")
        print()
