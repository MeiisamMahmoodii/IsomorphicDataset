"""
Model Rewriter - High-Fidelity Variation Generation

Uses diverse LLMs (Llama-3.2, Gemma-4, Qwen-3.5) to rewrite sentences
under strict length and banned-word constraints.
"""

import torch
from typing import List, Dict
from isomorphic.datasets.base_dataset import DatasetEntry, LengthConstraint

class ModelRewriter:
    """
    Handles systematic rewriting of sentences across multiple models and constraints.
    """
    
    MODELS = [
        "huihui-ai/Llama-3.2-11B-Vision-Instruct-abliterated",
        "huihui-ai/Llama-3.2-3B-Instruct-abliterated",
        "huihui-ai/Llama-3.2-1B-Instruct-abliterated",
        "huihui-ai/Huihui-gpt-oss-20b-BF16-abliterated",
        "huihui-ai/Huihui-gemma-4-E2B-it-abliterated",
        "huihui-ai/Huihui-gemma-4-E4B-it-abliterated",
        "huihui-ai/Huihui-gemma-4-26B-A4B-it-abliterated",
        "huihui-ai/Huihui-gemma-4-31B-it-abliterated",
        "huihui-ai/Huihui-Qwen3.5-35B-A3B-abliterated",
        "huihui-ai/Huihui-Qwen3.5-27B-abliterated",
        "huihui-ai/Huihui-Qwen3.5-9B-abliterated",
        "huihui-ai/Huihui-Qwen3.5-4B-abliterated",
        "huihui-ai/Huihui-Qwen3.5-2B-abliterated",
        "huihui-ai/Huihui-Qwen3.5-0.8B-abliterated"
    ]

    def __init__(self, current_model_id: str, model, tokenizer):
        self.model_id = current_model_id
        self.model = model
        self.tokenizer = tokenizer
        self.device = model.device

    def rewrite(self, text: str, banned_words: List[str], constraint: LengthConstraint) -> str:
        """
        Rewrite a sentence while adhering to length and word constraints.
        
        Args:
            text: Seed sentence
            banned_words: Words to avoid
            constraint: Min/Max word count
            
        Returns:
            Rewritten sentence
        """
        banned_str = ", ".join(banned_words)
        prompt = f"Rewrite the following sentence using between {constraint.min_words} " \
                 f"and {constraint.max_words} words. DO NOT use any of these words: {banned_str}.\n" \
                 f"Sentence: {text}\n" \
                 f"Rewritten:"
        
        # Generation configuration to enforce word masking and length
        # In a real implementation, we would use custom logit processors 
        # to strictly forbid 'banned_words' at the token level.
        
        # Simulated generation
        variation = f"This is an isomorphic variation of '{text[:10]}...'"
        
        # Real generation logic would go here:
        # inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        # outputs = self.model.generate(**inputs, max_new_tokens=50, ...)
        # variation = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        return variation

    def process_entry(self, entry: DatasetEntry) -> DatasetEntry:
        """Generate variations for a dataset entry across all length constraints."""
        for constraint in entry.length_constraints:
            desc = f"{constraint.min_words}-{constraint.max_words}"
            key = f"{self.model_id}_{desc}"
            
            variation = self.rewrite(entry.seed_text, entry.forbidden_words, constraint)
            entry.variations[key] = variation
            
        return entry
