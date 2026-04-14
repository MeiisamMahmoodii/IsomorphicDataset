"""
Set-ConCA Preprocessor - Dataset Constraint Generation

Uses a high-parameter model (e.g. Gemma-4-31B) to generate semantic constraints:
1. Banned Words (words LLMs should avoid to force semantic divergence)
2. Sentence Length Constraints (5-10 words, 15-20 words)
"""

import torch
from typing import List, Tuple
from isomorphic.datasets.base_dataset import DatasetEntry, LengthConstraint

class SetConCAPreprocessor:
    """
    Generates semantic constraints for a dataset using a reference model.
    """
    
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.device = model.device
        
    def generate_constraints(self, text: str) -> Tuple[List[str], List[LengthConstraint]]:
        """
        Generates banned words and defines length constraints for a seed text.
        """
        # 1. Generate Banned Words via Keyword Extraction & Proximity
        # In a production setting, this uses an NLP model to find semantic neighbors.
        # For our abliterated research, we use the reference model to identify 
        # words it would naturally use, then ban them to force divergence.
        
        prompt = f"Identify the top 5 most important words in this sentence: '{text}'.\n" \
                 f"Output only the words separated by commas."
        
        # Real generation logic (using self.model)
        # Using a simple prompt for the 31B reference model
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model.generate(**inputs, max_new_tokens=20)
        
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        # Simple extraction from model response
        banned_words = [w.strip().lower() for w in response.split(",") if len(w.strip()) > 2]
        
        # Ensure fallback
        if not banned_words:
            banned_words = [text.split()[0].lower()] if text.split() else ["the"]

        # 2. Define High-Fidelity Length Constraints
        length_constraints = [
            LengthConstraint(min_words=5, max_words=10),
            LengthConstraint(min_words=15, max_words=20)
        ]
        
        return banned_words, length_constraints

    def process_entry(self, entry: DatasetEntry) -> DatasetEntry:
        """Apply constraints to a dataset entry."""
        banned, lengths = self.generate_constraints(entry.seed_text)
        entry.forbidden_words = banned
        entry.length_constraints = lengths
        return entry
