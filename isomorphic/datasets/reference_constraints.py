"""
Reference constraint generation for isomorphic dataset construction.

Uses a high-parameter reference model (e.g. Gemma-4-31B) to produce:
- Banned words (surface-divergence pressure)
- Length bins (5–10 and 15–20 words)
- Per-bin rewrite prompts (persisted for reproducibility)
"""

from __future__ import annotations

import torch
from typing import List, Tuple

from isomorphic.datasets.base_dataset import DatasetEntry, LengthConstraint
from isomorphic.generation.rewriter import ModelRewriter


class ReferenceConstraintBuilder:
    """
    Builds constraint rows for each dataset entry using a reference model.
    """

    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.device = model.device

    def generate_constraints(self, text: str) -> Tuple[List[str], List[LengthConstraint]]:
        """Banned words + default length bins for a seed text."""
        prompt = (
            f"Identify the top 5 most important words in this sentence: '{text}'.\n"
            f"Output only the words separated by commas."
        )

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model.generate(**inputs, max_new_tokens=20)

        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        banned_words = [w.strip().lower() for w in response.split(",") if len(w.strip()) > 2]

        if not banned_words:
            banned_words = [text.split()[0].lower()] if text.split() else ["the"]

        length_constraints = [
            LengthConstraint(min_words=5, max_words=10),
            LengthConstraint(min_words=15, max_words=20),
        ]

        return banned_words, length_constraints

    def process_entry(self, entry: DatasetEntry) -> DatasetEntry:
        """Fill forbidden words, length constraints, and persisted rewrite prompts per bin."""
        banned, lengths = self.generate_constraints(entry.seed_text)
        entry.forbidden_words = banned
        entry.length_constraints = lengths
        entry.rewrite_specs = {}
        for constraint in lengths:
            desc = f"{constraint.min_words}-{constraint.max_words}"
            entry.rewrite_specs[desc] = ModelRewriter.build_rewrite_prompt(
                entry.seed_text, banned, constraint
            )
        return entry
