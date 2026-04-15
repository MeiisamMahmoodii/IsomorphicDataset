"""
Reference constraint generation for isomorphic dataset construction.

Uses a high-parameter reference model (e.g. Gemma-4-31B) to produce:
- Banned words (surface-divergence pressure)
- Length bins (5–10 and 15–20 words)
- Per-bin rewrite prompts (persisted for reproducibility)
"""

from __future__ import annotations

import re
import torch
from typing import List, Tuple

from isomorphic.datasets.base_dataset import DatasetEntry, LengthConstraint
from isomorphic.generation.rewriter import ModelRewriter
from isomorphic.utils.device_utils import get_model_input_device


class ReferenceConstraintBuilder:
    COMMON_KEYWORD_INSTRUCTION = (
        "You are a strict keyword extractor for constrained paraphrasing.\n"
        "Return only comma-separated keywords with no extra text.\n"
    )

    """
    Builds constraint rows for each dataset entry using a reference model.
    """

    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.device = get_model_input_device(model)

    @staticmethod
    def _normalize_word(w: str) -> str:
        return re.sub(r"[^a-z0-9]", "", w.lower())

    def _extract_seed_terms(self, text: str) -> List[str]:
        terms = []
        for raw in re.findall(r"[A-Za-z0-9']+", text):
            n = self._normalize_word(raw)
            if len(n) >= 3:
                terms.append(n)
        # preserve order while deduplicating
        seen = set()
        out = []
        for t in terms:
            if t not in seen:
                out.append(t)
                seen.add(t)
        return out

    def generate_constraints(self, text: str) -> Tuple[List[str], List[LengthConstraint]]:
        """Banned words (3-5 key words only) + default length bins for a seed text."""
        prompt = (
            f"{self.COMMON_KEYWORD_INSTRUCTION}"
            "You are building lexical constraints for paraphrasing.\n"
            "Task: find the 5 most important content words from the sentence.\n"
            "Rules:\n"
            "- Return ONLY a comma-separated list\n"
            "- Use words from the sentence itself\n"
            "- No numbering, no explanation, no extra text\n"
            f"Sentence: {text}\n"
            "Keywords:"
        )

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        if "input_ids" in inputs:
            inputs["input_ids"] = inputs["input_ids"].long()
        if "attention_mask" in inputs:
            inputs["attention_mask"] = inputs["attention_mask"].long()
        with torch.no_grad():
            outputs = self.model.generate(**inputs, max_new_tokens=32, do_sample=False)

        # Decode only generated continuation (avoid prompt echo artifacts).
        generated = outputs[0][inputs.input_ids.shape[-1] :]
        response = self.tokenizer.decode(generated, skip_special_tokens=True)

        seed_terms = self._extract_seed_terms(text)
        seed_set = set(seed_terms)
        parsed = []
        for raw in re.split(r"[,\n;|]+", response):
            w = self._normalize_word(raw.strip())
            if len(w) < 3:
                continue
            # Force banned keywords to come from source sentence.
            if w not in seed_set:
                continue
            parsed.append(w)

        banned_words = []
        seen = set()
        for w in parsed:
            if w not in seen:
                banned_words.append(w)
                seen.add(w)
            if len(banned_words) >= 5:
                break

        # Enforce: banned list contains ONLY key words from the sentence,
        # with target cardinality 3-5 whenever possible.
        if len(banned_words) < 3:
            for w in seed_terms:
                if w not in seen:
                    banned_words.append(w)
                    seen.add(w)
                if len(banned_words) >= 3:
                    break

        if len(banned_words) > 5:
            banned_words = banned_words[:5]

        if not banned_words:
            # Extremely rare fallback for empty/invalid seed text.
            banned_words = []

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
