"""
Validation rules for constrained rewrites (word count, banned lemmas).

Documented for reproducibility: case-insensitive substring match on word tokens
after splitting on whitespace (no stemming).
"""

from __future__ import annotations

import re
from typing import List, Tuple

WRITER_MAX_ATTEMPTS = 5
META_OUTPUT_PATTERNS = (
    "this sentence",
    "the sentence",
    "original sentence",
    "rewritten sentence",
    "core message",
    "provided words",
    "i cannot rewrite",
    "i'm sorry",
    "as requested",
    "according to",
    "human:",
    "assistant:",
    "system:",
    "user:",
    "word version",
    "recipe for",
    "i need a recipe",
    "rephrased sentence",
    "this rephrased sentence",
    "uses fewer than",
    "maintaining the original meaning",
    "analyze the request",
    "role: strict paraphrasing engine",
    "</think>",
    "<think>",
)

# Junk / instruction-echo outputs that must never pass validation.
GARBAGE_SUBSTRINGS = (
    "</rewrite>",
    "<rewrite",
    "one sentence here",
    "now produce only",
    "human resources",
    "job satisfaction",
    "employee retention",
    "let's rewrite",
    "i'll rewrite",
    "analyze the request",
    "role: strict paraphrasing engine",
    "</think>",
    "<think>",
    "**analyze",
)


def _looks_like_garbage_output(text: str) -> bool:
    """Heuristic: XML echoes, numeric spam, unrelated boilerplate."""
    lower = text.lower()
    if any(s in lower for s in GARBAGE_SUBSTRINGS):
        return True
    # Decimal spam like "0.1 0.2 0.3 ..."
    if re.search(r"\b0\.\d+\s+0\.\d+\s+0\.\d+", lower):
        return True
    return False


def word_count(text: str) -> int:
    if not text or not str(text).strip():
        return 0
    return len(str(text).split())


def violates_banned_words(text: str, banned_words: List[str]) -> bool:
    """True if any banned word appears as a case-insensitive whole-word token."""
    lower = text.lower()
    for w in banned_words:
        w = w.strip().lower()
        if len(w) < 2:
            continue
        for m in re.finditer(r"\b" + re.escape(w) + r"\b", lower):
            return True
    return False


def rewrite_passes_constraints(
    text: str, banned_words: List[str], min_words: int, max_words: int
) -> Tuple[bool, str]:
    """
    Returns (ok, reason). Empty or non-string text fails.
    """
    if text is None:
        return False, "empty_output"
    s = str(text).strip()
    if not s:
        return False, "empty_output"
    n = word_count(s)
    if n < min_words or n > max_words:
        return False, f"length_{n}_not_in_[{min_words},{max_words}]"
    lower = s.lower()
    if any(p in lower for p in META_OUTPUT_PATTERNS):
        return False, "meta_output_present"
    if _looks_like_garbage_output(s):
        return False, "garbage_output"
    if violates_banned_words(s, banned_words):
        return False, "banned_word_present"
    return True, "ok"
