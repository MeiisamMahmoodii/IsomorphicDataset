"""
Validation rules for constrained rewrites (word count, banned lemmas).

Documented for reproducibility: case-insensitive substring match on word tokens
after splitting on whitespace (no stemming).
"""

from __future__ import annotations

import re
from typing import List, Tuple

WRITER_MAX_ATTEMPTS = 5


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
    if violates_banned_words(s, banned_words):
        return False, "banned_word_present"
    return True, "ok"
