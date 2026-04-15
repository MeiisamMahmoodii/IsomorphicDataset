"""Unit tests for rewrite constraint validation (stdlib unittest; pytest optional)."""

import unittest

from isomorphic.generation.constraint_utils import (
    rewrite_passes_constraints,
    word_count,
    violates_banned_words,
)


class TestConstraintUtils(unittest.TestCase):
    def test_word_count(self):
        self.assertEqual(word_count("one two three"), 3)
        self.assertEqual(word_count(""), 0)

    def test_banned_whole_word(self):
        self.assertTrue(violates_banned_words("hello world", ["hello"]))
        self.assertFalse(violates_banned_words("the cat sat", ["hello"]))

    def test_rewrite_passes(self):
        ok, reason = rewrite_passes_constraints(
            "a b c d e e", ["foo"], min_words=5, max_words=10
        )
        self.assertTrue(ok)
        self.assertEqual(reason, "ok")
        bad, r = rewrite_passes_constraints("a b", ["foo"], min_words=5, max_words=10)
        self.assertFalse(bad)
        self.assertTrue(r.startswith("length_"))


if __name__ == "__main__":
    unittest.main()
