from __future__ import annotations

import unittest

from kaggle_research_agent.trial_memory_card import MAX_TEXT_CHARS, _compact_text


class TrialMemoryCardTextTest(unittest.TestCase):
    def test_compact_text_preserves_cjk_emoji_and_symbols(self):
        value = "客室 등급 Cabin★ 특성 😊 Δscore≤0.01"

        compact = _compact_text(value)

        self.assertEqual(value, compact)

    def test_compact_text_replaces_control_characters_and_keeps_length_cap(self):
        compact = _compact_text("feature\x00name\tvalue\n" + ("가" * 300))

        self.assertNotIn("\x00", compact)
        self.assertNotIn("\t", compact)
        self.assertNotIn("\n", compact)
        self.assertLessEqual(len(compact), MAX_TEXT_CHARS)
        self.assertTrue(compact.endswith("..."))


if __name__ == "__main__":
    unittest.main()
