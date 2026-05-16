from __future__ import annotations

import unittest

from backend.matching import adjusted_total_score, camelot_key_distance, key_label_to_camelot, parse_camelot, total_rank


class CamelotMatchingTests(unittest.TestCase):
    def test_same_key(self) -> None:
        result = camelot_key_distance("8A", "8A")

        self.assertEqual(result["relation"], "same")
        self.assertEqual(result["score"], 100)

    def test_adjacent_keys_include_wraparound(self) -> None:
        for source, target in [("8A", "7A"), ("8A", "9A"), ("1A", "12A"), ("12B", "1B")]:
            with self.subTest(source=source, target=target):
                result = camelot_key_distance(source, target)
                self.assertEqual(result["relation"], "adjacent")
                self.assertGreaterEqual(result["score"], 85)

    def test_relative_major_minor(self) -> None:
        result = camelot_key_distance("8A", "8B")

        self.assertEqual(result["relation"], "relative_major_minor")
        self.assertGreaterEqual(result["score"], 90)

    def test_clash(self) -> None:
        for source, target in [("8A", "2B"), ("3A", "10B")]:
            with self.subTest(source=source, target=target):
                result = camelot_key_distance(source, target)
                self.assertEqual(result["relation"], "clash")
                self.assertLess(result["score"], 35)

    def test_invalid_input_is_unknown(self) -> None:
        for value in ["0A", "13A", "8C", "A8", "", None]:
            with self.subTest(value=value):
                result = camelot_key_distance(value, "8A")
                self.assertEqual(result["relation"], "unknown")
                self.assertLess(result["score"], 85)

    def test_traditional_key_mapping(self) -> None:
        self.assertEqual(key_label_to_camelot("A minor"), "8A")
        self.assertEqual(key_label_to_camelot("C major"), "8B")
        self.assertEqual(key_label_to_camelot("E minor"), "9A")
        self.assertEqual(key_label_to_camelot("G major"), "9B")
        self.assertEqual(key_label_to_camelot("Bb major"), "6B")
        self.assertEqual(key_label_to_camelot("Ab minor"), "1A")

    def test_strict_parse(self) -> None:
        self.assertEqual(parse_camelot("12b"), (12, "B"))
        self.assertIsNone(parse_camelot("13A"))

    def test_low_energy_match_cannot_be_perfect_even_with_same_key_and_bpm(self) -> None:
        key_eval = {"score": 100, "relation": "same"}
        bpm_eval = {"score": 100}
        energy_eval = {"score": 50.4}
        structure_eval = {"score": 96}
        raw = 92.2

        adjusted = adjusted_total_score(raw, key_eval, bpm_eval, energy_eval, structure_eval)

        self.assertEqual(adjusted, 89.0)
        self.assertEqual(total_rank(adjusted, key_eval, bpm_eval, energy_eval, structure_eval), "recommended")


if __name__ == "__main__":
    unittest.main()
