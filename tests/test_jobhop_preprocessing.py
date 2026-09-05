from __future__ import annotations

import unittest

import pandas as pd

from src.jobhop_preprocessing import clean_jobhop_frame, parse_quarter


class JobHopPreprocessingTests(unittest.TestCase):
    def test_quarter_parser(self) -> None:
        self.assertEqual(parse_quarter("Q1 2000"), 8000)
        self.assertEqual(parse_quarter("Q4 2000"), 8003)
        self.assertIsNone(parse_quarter(None))
        with self.assertRaises(ValueError):
            parse_quarter("2000-Q1")

    def test_cleaning_maps_codes_and_skips_ambiguous_pairs(self) -> None:
        frame = pd.DataFrame(
            [
                (1, "a", "Q1 2020", "Q2 2020", "Secondary school"),
                (1, "a", "Q1 2020", "Q2 2020", "Secondary school"),
                (1, "b", "Q3 2020", "Q4 2020", "Bachelor"),
                (2, "a", "Q1 2020", "Q2 2020", "Master"),
                (2, "b", "Q1 2020", "Q2 2020", "PhD"),
                (3, "unknown", "Q1 2020", "Q2 2020", "None"),
                (4, "obsolete", "Q1 2020", "Q2 2020", "None"),
            ],
            columns=[
                "resume_id",
                "matched_code",
                "start_date",
                "end_date",
                "university_level",
            ],
        )

        cleaned = clean_jobhop_frame(frame, "train", {"a": "role:a", "b": "role:b"})

        self.assertEqual(cleaned.report.exact_duplicate_rows, 1)
        self.assertEqual(cleaned.report.unknown_code_rows, 1)
        self.assertEqual(cleaned.report.unmapped_code_rows, 1)
        self.assertEqual(cleaned.report.invalid_education_rows, 0)
        self.assertEqual(cleaned.report.ambiguous_same_start_pairs, 1)
        self.assertEqual(cleaned.pair_counts[("role:a", "role:b")], 1)
        self.assertIn("Secondary", set(cleaned.frame["university_level"]))


if __name__ == "__main__":
    unittest.main()
