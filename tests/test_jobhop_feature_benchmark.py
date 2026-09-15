from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import pandas as pd

from src.jobhop_feature_benchmark import (
    assert_disjoint_splits,
    education_bucket,
    gap_bucket,
    iter_jobhop_feature_prefixes,
    tenure_bucket,
)


class JobHopFeatureBenchmarkTests(unittest.TestCase):
    def test_buckets_have_explicit_missing_values(self) -> None:
        self.assertEqual(tenure_bucket(None, 10), 0)
        self.assertEqual(tenure_bucket(10, 10), 1)
        self.assertEqual(tenure_bucket(10, 14), 4)
        self.assertEqual(tenure_bucket(10, 100), 8)
        self.assertEqual(gap_bucket(None, 10), 0)
        self.assertEqual(gap_bucket(10, 9), 1)
        self.assertEqual(gap_bucket(10, 10), 2)
        self.assertEqual(gap_bucket(10, 11), 3)
        self.assertEqual(education_bucket("Master", include_static_resume_education=False), 0)
        self.assertEqual(education_bucket("Master", include_static_resume_education=True), 4)

    def test_prefixes_skip_ambiguous_and_duplicate_rows_and_never_read_target_features(self) -> None:
        frame = pd.DataFrame(
            [
                (1, "role:a", 10, 10, 14, "Bachelor"),
                (1, "role:b", 16, 16, None, "Bachelor"),
                (1, "role:c", 16, 16, 17, "Bachelor"),  # same-quarter ambiguity
                (1, "role:b", 18, 18, 19, "Bachelor"),  # duplicate current role
                (1, "role:d", 20, 20, 22, "Bachelor"),
            ],
            columns=["resume_id", "role_id", "order_quarter", "start_quarter", "end_quarter", "university_level"],
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "split.parquet"
            frame.to_parquet(path, index=False)
            prefixes = list(iter_jobhop_feature_prefixes(path, maximum_history_length=10))
            education = list(iter_jobhop_feature_prefixes(
                path, maximum_history_length=10, include_static_resume_education=True,
            ))
        self.assertEqual([prefix.target_role_id for prefix in prefixes], ["role:b", "role:d"])
        self.assertEqual(prefixes[0].history_role_ids, ("role:a",))
        self.assertEqual(prefixes[0].context_features, (4, 0, 0))
        self.assertEqual(prefixes[1].history_role_ids, ("role:a", "role:b"))
        self.assertEqual(prefixes[1].context_features, (0, 4, 0))
        self.assertEqual(education[0].context_features[2], 3)

    def test_split_overlap_is_rejected(self) -> None:
        frame = pd.DataFrame({"resume_id": [1], "role_id": ["role:a"], "order_quarter": [1],
                              "start_quarter": [1], "end_quarter": [2], "university_level": ["None"]})
        with tempfile.TemporaryDirectory() as directory:
            left, right = Path(directory) / "left.parquet", Path(directory) / "right.parquet"
            frame.to_parquet(left, index=False)
            frame.to_parquet(right, index=False)
            with self.assertRaises(ValueError):
                assert_disjoint_splits({"train": left, "validation": right})


if __name__ == "__main__":
    unittest.main()
