import unittest

import numpy as np

from train_link_prediction_cloud import neighbour_evidence_values, source_ranking_metrics


class SourceRankingMetricsTests(unittest.TestCase):
    def test_computes_hits_and_mrr_per_source(self):
        pairs = [("a", "1"), ("a", "2"), ("b", "1"), ("b", "2")]
        labels = np.asarray([0, 1, 1, 0], dtype=np.int32)
        scores = np.asarray([0.9, 0.8, 0.7, 0.1], dtype=np.float64)

        metrics = source_ranking_metrics(pairs, labels, scores)

        self.assertEqual(metrics["sources"], 2)
        self.assertEqual(metrics["hits_at_1"], 0.5)
        self.assertEqual(metrics["hits_at_3"], 1.0)
        self.assertEqual(metrics["mrr"], 0.75)

    def test_ties_are_deterministic_by_target_id(self):
        pairs = [("a", "z"), ("a", "a")]
        labels = np.asarray([1, 0], dtype=np.int32)
        scores = np.asarray([0.5, 0.5], dtype=np.float64)

        metrics = source_ranking_metrics(pairs, labels, scores)

        self.assertEqual(metrics["hits_at_1"], 0.0)
        self.assertEqual(metrics["mrr"], 0.5)

    def test_requires_a_positive_source_group(self):
        with self.assertRaisesRegex(RuntimeError, "no positive"):
            source_ranking_metrics(
                [("a", "1")],
                np.asarray([0], dtype=np.int32),
                np.asarray([0.1], dtype=np.float64),
            )


class NeighbourEvidenceTests(unittest.TestCase):
    def test_excludes_held_out_source_transitions(self):
        values = neighbour_evidence_values(
            [("query", "target")],
            {
                "held-out": {"target": 0.9},
                "training": {"target": 0.4},
            },
            {"query": ["held-out", "training"]},
            {"held-out"},
        )

        self.assertEqual(values.tolist(), [0.4])


if __name__ == "__main__":
    unittest.main()
