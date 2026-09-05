from __future__ import annotations

import unittest
from collections import Counter

import networkx as nx

from evaluation.transition_metrics import (
    evaluate_prediction_map,
    evaluate_transition_predictions,
    transition_predictions,
)
from src.karrierewege_preprocessing import SplitQualityReport, TransitionAggregate


class TransitionMetricTests(unittest.TestCase):
    def test_weighted_hits_mrr_and_missing_source_behavior(self) -> None:
        graph = nx.MultiDiGraph()
        for node_id, title in (("a", "Role A"), ("b", "Role B"), ("c", "Role C"), ("d", "Role D")):
            graph.add_node(node_id, type="role", source="esco", title=title)
        graph.add_edge(
            "a",
            "b",
            key="karrierewege:TRANSITIONS_TO",
            relation="TRANSITIONS_TO",
            source="karrierewege",
            probability=0.7,
            count=7,
        )
        graph.add_edge(
            "a",
            "c",
            key="karrierewege:TRANSITIONS_TO",
            relation="TRANSITIONS_TO",
            source="karrierewege",
            probability=0.3,
            count=3,
        )
        held_out = TransitionAggregate(
            split="test",
            pair_counts=Counter({("role a", "role b"): 2, ("role a", "role c"): 1, ("role d", "role b"): 1}),
            source_totals=Counter({"role a": 3, "role d": 1}),
            report=SplitQualityReport(split="test", source_file="fixture.csv"),
        )

        metrics = evaluate_transition_predictions(graph, held_out)
        map_metrics = evaluate_prediction_map(transition_predictions(graph), held_out)

        self.assertEqual(metrics["observations"], 4)
        self.assertEqual(metrics, map_metrics)
        self.assertAlmostEqual(metrics["source_role_coverage"], 0.5)
        self.assertAlmostEqual(metrics["observation_coverage"], 0.75)
        self.assertAlmostEqual(metrics["hits_at_1"], 0.5)
        self.assertAlmostEqual(metrics["hits_at_3"], 0.75)
        self.assertAlmostEqual(metrics["mrr"], 0.625)
        expected_source_ndcg = (2 + 1 / (3 ** 0.5)) / (2 + 1 / (3 ** 0.5))
        self.assertAlmostEqual(metrics["ndcg_at_3"], 0.75 * expected_source_ndcg)
        self.assertAlmostEqual(metrics["macro_ndcg_at_3"], 0.5 * expected_source_ndcg)

    def test_training_split_is_rejected(self) -> None:
        aggregate = TransitionAggregate(
            split="train",
            pair_counts=Counter(),
            source_totals=Counter(),
            report=SplitQualityReport(split="train", source_file="fixture.csv"),
        )
        with self.assertRaises(ValueError):
            evaluate_transition_predictions(nx.MultiDiGraph(), aggregate)


if __name__ == "__main__":
    unittest.main()
