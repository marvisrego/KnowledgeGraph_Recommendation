from __future__ import annotations

import tempfile
import unittest
from collections import Counter
from pathlib import Path

import networkx as nx
import pandas as pd

from src.karrierewege_preprocessing import (
    TRANSITION_EDGE_KEY,
    SplitQualityReport,
    TransitionAggregate,
    add_transition_edges,
    aggregate_split_transitions,
    build_esco_title_index,
    discover_split_files,
    transition_rows,
)


def _graph() -> nx.MultiDiGraph:
    graph = nx.MultiDiGraph()
    for node_id, title in (("esco:a", "Role A"), ("esco:b", "Role B"), ("esco:c", "Role C")):
        graph.add_node(node_id, type="role", source="esco", title=title)
    return graph


class KarrierewegePreprocessingTests(unittest.TestCase):
    def test_discovers_inconsistently_spelled_split_files(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            for name in ("karierewege_train.csv", "karierewege_validation.csv", "karrierewege_test.csv"):
                (root / name).write_text(
                    "_id,experience_order,preferredLabel_en\n", encoding="utf-8"
                )
            splits = discover_split_files(root)
            self.assertEqual(set(splits), {"train", "validation", "test"})

    def test_cleans_trajectories_and_uses_unfiltered_probability_denominator(self) -> None:
        records = [
            ("p1", 0, "Role A"),
            ("p1", 1, "Role A"),
            ("p1", 2, "Role B"),
            ("p2", 0, "Role A"),
            ("p2", 0, "Role A"),
            ("p2", 1, "Role B"),
            ("p3", 0, "Role A"),
            ("p3", 0, "Role C"),
            ("p3", 1, "Role B"),
            ("p4", 0, "Role A"),
            ("p4", 2, "Role B"),
            ("p5", 0, "Role A"),
            ("p5", 1, "Role C"),
        ]
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "karierewege_train.csv"
            pd.DataFrame(records, columns=["_id", "experience_order", "preferredLabel_en"]).to_csv(
                path, index=False
            )
            graph = _graph()
            index = build_esco_title_index(graph)
            aggregate = aggregate_split_transitions(path, "train", index, chunk_size=3)

        self.assertEqual(aggregate.report.trajectories, 5)
        self.assertEqual(aggregate.report.exact_duplicate_rows, 1)
        self.assertEqual(aggregate.report.ambiguous_positions, 1)
        self.assertEqual(aggregate.report.nonconsecutive_pairs, 1)
        self.assertEqual(aggregate.report.self_transitions, 1)
        self.assertEqual(aggregate.pair_counts[("role a", "role b")], 2)
        self.assertEqual(aggregate.pair_counts[("role a", "role c")], 1)

        rows = transition_rows(aggregate, index, min_support=2)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["count"], 2)
        self.assertEqual(rows[0]["source_total"], 3)
        self.assertAlmostEqual(rows[0]["probability"], 2 / 3)

    def test_transition_edges_are_idempotent_and_parallel_to_taxonomy(self) -> None:
        graph = _graph()
        graph.add_edge(
            "esco:a", "esco:b", key="esco:BROADER_THAN", relation="BROADER_THAN", source="esco"
        )
        row = {
            "src": "esco:a",
            "dst": "esco:b",
            "relation": "TRANSITIONS_TO",
            "source": "karrierewege",
            "count": 7,
            "probability": 0.5,
            "source_total": 14,
            "split": "train",
            "min_support": 5,
        }

        self.assertEqual(add_transition_edges(graph, [row]), 1)
        self.assertEqual(add_transition_edges(graph, [row]), 1)
        self.assertTrue(graph.has_edge("esco:a", "esco:b", key="esco:BROADER_THAN"))
        self.assertTrue(graph.has_edge("esco:a", "esco:b", key=TRANSITION_EDGE_KEY))
        self.assertEqual(graph.number_of_edges("esco:a", "esco:b"), 2)

    def test_invalid_row_tolerance_and_held_out_edge_guard(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "karrierewege_test.csv"
            pd.DataFrame(
                [("p1", 0, "Role A"), ("p1", "not-an-order", "Role B")],
                columns=["_id", "experience_order", "preferredLabel_en"],
            ).to_csv(path, index=False)
            index = build_esco_title_index(_graph())
            with self.assertRaisesRegex(ValueError, "invalid-row ratio"):
                aggregate_split_transitions(
                    path,
                    "test",
                    index,
                    chunk_size=2,
                    max_invalid_row_ratio=0.1,
                )

        held_out = TransitionAggregate(
            split="test",
            pair_counts=Counter(),
            source_totals=Counter(),
            report=SplitQualityReport(split="test", source_file="fixture.csv"),
        )
        with self.assertRaisesRegex(ValueError, "Only the training split"):
            transition_rows(held_out, index, min_support=1)


if __name__ == "__main__":
    unittest.main()
