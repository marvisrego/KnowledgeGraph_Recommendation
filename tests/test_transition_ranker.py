from __future__ import annotations

import tempfile
import unittest
from collections import Counter
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd

from src.transition_ranker import (
    FEATURE_NAMES,
    TransitionFeatureBuilder,
    aggregate_person_disjoint_folds,
    build_ranker_rows,
    build_transition_context,
    head_preserving_fusion,
    relevance_from_count,
    stable_person_fold,
)


def _graph() -> nx.MultiDiGraph:
    graph = nx.MultiDiGraph()
    for role, title, isco in (
        ("a", "Role A", "21"),
        ("b", "Role B", "21"),
        ("c", "Role C", "31"),
        ("unseen", "Unseen Positive", "41"),
    ):
        graph.add_node(role, type="role", source="esco", title=title, isco_2digit=isco)
    graph.add_node("s1", type="skill", source="esco", idf=2.0)
    graph.add_edge("a", "s1", relation="REQUIRES")
    graph.add_edge("b", "s1", relation="REQUIRES")
    return graph


class TransitionRankerTests(unittest.TestCase):
    def test_person_fold_is_stable_and_person_disjoint(self) -> None:
        self.assertEqual(stable_person_fold("person-1", 5), stable_person_fold("person-1", 5))
        rows = [
            ("p1", 0, "Role A"),
            ("p1", 1, "Role B"),
            ("p2", 0, "Role A"),
            ("p2", 1, "Role C"),
        ]
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "train.csv"
            pd.DataFrame(
                rows, columns=["_id", "experience_order", "preferredLabel_en"]
            ).to_csv(path, index=False)
            folded = aggregate_person_disjoint_folds(
                path,
                {"role a": "a", "role b": "b", "role c": "c"},
                n_folds=2,
                chunk_size=2,
            )
        self.assertEqual(sum(map(sum, (counts.values() for counts in folded.fold_pair_counts))), 2)
        pair_folds = [index for index, counts in enumerate(folded.fold_pair_counts) if counts]
        self.assertGreaterEqual(len(pair_folds), 1)

    def test_labels_never_expand_candidates(self) -> None:
        graph = _graph()
        embeddings = {
            "a": np.array([1.0, 0.0], dtype=np.float32),
            "b": np.array([0.9, 0.1], dtype=np.float32),
            "c": np.array([0.0, 1.0], dtype=np.float32),
            "unseen": np.array([-1.0, 0.0], dtype=np.float32),
        }
        builder = TransitionFeatureBuilder(
            graph,
            embeddings,
            source_neighbours={"a": [("c", 0.1)]},
            destination_neighbours={"a": [("b", 0.9)]},
            per_channel_limit=1,
        )
        context = build_transition_context(Counter({("a", "b"): 3, ("c", "b"): 2}))
        matrix, labels, groups, pairs, report = build_ranker_rows(
            ["a"],
            context,
            {("a", "unseen"): 5},
            builder,
        )

        self.assertEqual(matrix.shape[1], len(FEATURE_NAMES))
        self.assertEqual(groups, [len(pairs)])
        self.assertNotIn(("a", "unseen"), pairs)
        self.assertTrue(np.all(labels == 0))
        self.assertEqual(report["candidate_recall"], 0.0)

    def test_relevance_bins(self) -> None:
        self.assertEqual([relevance_from_count(i) for i in (0, 1, 2, 3, 4, 20)], [0, 1, 2, 2, 3, 3])

    def test_head_preserving_fusion_keeps_head_and_deduplicates(self) -> None:
        fused = head_preserving_fusion(
            {"a": ["p1", "p2", "p3", "p4"]},
            {"a": ["p3", "s1", "p1", "s2"]},
            head_size=2,
            secondary_slots=2,
        )
        self.assertEqual(fused["a"], ["p1", "p2", "p3", "s1", "p4", "s2"])


if __name__ == "__main__":
    unittest.main()
