from __future__ import annotations

import json
import unittest

import networkx as nx
import numpy as np

from evaluation.evaluate_embedding_transitions import decide_acceptance
from src.transition_embedding import (
    RuntimeTransitionSmoother,
    SmoothingConfig,
    compute_neighbour_index,
    hybrid_prediction_map,
    load_live_esco_embeddings,
    rank_hybrid_destinations,
    transition_distributions,
)


def _graph() -> nx.MultiDiGraph:
    graph = nx.MultiDiGraph()
    for node_id, title in (
        ("source", "Source Role"),
        ("neighbour", "Neighbour Role"),
        ("direct", "Direct Destination"),
        ("backoff", "Backoff Destination"),
        ("missing", "Missing Direct Role"),
    ):
        graph.add_node(node_id, type="role", source="esco", title=title)
    graph.add_node("onet", type="role", source="onet", title="ONET Role")
    graph.add_edge(
        "source",
        "direct",
        key="karrierewege:TRANSITIONS_TO",
        relation="TRANSITIONS_TO",
        source="karrierewege",
        split="train",
        probability=0.8,
        count=8,
        source_total=10,
    )
    graph.add_edge(
        "neighbour",
        "backoff",
        key="karrierewege:TRANSITIONS_TO",
        relation="TRANSITIONS_TO",
        source="karrierewege",
        split="train",
        probability=0.9,
        count=9,
        source_total=10,
    )
    return graph


class _Collection:
    def __init__(self, vectors: dict[str, list[float]]) -> None:
        self.vectors = vectors

    def get(self, ids, include):
        selected = [role_id for role_id in ids if role_id in self.vectors]
        return {
            "ids": selected,
            "embeddings": [self.vectors[role_id] for role_id in selected],
        }


class TransitionEmbeddingTests(unittest.TestCase):
    def test_transition_distributions_are_training_only(self) -> None:
        graph = _graph()
        distributions = transition_distributions(graph)
        self.assertAlmostEqual(distributions["source"]["direct"].probability, 0.8)

        graph["source"]["direct"]["karrierewege:TRANSITIONS_TO"]["split"] = "test"
        held_out_filtered = transition_distributions(graph)
        self.assertNotIn("source", held_out_filtered)
        self.assertIn("neighbour", held_out_filtered)

    def test_live_embedding_loader_filters_source_and_invalid_vectors(self) -> None:
        graph = _graph()
        collection = _Collection(
            {
                "source": [3.0, 4.0],
                "neighbour": [float("nan"), 1.0],
                "direct": [],
                "onet": [1.0, 0.0],
            }
        )
        vectors, report = load_live_esco_embeddings(collection, graph, batch_size=2)
        self.assertEqual(set(vectors), {"source"})
        self.assertAlmostEqual(float(np.linalg.norm(vectors["source"])), 1.0)
        self.assertEqual(report["loaded_vectors"], 1)
        self.assertEqual(report["rejected_vectors"], 2)
        self.assertEqual(report["missing_vectors"], 2)

    def test_neighbours_are_deterministic_and_exclude_self_and_stale_ids(self) -> None:
        vectors = {
            "source": np.array([1.0, 0.0], dtype=np.float32),
            "neighbour": np.array([0.8, 0.6], dtype=np.float32),
            "other": np.array([0.0, 1.0], dtype=np.float32),
        }
        index = compute_neighbour_index(
            vectors,
            {"source", "neighbour", "other", "stale"},
            max_neighbours=2,
            query_ids={"source"},
        )
        self.assertEqual([role_id for role_id, _ in index["source"]], ["neighbour", "other"])
        self.assertNotIn("source", [role_id for role_id, _ in index["source"]])

    def test_direct_neighbour_and_fallback_scoring(self) -> None:
        graph = _graph()
        distributions = transition_distributions(graph)
        neighbours = {
            "source": [("neighbour", 0.9)],
            "missing": [("neighbour", 0.9)],
        }
        config = SmoothingConfig(1, 0.75, 0.1)

        ranked = rank_hybrid_destinations("source", distributions, neighbours, config, graph)
        self.assertEqual([item.role_id for item in ranked], ["direct", "backoff"])
        self.assertAlmostEqual(ranked[0].score, 0.6)
        self.assertEqual(ranked[0].evidence_type, "direct_transition")
        self.assertAlmostEqual(ranked[1].score, 0.225)
        self.assertEqual(ranked[1].evidence_type, "semantic_transition_backoff")

        backoff = rank_hybrid_destinations("missing", distributions, neighbours, config, graph)
        self.assertEqual([item.role_id for item in backoff], ["backoff"])
        self.assertAlmostEqual(backoff[0].score, 0.9)
        self.assertEqual(rank_hybrid_destinations("unknown", distributions, neighbours, config), [])

    def test_prediction_map_is_normalized_and_json_safe(self) -> None:
        graph = _graph()
        distributions = transition_distributions(graph)
        config = SmoothingConfig(1, 0.75, 0.1)
        predictions = hybrid_prediction_map(
            graph,
            distributions,
            {"source": [("neighbour", 0.9)]},
            config,
            source_ids={"source"},
        )
        self.assertEqual(predictions["source role"], ["direct destination", "backoff destination"])
        json.dumps({"config": config.to_dict(), "predictions": predictions})

    def test_runtime_smoother_uses_stored_vectors_and_caches_rankings(self) -> None:
        graph = _graph()
        collection = _Collection(
            {
                "source": [1.0, 0.0],
                "neighbour": [0.9, 0.1],
                "missing": [0.95, 0.05],
            }
        )
        smoother = RuntimeTransitionSmoother(
            graph,
            collection,
            SmoothingConfig(1, 0.9, 0.05),
        )
        ranking = smoother.rank("missing", 3)
        self.assertEqual([item.role_id for item in ranking], ["direct"])
        self.assertEqual(ranking[0].evidence_type, "semantic_transition_backoff")
        self.assertEqual(smoother.rank("missing", 3), ranking)
        self.assertEqual(smoother.diagnostics()["cached_roles"], 1)

    def test_acceptance_gate_accepts_improvement_and_rejects_regression(self) -> None:
        baseline = {
            "mrr": 0.25,
            "hits_at_5": 0.37,
            "hits_at_10": 0.50,
            "source_role_coverage": 0.74,
            "destination_coverage": 0.87,
        }
        improved = {
            "mrr": 0.26,
            "hits_at_5": 0.375,
            "hits_at_10": 0.501,
            "source_role_coverage": 0.75,
            "destination_coverage": 0.88,
        }
        accepted = decide_acceptance(baseline, improved, baseline, improved, True)
        self.assertTrue(accepted.accepted)
        self.assertEqual(accepted.reasons, ())

        equal_mrr = {**improved, "mrr": baseline["mrr"]}
        rejected = decide_acceptance(baseline, equal_mrr, baseline, improved, True)
        self.assertFalse(rejected.accepted)
        self.assertIn("validation_mrr_did_not_improve", rejected.reasons)


if __name__ == "__main__":
    unittest.main()
