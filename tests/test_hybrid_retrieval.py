"""Tests for request-time hybrid retrieval and virtual predicted edges."""

from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import networkx as nx
import numpy as np

from src.link_prediction import FEATURE_NAMES
from src.hybrid_retrieval import (
    LinkPredictionRuntime,
    combine_prediction_maps,
    fuse_transition_candidates,
    graph_retrieve,
    hybrid_retrieve,
    reciprocal_rank_fusion,
)


def _graph() -> nx.MultiDiGraph:
    graph = nx.MultiDiGraph()
    for role_id, title in (
        ("source", "Developer"),
        ("direct", "Data Engineer"),
        ("predicted", "ML Engineer"),
        ("teacher", "English Teacher"),
    ):
        graph.add_node(role_id, type="role", source="esco", title=title)
    graph.add_node("python", type="skill", source="esco", title="Python", idf=4.0)
    graph.add_node("english", type="skill", source="esco", title="English", idf=1.0)
    graph.add_edge("direct", "python", relation="REQUIRES", source="esco", requirement_level="essential")
    graph.add_edge("direct", "english", relation="REQUIRES", source="esco", requirement_level="optional")
    graph.add_edge("teacher", "english", relation="REQUIRES", source="esco", requirement_level="essential")
    graph.add_edge(
        "source",
        "direct",
        relation="TRANSITIONS_TO",
        source="karrierewege",
        split="train",
        count=8,
        probability=0.4,
        source_total=20,
    )
    return graph


class _Predictor:
    def predict(self, source_id, top_k=20):
        return [("predicted", 0.91)]


class _Model:
    def __init__(self) -> None:
        self.features = None

    def predict(self, features):
        self.features = features.copy()
        return np.arange(len(features), dtype=np.float64) / 10.0


class _LegacyModel(_Model):
    def num_feature(self):
        return 9


class HybridRetrievalTests(unittest.TestCase):
    def test_graph_retrieval_keeps_language_when_role_essential(self) -> None:
        graph = _graph()
        results = graph_retrieve({"english"}, graph, top_k=10)
        self.assertEqual([item["id"] for item in results], ["teacher"])

    def test_runtime_prediction_excludes_observed_edges_and_uses_features(self) -> None:
        graph = _graph()
        model = _Model()
        embeddings = {
            "source": np.array([1.0, 0.0], dtype=np.float32),
            "direct": np.array([0.8, 0.2], dtype=np.float32),
            "predicted": np.array([0.6, 0.4], dtype=np.float32),
        }
        runtime = LinkPredictionRuntime(
            model=model,
            graph=graph,
            embeddings=embeddings,
            transition_index={"source": {"direct": 0.4}},
            neighbour_index={"source": ["source"]},
            idf_map={"python": 4.0, "english": 1.0},
            candidate_role_ids=("source", "direct", "predicted"),
            skills_cache={
                "source": set(),
                "direct": {"python", "english"},
                "predicted": set(),
            },
            skill_degrees={"python": 1, "english": 2},
            isco_cache={},
            vector_report={"loaded_vectors": 3},
        )

        predictions = runtime.predict("source", top_k=10)
        self.assertEqual([role_id for role_id, _ in predictions], ["predicted"])
        self.assertEqual(model.features.shape, (1, len(FEATURE_NAMES)))
        self.assertGreater(model.features[0, 5], 0.0)

    def test_runtime_respects_legacy_model_feature_schema(self) -> None:
        graph = _graph()
        model = _LegacyModel()
        runtime = LinkPredictionRuntime(
            model=model,
            graph=graph,
            embeddings={
                "source": np.array([1.0, 0.0], dtype=np.float32),
                "predicted": np.array([0.6, 0.4], dtype=np.float32),
            },
            transition_index={"source": {"direct": 0.4}},
            neighbour_index={},
            idf_map={},
            candidate_role_ids=("source", "direct", "predicted"),
            skills_cache={"source": set(), "direct": set(), "predicted": set()},
            skill_degrees={},
            isco_cache={},
            vector_report={"loaded_vectors": 2},
        )

        runtime.predict("source", top_k=1)

        self.assertEqual(model.features.shape, (1, 9))

    @patch("src.inference_pipeline.retrieve_candidates")
    def test_hybrid_retrieval_wires_vector_direct_graph_and_lp(self, retrieve) -> None:
        graph = _graph()
        retrieve.return_value = [
            {
                "id": "source",
                "metadata": {"title": "Developer"},
                "document": "Developer",
                "score": 0.9,
            },
            {
                "id": "teacher",
                "metadata": {"title": "English Teacher"},
                "document": "English Teacher",
                "score": 0.8,
            },
        ]
        settings = SimpleNamespace(
            retrieval_top_k=50,
            transition_candidate_limit=12,
            onet_importance_threshold=3.5,
        )

        results = hybrid_retrieve(
            "career query",
            object(),
            {"python"},
            "source",
            graph,
            settings,
            link_prediction_runtime=_Predictor(),
            vector_top_k=7,
        )
        by_id = {item["id"]: item for item in results}

        self.assertEqual(retrieve.call_args.kwargs["limit"], 7)
        self.assertNotIn("source", by_id)
        self.assertEqual(by_id["direct"]["transition"]["evidence_type"], "direct_transition")
        self.assertEqual(by_id["predicted"]["transition"]["evidence_type"], "predicted_transition")
        self.assertIn("vector", by_id["teacher"]["retrieval_sources"])
        self.assertIn("graph_structural", by_id["direct"]["retrieval_sources"])

    def test_rrf_retains_strongest_transition_evidence(self) -> None:
        fused = reciprocal_rank_fusion(
            [{"id": "r", "source": "link_prediction", "transition": {"evidence_type": "predicted_transition"}}],
            [{"id": "r", "source": "direct_transition", "transition": {"evidence_type": "direct_transition", "count": 3}}],
        )
        self.assertEqual(fused[0]["transition"]["evidence_type"], "direct_transition")
        self.assertEqual(fused[0]["retrieval_sources"], ["direct_transition", "link_prediction"])

    def test_priority_combination_does_not_truncate_empirical_ranking(self) -> None:
        direct = {"source": [f"role-{index}" for index in range(75)]}
        combined = combine_prediction_maps(
            direct,
            lp_map={"uncovered": ["predicted"]},
            top_k=50,
        )
        self.assertEqual(len(combined["source"]), 75)
        self.assertEqual(combined["uncovered"], ["predicted"])

    def test_validation_accepted_smoothing_has_priority(self) -> None:
        combined = combine_prediction_maps(
            {"source": ["direct"]},
            smoothed_map={"source": ["smoothed", "direct"]},
            lp_map={"source": ["predicted"]},
        )
        self.assertEqual(combined["source"], ["smoothed", "direct"])

    def test_sequential_fusion_happens_inside_transition_channel(self) -> None:
        class Prediction:
            role_id = "predicted"
            score = 0.8
            model = "step_gru"
            history_length = 2

        graph = _graph()
        fused = fuse_transition_candidates(
            [{"id": "direct", "score": 0.9, "source": "direct_transition", "retrieval_sources": ["direct_transition"], "transition": {"evidence_type": "direct_transition"}}],
            [Prediction()],
            sequential_weight=0.5,
            G=graph,
            limit=5,
        )
        by_id = {item["id"]: item for item in fused}
        self.assertIn("sequential_transition", by_id["predicted"]["retrieval_sources"])
        self.assertEqual(by_id["direct"]["transition"]["evidence_type"], "direct_transition")


if __name__ == "__main__":
    unittest.main()
