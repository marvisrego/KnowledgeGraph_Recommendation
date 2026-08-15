from __future__ import annotations

import json
import unittest

import networkx as nx

from src.inference_pipeline import (
    _build_context_block,
    _build_path_data,
    augment_candidates_with_transitions,
    filter_candidates_to_graph,
    traverse_graph,
)


def _graph() -> nx.MultiDiGraph:
    graph = nx.MultiDiGraph()
    graph.add_node("analyst", type="role", source="esco", title="data analyst")
    graph.add_node("scientist", type="role", source="esco", title="data scientist")
    graph.add_node("engineer", type="role", source="esco", title="data engineer")
    graph.add_node("python", type="skill", source="esco", title="Python")
    graph.add_edge(
        "scientist",
        "python",
        key="esco:REQUIRES",
        relation="REQUIRES",
        source="esco",
        requirement_level="essential",
    )
    for target, probability, count in (("scientist", 0.4, 8), ("engineer", 0.2, 4)):
        graph.add_edge(
            "analyst",
            target,
            key="karrierewege:TRANSITIONS_TO",
            relation="TRANSITIONS_TO",
            source="karrierewege",
            probability=probability,
            count=count,
            source_total=20,
        )
    return graph


class InferenceEvidenceTests(unittest.TestCase):
    def test_stale_vector_candidates_are_removed(self) -> None:
        graph = _graph()
        filtered = filter_candidates_to_graph(
            [{"id": "scientist"}, {"id": "removed-role"}], graph
        )
        self.assertEqual(filtered, [{"id": "scientist"}])

    def test_transition_augmentation_merges_and_excludes_current_role(self) -> None:
        candidates = [
            {"id": "analyst", "document": "current", "score": 0.9},
            {"id": "scientist", "document": "target", "score": 0.8},
        ]
        augmented = augment_candidates_with_transitions(candidates, "analyst", _graph(), limit=2)
        by_id = {item["id"]: item for item in augmented}
        self.assertNotIn("analyst", by_id)
        self.assertEqual(by_id["scientist"]["transition"]["count"], 8)
        self.assertIn("engineer", by_id)
        self.assertTrue(by_id["engineer"]["document"])

    def test_transition_traversal_is_probability_bounded_and_contextualized(self) -> None:
        graph = _graph()
        triples = traverse_graph(["analyst"], graph, transition_limit=1)
        transition_triples = [item for item in triples if item[1] == "TRANSITIONS_TO"]
        self.assertEqual(len(transition_triples), 1)
        self.assertEqual(transition_triples[0][2], "scientist")
        context = _build_context_block(["analyst"], triples, graph)
        self.assertIn("observed_count=8", context)
        self.assertIn("observed_probability=0.4000", context)

    def test_path_payload_removes_internal_sets(self) -> None:
        graph = _graph()
        candidates = [
            {
                "id": "scientist",
                "semantic_score": 0.8,
                "transition": {
                    "from_role_id": "analyst",
                    "count": 8,
                    "probability": 0.4,
                    "source_total": 20,
                },
                "skill_gap": {
                    "accessibility": 1.0,
                    "gap": 0.0,
                    "gap_evidence": "direct_esco",
                    "required_skill_count": 1,
                    "have_count": 1,
                    "need_count": 0,
                    "have": [{"id": "python", "title": "Python"}],
                    "need": [],
                    "have_ids": {"python"},
                    "need_ids": set(),
                },
            }
        ]
        triples = traverse_graph(["scientist"], graph)
        payload = _build_path_data(["scientist"], triples, graph, candidates)
        serialized = json.dumps(payload)
        self.assertIn('"accessibility": 1.0', serialized)
        self.assertEqual(payload["roles"][0]["have"][0]["title"], "Python")


if __name__ == "__main__":
    unittest.main()
