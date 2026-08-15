from __future__ import annotations

import unittest

import networkx as nx

from src.skill_gap import (
    compute_skill_gap,
    rank_roles_by_gap,
    resolve_current_role,
    resolve_user_skills,
    transition_destinations,
)


def _graph() -> nx.MultiDiGraph:
    graph = nx.MultiDiGraph()
    graph.add_node("python", type="skill", source="esco", title="Python (computer programming)")
    graph.add_node("sql", type="skill", source="esco", title="SQL")
    graph.add_node("ml", type="skill", source="esco", title="machine learning")
    graph.add_node("analyst", type="role", source="esco", title="data analyst")
    graph.add_node("scientist", type="role", source="esco", title="data scientist")
    graph.add_node("developer", type="role", source="esco", title="software developer")
    graph.add_node("onet", type="role", source="onet", title="Data Scientists")
    for skill in ("python", "sql", "ml"):
        graph.add_edge(
            "scientist",
            skill,
            key="esco:REQUIRES",
            relation="REQUIRES",
            source="esco",
            requirement_level="essential",
        )
    graph.add_edge(
        "developer",
        "python",
        key="esco:REQUIRES",
        relation="REQUIRES",
        source="esco",
        requirement_level="essential",
    )
    graph.add_edge(
        "onet",
        "scientist",
        key="alignment:SIMILAR_TO",
        relation="SIMILAR_TO",
        source="alignment",
        similarity=0.9,
    )
    graph.add_edge(
        "analyst",
        "scientist",
        key="karrierewege:TRANSITIONS_TO",
        relation="TRANSITIONS_TO",
        source="karrierewege",
        count=8,
        probability=0.4,
        source_total=20,
    )
    return graph


class SkillGapTests(unittest.TestCase):
    def test_explicit_matching_parenthetical_alias_and_negation(self) -> None:
        graph = _graph()
        result = resolve_user_skills(
            ["Python", "SQL"],
            ["I know SQL but do not know Python."],
            graph,
        )
        self.assertEqual(result["skill_ids"], {"sql"})
        self.assertIn("Python", result["excluded"])

    def test_goal_skills_are_not_treated_as_owned(self) -> None:
        graph = _graph()
        result = resolve_user_skills(
            ["Python", "SQL", "machine learning"],
            ["I know Python and SQL, and I want to move into machine learning."],
            graph,
        )
        self.assertEqual(result["skill_ids"], {"python", "sql"})
        self.assertIn("machine learning", result["excluded"])

    def test_embedded_possession_phrases_are_recognized(self) -> None:
        graph = _graph()
        experienced = resolve_user_skills(
            ["Python"],
            ["I am a software developer experienced in Python and want to progress."],
            graph,
        )
        graduate = resolve_user_skills(
            ["Python"],
            ["I completed computer science and know Python. I want an AI role."],
            graph,
        )
        self.assertEqual(experienced["skill_ids"], {"python"})
        self.assertEqual(graduate["skill_ids"], {"python"})

    def test_gap_partitions_requirements_and_alignment_fallback(self) -> None:
        graph = _graph()
        direct = compute_skill_gap({"python", "sql"}, "scientist", graph)
        aligned = compute_skill_gap({"python", "sql"}, "onet", graph)
        self.assertAlmostEqual(direct["accessibility"], 2 / 3)
        self.assertEqual(direct["have_ids"] | direct["need_ids"], {"python", "sql", "ml"})
        self.assertFalse(direct["have_ids"] & direct["need_ids"])
        self.assertEqual(aligned["gap_evidence"], "aligned_esco")
        self.assertAlmostEqual(aligned["accessibility"], 2 / 3)

    def test_missing_user_evidence_is_not_zero_accessibility(self) -> None:
        gap = compute_skill_gap(set(), "scientist", _graph())
        self.assertIsNone(gap["accessibility"])
        self.assertEqual(gap["gap_evidence"], "no_user_skills")

    def test_accessibility_ranking_then_transition_and_semantic_ties(self) -> None:
        graph = _graph()
        candidates = [
            {"id": "scientist", "rerank_score": 0.95},
            {"id": "developer", "rerank_score": 0.60},
        ]
        ranked = rank_roles_by_gap({"python"}, candidates, graph, current_role_id="analyst")
        self.assertEqual([item["id"] for item in ranked], ["developer", "scientist"])
        self.assertEqual(ranked[1]["transition"]["count"], 8)

    def test_role_resolution_and_transition_destinations(self) -> None:
        graph = _graph()
        self.assertEqual(resolve_current_role("Data Analyst", graph), "analyst")
        destinations = transition_destinations("analyst", graph)
        self.assertEqual(destinations[0]["id"], "scientist")


if __name__ == "__main__":
    unittest.main()
