from __future__ import annotations

import unittest

import networkx as nx

from src.skill_gap import (
    compute_skill_gap,
    rank_roles_by_gap,
    role_requirements,
    resolve_current_role,
    resolve_user_skills,
    transition_destinations,
    transition_evidence,
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

    def test_normalized_contractions_and_no_are_negated(self) -> None:
        graph = _graph()
        for text in (
            "I don't use Python.",
            "I dont use Python.",
            "I have no Python experience.",
        ):
            with self.subTest(text=text):
                result = resolve_user_skills(["Python"], [text], graph)
                self.assertNotIn("python", result["skill_ids"])
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

    def test_role_relevance_uses_essential_requirements(self) -> None:
        graph = _graph()
        graph.add_node("english", type="skill", source="esco", title="English", idf=1.0)
        graph.add_node("pedagogy", type="skill", source="esco", title="pedagogy", idf=1.0)
        graph.nodes["python"]["idf"] = 5.0
        graph.add_node("teacher", type="role", source="esco", title="English teacher")
        graph.add_edge(
            "developer", "english", key="esco:OPTIONAL",
            relation="REQUIRES", source="esco", requirement_level="optional",
        )
        for skill in ("english", "pedagogy"):
            graph.add_edge(
                "teacher", skill, key=f"esco:{skill}",
                relation="REQUIRES", source="esco", requirement_level="essential",
            )

        technical = compute_skill_gap({"python"}, "developer", graph)
        teaching = compute_skill_gap({"pedagogy"}, "teacher", graph)
        self.assertEqual(technical["accessibility"], 1.0)
        self.assertNotIn("english", technical["need_ids"])
        self.assertEqual(teaching["accessibility"], 0.5)
        self.assertIn("english", teaching["need_ids"])

    def test_unaligned_onet_uses_only_high_importance_requirements(self) -> None:
        graph = _graph()
        graph.add_node("raw-onet", type="role", source="onet", title="Technical role")
        graph.add_node("high", type="element", source="onet", title="Programming")
        graph.add_node("low", type="element", source="onet", title="General awareness")
        graph.add_edge(
            "raw-onet", "high", key="onet:high", relation="REQUIRES",
            source="onet", requirement_level=4.2,
        )
        graph.add_edge(
            "raw-onet", "low", key="onet:low", relation="REQUIRES",
            source="onet", requirement_level=3.0,
        )
        required, evidence, evidence_role = role_requirements(
            "raw-onet", graph, onet_importance_threshold=3.5
        )
        self.assertEqual(required, {"high"})
        self.assertEqual(evidence, "direct_onet")
        self.assertEqual(evidence_role, "raw-onet")

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

    def test_held_out_transitions_are_not_runtime_evidence(self) -> None:
        graph = _graph()
        graph.add_node("held-out", type="role", source="esco", title="Held Out")
        graph.add_edge(
            "analyst",
            "held-out",
            key="karrierewege:test",
            relation="TRANSITIONS_TO",
            source="karrierewege",
            split="test",
            probability=0.99,
            count=999,
            source_total=999,
        )
        self.assertIsNone(transition_evidence("analyst", "held-out", graph))
        self.assertNotIn(
            "held-out",
            {row["id"] for row in transition_destinations("analyst", graph)},
        )


if __name__ == "__main__":
    unittest.main()
