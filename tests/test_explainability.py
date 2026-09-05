"""Unit tests for src/explainability.py — provenance-traced explanation chains."""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import networkx as nx
from src.explainability import (
    explain_career_path,
    format_explanation,
    trace_recommendation_path,
)


def _build_explain_graph() -> nx.MultiDiGraph:
    G = nx.MultiDiGraph()
    G.add_node("analyst", type="role", source="esco", title="Data Analyst")
    G.add_node("scientist", type="role", source="esco", title="Data Scientist")
    G.add_node("onet_sci", type="role", source="onet", title="Data Scientists")
    G.add_node("s_python", type="skill", source="esco", title="Python")
    G.add_node("s_sql", type="skill", source="esco", title="SQL")
    G.add_node("s_ml", type="skill", source="esco", title="machine learning")

    # Analyst requires python, sql
    G.add_edge("analyst", "s_python", key="esco:REQUIRES", relation="REQUIRES", source="esco", requirement_level="essential")
    G.add_edge("analyst", "s_sql", key="esco:REQUIRES", relation="REQUIRES", source="esco", requirement_level="essential")

    # Scientist requires python, ml
    G.add_edge("scientist", "s_python", key="esco:REQUIRES", relation="REQUIRES", source="esco", requirement_level="essential")
    G.add_edge("scientist", "s_ml", key="esco:REQUIRES", relation="REQUIRES", source="esco", requirement_level="essential")

    # Transition: analyst → scientist
    G.add_edge("analyst", "scientist", key="karrierewege:TRANSITIONS_TO",
               relation="TRANSITIONS_TO", source="karrierewege",
               probability=0.12, count=847, source_total=7058)

    # SIMILAR_TO: scientist ↔ onet_sci
    G.add_edge("scientist", "onet_sci", key="alignment:SIMILAR_TO",
               relation="SIMILAR_TO", source="alignment", similarity=0.72)

    return G


class TestTraceRecommendationPath(unittest.TestCase):

    def test_full_trace(self):
        G = _build_explain_graph()
        owned = {"s_python"}  # User has python
        steps = trace_recommendation_path("analyst", "scientist", owned, G)
        # Should have: TRANSITIONS_TO, SIMILAR_TO, REQUIRES (owned), REQUIRES (missing)
        relations = [s.relation for s in steps]
        self.assertIn("TRANSITIONS_TO", relations)
        self.assertIn("SIMILAR_TO", relations)
        self.assertIn("REQUIRES", relations)

    def test_transition_evidence(self):
        G = _build_explain_graph()
        steps = trace_recommendation_path("analyst", "scientist", set(), G)
        trans_steps = [s for s in steps if s.relation == "TRANSITIONS_TO"]
        self.assertEqual(len(trans_steps), 1)
        self.assertAlmostEqual(trans_steps[0].attributes["probability"], 0.12)
        self.assertEqual(trans_steps[0].attributes["count"], 847)

    def test_held_out_transition_is_not_explained_as_observed(self):
        G = _build_explain_graph()
        G.add_node("held-out", type="role", source="esco", title="Held Out")
        G.add_edge(
            "analyst", "held-out", key="karrierewege:test",
            relation="TRANSITIONS_TO", source="karrierewege", split="test",
            probability=0.99, count=999, source_total=999,
        )
        steps = trace_recommendation_path("analyst", "held-out", set(), G)
        self.assertNotIn("TRANSITIONS_TO", [step.relation for step in steps])

    def test_strongest_similarity_alignment_is_selected(self):
        G = _build_explain_graph()
        G.add_node("onet_strong", type="role", source="onet", title="Strong Alignment")
        G.add_edge(
            "scientist", "onet_strong", key="alignment:strong",
            relation="SIMILAR_TO", source="alignment", similarity=0.95,
        )
        steps = trace_recommendation_path("analyst", "scientist", set(), G)
        similarity = [step for step in steps if step.relation == "SIMILAR_TO"]
        self.assertEqual(len(similarity), 1)
        self.assertEqual(similarity[0].target_id, "onet_strong")
        self.assertEqual(similarity[0].attributes["similarity"], 0.95)

    def test_owned_vs_missing_skills(self):
        G = _build_explain_graph()
        owned = {"s_python"}
        steps = trace_recommendation_path("analyst", "scientist", owned, G)
        req_steps = [s for s in steps if s.relation == "REQUIRES"]
        statuses = [s.attributes["status"] for s in req_steps]
        self.assertIn("you_have_this", statuses)
        self.assertIn("you_need_this", statuses)

    def test_missing_nodes(self):
        G = _build_explain_graph()
        steps = trace_recommendation_path("nonexistent", "scientist", set(), G)
        self.assertEqual(steps, [])


class TestFormatExplanation(unittest.TestCase):

    def test_format_output(self):
        G = _build_explain_graph()
        steps = trace_recommendation_path("analyst", "scientist", {"s_python"}, G)
        text = format_explanation(steps)
        self.assertIn("TRANSITIONS_TO", text)
        self.assertIn("12.00%", text)
        self.assertIn("you already have this", text)
        self.assertIn("you need this", text)

    def test_empty_steps(self):
        text = format_explanation([])
        self.assertEqual(text, "No traceable evidence path available.")


class TestExplainCareerPath(unittest.TestCase):

    def test_generates_explanations(self):
        G = _build_explain_graph()
        path_roles = [{"id": "scientist", "title": "Data Scientist"}]
        explanations = explain_career_path(path_roles, "analyst", {"s_python"}, G)
        self.assertEqual(len(explanations), 1)
        self.assertEqual(explanations[0]["role_id"], "scientist")
        self.assertIn("steps", explanations[0])
        self.assertTrue(len(explanations[0]["steps"]) > 0)

    def test_no_source_role(self):
        G = _build_explain_graph()
        path_roles = [{"id": "scientist", "title": "Data Scientist"}]
        explanations = explain_career_path(path_roles, None, set(), G)
        self.assertEqual(explanations, [])

    def test_skips_self(self):
        G = _build_explain_graph()
        path_roles = [
            {"id": "analyst", "title": "Data Analyst"},
            {"id": "scientist", "title": "Data Scientist"},
        ]
        explanations = explain_career_path(path_roles, "analyst", set(), G)
        role_ids = [e["role_id"] for e in explanations]
        self.assertNotIn("analyst", role_ids)
        self.assertIn("scientist", role_ids)

    def test_virtual_prediction_is_labeled_as_inferred(self):
        G = _build_explain_graph()
        G.add_node("engineer", type="role", source="esco", title="AI Engineer")
        path_roles = [{
            "id": "engineer",
            "title": "AI Engineer",
            "transition": {
                "evidence_type": "predicted_transition",
                "score": 0.82,
                "model": "lightgbm",
            },
        }]
        explanations = explain_career_path(path_roles, "analyst", set(), G)
        self.assertEqual(explanations[0]["steps"][0]["relation"], "PREDICTED_TRANSITION")
        self.assertIn("model score", explanations[0]["explanation_text"])


if __name__ == "__main__":
    unittest.main()
