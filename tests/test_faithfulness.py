"""Unit tests for src/faithfulness.py — graph-provenance verification."""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import networkx as nx
from src.faithfulness import (
    check_reachability,
    compute_faithfulness,
    extract_entities,
    match_entities_to_graph,
)


def _build_faith_graph() -> nx.MultiDiGraph:
    G = nx.MultiDiGraph()
    G.add_node("r1", type="role", source="esco", title="Data Scientist")
    G.add_node("r2", type="role", source="esco", title="Machine Learning Engineer")
    G.add_node("r3", type="role", source="esco", title="Surgeon")
    G.add_node("s1", type="skill", source="esco", title="Python")
    G.add_node("s2", type="skill", source="esco", title="deep learning")
    # r1 → s1 (REQUIRES)
    G.add_edge("r1", "s1", key="esco:REQUIRES", relation="REQUIRES", source="esco")
    # r1 → r2 (TRANSITIONS_TO)
    G.add_edge("r1", "r2", key="karrierewege:TRANSITIONS_TO", relation="TRANSITIONS_TO", source="karrierewege")
    # r2 → s2 (REQUIRES)
    G.add_edge("r2", "s2", key="esco:REQUIRES", relation="REQUIRES", source="esco")
    return G


class TestExtractEntities(unittest.TestCase):

    def test_bold_entities(self):
        text = "You should consider **Data Scientist** or **Machine Learning Engineer**."
        entities = extract_entities(text)
        self.assertEqual(len(entities), 2)
        self.assertIn("Data Scientist", entities)
        self.assertIn("Machine Learning Engineer", entities)

    def test_no_entities(self):
        text = "This is a plain response with no bold text."
        entities = extract_entities(text)
        self.assertEqual(entities, [])

    def test_deduplication(self):
        text = "Consider **data scientist** or **Data Scientist** for your path."
        entities = extract_entities(text)
        self.assertEqual(len(entities), 1)

    def test_unbolded_skill_candidates(self):
        entities = extract_entities(
            "You could consider **Data Scientist** and develop quantum tensor alchemy."
        )
        self.assertIn("Data Scientist", entities)
        self.assertIn("quantum tensor alchemy", entities)

    def test_slash_separated_technology_names_are_split(self):
        entities = extract_entities("Your Python/SQL background is useful.")
        self.assertIn("Python", entities)
        self.assertIn("SQL", entities)
        self.assertNotIn("Python/SQL", entities)


class TestMatchEntities(unittest.TestCase):

    def test_match_found(self):
        G = _build_faith_graph()
        matches = match_entities_to_graph(["Data Scientist"], G)
        self.assertEqual(len(matches), 1)
        self.assertEqual(matches[0].node_id, "r1")
        self.assertEqual(matches[0].node_type, "role")

    def test_no_match(self):
        G = _build_faith_graph()
        matches = match_entities_to_graph(["Astronaut"], G)
        self.assertEqual(len(matches), 1)
        self.assertIsNone(matches[0].node_id)

    def test_skill_match(self):
        G = _build_faith_graph()
        matches = match_entities_to_graph(["Python"], G)
        self.assertEqual(matches[0].node_id, "s1")
        self.assertEqual(matches[0].node_type, "skill")

    def test_parenthetical_skill_title_matches_short_surface_form(self):
        G = _build_faith_graph()
        G.nodes["s1"]["title"] = "Python (computer programming)"
        matches = match_entities_to_graph(["Python"], G)
        self.assertEqual(matches[0].node_id, "s1")


class TestReachability(unittest.TestCase):

    def test_direct_neighbour(self):
        G = _build_faith_graph()
        self.assertTrue(check_reachability("s1", ["r1"], G, max_hops=1))

    def test_two_hops(self):
        G = _build_faith_graph()
        # r1 → r2 → s2 (2 hops from r1)
        self.assertTrue(check_reachability("s2", ["r1"], G, max_hops=2))

    def test_unreachable(self):
        G = _build_faith_graph()
        # Surgeon has no edges to/from r1 or r2
        self.assertFalse(check_reachability("r3", ["r1"], G, max_hops=3))

    def test_self_is_reachable(self):
        G = _build_faith_graph()
        self.assertTrue(check_reachability("r1", ["r1"], G, max_hops=0))


class TestComputeFaithfulness(unittest.TestCase):

    def test_fully_faithful(self):
        G = _build_faith_graph()
        text = "You should consider **Data Scientist** and develop your **Python** skills."
        result = compute_faithfulness(text, ["r1"], G)
        self.assertEqual(result.score, 1.0)
        self.assertEqual(result.total_entities, 2)
        self.assertEqual(result.reachable_count, 2)
        self.assertEqual(result.unmatched, [])

    def test_hallucinated_entity(self):
        G = _build_faith_graph()
        text = "You should consider **Data Scientist** or **Quantum Physicist**."
        result = compute_faithfulness(text, ["r1"], G)
        self.assertLess(result.score, 1.0)
        self.assertEqual(len(result.unmatched), 1)
        self.assertIn("Quantum Physicist", result.unmatched)

    def test_unbolded_hallucinated_skill_reduces_score(self):
        G = _build_faith_graph()
        text = "Consider **Data Scientist** and develop quantum tensor alchemy."
        result = compute_faithfulness(text, ["r1"], G)
        self.assertLess(result.score, 1.0)
        self.assertIn("quantum tensor alchemy", result.unmatched)

    def test_unreachable_entity(self):
        G = _build_faith_graph()
        text = "You could become a **Surgeon**."
        result = compute_faithfulness(text, ["r1"], G)
        self.assertEqual(result.score, 0.0)
        self.assertEqual(result.matched_count, 1)
        self.assertEqual(result.reachable_count, 0)

    def test_empty_response(self):
        G = _build_faith_graph()
        result = compute_faithfulness("No bold entities here.", ["r1"], G)
        self.assertEqual(result.score, 1.0)
        self.assertEqual(result.total_entities, 0)

    def test_to_dict(self):
        G = _build_faith_graph()
        text = "Consider **Data Scientist**."
        result = compute_faithfulness(text, ["r1"], G)
        d = result.to_dict()
        self.assertIn("score", d)
        self.assertIn("total_entities", d)
        self.assertIn("matched_entities", d)


if __name__ == "__main__":
    unittest.main()
