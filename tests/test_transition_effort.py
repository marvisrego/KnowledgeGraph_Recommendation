"""Unit tests for src/transition_effort.py — Transition Effort Score."""

import math
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import networkx as nx
from src.kg_enrichment import compute_skill_idf, compute_isco_2digit
from src.transition_effort import (
    EffortWeights,
    domain_distance,
    effort_band,
    empirical_support,
    rank_roles_by_effort,
    skill_gap_magnitude,
    transferability,
    transition_effort_score,
)


def _build_effort_graph() -> nx.MultiDiGraph:
    """Graph with roles in different domains and varying skill overlap."""
    G = nx.MultiDiGraph()

    # Roles
    G.add_node("analyst", type="role", source="esco", title="Data Analyst", isco_group="2511")
    G.add_node("scientist", type="role", source="esco", title="Data Scientist", isco_group="2120")
    G.add_node("surgeon", type="role", source="esco", title="Surgeon", isco_group="2212")
    G.add_node("ai_eng", type="role", source="esco", title="AI Engineer", isco_group="2511")

    # Skills
    G.add_node("s_python", type="skill", source="esco", title="Python")
    G.add_node("s_sql", type="skill", source="esco", title="SQL")
    G.add_node("s_ml", type="skill", source="esco", title="machine learning")
    G.add_node("s_stats", type="skill", source="esco", title="statistics")
    G.add_node("s_surgery", type="skill", source="esco", title="surgical procedures")
    G.add_node("s_anatomy", type="skill", source="esco", title="human anatomy")
    G.add_node("s_dlp", type="skill", source="esco", title="deep learning")

    # Analyst requires: python, sql, stats
    for skill in ("s_python", "s_sql", "s_stats"):
        G.add_edge("analyst", skill, key="esco:REQUIRES", relation="REQUIRES", source="esco", requirement_level="essential")

    # Scientist requires: python, sql, ml, stats, deep learning
    for skill in ("s_python", "s_sql", "s_ml", "s_stats", "s_dlp"):
        G.add_edge("scientist", skill, key="esco:REQUIRES", relation="REQUIRES", source="esco", requirement_level="essential")

    # Surgeon requires: surgery, anatomy
    for skill in ("s_surgery", "s_anatomy"):
        G.add_edge("surgeon", skill, key="esco:REQUIRES", relation="REQUIRES", source="esco", requirement_level="essential")

    # AI Engineer requires: python, ml, deep learning
    for skill in ("s_python", "s_ml", "s_dlp"):
        G.add_edge("ai_eng", skill, key="esco:REQUIRES", relation="REQUIRES", source="esco", requirement_level="essential")

    # Transition: analyst → scientist (high frequency)
    G.add_edge("analyst", "scientist", key="karrierewege:TRANSITIONS_TO",
               relation="TRANSITIONS_TO", source="karrierewege",
               probability=0.12, count=847, source_total=7058, split="train")

    # No transition analyst → surgeon

    # Enrich
    compute_skill_idf(G)
    compute_isco_2digit(G)

    return G


class TestEffortBand(unittest.TestCase):

    def test_low(self):
        self.assertEqual(effort_band(0.0), "low")
        self.assertEqual(effort_band(0.3), "low")

    def test_moderate(self):
        self.assertEqual(effort_band(0.31), "moderate")
        self.assertEqual(effort_band(0.6), "moderate")

    def test_high(self):
        self.assertEqual(effort_band(0.61), "high")
        self.assertEqual(effort_band(1.0), "high")


class TestSkillGapMagnitude(unittest.TestCase):

    def test_full_coverage(self):
        idf = {"a": 1.0, "b": 2.0}
        self.assertEqual(skill_gap_magnitude({"a", "b"}, {"a", "b"}, idf), 0.0)

    def test_no_coverage(self):
        idf = {"a": 1.0, "b": 2.0}
        result = skill_gap_magnitude(set(), {"a", "b"}, idf)
        self.assertAlmostEqual(result, 1.0)

    def test_partial_coverage_weighted(self):
        idf = {"a": 1.0, "b": 3.0}
        # Own "a" (low IDF), missing "b" (high IDF)
        result = skill_gap_magnitude({"a"}, {"a", "b"}, idf)
        expected = 3.0 / 4.0  # missing_idf / total_idf
        self.assertAlmostEqual(result, expected)

    def test_empty_requirements(self):
        self.assertEqual(skill_gap_magnitude({"a"}, set(), {}), 1.0)


class TestDomainDistance(unittest.TestCase):

    def test_same_isco(self):
        G = _build_effort_graph()
        # analyst and ai_eng are both ISCO 25
        self.assertAlmostEqual(domain_distance("analyst", "ai_eng", G), 0.0)

    def test_different_isco(self):
        G = _build_effort_graph()
        # analyst (25) vs surgeon (22)
        self.assertAlmostEqual(domain_distance("analyst", "surgeon", G), 1.0)

    def test_missing_isco(self):
        G = nx.MultiDiGraph()
        G.add_node("r1", type="role", source="esco")
        G.add_node("r2", type="role", source="esco", isco_2digit="25")
        self.assertAlmostEqual(domain_distance("r1", "r2", G), 0.5)


class TestEmpiricalSupport(unittest.TestCase):

    def test_direct_transition(self):
        G = _build_effort_graph()
        support = empirical_support("analyst", "scientist", G)
        expected = min(1.0, 0.12 * math.log2(1 + 847))
        self.assertAlmostEqual(support, expected, places=4)
        self.assertGreater(support, 0.5)

    def test_no_transition(self):
        G = _build_effort_graph()
        self.assertEqual(empirical_support("analyst", "surgeon", G), 0.0)

    def test_missing_source(self):
        G = _build_effort_graph()
        self.assertEqual(empirical_support("nonexistent", "scientist", G), 0.0)

    def test_held_out_transition_is_ignored(self):
        G = _build_effort_graph()
        G.add_edge(
            "analyst", "surgeon", key="karrierewege:test",
            relation="TRANSITIONS_TO", source="karrierewege", split="test",
            probability=0.9, count=100, source_total=100,
        )
        self.assertEqual(empirical_support("analyst", "surgeon", G), 0.0)

    def test_smoother_receives_an_explicit_limit(self):
        G = _build_effort_graph()

        class Smoother:
            def __init__(self):
                self.limit = None

            def rank(self, source_id, limit):
                self.limit = limit
                return []

        smoother = Smoother()
        self.assertEqual(empirical_support("analyst", "surgeon", G, smoother), 0.0)
        self.assertEqual(smoother.limit, 50)


class TestTransferability(unittest.TestCase):

    def test_high_overlap(self):
        G = _build_effort_graph()
        idf = {nid: d["idf"] for nid, d in G.nodes(data=True) if "idf" in d}
        # analyst → scientist: share python, sql, stats (3 of 5 target skills)
        tf = transferability("analyst", "scientist", G, idf)
        self.assertGreater(tf, 0.3)

    def test_no_overlap(self):
        G = _build_effort_graph()
        idf = {nid: d["idf"] for nid, d in G.nodes(data=True) if "idf" in d}
        # analyst → surgeon: no shared skills
        tf = transferability("analyst", "surgeon", G, idf)
        self.assertEqual(tf, 0.0)

    def test_empty_target(self):
        G = _build_effort_graph()
        G.add_node("empty_role", type="role", source="esco")
        idf = {nid: d["idf"] for nid, d in G.nodes(data=True) if "idf" in d}
        self.assertEqual(transferability("analyst", "empty_role", G, idf), 0.0)


class TestTransitionEffortScore(unittest.TestCase):

    def test_easy_transition(self):
        G = _build_effort_graph()
        # Analyst → Scientist: different ISCO (25 vs 21), but high empirical support + transferability
        owned = {"s_python", "s_sql", "s_stats"}
        result = transition_effort_score("analyst", "scientist", owned, G)
        self.assertIn(result.band, ("low", "moderate"))
        self.assertLess(result.score, 0.5)

    def test_hard_transition(self):
        G = _build_effort_graph()
        # Analyst → Surgeon: different domain, no empirical, no transferability
        owned = {"s_python", "s_sql", "s_stats"}
        result = transition_effort_score("analyst", "surgeon", owned, G)
        self.assertEqual(result.band, "high")
        self.assertGreater(result.score, 0.7)

    def test_moderate_transition(self):
        G = _build_effort_graph()
        # Analyst → AI Engineer: same ISCO domain (25), no direct transition, missing ML/DL skills
        owned = {"s_python", "s_sql", "s_stats"}
        result = transition_effort_score("analyst", "ai_eng", owned, G)
        self.assertGreater(result.score, 0.4)
        self.assertLess(result.score, 0.9)

    def test_score_bounded(self):
        G = _build_effort_graph()
        result = transition_effort_score("analyst", "surgeon", set(), G)
        self.assertGreaterEqual(result.score, 0.0)
        self.assertLessEqual(result.score, 1.0)

    def test_to_dict_format(self):
        G = _build_effort_graph()
        result = transition_effort_score("analyst", "scientist", {"s_python"}, G)
        d = result.to_dict()
        self.assertIn("effort_score", d)
        self.assertIn("effort_band", d)
        self.assertIn("components", d)
        self.assertIn("skill_gap_magnitude", d["components"])

    def test_optional_generic_requirement_does_not_inflate_effort(self):
        G = _build_effort_graph()
        owned = {"s_python", "s_sql", "s_stats"}
        baseline = transition_effort_score("analyst", "scientist", owned, G)
        G.add_node("s_english", type="skill", source="esco", title="English", idf=100.0)
        G.add_edge(
            "scientist", "s_english", key="esco:optional-english",
            relation="REQUIRES", source="esco", requirement_level="optional",
        )
        with_optional = transition_effort_score("analyst", "scientist", owned, G)
        self.assertAlmostEqual(baseline.score, with_optional.score)
        self.assertAlmostEqual(
            baseline.skill_gap_magnitude, with_optional.skill_gap_magnitude
        )


class TestRankByEffort(unittest.TestCase):

    def test_low_effort_first(self):
        G = _build_effort_graph()
        candidates = [
            {"id": "surgeon"},
            {"id": "scientist"},
            {"id": "ai_eng"},
        ]
        owned = {"s_python", "s_sql", "s_stats"}
        ranked = rank_roles_by_effort(candidates, "analyst", owned, G)
        # Scientist should be first (easiest), surgeon last (hardest)
        self.assertEqual(ranked[0]["id"], "scientist")
        self.assertEqual(ranked[-1]["id"], "surgeon")

    def test_missing_source(self):
        G = _build_effort_graph()
        candidates = [{"id": "scientist"}]
        ranked = rank_roles_by_effort(candidates, None, set(), G)
        self.assertIsNone(ranked[0]["effort"])


class TestEffortWeights(unittest.TestCase):

    def test_valid_weights(self):
        w = EffortWeights(0.4, 0.2, 0.2, 0.2)
        self.assertAlmostEqual(w.skill_gap + w.domain + w.empirical + w.transferability, 1.0)

    def test_invalid_weights(self):
        with self.assertRaises(ValueError):
            EffortWeights(0.5, 0.5, 0.5, 0.5)


if __name__ == "__main__":
    unittest.main()
