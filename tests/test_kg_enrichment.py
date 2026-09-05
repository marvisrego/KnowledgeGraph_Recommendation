"""Unit tests for src/kg_enrichment.py — skill IDF and ISCO enrichment."""

import math
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import networkx as nx
from src.kg_enrichment import (
    compute_isco_2digit,
    compute_skill_idf,
    enrich_graph,
    role_skills,
    skill_degree,
)


def _build_mini_graph() -> nx.MultiDiGraph:
    """Small graph: 3 roles, 4 skills, various REQUIRES edges."""
    G = nx.MultiDiGraph()

    # Roles
    G.add_node("role_a", type="role", source="esco", title="Data Analyst", isco_group="2511")
    G.add_node("role_b", type="role", source="esco", title="Data Scientist", isco_group="2120")
    G.add_node("role_c", type="role", source="onet", title="Software Dev", soc_code="15-1252.00")

    # Skills
    G.add_node("skill_python", type="skill", source="esco", title="Python")
    G.add_node("skill_sql", type="skill", source="esco", title="SQL")
    G.add_node("skill_ml", type="skill", source="esco", title="machine learning")
    G.add_node("skill_rare", type="skill", source="esco", title="quantum computing")

    # REQUIRES edges
    # role_a requires python, sql (2 skills)
    G.add_edge("role_a", "skill_python", key="esco:REQUIRES", relation="REQUIRES", source="esco", requirement_level="essential")
    G.add_edge("role_a", "skill_sql", key="esco:REQUIRES", relation="REQUIRES", source="esco", requirement_level="essential")

    # role_b requires python, sql, ml (3 skills)
    G.add_edge("role_b", "skill_python", key="esco:REQUIRES", relation="REQUIRES", source="esco", requirement_level="essential")
    G.add_edge("role_b", "skill_sql", key="esco:REQUIRES", relation="REQUIRES", source="esco", requirement_level="essential")
    G.add_edge("role_b", "skill_ml", key="esco:REQUIRES", relation="REQUIRES", source="esco", requirement_level="essential")

    # role_c requires python, rare (2 skills)
    G.add_edge("role_c", "skill_python", key="onet:REQUIRES", relation="REQUIRES", source="onet", requirement_level=3.5)
    G.add_edge("role_c", "skill_rare", key="onet:REQUIRES", relation="REQUIRES", source="onet", requirement_level=4.0)

    return G


class TestSkillIDF(unittest.TestCase):

    def test_idf_common_vs_rare(self):
        G = _build_mini_graph()
        idf_map = compute_skill_idf(G)
        # Python is required by 3/3 roles → IDF = log(3/3) = 0
        self.assertAlmostEqual(idf_map["skill_python"], math.log(3 / 3), places=6)
        # SQL is required by 2/3 roles → IDF = log(3/2)
        self.assertAlmostEqual(idf_map["skill_sql"], math.log(3 / 2), places=6)
        # ML is required by 1/3 roles → IDF = log(3/1)
        self.assertAlmostEqual(idf_map["skill_ml"], math.log(3 / 1), places=6)
        # Rare is required by 1/3 roles → IDF = log(3/1)
        self.assertAlmostEqual(idf_map["skill_rare"], math.log(3 / 1), places=6)

    def test_idf_stored_on_nodes(self):
        G = _build_mini_graph()
        compute_skill_idf(G)
        self.assertIn("idf", G.nodes["skill_python"])
        self.assertAlmostEqual(G.nodes["skill_python"]["idf"], 0.0, places=6)
        self.assertGreater(G.nodes["skill_ml"]["idf"], 0.0)

    def test_empty_graph(self):
        G = nx.MultiDiGraph()
        idf_map = compute_skill_idf(G)
        self.assertEqual(idf_map, {})


class TestISCO2Digit(unittest.TestCase):

    def test_esco_roles_get_isco_2digit(self):
        G = _build_mini_graph()
        isco_map = compute_isco_2digit(G)
        self.assertEqual(isco_map["role_a"], "25")
        self.assertEqual(isco_map["role_b"], "21")
        # ONET role should NOT get isco_2digit from this function
        self.assertNotIn("role_c", isco_map)

    def test_stored_on_nodes(self):
        G = _build_mini_graph()
        compute_isco_2digit(G)
        self.assertEqual(G.nodes["role_a"]["isco_2digit"], "25")

    def test_missing_isco_group(self):
        G = nx.MultiDiGraph()
        G.add_node("r1", type="role", source="esco", title="No ISCO")
        isco_map = compute_isco_2digit(G)
        self.assertNotIn("r1", isco_map)


class TestRoleSkills(unittest.TestCase):

    def test_returns_correct_skills(self):
        G = _build_mini_graph()
        skills = role_skills("role_b", G)
        self.assertEqual(skills, {"skill_python", "skill_sql", "skill_ml"})

    def test_missing_node(self):
        G = _build_mini_graph()
        self.assertEqual(role_skills("nonexistent", G), set())


class TestSkillDegree(unittest.TestCase):

    def test_python_degree(self):
        G = _build_mini_graph()
        self.assertEqual(skill_degree("skill_python", G), 3)

    def test_rare_skill_degree(self):
        G = _build_mini_graph()
        self.assertEqual(skill_degree("skill_rare", G), 1)


class TestEnrichGraph(unittest.TestCase):

    def test_enrich_adds_all_attributes(self):
        G = _build_mini_graph()
        enrich_graph(G)
        # IDF attributes
        self.assertIn("idf", G.nodes["skill_python"])
        # ISCO attributes
        self.assertEqual(G.nodes["role_a"].get("isco_2digit"), "25")


if __name__ == "__main__":
    unittest.main()
