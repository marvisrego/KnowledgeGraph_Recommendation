from __future__ import annotations

import unittest

import networkx as nx

from src.graph_quality import improve_graph_quality


class GraphQualityCloudTests(unittest.TestCase):
    def test_cleans_invalid_evidence_and_adds_source_backed_entities(self) -> None:
        graph = nx.MultiDiGraph()
        graph.add_node(
            "role",
            type="role",
            source="onet",
            title="  Data   Scientist ",
            typical_education_level="Bachelor's degree",
            job_zone=4,
            job_zone_title="Considerable Preparation Needed",
        )
        graph.add_node("skill", type="element", source="onet", title="Mathematics")
        graph.add_node("bad", type="unknown", source="test", title="Bad")
        graph.add_node("isolated", type="skill", source="esco", title="Unused")
        graph.add_edge("role", "skill", key="a", relation="REQUIRES", source="onet", requirement_level=4.0)
        graph.add_edge("role", "skill", key="b", relation="REQUIRES", source="onet", requirement_level=3.0)
        graph.add_edge("role", "role", relation="SIMILAR_TO", source="alignment", similarity=1.0)
        graph.add_edge(
            "role",
            "skill",
            key="held-out",
            relation="TRANSITIONS_TO",
            source="karrierewege",
            split="test",
            probability=0.5,
            count=1,
        )

        cleaned, report = improve_graph_quality(graph)

        self.assertNotIn("bad", cleaned)
        self.assertNotIn("isolated", cleaned)
        self.assertEqual(cleaned.nodes["role"]["title"], "Data Scientist")
        self.assertEqual(cleaned.nodes["role"]["normalized_title"], "data scientist")
        self.assertEqual(report.removed_duplicate_edges, 1)
        self.assertEqual(report.removed_self_loops, 1)
        self.assertEqual(report.removed_held_out_transitions, 1)
        self.assertEqual(report.added_qualification_nodes, 1)
        self.assertEqual(report.added_job_zone_nodes, 1)
        relations = {data["relation"] for _, _, data in cleaned.edges(data=True)}
        self.assertIn("TYPICALLY_REQUIRES_QUALIFICATION", relations)
        self.assertIn("IN_JOB_ZONE", relations)


if __name__ == "__main__":
    unittest.main()
