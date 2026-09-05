from __future__ import annotations

import unittest

import networkx as nx

from src.neo4j_store import (
    entity_key,
    node_upload_rows,
    normalize_properties,
    relationship_upload_rows,
)


class Neo4jStoreTests(unittest.TestCase):
    def test_property_normalization_and_entity_key(self) -> None:
        properties = normalize_properties(
            {"good": 1, "bad": float("nan"), "items": ["a", "b"], "mapping": {"b": 2, "a": 1}}
        )
        self.assertEqual(properties["good"], 1)
        self.assertNotIn("bad", properties)
        self.assertEqual(properties["items"], ["a", "b"])
        self.assertEqual(properties["mapping"], '{"a": 1, "b": 2}')
        self.assertEqual(entity_key("graph", "role"), "graph::role")

    def test_upload_rows_preserve_typed_parallel_edges(self) -> None:
        graph = nx.MultiDiGraph()
        graph.add_node("r", type="role", source="esco", title="Role")
        graph.add_node("s", type="skill", source="esco", title="Skill")
        graph.add_edge("r", "s", key="req", relation="REQUIRES", source="esco")
        graph.add_edge("r", "s", key="broad", relation="BROADER_THAN", source="esco")

        nodes = node_upload_rows(graph, "g")
        relationships = relationship_upload_rows(graph, "g")

        self.assertEqual(set(nodes), {"Role", "Skill"})
        self.assertEqual(set(relationships), {"REQUIRES", "BROADER_THAN"})
        self.assertNotEqual(
            relationships["REQUIRES"][0]["edge_key"],
            relationships["BROADER_THAN"][0]["edge_key"],
        )

    def test_rejects_untrusted_dynamic_types(self) -> None:
        graph = nx.MultiDiGraph()
        graph.add_node("x", type="Role`) MATCH (n) DETACH DELETE n //", title="Bad")
        with self.assertRaises(ValueError):
            node_upload_rows(graph, "g")


if __name__ == "__main__":
    unittest.main()
