from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import networkx as nx

from src.graph_build import add_alignment_edges, ensure_multidigraph, load_graph, save_graph


class GraphBuildTests(unittest.TestCase):
    def test_legacy_graph_migration_preserves_attributes(self) -> None:
        legacy = nx.DiGraph(project="career-kg")
        legacy.add_node("a", type="role", title="A")
        legacy.add_node("b", type="role", title="B")
        legacy.add_edge("a", "b", relation="BROADER_THAN", source="esco", pillar="occupations")

        graph = ensure_multidigraph(legacy)

        self.assertIsInstance(graph, nx.MultiDiGraph)
        self.assertEqual(graph.graph["project"], "career-kg")
        self.assertEqual(graph.nodes["a"]["title"], "A")
        self.assertEqual(
            graph["a"]["b"]["esco:BROADER_THAN"]["pillar"], "occupations"
        )

    def test_alignment_replacement_is_idempotent(self) -> None:
        graph = nx.MultiDiGraph()
        graph.add_node("onet", type="role", source="onet")
        graph.add_node("esco", type="role", source="esco")

        add_alignment_edges(graph, [("onet", "esco", 0.8)])
        add_alignment_edges(graph, [("onet", "esco", 0.9)])

        self.assertEqual(graph.number_of_edges(), 2)
        self.assertEqual(
            graph["onet"]["esco"]["alignment:SIMILAR_TO"]["similarity"], 0.9
        )

    def test_atomic_round_trip_loads_a_multidigraph(self) -> None:
        graph = nx.MultiDiGraph()
        graph.add_node("a", type="role")
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "graph.gpickle"
            save_graph(graph, path)
            loaded = load_graph(path)
        self.assertIsInstance(loaded, nx.MultiDiGraph)
        self.assertTrue(loaded.has_node("a"))


if __name__ == "__main__":
    unittest.main()
