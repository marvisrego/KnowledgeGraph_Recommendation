from __future__ import annotations

import unittest

import networkx as nx

from src.career_history import resolve_career_history


class CareerHistoryTests(unittest.TestCase):
    def test_order_duplicates_unknowns_and_current_alignment(self) -> None:
        graph = nx.MultiDiGraph()
        graph.add_node("analyst", type="role", source="esco", title="Data Analyst")
        graph.add_node("engineer", type="role", source="esco", title="Data Engineer")
        history = resolve_career_history(
            ["Data Analyst", "Data Analyst", "unknown role"], "engineer", graph
        )
        self.assertEqual(history, ["analyst", "engineer"])

    def test_history_is_capped_from_the_newest_side(self) -> None:
        graph = nx.MultiDiGraph()
        for index in range(4):
            graph.add_node(f"r{index}", type="role", source="esco", title=f"Role {index}")
        history = resolve_career_history([f"Role {index}" for index in range(4)], "r3", graph, maximum_length=2)
        self.assertEqual(history, ["r2", "r3"])


if __name__ == "__main__":
    unittest.main()
