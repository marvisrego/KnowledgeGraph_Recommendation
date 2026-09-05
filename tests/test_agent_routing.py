"""Regression tests for full versus partial LangGraph routing."""

from __future__ import annotations

import unittest

import networkx as nx

from agents.graph import build_career_graph, _route_after_traversal


class AgentRoutingTests(unittest.TestCase):
    def test_partial_context_routes_to_explore(self) -> None:
        self.assertEqual(_route_after_traversal({"has_context": False}), "explore")
        self.assertEqual(_route_after_traversal({"has_context": True}), "effort")

    def test_workflow_compiles_with_conditional_partial_route(self) -> None:
        app = build_career_graph(object(), nx.MultiDiGraph(), object())
        destinations = {
            edge.target
            for edge in app.get_graph().edges
            if edge.source == "traversal"
        }
        self.assertEqual(destinations, {"effort", "explore"})


if __name__ == "__main__":
    unittest.main()
