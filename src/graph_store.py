"""Runtime-only knowledge-graph loading utilities."""

from __future__ import annotations

import pickle
from pathlib import Path

import networkx as nx


def ensure_multidigraph(graph: nx.Graph) -> nx.MultiDiGraph:
    """Return ``graph`` as a typed-edge MultiDiGraph without changing IDs."""
    if isinstance(graph, nx.MultiDiGraph):
        return graph
    if not isinstance(graph, nx.DiGraph):
        raise TypeError(f"Expected a directed graph, got {type(graph)}")

    migrated = nx.MultiDiGraph()
    migrated.graph.update(graph.graph)
    migrated.add_nodes_from(graph.nodes(data=True))
    for source, target, data in graph.edges(data=True):
        attrs = dict(data)
        relation = str(attrs.get("relation", "UNKNOWN"))
        edge_source = str(attrs.get("source", "unknown"))
        migrated.add_edge(
            str(source),
            str(target),
            key=f"{edge_source}:{relation}",
            **attrs,
        )
    return migrated


def load_graph(path: Path) -> nx.MultiDiGraph:
    """Load the packaged NetworkX graph from disk."""
    path = Path(path)
    try:
        with path.open("rb") as handle:
            graph = pickle.load(handle)
        return ensure_multidigraph(graph)
    except FileNotFoundError:
        raise RuntimeError(f"[graph_store] Packaged graph not found: {path}.") from None
    except Exception as exc:
        raise RuntimeError(f"[graph_store] Failed to load graph from {path}: {exc}") from exc
