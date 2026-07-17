"""
Graph construction: merges ONET and ESCO nodes/edges into a single NetworkX
DiGraph, and provides save/load utilities.

NetworkX 3.x removed nx.write_gpickle — we use stdlib pickle directly.
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import TYPE_CHECKING

import networkx as nx
import pandas as pd

if TYPE_CHECKING:
    from config import Settings


# ---------------------------------------------------------------------------
# Build
# ---------------------------------------------------------------------------

def build_graph(
    onet_nodes: pd.DataFrame,
    onet_edges: pd.DataFrame,
    esco_nodes: pd.DataFrame,
    esco_edges: pd.DataFrame,
) -> nx.DiGraph:
    """Merge all nodes and edges into a single directed graph."""
    G = nx.DiGraph()

    def _is_present(v) -> bool:
        """Return False only for scalar NA/None; True for arrays and non-null scalars."""
        if isinstance(v, (list, dict)):
            return True
        try:
            return bool(pd.notna(v))
        except (ValueError, TypeError):
            return True

    # --- Nodes ---
    for df in (onet_nodes, esco_nodes):
        for _, row in df.iterrows():
            node_id = str(row["id"])
            attrs = {k: v for k, v in row.items() if k != "id" and _is_present(v)}
            # Convert list columns (interests, interests_keywords) that may
            # have been serialised as strings back to lists
            for list_col in ("interests", "interests_keywords"):
                if list_col in attrs and isinstance(attrs[list_col], str):
                    try:
                        import ast
                        attrs[list_col] = ast.literal_eval(attrs[list_col])
                    except Exception:
                        attrs[list_col] = []
            G.add_node(node_id, **attrs)

    # --- Edges ---
    for df in (onet_edges, esco_edges):
        if df.empty:
            continue
        for _, row in df.iterrows():
            src = str(row["src"])
            dst = str(row["dst"])
            if not G.has_node(src) or not G.has_node(dst):
                continue
            attrs = {k: v for k, v in row.items() if k not in ("src", "dst") and _is_present(v)}
            G.add_edge(src, dst, **attrs)

    return G


# ---------------------------------------------------------------------------
# SIMILAR_TO alignment edges (added after ChromaDB embedding step)
# ---------------------------------------------------------------------------

def add_alignment_edges(
    G: nx.DiGraph,
    alignment_pairs: list[tuple[str, str, float]],
) -> None:
    """Add bidirectional SIMILAR_TO edges between ONET and ESCO role nodes.

    alignment_pairs: list of (onet_node_id, esco_node_id, similarity_score)
    """
    for onet_id, esco_id, score in alignment_pairs:
        if G.has_node(onet_id) and G.has_node(esco_id):
            G.add_edge(
                onet_id,
                esco_id,
                relation="SIMILAR_TO",
                source="alignment",
                similarity=float(score),
            )
            G.add_edge(
                esco_id,
                onet_id,
                relation="SIMILAR_TO",
                source="alignment",
                similarity=float(score),
            )


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

def save_graph(G: nx.DiGraph, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with open(path, "wb") as fh:
            pickle.dump(G, fh, protocol=pickle.HIGHEST_PROTOCOL)
    except Exception as exc:
        raise RuntimeError(f"[graph_build] Failed to save graph to {path}: {exc}") from exc


def load_graph(path: Path) -> nx.DiGraph:
    path = Path(path)
    try:
        with open(path, "rb") as fh:
            G = pickle.load(fh)
        if not isinstance(G, nx.DiGraph):
            raise TypeError(f"Expected nx.DiGraph, got {type(G)}")
        return G
    except FileNotFoundError:
        raise RuntimeError(
            f"[graph_build] Graph file not found: {path}. "
            "Run 'python build_graph.py' first."
        ) from None
    except Exception as exc:
        raise RuntimeError(f"[graph_build] Failed to load graph from {path}: {exc}") from exc


# ---------------------------------------------------------------------------
# Orchestration helper
# ---------------------------------------------------------------------------

def build_all(settings: "Settings") -> nx.DiGraph:
    """Full offline graph build: ONET + ESCO → NetworkX DiGraph → disk."""
    from src.onet_preprocessing import build_onet
    from src.esco_preprocessing import build_esco

    print("[graph_build] Processing ONET …")
    onet_nodes, onet_edges = build_onet(settings.onet_dir)
    print(
        f"[graph_build] ONET: {len(onet_nodes)} nodes, {len(onet_edges)} edges"
    )

    print("[graph_build] Processing ESCO …")
    esco_nodes, esco_edges = build_esco(settings.esco_dir)
    print(
        f"[graph_build] ESCO: {len(esco_nodes)} nodes, {len(esco_edges)} edges"
    )

    print("[graph_build] Building NetworkX graph …")
    G = build_graph(onet_nodes, onet_edges, esco_nodes, esco_edges)
    print(
        f"[graph_build] Graph: {G.number_of_nodes()} nodes, "
        f"{G.number_of_edges()} edges"
    )

    print(f"[graph_build] Saving to {settings.graph_path} …")
    save_graph(G, settings.graph_path)
    print("[graph_build] Done.")
    return G
