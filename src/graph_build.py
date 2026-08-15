"""
Graph construction: merges ONET and ESCO nodes/edges into a single NetworkX
DiGraph, and provides save/load utilities.

NetworkX 3.x removed nx.write_gpickle — we use stdlib pickle directly.
"""

from __future__ import annotations

import os
import pickle
import tempfile
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
) -> nx.MultiDiGraph:
    """Merge all nodes and edges into a single directed graph."""
    G = nx.MultiDiGraph()

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
            relation = str(attrs.get("relation", "UNKNOWN"))
            source = str(attrs.get("source", "unknown"))
            G.add_edge(src, dst, key=f"{source}:{relation}", **attrs)

    return G


# ---------------------------------------------------------------------------
# SIMILAR_TO alignment edges (added after ChromaDB embedding step)
# ---------------------------------------------------------------------------

def add_alignment_edges(
    G: nx.MultiDiGraph,
    alignment_pairs: list[tuple[str, str, float]],
) -> None:
    """Add bidirectional SIMILAR_TO edges between ONET and ESCO role nodes.

    alignment_pairs: list of (onet_node_id, esco_node_id, similarity_score)
    """
    existing = [
        (source, target, key)
        for source, target, key, data in G.edges(keys=True, data=True)
        if data.get("relation") == "SIMILAR_TO" and data.get("source") == "alignment"
    ]
    G.remove_edges_from(existing)

    for onet_id, esco_id, score in alignment_pairs:
        if G.has_node(onet_id) and G.has_node(esco_id):
            G.add_edge(
                onet_id,
                esco_id,
                key="alignment:SIMILAR_TO",
                relation="SIMILAR_TO",
                source="alignment",
                similarity=float(score),
            )
            G.add_edge(
                esco_id,
                onet_id,
                key="alignment:SIMILAR_TO",
                relation="SIMILAR_TO",
                source="alignment",
                similarity=float(score),
            )


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

def ensure_multidigraph(G: nx.Graph) -> nx.MultiDiGraph:
    """Return ``G`` as a typed-edge MultiDiGraph without changing node IDs."""
    if isinstance(G, nx.MultiDiGraph):
        return G
    if not isinstance(G, nx.DiGraph):
        raise TypeError(f"Expected a directed graph, got {type(G)}")

    migrated = nx.MultiDiGraph()
    migrated.graph.update(G.graph)
    migrated.add_nodes_from(G.nodes(data=True))
    for source, target, data in G.edges(data=True):
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


def save_graph(G: nx.MultiDiGraph, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    handle: int | None = None
    temp_name: str | None = None
    try:
        handle, temp_name = tempfile.mkstemp(
            prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
        )
        with os.fdopen(handle, "wb") as fh:
            handle = None
            pickle.dump(G, fh, protocol=pickle.HIGHEST_PROTOCOL)
        os.replace(temp_name, path)
        temp_name = None
    except Exception as exc:
        raise RuntimeError(f"[graph_build] Failed to save graph to {path}: {exc}") from exc
    finally:
        if handle is not None:
            os.close(handle)
        if temp_name is not None:
            try:
                os.unlink(temp_name)
            except FileNotFoundError:
                pass


def load_graph(path: Path) -> nx.MultiDiGraph:
    path = Path(path)
    try:
        with open(path, "rb") as fh:
            G = pickle.load(fh)
        return ensure_multidigraph(G)
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

def prune_graph(G: nx.MultiDiGraph, min_role_edges: int = 3) -> nx.MultiDiGraph:
    """Remove low-quality nodes to keep the graph compact and high-signal.

    Steps:
      1. Remove role nodes with fewer than min_role_edges REQUIRES edges —
         they lack enough information to generate useful recommendations.
      2. Remove all isolated nodes (degree 0) left after step 1.

    Returns the pruned graph (mutates in-place and returns it).
    """
    # Step 1: thin roles
    thin_roles = [
        nid for nid, d in G.nodes(data=True)
        if d.get("type") == "role"
        and sum(1 for _, _, ed in G.out_edges(nid, data=True) if ed.get("relation") == "REQUIRES") < min_role_edges
    ]
    G.remove_nodes_from(thin_roles)
    print(f"[graph_build] Pruned {len(thin_roles)} thin role nodes (< {min_role_edges} REQUIRES edges).")

    # Step 2: isolates
    isolates = list(nx.isolates(G))
    G.remove_nodes_from(isolates)
    print(f"[graph_build] Pruned {len(isolates)} isolated nodes.")

    return G


def build_all(settings: "Settings") -> nx.MultiDiGraph:
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

    print("[graph_build] Pruning low-quality nodes …")
    G = prune_graph(G)
    print(
        f"[graph_build] After pruning: {G.number_of_nodes()} nodes, "
        f"{G.number_of_edges()} edges"
    )

    print(f"[graph_build] Saving to {settings.graph_path} …")
    save_graph(G, settings.graph_path)
    print("[graph_build] Done.")
    return G
