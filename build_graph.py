"""
Offline graph build CLI.

Usage:
  python build_graph.py           # build graph only (graph/graph.gpickle)
  python build_graph.py --embed   # also build ChromaDB index + alignment edges
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Ensure project root is on sys.path when called directly
sys.path.insert(0, str(Path(__file__).parent))

from config import Settings
from src.graph_build import build_all, load_graph, save_graph, add_alignment_edges
from src.embeddings_index import build_chroma_index, compute_alignment_edges


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the GraphRAG knowledge graph.")
    parser.add_argument(
        "--embed",
        action="store_true",
        help="Also build the ChromaDB embedding index and compute alignment edges.",
    )
    parser.add_argument(
        "--align-only",
        action="store_true",
        help=(
            "Assume the graph pickle and ChromaDB index already exist; "
            "only recompute ONET–ESCO alignment edges and re-save the graph."
        ),
    )
    args = parser.parse_args()

    settings = Settings.from_env()

    if args.align_only:
        print("[build_graph] Loading existing graph …")
        G = load_graph(settings.graph_path)
        print(f"[build_graph] Graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
        pairs = compute_alignment_edges(G, settings)
        add_alignment_edges(G, pairs)
        save_graph(G, settings.graph_path)
        print("[build_graph] Alignment edges added and graph saved.")
        return

    # Full build
    G = build_all(settings)

    if args.embed:
        print("[build_graph] Building ChromaDB embedding index …")
        build_chroma_index(G, settings)

        print("[build_graph] Computing ONET–ESCO alignment edges …")
        pairs = compute_alignment_edges(G, settings)
        add_alignment_edges(G, pairs)
        print(f"[build_graph] Added {len(pairs)} SIMILAR_TO alignment edges.")

        # Re-save the graph with alignment edges included
        save_graph(G, settings.graph_path)
        print(f"[build_graph] Graph with alignment re-saved to {settings.graph_path}")

    # Summary
    role_nodes = sum(
        1 for _, d in G.nodes(data=True) if d.get("type") == "role"
    )
    print(
        f"\n[build_graph] Final graph: "
        f"{G.number_of_nodes()} nodes ({role_nodes} roles), "
        f"{G.number_of_edges()} edges"
    )


if __name__ == "__main__":
    main()
