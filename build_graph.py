"""
Offline graph build CLI.

Usage:
  python build_graph.py           # build graph only (graph/graph.gpickle)
  python build_graph.py --embed   # also build ChromaDB index + alignment edges
  python build_graph.py --transitions-only  # enrich existing graph, keep index/alignment
  python build_graph.py --transitions        # include transitions in a full build
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
from src.karrierewege_preprocessing import (
    add_transition_edges,
    aggregate_split_transitions,
    build_esco_title_index,
    discover_split_files,
    transition_rows,
    write_json_atomic,
)


def _enrich_with_transitions(G, settings: Settings, min_support: int, chunk_size: int) -> dict:
    split_files = discover_split_files(settings.karrierewege_dir)
    title_index = build_esco_title_index(G)
    print(f"[build_graph] Processing Karrierewege training split: {split_files['train']}")
    aggregate = aggregate_split_transitions(
        split_files["train"],
        "train",
        title_index,
        chunk_size=chunk_size,
        strict_mapping=True,
        max_invalid_row_ratio=settings.karrierewege_max_invalid_row_ratio,
    )
    rows = transition_rows(aggregate, title_index, min_support=min_support)
    added = add_transition_edges(G, rows)
    report = {
        "policy": {
            "graph_training_split": "train",
            "held_out_splits": ["validation", "test"],
            "min_support": min_support,
            "chunk_size": chunk_size,
            "max_invalid_row_ratio": settings.karrierewege_max_invalid_row_ratio,
        },
        "train": aggregate.report.to_dict(),
        "graph": {
            "transition_edges": added,
            "nodes": G.number_of_nodes(),
            "edges": G.number_of_edges(),
        },
    }
    write_json_atomic(report, settings.karrierewege_report_path)
    print(
        f"[build_graph] Added {added} TRANSITIONS_TO edges; "
        f"report: {settings.karrierewege_report_path}"
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the GraphRAG knowledge graph.")
    parser.add_argument(
        "--embed",
        action="store_true",
        help="Also build the ChromaDB embedding index and compute alignment edges.",
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--align-only",
        action="store_true",
        help=(
            "Assume the graph pickle and ChromaDB index already exist; "
            "only recompute ONET–ESCO alignment edges and re-save the graph."
        ),
    )
    mode.add_argument(
        "--transitions-only",
        action="store_true",
        help=(
            "Load the existing graph and replace only Karrierewege transition edges; "
            "preserves nodes, ChromaDB, and existing alignment edges."
        ),
    )
    mode.add_argument(
        "--transitions",
        action="store_true",
        help="Include Karrierewege training transitions in a full graph build.",
    )
    parser.add_argument(
        "--transition-min-count",
        type=int,
        default=None,
        help="Minimum observed training support for a TRANSITIONS_TO edge.",
    )
    parser.add_argument(
        "--transition-chunk-size",
        type=int,
        default=None,
        help="CSV rows processed per Karrierewege chunk.",
    )
    args = parser.parse_args()

    settings = Settings.from_env()
    min_support = (
        args.transition_min_count
        if args.transition_min_count is not None
        else settings.transition_min_count
    )
    chunk_size = (
        args.transition_chunk_size
        if args.transition_chunk_size is not None
        else settings.transition_chunk_size
    )

    if min_support < 1:
        parser.error("--transition-min-count must be at least 1")
    if chunk_size < 2:
        parser.error("--transition-chunk-size must be at least 2")

    if args.transitions_only:
        print("[build_graph] Loading existing graph …")
        G = load_graph(settings.graph_path)
        print(f"[build_graph] Graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
        _enrich_with_transitions(G, settings, min_support, chunk_size)
        save_graph(G, settings.graph_path)
        print("[build_graph] Transition-enriched graph saved.")
        return

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

    if args.transitions:
        _enrich_with_transitions(G, settings, min_support, chunk_size)
        save_graph(G, settings.graph_path)
        print(f"[build_graph] Graph with transitions re-saved to {settings.graph_path}")

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
