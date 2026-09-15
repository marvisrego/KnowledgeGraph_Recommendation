"""Evaluate adaptive embedding smoothing on ordered, full-candidate prefixes."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import Settings
from evaluation.research_resources import load_research_embeddings, load_research_graph
from src.karrierewege_preprocessing import build_esco_title_index, discover_split_files, write_json_atomic
from src.trajectory_benchmark import evaluate_prefix_rankings, iter_karrierewege_prefixes
from src.transition_embedding import SmoothingConfig, compute_neighbour_index, rank_hybrid_destinations, transition_distributions


def main() -> None:
    parser = argparse.ArgumentParser(description="Full-candidate prefix benchmark for adaptive transition smoothing")
    parser.add_argument("--output", type=Path, default=Path("artifacts/karrierewege/prefix_smoothing_evaluation.json"))
    parser.add_argument("--neighbours", type=int, default=25)
    parser.add_argument("--temperature", type=float, default=0.06)
    parser.add_argument("--prior-strength", type=float, required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=500)
    args = parser.parse_args()

    settings = Settings.from_env()
    graph, graph_source = load_research_graph(settings)
    title_index = build_esco_title_index(graph)
    split_files = discover_split_files(settings.karrierewege_dir)
    embeddings, embedding_report = load_research_embeddings(settings, graph)
    distributions = transition_distributions(graph)
    live_roles = sorted(
        str(node_id) for node_id, data in graph.nodes(data=True)
        if data.get("type") == "role" and data.get("source") == "esco"
    )
    config = SmoothingConfig(args.neighbours, 0.0, args.temperature, args.prior_strength)

    def ranker(history):
        source_id = str(history[-1])
        neighbours = compute_neighbour_index(
            embeddings, distributions, max_neighbours=config.neighbours, query_ids=[source_id]
        )
        observed = [item.role_id for item in rank_hybrid_destinations(source_id, distributions, neighbours, config, graph)]
        observed_set = set(observed)
        # The metric is genuinely full-candidate: unscored live occupations are
        # retained deterministically after scored destinations.
        return observed + [role_id for role_id in live_roles if role_id not in observed_set and role_id != source_id]

    results = {}
    for split in ("validation", "test"):
        examples = iter_karrierewege_prefixes(split_files[split], title_index, split, settings.transition_chunk_size)
        results[split] = evaluate_prefix_rankings(examples, ranker, bootstrap_samples=args.bootstrap_samples)
    payload = {
        "model": "adaptive_embedding_smoother",
        "config": config.to_dict(),
        "candidate_roles": len(live_roles),
        "embedding_report": embedding_report,
        "graph_source": graph_source,
        "results": results,
    }
    write_json_atomic(payload, args.output)
    print(f"[prefix_ranking] Wrote {args.output}")


if __name__ == "__main__":
    main()
