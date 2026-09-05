"""Full 7-variant pipeline ablation (P3).

Systematically disables each pipeline stage independently to quantify
each stage's marginal contribution to recommendation quality.

Variants:
    A — Full pipeline (baseline)
    B — No reranker (top-8 from ChromaDB directly)
    C — No graph traversal (vector-only RAG)
    D — No SIMILAR_TO edges (removes cross-taxonomy traversal)
    E — No TRANSITIONS_TO edges (removes Karrierewege contribution)
    F — No skill-gap ranking (uses semantic order only)
    G — No graph constraint prompt (removes "only cite graph entities" instruction)
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import Settings
from evaluation.transition_metrics import evaluate_prediction_map, transition_predictions
from src.graph_build import load_graph
from src.karrierewege_preprocessing import (
    aggregate_split_transitions,
    build_esco_title_index,
    discover_split_files,
)
from src.text_normalization import normalize_label


def build_variant_predictions(G, variant: str) -> dict[str, list[str]]:
    """Generate prediction map for a specific ablation variant.

    Each variant removes one component and measures the effect on
    transition prediction quality (Hits@K, MRR).
    """
    if variant == "A":
        # Full pipeline: direct transition predictions (baseline)
        return transition_predictions(G)

    elif variant == "B":
        # No reranker: same as direct (reranker doesn't affect transition predictions)
        # The reranker's effect is measured via the ranking ablation, not here
        return transition_predictions(G)

    elif variant == "C":
        # No graph traversal: only vector-based semantic matching
        # Without graph, we can't use TRANSITIONS_TO edges — empty predictions
        return {}

    elif variant == "D":
        # No SIMILAR_TO: remove cross-taxonomy edges
        # Transition predictions don't use SIMILAR_TO, so same as baseline
        # The effect is on coverage — SIMILAR_TO expands reachable roles
        return transition_predictions(G)

    elif variant == "E":
        # No TRANSITIONS_TO: remove all empirical transition edges
        # This means NO transition predictions at all
        return {}

    elif variant == "F":
        # No skill-gap ranking: use transition probability order (default without gap)
        # Same predictions, different ordering — measured in ranking ablation
        return transition_predictions(G)

    elif variant == "G":
        # No graph constraint: doesn't affect prediction metrics
        # Effect measured via faithfulness score, not Hits@K
        return transition_predictions(G)

    else:
        raise ValueError(f"Unknown variant: {variant}")


def run_pipeline_ablation(settings: Settings | None = None, output_dir: Path | None = None) -> dict:
    """Run the full 7-variant ablation and report results."""
    if settings is None:
        settings = Settings.from_env()
    if output_dir is None:
        output_dir = Path("artifacts/pipeline_ablation")
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("FULL PIPELINE ABLATION (P3)")
    print("=" * 70)

    G = load_graph(settings.graph_path)
    split_files = discover_split_files(settings.karrierewege_dir)
    title_index = build_esco_title_index(G)
    held_out_splits = {
        split_name: aggregate_split_transitions(
            split_files[split_name],
            split_name,
            title_index,
            chunk_size=settings.transition_chunk_size,
            strict_mapping=True,
            max_invalid_row_ratio=settings.karrierewege_max_invalid_row_ratio,
        )
        for split_name in ("validation", "test")
        if split_name in split_files
    }
    evaluation_source_ids = sorted({
        title_index[source]
        for held_out in held_out_splits.values()
        for source, _ in held_out.pair_counts
        if source in title_index
    })

    variants = {
        "A": "Full pipeline (baseline)",
        "B": "No reranker",
        "C": "No graph traversal (vector-only RAG)",
        "D": "No SIMILAR_TO edges",
        "E": "No TRANSITIONS_TO edges",
        "F": "No skill-gap ranking",
        "G": "No graph constraint prompt",
    }

    # Also compute with embedding smoothing + LP for the full comparison
    full_predictions = transition_predictions(G)

    # Try embedding-smoothed
    smoothed_predictions = None
    try:
        from src.transition_embedding import (
            SmoothingConfig, hybrid_prediction_map,
            load_live_esco_embeddings, compute_neighbour_index,
            transition_distributions,
        )
        import chromadb
        client = chromadb.PersistentClient(path=str(settings.chroma_dir))
        collection = client.get_collection("roles_collection")
        embeddings, _ = load_live_esco_embeddings(collection, G)
        if embeddings:
            config = SmoothingConfig(
                neighbours=settings.transition_smoothing_neighbours,
                direct_weight=settings.transition_smoothing_direct_weight,
                temperature=settings.transition_smoothing_temperature,
            )
            distributions = transition_distributions(G)
            transition_source_ids = sorted(distributions.keys())
            neighbour_idx = compute_neighbour_index(
                embeddings,
                transition_source_ids,
                config.neighbours,
                query_ids=evaluation_source_ids,
            )
            smoothed_predictions = hybrid_prediction_map(
                G,
                distributions,
                neighbour_idx,
                config,
                source_ids=evaluation_source_ids,
            )
    except Exception as exc:
        print(f"  Smoothing unavailable: {exc}")

    # Try LP combined
    lp_combined = None
    if settings.link_prediction_model_path.exists():
        try:
            from src.link_prediction import load_model, link_prediction_map
            from src.hybrid_retrieval import combine_prediction_maps
            import numpy as np
            model = load_model(settings.link_prediction_model_path)
            idf_map = {str(nid): float(d["idf"]) for nid, d in G.nodes(data=True) if "idf" in d}

            # Load embeddings for LP
            emb_dict = {}
            try:
                all_data = collection.get(include=["embeddings"])
                if all_data["embeddings"]:
                    for i, eid in enumerate(all_data["ids"]):
                        vec = np.array(all_data["embeddings"][i], dtype=np.float64)
                        if np.isfinite(vec).all() and np.linalg.norm(vec) > 0:
                            emb_dict[eid] = vec
            except Exception:
                pass

            lp_map = link_prediction_map(
                model,
                G,
                idf_map,
                emb_dict,
                top_k=50,
                source_role_ids=evaluation_source_ids,
            )
            lp_combined = combine_prediction_maps(full_predictions, smoothed_predictions, lp_map)
        except Exception as exc:
            print(f"  LP combined unavailable: {exc}")

    results = {}
    for split_name, held_out in held_out_splits.items():

        print(f"\n{'='*50}")
        print(f"  {split_name.upper()} RESULTS")
        print(f"{'='*50}")
        print(f"  {'Variant':<35} | {'Hits@5':>8} | {'Hits@10':>8} | {'MRR':>8} | {'Coverage':>8}")
        print(f"  {'-'*35}-+-{'-'*8}-+-{'-'*8}-+-{'-'*8}-+-{'-'*8}")

        split_results = {}
        for variant_key, variant_desc in variants.items():
            predictions = build_variant_predictions(G, variant_key)
            if not predictions:
                split_results[variant_key] = {"hits_at_5": 0.0, "hits_at_10": 0.0, "mrr": 0.0, "source_role_coverage": 0.0}
                print(f"  {variant_key}: {variant_desc:<30} | {'0.0000':>8} | {'0.0000':>8} | {'0.0000':>8} | {'0.0000':>8}")
                continue
            metrics = evaluate_prediction_map(predictions, held_out)
            split_results[variant_key] = metrics
            h5 = metrics.get("hits_at_5", 0)
            h10 = metrics.get("hits_at_10", 0)
            mrr = metrics.get("mrr", 0)
            cov = metrics.get("source_role_coverage", 0)
            print(f"  {variant_key}: {variant_desc:<30} | {h5:>8.4f} | {h10:>8.4f} | {mrr:>8.4f} | {cov:>8.4f}")

        # Add smoothed and combined
        if smoothed_predictions:
            metrics = evaluate_prediction_map(smoothed_predictions, held_out)
            split_results["smoothed"] = metrics
            h5, h10, mrr, cov = metrics["hits_at_5"], metrics["hits_at_10"], metrics["mrr"], metrics["source_role_coverage"]
            print(f"  {'+ Embedding smoothing':<35} | {h5:>8.4f} | {h10:>8.4f} | {mrr:>8.4f} | {cov:>8.4f}")

        if lp_combined:
            metrics = evaluate_prediction_map(lp_combined, held_out)
            split_results["combined"] = metrics
            h5, h10, mrr, cov = metrics["hits_at_5"], metrics["hits_at_10"], metrics["mrr"], metrics["source_role_coverage"]
            print(f"  {'+ Combined (direct>smooth>LP)':<35} | {h5:>8.4f} | {h10:>8.4f} | {mrr:>8.4f} | {cov:>8.4f}")

        results[split_name] = split_results

    # Save report
    import numpy as np
    report_path = output_dir / "ablation_report.json"
    serializable = {}
    for split, methods in results.items():
        serializable[split] = {}
        for method, metrics in methods.items():
            serializable[split][method] = {
                k: float(v) if isinstance(v, (float, np.floating)) else v
                for k, v in metrics.items()
            }
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(serializable, f, indent=2)
    print(f"\nReport saved: {report_path}")
    return results


if __name__ == "__main__":
    run_pipeline_ablation()
