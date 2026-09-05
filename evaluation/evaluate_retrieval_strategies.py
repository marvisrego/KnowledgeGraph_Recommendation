"""Compare direct, smoothed, LP, and validation-selected fallback rankings.

Measures Hits@K and MRR for each method against held-out splits.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import Settings
from evaluation.transition_metrics import (
    evaluate_prediction_map,
    transition_predictions,
)
from src.graph_build import load_graph
from src.hybrid_retrieval import combine_prediction_maps
from src.karrierewege_preprocessing import (
    aggregate_split_transitions,
    build_esco_title_index,
    discover_split_files,
)
from src.text_normalization import normalize_label
from src.transition_embedding import (
    SmoothingConfig,
    hybrid_prediction_map,
    load_live_esco_embeddings,
    compute_neighbour_index,
    transition_distributions,
)


def evaluate_retrieval_strategies(settings: Settings | None = None, output_dir: Path | None = None) -> dict:
    """Run all retrieval strategies and compare metrics."""
    if settings is None:
        settings = Settings.from_env()
    if output_dir is None:
        output_dir = Path("artifacts/retrieval_comparison")
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("RETRIEVAL STRATEGY COMPARISON")
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
    evaluation_source_titles = {
        source
        for held_out in held_out_splits.values()
        for source, _ in held_out.pair_counts
    }
    evaluation_source_ids = sorted(
        title_index[title]
        for title in evaluation_source_titles
        if title in title_index
    )

    # Method 1: Direct edges only
    print("\n[1/4] Direct-edge predictions...")
    direct_map = transition_predictions(G)
    print(f"  Source roles: {len(direct_map)}")

    # Method 2: Embedding-smoothed
    print("\n[2/4] Embedding-smoothed predictions...")
    smoothed_map = None
    collection = None
    try:
        import chromadb
        client = chromadb.PersistentClient(path=str(settings.chroma_dir))
        collection = client.get_collection("roles_collection")
        embeddings, _report = load_live_esco_embeddings(collection, G)
        if embeddings and len(embeddings) > 0:
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
            smoothed_map = hybrid_prediction_map(
                G,
                distributions,
                neighbour_idx,
                config,
                source_ids=evaluation_source_ids,
            )
            print(f"  Source roles: {len(smoothed_map)}")
    except Exception as exc:
        print(f"  Smoothing failed: {exc}")

    # Method 3: Link prediction
    print("\n[3/4] Link prediction predictions...")
    lp_map = None
    lp_model_path = settings.link_prediction_model_path
    if lp_model_path.exists():
        try:
            from src.link_prediction import load_model, link_prediction_map
            model = load_model(lp_model_path)
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

            idf_map = {str(nid): float(d["idf"]) for nid, d in G.nodes(data=True) if "idf" in d}
            lp_map = link_prediction_map(
                model,
                G,
                idf_map,
                emb_dict,
                top_k=50,
                source_role_ids=evaluation_source_ids,
            )
            print(f"  Source roles: {len(lp_map)}")
        except Exception as exc:
            print(f"  LP failed: {exc}")
    else:
        print(f"  Model not found at {lp_model_path}")

    # Method 4: Priority fallback
    print("\n[4/4] Combined priority-fallback predictions...")
    combined_map = combine_prediction_maps(direct_map, smoothed_map, lp_map)
    print(f"  Source roles: {len(combined_map)}")

    # Evaluate all methods on validation and test
    results = {}
    for split_name, held_out in held_out_splits.items():

        print(f"\n{'='*40}")
        print(f"  {split_name.upper()} RESULTS")
        print(f"{'='*40}")

        methods = {"direct": direct_map}
        if smoothed_map:
            methods["smoothed"] = smoothed_map
        if lp_map:
            methods["link_prediction"] = lp_map
        methods["combined_priority"] = combined_map

        split_results = {}
        for method_name, pred_map in methods.items():
            metrics = evaluate_prediction_map(pred_map, held_out)
            split_results[method_name] = metrics
            h5 = metrics.get("hits_at_5", 0)
            h10 = metrics.get("hits_at_10", 0)
            mrr = metrics.get("mrr", 0)
            cov = metrics.get("source_role_coverage", 0)
            print(f"  {method_name:20s} | Hits@5={h5:.4f} | Hits@10={h10:.4f} | MRR={mrr:.4f} | Coverage={cov:.4f}")

        results[split_name] = split_results

    # Save report
    report_path = output_dir / "comparison_report.json"
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
    evaluate_retrieval_strategies()
