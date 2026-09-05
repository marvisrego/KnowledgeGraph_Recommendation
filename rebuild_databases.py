"""Fully rebuild the dedicated Neo4j Aura and Qdrant Cloud databases."""

from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import Settings
from evaluation.transition_metrics import evaluate_prediction_map, evaluate_transition_predictions
from src.graph_build import add_alignment_edges, build_all, save_graph
from src.graph_quality import improve_graph_quality
from src.hybrid_retrieval import build_link_prediction_runtime
from src.isco_edges import add_isco_group_edges
from src.karrierewege_preprocessing import (
    add_transition_edges,
    aggregate_split_transitions,
    build_esco_title_index,
    discover_split_files,
    transition_rows,
    write_json_atomic,
)
from src.kg_enrichment import enrich_graph
from src.neo4j_store import Neo4jGraphStore, validate_neo4j_settings
from src.qdrant_store import (
    PreparedVectorCollection,
    QdrantVectorStore,
    alignment_from_prepared_vectors,
    load_prepared_vectors,
    prepare_role_vectors,
    save_prepared_vectors,
    validate_qdrant_settings,
)
from src.transition_embedding import (
    SmoothingConfig,
    compute_neighbour_index,
    hybrid_prediction_map,
    transition_distributions,
)


ACCEPTED_LOCAL_BASELINE = {
    "mrr": 0.2618052531093985,
    "hits_at_5": 0.37227257195854146,
    "hits_at_10": 0.5058494280740006,
    "source_role_coverage": 1.0,
    "destination_coverage": 0.9583218080346247,
}


def _source_ids(aggregate, title_index: dict[str, str]) -> set[str]:
    return {
        title_index[source]
        for source, _ in aggregate.pair_counts
        if source in title_index
    }


def _evaluate_prepared_graph(graph, points, settings, split_files, title_index) -> dict:
    embeddings = {}
    for point in points:
        norm = float(np.linalg.norm(point.vector))
        if norm > 1e-12 and math.isfinite(norm):
            embeddings[point.record_id] = point.vector / norm
    distributions = transition_distributions(graph)
    config = SmoothingConfig(
        settings.transition_smoothing_neighbours,
        settings.transition_smoothing_direct_weight,
        settings.transition_smoothing_temperature,
    )
    results = {}
    for split in ("validation", "test"):
        aggregate = aggregate_split_transitions(
            split_files[split],
            split,
            title_index,
            chunk_size=settings.transition_chunk_size,
            strict_mapping=True,
            max_invalid_row_ratio=settings.karrierewege_max_invalid_row_ratio,
        )
        source_ids = _source_ids(aggregate, title_index)
        neighbours = compute_neighbour_index(
            embeddings,
            distributions,
            max_neighbours=config.neighbours,
            query_ids=source_ids,
        )
        predictions = hybrid_prediction_map(
            graph,
            distributions,
            neighbours,
            config,
            source_ids=source_ids,
        )
        results[split] = {
            "direct": evaluate_transition_predictions(graph, aggregate),
            "smoothed": evaluate_prediction_map(predictions, aggregate),
        }
    results["test"]["delta_from_accepted_local"] = {
        key: float(results["test"]["smoothed"][key] - value)
        for key, value in ACCEPTED_LOCAL_BASELINE.items()
    }
    return results


def _representative_predictions(graph, prepared_collection, settings) -> dict:
    if not settings.link_prediction_enabled or not settings.link_prediction_model_path.is_file():
        return {"available": False, "reason": "model_not_available"}
    runtime = build_link_prediction_runtime(
        settings.link_prediction_model_path,
        graph,
        prepared_collection,
    )
    source_ids = sorted(runtime.transition_index, key=lambda source: (-len(runtime.transition_index[source]), source))
    if not source_ids:
        return {"available": False, "reason": "no_transition_sources"}
    source_id = source_ids[0]
    predictions = runtime.predict(source_id, top_k=5)
    return {
        "available": True,
        "source_id": source_id,
        "source_title": graph.nodes[source_id].get("title", source_id),
        "predictions": [
            {
                "role_id": role_id,
                "title": graph.nodes[role_id].get("title", role_id),
                "score": score,
                "evidence_type": "predicted_transition",
                "persisted": False,
            }
            for role_id, score in predictions
        ],
    }


def prepare(settings: Settings, output_dir: Path, *, reuse_prepared_vectors: bool = False) -> tuple:
    print("[rebuild] Building the graph from source datasets ...", flush=True)
    graph = build_all(settings)
    crosswalk = Path("Data/crosswalks/soc_isco08_crosswalk.csv")
    enrich_graph(graph, crosswalk if crosswalk.exists() else None)
    isco_edges = add_isco_group_edges(graph)

    split_files = discover_split_files(settings.karrierewege_dir)
    title_index = build_esco_title_index(graph)
    print("[rebuild] Processing train-only Karrierewege transitions ...", flush=True)
    training = aggregate_split_transitions(
        split_files["train"],
        "train",
        title_index,
        chunk_size=settings.transition_chunk_size,
        strict_mapping=True,
        max_invalid_row_ratio=settings.karrierewege_max_invalid_row_ratio,
    )
    transitions = transition_rows(training, title_index, min_support=settings.transition_min_count)
    add_transition_edges(graph, transitions)

    graph, quality = improve_graph_quality(graph)
    if graph.number_of_nodes() < 1000 or graph.number_of_edges() < 1000:
        raise RuntimeError("Prepared graph is implausibly small; refusing cloud rebuild")
    title_index = build_esco_title_index(graph)

    role_count = sum(1 for _, data in graph.nodes(data=True) if data.get("type") == "role")
    vector_cache = output_dir / "prepared_vectors.pkl"
    points = []
    if reuse_prepared_vectors:
        points = load_prepared_vectors(
            vector_cache,
            graph,
            graph_id=str(settings.kg_id),
            embedding_model=settings.embed_model,
        )
        if points:
            print(f"[rebuild] Reusing {len(points)} validated prepared role vectors ...", flush=True)
    if not points:
        print("[rebuild] Generating role embeddings with the configured embedding model ...", flush=True)
        points = prepare_role_vectors(graph, settings)
        save_prepared_vectors(
            points,
            vector_cache,
            graph_id=str(settings.kg_id),
            embedding_model=settings.embed_model,
        )
    if len(points) != role_count:
        raise RuntimeError(f"Vector coverage mismatch: expected {role_count}, prepared {len(points)}")

    print("[rebuild] Computing embedding-derived cross-taxonomy alignment ...", flush=True)
    alignment = alignment_from_prepared_vectors(graph, points, settings.similarity_threshold)
    add_alignment_edges(graph, alignment)
    graph, final_quality = improve_graph_quality(graph)
    save_graph(graph, output_dir / "prepared_graph.gpickle")

    print("[rebuild] Evaluating the prepared graph before cloud mutation ...", flush=True)
    metrics = _evaluate_prepared_graph(graph, points, settings, split_files, title_index)
    predictions = _representative_predictions(
        graph,
        PreparedVectorCollection(points),
        settings,
    )
    preparation = {
        "nodes": graph.number_of_nodes(),
        "relationships": graph.number_of_edges(),
        "roles": role_count,
        "vectors": len(points),
        "vector_dimension": int(points[0].vector.size),
        "transition_edges": len(transitions),
        "isco_edges_added": isco_edges,
        "alignment_pairs": len(alignment),
        "initial_quality": quality.to_dict(),
        "final_quality": final_quality.to_dict(),
        "training_quality": training.report.to_dict(),
        "metrics": metrics,
        "link_prediction": predictions,
    }
    write_json_atomic(preparation, output_dir / "preparation_report.json")
    return graph, points, preparation


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true", help="Prepare and validate locally without cloud mutation.")
    parser.add_argument(
        "--reuse-prepared-vectors",
        action="store_true",
        help="Reuse a locally cached vector set only when graph IDs and embedding model match.",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/cloud_rebuild"))
    args = parser.parse_args()

    settings = Settings.from_env()
    validate_neo4j_settings(settings)
    validate_qdrant_settings(settings)
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    graph, points, preparation = prepare(
        settings,
        output_dir,
        reuse_prepared_vectors=args.reuse_prepared_vectors,
    )
    if args.dry_run:
        print(
            f"[rebuild] Dry run complete: {graph.number_of_nodes()} nodes, "
            f"{graph.number_of_edges()} relationships, {len(points)} vectors.",
            flush=True,
        )
        return

    print("[rebuild] Verifying both cloud connections before destructive operations ...", flush=True)
    neo4j_store = Neo4jGraphStore.from_settings(settings)
    try:
        qdrant_store = QdrantVectorStore.from_settings(settings)
    except Exception:
        neo4j_store.close()
        raise

    try:
        print("[rebuild] Clearing and rebuilding the dedicated Neo4j Aura database ...", flush=True)
        neo4j_store.clear_database()
        neo4j_store.ensure_schema()
        neo4j_upload = neo4j_store.upload_graph(graph)
        neo4j_validation = neo4j_store.validate_graph(graph)
        reconstructed = neo4j_store.load_networkx()
        if reconstructed.number_of_nodes() != graph.number_of_nodes() or reconstructed.number_of_edges() != graph.number_of_edges():
            raise RuntimeError("Aura-to-NetworkX reconstruction count mismatch")

        print("[rebuild] Recreating and uploading the Qdrant collection ...", flush=True)
        qdrant_upload = qdrant_store.rebuild(points)
        qdrant_validation = qdrant_store.validate(points)
    finally:
        neo4j_store.close()

    report = {
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "graph_id": settings.kg_id,
        "qdrant_collection": settings.qdrant_collection,
        "embedding_model": settings.embed_model,
        "preparation": preparation,
        "neo4j": {
            "upload": neo4j_upload,
            "validation": neo4j_validation,
            "networkx_snapshot": {
                "nodes": reconstructed.number_of_nodes(),
                "relationships": reconstructed.number_of_edges(),
            },
        },
        "qdrant": {"upload": qdrant_upload, "validation": qdrant_validation},
        "secrets_recorded": False,
    }
    write_json_atomic(report, output_dir / "rebuild_report.json")
    print(
        f"[rebuild] Complete: Neo4j {neo4j_upload['nodes']} nodes/"
        f"{neo4j_upload['relationships']} relationships; Qdrant {qdrant_upload['vectors']} vectors.",
        flush=True,
    )


if __name__ == "__main__":
    main()
