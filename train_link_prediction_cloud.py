"""Train and validate link prediction using only AuraDB and Qdrant Cloud.

This entry point is intended for GitHub Actions. It never writes predicted
relationships to AuraDB and never logs connection details or credentials.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from config import Settings
from src.link_prediction import (
    FEATURE_NAMES,
    _get_idf,
    build_neighbour_index,
    build_training_data,
    build_transition_index,
    save_portable_model,
    train_link_predictor,
)
from src.neo4j_store import load_graph_from_neo4j
from src.portable_lightgbm import PortableLightGBMBooster
from src.qdrant_store import load_qdrant_collection
from src.transition_embedding import load_live_esco_embeddings


def neighbour_evidence_values(
    pairs: list[tuple[str, str]],
    transition_index: dict[str, dict[str, float]],
    neighbour_index: dict[str, list[str]],
    excluded_transition_sources: set[str],
) -> np.ndarray:
    """Compute neighbour evidence without reading transitions from held-out sources."""
    values = np.zeros(len(pairs), dtype=np.float64)
    for index, (source, target) in enumerate(pairs):
        values[index] = max(
            (
                float(transition_index.get(neighbour, {}).get(target, 0.0))
                for neighbour in neighbour_index.get(source, [])[:10]
                if neighbour not in excluded_transition_sources
            ),
            default=0.0,
        )
    return values


def source_ranking_metrics(
    pairs: list[tuple[str, str]],
    labels: np.ndarray,
    scores: np.ndarray,
) -> dict[str, float | int]:
    """Evaluate first-positive rank independently for each source role."""
    source_rows: dict[str, list[int]] = {}
    for index, (source, _) in enumerate(pairs):
        source_rows.setdefault(source, []).append(index)

    reciprocal_ranks: list[float] = []
    hits = {1: 0, 3: 0, 5: 0, 10: 0}
    ndcg = {5: [], 10: []}

    for rows in source_rows.values():
        positive_count = int(sum(int(labels[index]) for index in rows))
        if positive_count == 0:
            continue
        ranked = sorted(rows, key=lambda index: (-float(scores[index]), pairs[index][1]))
        first_positive_rank = next(
            rank for rank, index in enumerate(ranked, start=1) if labels[index] == 1
        )
        reciprocal_ranks.append(1.0 / first_positive_rank)
        for cutoff in hits:
            hits[cutoff] += int(first_positive_rank <= cutoff)
        for cutoff in ndcg:
            dcg = sum(
                int(labels[index]) / np.log2(rank + 1)
                for rank, index in enumerate(ranked[:cutoff], start=1)
            )
            ideal_count = min(positive_count, cutoff)
            ideal = sum(1.0 / np.log2(rank + 1) for rank in range(1, ideal_count + 1))
            ndcg[cutoff].append(float(dcg / ideal) if ideal else 0.0)

    source_count = len(reciprocal_ranks)
    if source_count == 0:
        raise RuntimeError("Source-grouped evaluation found no positive validation groups")
    return {
        "sources": source_count,
        "hits_at_1": hits[1] / source_count,
        "hits_at_3": hits[3] / source_count,
        "hits_at_5": hits[5] / source_count,
        "hits_at_10": hits[10] / source_count,
        "mrr": float(np.mean(reciprocal_ranks)),
        "ndcg_at_5": float(np.mean(ndcg[5])),
        "ndcg_at_10": float(np.mean(ndcg[10])),
    }


def source_grouped_cross_validation(
    features: np.ndarray,
    labels: np.ndarray,
    pairs: list[tuple[str, str]],
    folds: int,
    transition_index: dict[str, dict[str, float]] | None = None,
    neighbour_index: dict[str, list[str]] | None = None,
) -> tuple[dict[str, float | int], list[dict[str, float | int]], np.ndarray]:
    """Return source-role-disjoint classification and ranking metrics."""
    from sklearn.metrics import average_precision_score, roc_auc_score
    from sklearn.model_selection import GroupKFold

    groups = np.asarray([source for source, _ in pairs])
    source_count = len(set(groups.tolist()))
    fold_count = min(folds, source_count)
    if fold_count < 2:
        raise RuntimeError("At least two transition-source roles are required")

    out_of_fold = np.zeros(len(labels), dtype=np.float64)
    fold_reports: list[dict[str, float | int]] = []
    splitter = GroupKFold(n_splits=fold_count)
    for fold, (train_indices, validation_indices) in enumerate(
        splitter.split(features, labels, groups), start=1
    ):
        training_features = features[train_indices].copy()
        validation_features = features[validation_indices].copy()
        if transition_index is not None and neighbour_index is not None:
            held_out_sources = set(groups[validation_indices].tolist())
            evidence_column = FEATURE_NAMES.index("neighbour_evidence")
            training_features[:, evidence_column] = neighbour_evidence_values(
                [pairs[index] for index in train_indices],
                transition_index,
                neighbour_index,
                held_out_sources,
            )
            validation_features[:, evidence_column] = neighbour_evidence_values(
                [pairs[index] for index in validation_indices],
                transition_index,
                neighbour_index,
                held_out_sources,
            )

        model = train_link_predictor(training_features, labels[train_indices])
        predictions = np.asarray(model.predict(validation_features), dtype=np.float64)
        out_of_fold[validation_indices] = predictions
        validation_pairs = [pairs[index] for index in validation_indices]
        ranking = source_ranking_metrics(
            validation_pairs,
            labels[validation_indices],
            predictions,
        )
        fold_reports.append(
            {
                "fold": fold,
                "training_rows": len(train_indices),
                "validation_rows": len(validation_indices),
                "auc": float(roc_auc_score(labels[validation_indices], predictions)),
                "average_precision": float(
                    average_precision_score(labels[validation_indices], predictions)
                ),
                **ranking,
            }
        )

    overall = {
        "folds": fold_count,
        "auc": float(roc_auc_score(labels, out_of_fold)),
        "average_precision": float(average_precision_score(labels, out_of_fold)),
        **source_ranking_metrics(pairs, labels, out_of_fold),
    }
    return overall, fold_reports, out_of_fold


def write_report(path: Path, report: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(f"{path.suffix}.tmp")
    with temporary_path.open("w", encoding="utf-8") as report_file:
        json.dump(report, report_file, indent=2, sort_keys=True)
    temporary_path.replace(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/link_prediction/cloud_run"))
    parser.add_argument("--model-output", type=Path, default=None)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--negative-ratio", type=int, default=5)
    parser.add_argument("--min-auc", type=float, default=0.87)
    parser.add_argument("--min-average-precision", type=float, default=0.68)
    parser.add_argument("--min-hits-at-5", type=float, default=0.89)
    parser.add_argument("--min-mrr", type=float, default=0.78)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.folds < 2 or args.negative_ratio < 1:
        raise SystemExit("folds must be >= 2 and negative-ratio must be >= 1")

    settings = Settings.from_env(Path(__file__).resolve().parent)
    output_dir = args.output_dir.resolve()
    report_path = output_dir / "evaluation_report.json"
    model_path = (args.model_output or output_dir / "link_predictor.json").resolve()
    started = time.perf_counter()

    print("[cloud-training] Loading the production graph from Neo4j AuraDB")
    graph = load_graph_from_neo4j(settings)
    print(
        f"[cloud-training] Graph loaded: {graph.number_of_nodes()} nodes, "
        f"{graph.number_of_edges()} relationships"
    )

    print("[cloud-training] Loading role embeddings from Qdrant Cloud")
    collection = load_qdrant_collection(settings)
    try:
        embeddings, vector_report = load_live_esco_embeddings(collection, graph)
    finally:
        close = getattr(collection.client, "close", None)
        if callable(close):
            close()
    if vector_report["missing_vectors"] or vector_report["rejected_vectors"]:
        raise RuntimeError(f"Qdrant vectors failed validation: {vector_report}")
    print(f"[cloud-training] Validated {len(embeddings)} ESCO vectors")

    idf_map = _get_idf(graph)
    if not idf_map:
        raise RuntimeError("Aura graph does not contain skill IDF values")

    features, labels, pairs = build_training_data(
        graph,
        idf_map,
        embeddings,
        neg_ratio=args.negative_ratio,
        seed=42,
    )
    print(
        f"[cloud-training] Training matrix: {len(labels)} rows, "
        f"{int(labels.sum())} positives, {features.shape[1]} features"
    )

    transition_index = build_transition_index(graph)
    neighbour_index = build_neighbour_index(
        graph,
        embeddings,
        sorted(transition_index),
        k=40,
    )
    metrics, fold_reports, _ = source_grouped_cross_validation(
        features,
        labels,
        pairs,
        args.folds,
        transition_index,
        neighbour_index,
    )
    thresholds = {
        "auc": args.min_auc,
        "average_precision": args.min_average_precision,
        "hits_at_5": args.min_hits_at_5,
        "mrr": args.min_mrr,
    }
    failures = [
        f"{name}={float(metrics[name]):.6f} < {minimum:.6f}"
        for name, minimum in thresholds.items()
        if float(metrics[name]) < minimum
    ]
    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "git_sha": os.getenv("GITHUB_SHA", "local"),
        "graph_nodes": graph.number_of_nodes(),
        "graph_relationships": graph.number_of_edges(),
        "vector_report": vector_report,
        "feature_names": FEATURE_NAMES,
        "training_rows": len(labels),
        "positive_rows": int(labels.sum()),
        "negative_rows": int(len(labels) - labels.sum()),
        "validation_protocol": (
            "source-role-disjoint GroupKFold over production training edges; "
            "held-out source transitions excluded from neighbour-evidence features"
        ),
        "metrics": metrics,
        "folds": fold_reports,
        "thresholds": thresholds,
        "accepted": not failures,
        "rejection_reasons": failures,
    }
    write_report(report_path, report)
    print("[cloud-training] Validation metrics:")
    for name in ("auc", "average_precision", "hits_at_1", "hits_at_3", "hits_at_5", "hits_at_10", "mrr"):
        print(f"  {name}: {float(metrics[name]):.6f}")

    if failures:
        print("[cloud-training] Candidate rejected by validation gates")
        for failure in failures:
            print(f"  {failure}")
        return 2

    print("[cloud-training] Training accepted final model")
    final_model = train_link_predictor(features, labels)
    save_portable_model(final_model, model_path)
    portable = PortableLightGBMBooster.from_file(model_path)
    sample_size = min(256, len(features))
    native_scores = np.asarray(final_model.predict(features[:sample_size]), dtype=np.float64)
    portable_scores = portable.predict(features[:sample_size])
    max_error = float(np.max(np.abs(native_scores - portable_scores)))
    if max_error > 1e-12:
        model_path.unlink(missing_ok=True)
        raise RuntimeError(f"Portable model parity check failed: max_abs_error={max_error}")

    report["portable_model"] = {
        "path": model_path.name,
        "bytes": model_path.stat().st_size,
        "feature_count": portable.num_feature(),
        "parity_max_abs_error": max_error,
    }
    report["elapsed_seconds"] = round(time.perf_counter() - started, 3)
    write_report(report_path, report)
    print(f"[cloud-training] Accepted model written to {model_path}")
    print(f"[cloud-training] Report written to {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
