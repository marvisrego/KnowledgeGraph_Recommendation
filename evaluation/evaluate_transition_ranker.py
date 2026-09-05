"""Train and evaluate the person-disjoint LambdaMART transition ranker."""

from __future__ import annotations

import argparse
import gc
import gzip
import json
import os
import pickle
import shutil
import sys
import tempfile
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import Settings
from evaluation.transition_metrics import evaluate_prediction_map
from src.graph_build import load_graph
from src.karrierewege_preprocessing import (
    aggregate_split_transitions,
    build_esco_title_index,
    discover_split_files,
    write_json_atomic,
)
from src.transition_embedding import (
    SmoothingConfig,
    compute_neighbour_index,
    hybrid_prediction_map,
    load_live_esco_embeddings,
    transition_distributions,
)
from src.transition_ranker import (
    FEATURE_NAMES,
    FoldedTransitionCounts,
    TransitionFeatureBuilder,
    aggregate_person_disjoint_folds,
    build_ranker_rows,
    build_transition_context,
    head_preserving_fusion,
    prediction_map_from_scores,
)


SMOOTHER_CONFIG = SmoothingConfig(neighbours=20, direct_weight=0.90, temperature=0.05)
MODEL_GRID = (
    {"n_estimators": 200, "learning_rate": 0.05, "num_leaves": 15, "min_child_samples": 30},
    {"n_estimators": 300, "learning_rate": 0.04, "num_leaves": 31, "min_child_samples": 40},
    {"n_estimators": 400, "learning_rate": 0.03, "num_leaves": 31, "min_child_samples": 25},
)


def _load_embeddings_from_copy(settings: Settings, graph) -> tuple[dict, dict]:
    if not settings.chroma_dir.is_dir():
        raise FileNotFoundError(f"ChromaDB directory not found: {settings.chroma_dir}")
    import chromadb

    with tempfile.TemporaryDirectory(prefix="career-kg-ranker-chroma-", ignore_cleanup_errors=True) as root:
        copied = Path(root) / "chroma"
        shutil.copytree(settings.chroma_dir, copied)
        client = chromadb.PersistentClient(path=str(copied))
        collection = client.get_collection("roles_collection")
        embeddings, report = load_live_esco_embeddings(collection, graph)
        del collection
        del client
        gc.collect()
    return embeddings, report


def _cache_signature(path: Path, folds: int) -> dict[str, int | str]:
    stat = path.stat()
    return {
        "path": str(path.resolve()),
        "bytes": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
        "folds": folds,
    }


def _load_or_build_folds(
    train_path: Path,
    title_index: dict[str, str],
    folds: int,
    chunk_size: int,
    cache_path: Path,
) -> tuple[FoldedTransitionCounts, bool]:
    signature = _cache_signature(train_path, folds)
    if cache_path.is_file():
        try:
            with gzip.open(cache_path, "rb") as stream:
                payload = pickle.load(stream)
            if payload.get("signature") == signature:
                return payload["folded"], True
        except (EOFError, OSError, pickle.PickleError, AttributeError, KeyError):
            pass

    folded = aggregate_person_disjoint_folds(
        train_path,
        title_index,
        n_folds=folds,
        chunk_size=chunk_size,
    )
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temp_name = tempfile.mkstemp(prefix=f".{cache_path.name}.", dir=cache_path.parent)
    os.close(descriptor)
    try:
        with gzip.open(temp_name, "wb", compresslevel=5) as stream:
            pickle.dump({"signature": signature, "folded": folded}, stream, protocol=5)
        os.replace(temp_name, cache_path)
    finally:
        if os.path.exists(temp_name):
            os.unlink(temp_name)
    return folded, False


def _id_counts(aggregate, title_index: dict[str, str]) -> Counter[tuple[str, str]]:
    return Counter(
        {
            (title_index[source], title_index[target]): int(count)
            for (source, target), count in aggregate.pair_counts.items()
        }
    )


def _source_ids(counts: Counter[tuple[str, str]]) -> set[str]:
    return {source for source, _ in counts}


def _load_jobhop_train_context(root: Path):
    path = root / "Data" / "JobHop_v2" / "processed" / "train_transition_counts.parquet"
    if not path.is_file():
        raise FileNotFoundError(
            f"JobHop transition counts not found: {path}. Run JobHop preprocessing first."
        )
    frame = pd.read_parquet(path)
    counts = Counter(
        {
            (str(source), str(target)): int(count)
            for source, target, count in frame[
                ["source_role_id", "target_role_id", "count"]
            ].itertuples(index=False, name=None)
            if int(count) > 0
        }
    )
    return build_transition_context(counts), path


def _selection_key(metrics: dict) -> tuple[float, ...]:
    return (
        float(metrics["mrr"]),
        float(metrics["hits_at_5"]),
        float(metrics["hits_at_10"]),
        float(metrics["macro_mrr"]),
    )


def _metric_delta(candidate: dict, baseline: dict) -> dict[str, float]:
    keys = (
        "mrr",
        "hits_at_1",
        "hits_at_3",
        "hits_at_5",
        "hits_at_10",
        "ndcg_at_5",
        "ndcg_at_10",
        "source_role_coverage",
        "destination_coverage",
        "macro_mrr",
    )
    return {key: float(candidate[key] - baseline[key]) for key in keys}


def _acceptance(validation_baseline: dict, validation_ranker: dict, test_baseline: dict, test_ranker: dict) -> dict:
    reasons: list[str] = []
    if validation_ranker["mrr"] <= validation_baseline["mrr"]:
        reasons.append("validation_mrr_did_not_improve")
    if test_ranker["mrr"] <= test_baseline["mrr"]:
        reasons.append("test_mrr_did_not_improve")
    if not (
        test_ranker["hits_at_5"] > test_baseline["hits_at_5"]
        or test_ranker["hits_at_10"] > test_baseline["hits_at_10"]
    ):
        reasons.append("no_test_hits_metric_improved")
    for metric in ("hits_at_5", "hits_at_10"):
        if test_baseline[metric] - test_ranker[metric] > 0.001:
            reasons.append(f"{metric}_decreased_beyond_tolerance")
    for metric in ("source_role_coverage", "destination_coverage", "macro_mrr"):
        if test_ranker[metric] + 1e-12 < test_baseline[metric]:
            reasons.append(f"{metric}_decreased")
    return {"accepted": not reasons, "reasons": reasons}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("artifacts/karrierewege/transition_ranker_evaluation.json"))
    parser.add_argument("--model-output", type=Path, default=Path("artifacts/transition_ranker/model.txt"))
    parser.add_argument("--fold-cache", type=Path, default=Path("artifacts/karrierewege/ranker_oof_counts.pkl.gz"))
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--chunk-size", type=int, default=None)
    parser.add_argument("--per-channel-limit", type=int, default=40)
    args = parser.parse_args()
    if args.folds < 2 or args.per_channel_limit < 1:
        parser.error("--folds must be >=2 and --per-channel-limit must be >=1")

    from lightgbm import LGBMRanker

    settings = Settings.from_env()
    root = Path(__file__).resolve().parents[1]
    output = args.output if args.output.is_absolute() else root / args.output
    model_output = args.model_output if args.model_output.is_absolute() else root / args.model_output
    fold_cache = args.fold_cache if args.fold_cache.is_absolute() else root / args.fold_cache
    chunk_size = args.chunk_size or settings.transition_chunk_size

    print("[ranker] Loading graph and stored ESCO embeddings ...", flush=True)
    graph = load_graph(settings.graph_path)
    title_index = build_esco_title_index(graph)
    embeddings, vector_report = _load_embeddings_from_copy(settings, graph)
    split_files = discover_split_files(settings.karrierewege_dir)

    print("[ranker] Building/loading deterministic person-disjoint train folds ...", flush=True)
    folded, cache_hit = _load_or_build_folds(
        split_files["train"], title_index, args.folds, chunk_size, fold_cache
    )
    full_context = build_transition_context(folded.pair_counts, folded.source_totals)
    jobhop_context, jobhop_counts_path = _load_jobhop_train_context(root)
    all_query_ids = set(folded.source_totals)

    print("[ranker] Precomputing semantic and destination neighbour channels ...", flush=True)
    source_neighbours = compute_neighbour_index(
        embeddings,
        full_context.distributions,
        max_neighbours=SMOOTHER_CONFIG.neighbours + 10,
        query_ids=all_query_ids,
    )
    destination_neighbours = compute_neighbour_index(
        embeddings,
        embeddings,
        max_neighbours=args.per_channel_limit,
        query_ids=all_query_ids,
    )
    builder = TransitionFeatureBuilder(
        graph,
        embeddings,
        source_neighbours,
        destination_neighbours,
        semantic_neighbours=SMOOTHER_CONFIG.neighbours,
        semantic_temperature=SMOOTHER_CONFIG.temperature,
        per_channel_limit=args.per_channel_limit,
        auxiliary_context=jobhop_context,
    )

    print("[ranker] Creating out-of-fold grouped ranking rows ...", flush=True)
    matrices: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    groups: list[int] = []
    fold_reports: list[dict] = []
    for fold_id in range(folded.n_folds):
        train_pairs, train_totals = folded.training_without(fold_id)
        context = build_transition_context(train_pairs, train_totals)
        labels = folded.fold_pair_counts[fold_id]
        matrix, target, fold_groups, _, report = build_ranker_rows(
            _source_ids(labels), context, labels, builder
        )
        report["fold"] = fold_id
        fold_reports.append(report)
        matrices.append(matrix)
        targets.append(target)
        groups.extend(fold_groups)
        print(
            f"[ranker] Fold {fold_id}: rows={len(target):,}, positives={int((target > 0).sum()):,}, "
            f"candidate_recall={report['candidate_recall']:.4f}",
            flush=True,
        )
    X_train = np.concatenate(matrices, axis=0)
    y_train = np.concatenate(targets, axis=0)
    if not len(X_train) or not np.any(y_train > 0):
        raise RuntimeError("OOF ranker data contains no positive candidate labels")

    print("[ranker] Cleaning official validation split ...", flush=True)
    validation = aggregate_split_transitions(
        split_files["validation"],
        "validation",
        title_index,
        chunk_size=chunk_size,
        strict_mapping=True,
        max_invalid_row_ratio=settings.karrierewege_max_invalid_row_ratio,
    )
    validation_labels = _id_counts(validation, title_index)
    validation_sources = _source_ids(validation_labels)
    missing_queries = validation_sources - set(source_neighbours)
    if missing_queries:
        source_neighbours.update(
            compute_neighbour_index(
                embeddings,
                full_context.distributions,
                max_neighbours=SMOOTHER_CONFIG.neighbours + 10,
                query_ids=missing_queries,
            )
        )
        destination_neighbours.update(
            compute_neighbour_index(
                embeddings,
                embeddings,
                max_neighbours=args.per_channel_limit,
                query_ids=missing_queries,
            )
        )
    X_validation, _, _, validation_pairs, validation_candidates = build_ranker_rows(
        validation_sources, full_context, validation_labels, builder
    )

    graph_distributions = transition_distributions(graph)
    baseline_neighbours = compute_neighbour_index(
        embeddings,
        graph_distributions,
        max_neighbours=SMOOTHER_CONFIG.neighbours,
        query_ids=validation_sources,
    )
    validation_baseline_predictions = hybrid_prediction_map(
        graph,
        graph_distributions,
        baseline_neighbours,
        SMOOTHER_CONFIG,
        source_ids=validation_sources,
    )
    validation_baseline = evaluate_prediction_map(validation_baseline_predictions, validation)

    print(f"[ranker] Tuning {len(MODEL_GRID)} LambdaMART configurations on validation ...", flush=True)
    trials: list[tuple[dict, object, dict]] = []
    for params in MODEL_GRID:
        model = LGBMRanker(
            objective="lambdarank",
            metric="ndcg",
            label_gain=(0, 1, 3, 7),
            random_state=42,
            bagging_seed=42,
            data_random_seed=42,
            feature_fraction_seed=42,
            n_jobs=-1,
            verbosity=-1,
            deterministic=True,
            force_col_wise=True,
            reg_lambda=1.0,
            **params,
        )
        model.fit(X_train, y_train, group=groups, eval_at=(5, 10))
        scores = model.predict(X_validation)
        predictions = prediction_map_from_scores(validation_pairs, scores, graph)
        metrics = evaluate_prediction_map(predictions, validation)
        trials.append((params, model, metrics))
        print(
            f"[ranker] leaves={params['num_leaves']} trees={params['n_estimators']}: "
            f"MRR={metrics['mrr']:.6f}, H5={metrics['hits_at_5']:.6f}, H10={metrics['hits_at_10']:.6f}",
            flush=True,
        )
    selected_params, selected_model, validation_ranker = max(
        trials, key=lambda item: _selection_key(item[2])
    )
    selected_validation_scores = selected_model.predict(X_validation)
    selected_validation_predictions = prediction_map_from_scores(
        validation_pairs, selected_validation_scores, graph
    )
    fusion_trials: list[tuple[dict, dict]] = []
    for head_size in (1, 3, 5, 7, 10):
        for secondary_slots in (1, 3, 5, 10, 20):
            config = {"head_size": head_size, "secondary_slots": secondary_slots}
            predictions = head_preserving_fusion(
                validation_baseline_predictions,
                selected_validation_predictions,
                **config,
            )
            fusion_trials.append((config, evaluate_prediction_map(predictions, validation)))
    selected_fusion, validation_fused = max(
        fusion_trials, key=lambda item: _selection_key(item[1])
    )
    print(
        f"[ranker] Selected fusion head={selected_fusion['head_size']} "
        f"slots={selected_fusion['secondary_slots']}: MRR={validation_fused['mrr']:.6f}, "
        f"H5={validation_fused['hits_at_5']:.6f}, H10={validation_fused['hits_at_10']:.6f}",
        flush=True,
    )

    print("[ranker] Configuration frozen; evaluating official test once ...", flush=True)
    test = aggregate_split_transitions(
        split_files["test"],
        "test",
        title_index,
        chunk_size=chunk_size,
        strict_mapping=True,
        max_invalid_row_ratio=settings.karrierewege_max_invalid_row_ratio,
    )
    test_labels = _id_counts(test, title_index)
    test_sources = _source_ids(test_labels)
    missing_test_queries = test_sources - set(source_neighbours)
    if missing_test_queries:
        source_neighbours.update(
            compute_neighbour_index(
                embeddings,
                full_context.distributions,
                max_neighbours=SMOOTHER_CONFIG.neighbours + 10,
                query_ids=missing_test_queries,
            )
        )
        destination_neighbours.update(
            compute_neighbour_index(
                embeddings,
                embeddings,
                max_neighbours=args.per_channel_limit,
                query_ids=missing_test_queries,
            )
        )
    X_test, _, _, test_pairs, test_candidates = build_ranker_rows(
        test_sources, full_context, test_labels, builder
    )
    test_scores = selected_model.predict(X_test)
    test_predictions = prediction_map_from_scores(test_pairs, test_scores, graph)
    test_ranker = evaluate_prediction_map(test_predictions, test)

    missing_baseline_queries = test_sources - set(baseline_neighbours)
    if missing_baseline_queries:
        baseline_neighbours.update(
            compute_neighbour_index(
                embeddings,
                graph_distributions,
                max_neighbours=SMOOTHER_CONFIG.neighbours,
                query_ids=missing_baseline_queries,
            )
        )
    test_baseline_predictions = hybrid_prediction_map(
        graph, graph_distributions, baseline_neighbours, SMOOTHER_CONFIG, source_ids=test_sources
    )
    test_baseline = evaluate_prediction_map(test_baseline_predictions, test)
    test_fused_predictions = head_preserving_fusion(
        test_baseline_predictions,
        test_predictions,
        **selected_fusion,
    )
    test_fused = evaluate_prediction_map(test_fused_predictions, test)
    decision = _acceptance(validation_baseline, validation_fused, test_baseline, test_fused)

    model_output.parent.mkdir(parents=True, exist_ok=True)
    selected_model.booster_.save_model(str(model_output))
    importance = sorted(
        zip(FEATURE_NAMES, selected_model.feature_importances_.tolist()),
        key=lambda item: (-item[1], item[0]),
    )
    payload = {
        "policy": {
            "training": "deterministic person-disjoint OOF folds within Karrierewege train",
            "validation": "official validation used for hyperparameter selection",
            "test": "frozen historical test used once after selection",
            "positive_injection": False,
            "jobhop_used_for_training": True,
            "jobhop_policy": "train-only separate prior and candidate feature; never merged into KG edges",
            "features": list(FEATURE_NAMES),
            "folds": args.folds,
            "per_channel_limit": args.per_channel_limit,
        },
        "vectors": vector_report,
        "training": {
            "cache_hit": cache_hit,
            "rows": len(y_train),
            "positive_rows": int((y_train > 0).sum()),
            "queries": len(groups),
            "fold_reports": fold_reports,
            "jobhop_transition_pairs": len(jobhop_context.pair_counts),
            "jobhop_transition_observations": jobhop_context.total_transitions,
            "jobhop_counts_path": str(jobhop_counts_path),
        },
        "selection": {
            "selected_params": selected_params,
            "validation_candidates": validation_candidates,
            "baseline": validation_baseline,
            "ranker": validation_ranker,
            "delta": _metric_delta(validation_ranker, validation_baseline),
            "trials": [{"params": params, "metrics": metrics} for params, _, metrics in trials],
            "fusion_config": selected_fusion,
            "fused": validation_fused,
            "fused_delta": _metric_delta(validation_fused, validation_baseline),
            "fusion_trials": [
                {"config": config, "metrics": metrics} for config, metrics in fusion_trials
            ],
        },
        "test": {
            "candidates": test_candidates,
            "baseline": test_baseline,
            "ranker": test_ranker,
            "delta": _metric_delta(test_ranker, test_baseline),
            "fused": test_fused,
            "fused_delta": _metric_delta(test_fused, test_baseline),
        },
        "model": {
            "path": str(model_output),
            "feature_importance": [{"feature": name, "gain": gain} for name, gain in importance],
        },
        "decision": decision,
    }
    write_json_atomic(payload, output)
    print(
        f"[ranker] Fused test MRR={test_fused['mrr']:.6f} ({payload['test']['fused_delta']['mrr']:+.6f}), "
        f"H5={test_fused['hits_at_5']:.6f} ({payload['test']['fused_delta']['hits_at_5']:+.6f}), "
        f"H10={test_fused['hits_at_10']:.6f} ({payload['test']['fused_delta']['hits_at_10']:+.6f})",
        flush=True,
    )
    print(f"[ranker] Accepted={decision['accepted']}; reasons={decision['reasons']}", flush=True)
    print(f"[ranker] Wrote {output}", flush=True)


if __name__ == "__main__":
    main()
