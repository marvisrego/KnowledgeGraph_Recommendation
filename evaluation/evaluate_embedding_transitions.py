"""Tune and test embedding-smoothed Karrierewege transition predictions."""

from __future__ import annotations

import argparse
import gc
import json
import shutil
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import Settings
from evaluation.transition_metrics import (
    evaluate_prediction_map,
    evaluate_transition_predictions,
    transition_predictions,
)
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


RECORDED_TEST_BASELINE = {
    "mrr": 0.2590564107948408,
    "hits_at_5": 0.37023869571584306,
    "hits_at_10": 0.5013586293301225,
}
# Fine search around the previously accepted (20, 0.90, 0.05) setting.  The
# original broad grid remains documented in the historical evaluation artifact.
NEIGHBOUR_GRID = (15, 20, 25)
DIRECT_WEIGHT_GRID = (0.86, 0.88, 0.89, 0.90, 0.91, 0.92, 0.94)
TEMPERATURE_GRID = (0.04, 0.05, 0.06)
METRIC_TOLERANCE = 1e-12


@dataclass(frozen=True)
class AcceptanceDecision:
    accepted: bool
    reasons: tuple[str, ...]

    def to_dict(self) -> dict[str, bool | list[str]]:
        return {"accepted": self.accepted, "reasons": list(self.reasons)}


def decide_acceptance(
    validation_baseline: dict,
    validation_hybrid: dict,
    test_baseline: dict,
    test_hybrid: dict,
    deterministic: bool,
) -> AcceptanceDecision:
    """Apply the pre-registered acceptance gate without rounding metrics."""
    reasons: list[str] = []
    if validation_hybrid["mrr"] <= validation_baseline["mrr"]:
        reasons.append("validation_mrr_did_not_improve")
    if test_hybrid["mrr"] <= test_baseline["mrr"]:
        reasons.append("test_mrr_did_not_improve")
    if not (
        test_hybrid["hits_at_5"] > test_baseline["hits_at_5"]
        or test_hybrid["hits_at_10"] > test_baseline["hits_at_10"]
    ):
        reasons.append("no_test_hits_metric_improved")
    for metric in ("hits_at_5", "hits_at_10"):
        if test_baseline[metric] - test_hybrid[metric] > 0.005:
            reasons.append(f"{metric}_decreased_beyond_tolerance")
    for metric in ("source_role_coverage", "destination_coverage"):
        if test_hybrid[metric] + METRIC_TOLERANCE < test_baseline[metric]:
            reasons.append(f"{metric}_decreased")
    if not deterministic:
        reasons.append("results_are_not_deterministic")
    return AcceptanceDecision(accepted=not reasons, reasons=tuple(reasons))


def _metric_delta(candidate: dict, baseline: dict) -> dict[str, float]:
    keys = (
        "mrr",
        "hits_at_1",
        "hits_at_3",
        "hits_at_5",
        "hits_at_10",
        "source_role_coverage",
        "observation_coverage",
        "destination_coverage",
        "macro_mrr",
    )
    return {key: float(candidate[key] - baseline[key]) for key in keys}


def _selection_key(item: tuple[SmoothingConfig, dict]) -> tuple:
    config, metrics = item
    return (
        -metrics["mrr"],
        -metrics["hits_at_5"],
        -metrics["hits_at_10"],
        -metrics["destination_coverage"],
        -config.direct_weight,
        config.neighbours,
        -config.temperature,
    )


def _load_embeddings_from_copy(settings: Settings, graph) -> tuple[dict, dict]:
    """Open Chroma only from a disposable copy because its client writes on open."""
    if not settings.chroma_dir.is_dir():
        raise FileNotFoundError(f"ChromaDB directory not found: {settings.chroma_dir}")
    try:
        import chromadb
    except ImportError:
        raise RuntimeError("chromadb is required for the smoothing experiment") from None

    with tempfile.TemporaryDirectory(prefix="career-kg-chroma-eval-", ignore_cleanup_errors=True) as temp_root:
        copied_dir = Path(temp_root) / "chroma"
        shutil.copytree(settings.chroma_dir, copied_dir)
        client = chromadb.PersistentClient(path=str(copied_dir))
        collection = client.get_collection("roles_collection")
        embeddings, report = load_live_esco_embeddings(collection, graph)
        del collection
        del client
        gc.collect()
    return embeddings, report


def _source_ids(aggregate, title_index: dict[str, str]) -> set[str]:
    return {
        title_index[source_title]
        for source_title, _ in aggregate.pair_counts
        if source_title in title_index
    }


def _assert_recorded_baseline(metrics: dict) -> None:
    for key, expected in RECORDED_TEST_BASELINE.items():
        if abs(float(metrics[key]) - expected) > METRIC_TOLERANCE:
            raise RuntimeError(
                f"Current {key} baseline {metrics[key]!r} does not match the "
                f"pre-registered value {expected!r}."
            )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate semantic-neighbour smoothing without split leakage."
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/karrierewege/embedding_transition_evaluation.json"),
    )
    parser.add_argument("--chunk-size", type=int, default=None)
    args = parser.parse_args()

    settings = Settings.from_env()
    chunk_size = args.chunk_size or settings.transition_chunk_size
    if chunk_size < 2:
        parser.error("--chunk-size must be at least 2")

    print("[embedding_transitions] Loading graph and a disposable Chroma copy …")
    graph = load_graph(settings.graph_path)
    title_index = build_esco_title_index(graph)
    distributions = transition_distributions(graph)
    embeddings, vector_report = _load_embeddings_from_copy(settings, graph)
    split_files = discover_split_files(settings.karrierewege_dir)

    print("[embedding_transitions] Cleaning validation split …")
    validation = aggregate_split_transitions(
        split_files["validation"],
        "validation",
        title_index,
        chunk_size=chunk_size,
        strict_mapping=True,
        max_invalid_row_ratio=settings.karrierewege_max_invalid_row_ratio,
    )
    validation_baseline = evaluate_transition_predictions(graph, validation)
    equivalent_baseline = evaluate_prediction_map(transition_predictions(graph), validation)
    if validation_baseline != equivalent_baseline:
        raise RuntimeError("Refactored direct metrics do not reproduce the existing baseline.")

    validation_sources = _source_ids(validation, title_index)
    neighbour_index = compute_neighbour_index(
        embeddings,
        distributions,
        max_neighbours=max(NEIGHBOUR_GRID),
        query_ids=validation_sources,
    )

    print(f"[embedding_transitions] Tuning {len(NEIGHBOUR_GRID) * len(DIRECT_WEIGHT_GRID) * len(TEMPERATURE_GRID)} validation configurations …")
    trials: list[tuple[SmoothingConfig, dict]] = []
    for neighbours in NEIGHBOUR_GRID:
        for direct_weight in DIRECT_WEIGHT_GRID:
            for temperature in TEMPERATURE_GRID:
                config = SmoothingConfig(neighbours, direct_weight, temperature)
                predictions = hybrid_prediction_map(
                    graph,
                    distributions,
                    neighbour_index,
                    config,
                    source_ids=validation_sources,
                )
                metrics = evaluate_prediction_map(predictions, validation)
                trials.append((config, metrics))

    selected_config, validation_hybrid = min(trials, key=_selection_key)
    print(
        "[embedding_transitions] Selected on validation: "
        f"k={selected_config.neighbours}, direct_weight={selected_config.direct_weight:.2f}, "
        f"temperature={selected_config.temperature:.2f}, "
        f"MRR={validation_hybrid['mrr']:.6f}"
    )

    print("[embedding_transitions] Configuration frozen; cleaning and evaluating test once …")
    test = aggregate_split_transitions(
        split_files["test"],
        "test",
        title_index,
        chunk_size=chunk_size,
        strict_mapping=True,
        max_invalid_row_ratio=settings.karrierewege_max_invalid_row_ratio,
    )
    test_baseline = evaluate_transition_predictions(graph, test)
    _assert_recorded_baseline(test_baseline)
    test_sources = _source_ids(test, title_index)
    missing_test_sources = test_sources - set(neighbour_index)
    if missing_test_sources:
        test_neighbours = compute_neighbour_index(
            embeddings,
            distributions,
            max_neighbours=max(NEIGHBOUR_GRID),
            query_ids=missing_test_sources,
        )
        neighbour_index.update(test_neighbours)

    test_predictions = hybrid_prediction_map(
        graph,
        distributions,
        neighbour_index,
        selected_config,
        source_ids=test_sources,
    )
    test_hybrid = evaluate_prediction_map(test_predictions, test)
    repeated_predictions = hybrid_prediction_map(
        graph,
        distributions,
        neighbour_index,
        selected_config,
        source_ids=test_sources,
    )
    repeated_metrics = evaluate_prediction_map(repeated_predictions, test)
    deterministic = test_predictions == repeated_predictions and test_hybrid == repeated_metrics
    decision = decide_acceptance(
        validation_baseline,
        validation_hybrid,
        test_baseline,
        test_hybrid,
        deterministic,
    )

    payload = {
        "policy": {
            "training_source": "graph TRANSITIONS_TO edges with split=train",
            "tuning_split": "validation",
            "locked_evaluation_split": "test",
            "embedding_model": settings.embed_model,
            "embedding_api_calls": 0,
            "raw_karrierewege_rows_embedded": 0,
            "grid": {
                "neighbours": list(NEIGHBOUR_GRID),
                "direct_weight": list(DIRECT_WEIGHT_GRID),
                "temperature": list(TEMPERATURE_GRID),
            },
        },
        "vectors": vector_report,
        "training": {
            "transition_sources": len(distributions),
            "transition_edges": sum(len(rows) for rows in distributions.values()),
        },
        "selection": {
            "config": selected_config.to_dict(),
            "validation_baseline": validation_baseline,
            "validation_hybrid": validation_hybrid,
            "validation_delta": _metric_delta(validation_hybrid, validation_baseline),
            "trial_count": len(trials),
        },
        "test": {
            "baseline": test_baseline,
            "hybrid": test_hybrid,
            "delta": _metric_delta(test_hybrid, test_baseline),
            "deterministic": deterministic,
        },
        "decision": decision.to_dict(),
    }

    output = args.output if args.output.is_absolute() else Path(__file__).resolve().parents[1] / args.output
    write_json_atomic(payload, output)
    print(
        "[embedding_transitions] Test: "
        f"MRR={test_hybrid['mrr']:.6f} ({payload['test']['delta']['mrr']:+.6f}), "
        f"Hits@5={test_hybrid['hits_at_5']:.6f} ({payload['test']['delta']['hits_at_5']:+.6f}), "
        f"Hits@10={test_hybrid['hits_at_10']:.6f} ({payload['test']['delta']['hits_at_10']:+.6f})"
    )
    print(f"[embedding_transitions] Accepted: {decision.accepted}; reasons={list(decision.reasons)}")
    print(f"[embedding_transitions] Wrote {output}")


if __name__ == "__main__":
    main()
