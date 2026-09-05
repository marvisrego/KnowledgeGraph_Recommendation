"""Evaluate train-derived Karrierewege edges on validation and test splits."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import Settings
from evaluation.transition_metrics import evaluate_transition_predictions
from src.transition_policy import is_training_transition
from src.graph_build import load_graph
from src.karrierewege_preprocessing import (
    aggregate_split_transitions,
    build_esco_title_index,
    discover_split_files,
    write_json_atomic,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate Karrierewege next-role predictions without split leakage."
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/karrierewege/evaluation.json"),
    )
    parser.add_argument("--chunk-size", type=int, default=None)
    args = parser.parse_args()

    settings = Settings.from_env()
    chunk_size = (
        args.chunk_size if args.chunk_size is not None else settings.transition_chunk_size
    )
    if chunk_size < 2:
        parser.error("--chunk-size must be at least 2")

    graph = load_graph(settings.graph_path)
    title_index = build_esco_title_index(graph)
    split_files = discover_split_files(settings.karrierewege_dir)

    payload = {
        "policy": {
            "training_split": "train",
            "evaluated_splits": ["validation", "test"],
            "graph_transition_edges": sum(
                1
                for _, _, _, data in graph.edges(keys=True, data=True)
                if is_training_transition(data)
            ),
            "max_invalid_row_ratio": settings.karrierewege_max_invalid_row_ratio,
        },
        "splits": {},
    }

    for split in ("validation", "test"):
        print(f"[evaluate_karrierewege] Cleaning {split}: {split_files[split]}")
        aggregate = aggregate_split_transitions(
            split_files[split],
            split,
            title_index,
            chunk_size=chunk_size,
            strict_mapping=True,
            max_invalid_row_ratio=settings.karrierewege_max_invalid_row_ratio,
        )
        metrics = evaluate_transition_predictions(graph, aggregate)
        payload["splits"][split] = {
            "quality": aggregate.report.to_dict(),
            "metrics": metrics,
        }
        print(
            f"[evaluate_karrierewege] {split}: "
            f"Hits@5={metrics['hits_at_5']:.4f}, MRR={metrics['mrr']:.4f}, "
            f"source coverage={metrics['source_role_coverage']:.4f}"
        )

    output = args.output if args.output.is_absolute() else Path(__file__).resolve().parents[1] / args.output
    write_json_atomic(payload, output)
    quality_payload = {}
    if settings.karrierewege_report_path.exists():
        with settings.karrierewege_report_path.open(encoding="utf-8") as stream:
            quality_payload = json.load(stream)
    for split in ("validation", "test"):
        quality_payload[split] = payload["splits"][split]["quality"]
    write_json_atomic(quality_payload, settings.karrierewege_report_path)
    print(f"[evaluate_karrierewege] Wrote {output}")
    print(f"[evaluate_karrierewege] Updated {settings.karrierewege_report_path}")


if __name__ == "__main__":
    main()
