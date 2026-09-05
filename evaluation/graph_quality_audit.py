"""Non-mutating knowledge-graph quality audit for the career GraphRAG graph."""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path

import networkx as nx

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import Settings
from src.graph_build import load_graph
from src.text_normalization import normalize_label
from src.transition_policy import TRANSITION_RELATION, is_training_transition


def audit_graph(graph: nx.MultiDiGraph) -> dict:
    """Return objective graph-integrity findings without changing the graph."""
    node_types = Counter(str(data.get("type", "missing")) for _, data in graph.nodes(data=True))
    edge_relations = Counter(
        str(data.get("relation", "missing")) for _, _, _, data in graph.edges(keys=True, data=True)
    )
    isolates = sorted(str(node_id) for node_id in nx.isolates(graph))

    labels: defaultdict[tuple[str, str, str], list[str]] = defaultdict(list)
    for node_id, data in graph.nodes(data=True):
        label = normalize_label(data.get("title", ""))
        if label:
            labels[(str(data.get("type", "")), str(data.get("source", "")), label)].append(
                str(node_id)
            )
    duplicate_labels = {
        "|".join(key): sorted(ids)
        for key, ids in labels.items()
        if len(ids) > 1
    }

    invalid_transition_edges: list[dict] = []
    held_out_transition_edges: list[dict] = []
    transition_sources: set[str] = set()
    transition_targets: set[str] = set()
    probability_sums: defaultdict[str, float] = defaultdict(float)
    role_requirement_counts: Counter[str] = Counter()
    invalid_endpoint_counts: Counter[str] = Counter()

    for source, target, key, data in graph.edges(keys=True, data=True):
        relation = str(data.get("relation", ""))
        source_data = graph.nodes.get(source, {})
        target_data = graph.nodes.get(target, {})
        if relation == "REQUIRES":
            role_requirement_counts[str(source)] += 1
            if source_data.get("type") != "role" or target_data.get("type") not in {
                "skill",
                "element",
            }:
                invalid_endpoint_counts[relation] += 1
        elif relation in {TRANSITION_RELATION, "SIMILAR_TO", "SAME_ISCO_GROUP"}:
            if source_data.get("type") != "role" or target_data.get("type") != "role":
                invalid_endpoint_counts[relation] += 1

        if relation != TRANSITION_RELATION:
            continue
        if not is_training_transition(data):
            held_out_transition_edges.append(
                {"source": str(source), "target": str(target), "key": str(key), "split": data.get("split")}
            )
            continue
        source_id, target_id = str(source), str(target)
        transition_sources.add(source_id)
        transition_targets.add(target_id)
        probability = float(data.get("probability", 0.0))
        count = int(data.get("count", 0))
        source_total = int(data.get("source_total", 0))
        probability_sums[source_id] += probability
        if (
            source_id == target_id
            or count <= 0
            or source_total < count
            or probability <= 0.0
            or probability > 1.0
            or not math.isfinite(probability)
        ):
            invalid_transition_edges.append(
                {
                    "source": source_id,
                    "target": target_id,
                    "key": str(key),
                    "count": count,
                    "source_total": source_total,
                    "probability": probability,
                }
            )

    roles = [
        str(node_id)
        for node_id, data in graph.nodes(data=True)
        if data.get("type") == "role"
    ]
    esco_roles = [
        str(node_id)
        for node_id, data in graph.nodes(data=True)
        if data.get("type") == "role" and data.get("source") == "esco"
    ]
    probability_overflow = {
        source: total for source, total in probability_sums.items() if total > 1.0 + 1e-9
    }
    return {
        "nodes": graph.number_of_nodes(),
        "edges": graph.number_of_edges(),
        "node_types": dict(sorted(node_types.items())),
        "edge_relations": dict(sorted(edge_relations.items())),
        "isolates": {"count": len(isolates), "sample": isolates[:20]},
        "duplicate_normalized_labels": {
            "count": len(duplicate_labels),
            "sample": dict(list(sorted(duplicate_labels.items()))[:20]),
        },
        "invalid_relation_endpoints": dict(sorted(invalid_endpoint_counts.items())),
        "roles_without_requirements": sum(role not in role_requirement_counts for role in roles),
        "esco_roles_without_requirements": sum(
            role not in role_requirement_counts for role in esco_roles
        ),
        "training_transitions": {
            "source_roles": len(transition_sources),
            "target_roles": len(transition_targets),
            "incident_roles": len(transition_sources | transition_targets),
            "invalid_edges": invalid_transition_edges[:50],
            "invalid_edge_count": len(invalid_transition_edges),
            "probability_overflow_sources": probability_overflow,
        },
        "held_out_transition_edge_count": len(held_out_transition_edges),
        "held_out_transition_edges": held_out_transition_edges[:50],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/graph_quality/audit.json"),
    )
    args = parser.parse_args()
    settings = Settings.from_env()
    report = audit_graph(load_graph(settings.graph_path))
    output = args.output if args.output.is_absolute() else Path(__file__).resolve().parents[1] / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
