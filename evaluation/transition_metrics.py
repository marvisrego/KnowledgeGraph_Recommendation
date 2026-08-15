"""Leakage-free metrics for Karrierewege next-occupation prediction."""

from __future__ import annotations

from collections import defaultdict

import networkx as nx

from src.karrierewege_preprocessing import (
    TRANSITION_RELATION,
    TRANSITION_SOURCE,
    TransitionAggregate,
    normalize_label,
)


def transition_predictions(G: nx.MultiDiGraph) -> dict[str, list[str]]:
    """Return normalized destinations per source, ranked by training evidence."""
    ranked: dict[str, list[tuple[str, float, int, str]]] = defaultdict(list)
    for source, target, _, data in G.edges(keys=True, data=True):
        if (
            data.get("relation") != TRANSITION_RELATION
            or data.get("source") != TRANSITION_SOURCE
        ):
            continue
        source_title = normalize_label(G.nodes[source].get("title", ""))
        target_title = normalize_label(G.nodes[target].get("title", ""))
        if not source_title or not target_title:
            continue
        ranked[source_title].append(
            (
                target_title,
                float(data.get("probability", 0.0)),
                int(data.get("count", 0)),
                str(target),
            )
        )

    predictions: dict[str, list[str]] = {}
    for source_title, rows in ranked.items():
        rows.sort(key=lambda row: (-row[1], -row[2], row[0], row[3]))
        predictions[source_title] = [target_title for target_title, _, _, _ in rows]
    return predictions


def evaluate_transition_predictions(
    G: nx.MultiDiGraph,
    held_out: TransitionAggregate,
    ks: tuple[int, ...] = (1, 3, 5, 10),
) -> dict:
    """Evaluate training-derived graph edges on one held-out split.

    Metrics are observation-weighted unless prefixed with ``macro``. Missing
    sources and destinations contribute zero to overall ranking metrics.
    """
    if held_out.split == "train":
        raise ValueError("Held-out transition evaluation cannot use the training split.")
    if not ks or any(k < 1 for k in ks):
        raise ValueError("ks must contain positive integers")

    predictions = transition_predictions(G)
    total_observations = sum(held_out.pair_counts.values())
    sources = sorted({source for source, _ in held_out.pair_counts})
    covered_sources = {source for source in sources if predictions.get(source)}
    covered_observations = 0
    ranked_observations = 0
    reciprocal_rank_sum = 0.0
    hits = {k: 0 for k in ks}
    covered_hits = {k: 0 for k in ks}
    per_source: dict[str, dict[str, float]] = {}

    pairs_by_source: dict[str, list[tuple[str, int]]] = defaultdict(list)
    for (source, target), count in held_out.pair_counts.items():
        pairs_by_source[source].append((target, count))

    for source in sources:
        source_pairs = pairs_by_source[source]
        source_total = sum(count for _, count in source_pairs)
        destination_rank = {
            destination: index + 1
            for index, destination in enumerate(predictions.get(source, []))
        }
        source_rr = 0.0
        source_hits = {k: 0 for k in ks}

        if destination_rank:
            covered_observations += source_total

        for target, count in source_pairs:
            rank = destination_rank.get(target)
            if rank is None:
                continue
            ranked_observations += count
            reciprocal_rank_sum += count / rank
            source_rr += count / rank
            for k in ks:
                if rank <= k:
                    hits[k] += count
                    covered_hits[k] += count
                    source_hits[k] += count

        per_source[source] = {
            "observations": source_total,
            "covered": bool(destination_rank),
            "mrr": source_rr / source_total if source_total else 0.0,
            **{
                f"hits_at_{k}": source_hits[k] / source_total if source_total else 0.0
                for k in ks
            },
        }

    denominator = total_observations or 1
    covered_denominator = covered_observations or 1
    source_denominator = len(sources) or 1
    return {
        "split": held_out.split,
        "observations": total_observations,
        "distinct_pairs": len(held_out.pair_counts),
        "source_roles": len(sources),
        "covered_source_roles": len(covered_sources),
        "source_role_coverage": len(covered_sources) / source_denominator,
        "covered_observations": covered_observations,
        "observation_coverage": covered_observations / denominator,
        "ranked_destination_observations": ranked_observations,
        "destination_coverage": ranked_observations / denominator,
        "mrr": reciprocal_rank_sum / denominator,
        "covered_only_mrr": reciprocal_rank_sum / covered_denominator,
        **{f"hits_at_{k}": hits[k] / denominator for k in ks},
        **{
            f"covered_only_hits_at_{k}": covered_hits[k] / covered_denominator
            for k in ks
        },
        "macro_mrr": sum(item["mrr"] for item in per_source.values()) / source_denominator,
        **{
            f"macro_hits_at_{k}": (
                sum(item[f"hits_at_{k}"] for item in per_source.values())
                / source_denominator
            )
            for k in ks
        },
    }
