"""Validation-only reciprocal-rank fusion for trajectory rankers.

This command intentionally cannot read the test split.  It combines an
unpromoted causal ranker, train-only second-order counts, and the accepted
embedding smoother over exactly the artifact's full ESCO candidate set.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from config import Settings
from evaluation.evaluate_sequential_ranker import _load_artifact
from evaluation.research_resources import load_research_embeddings, load_research_graph
from src.karrierewege_preprocessing import build_esco_title_index, discover_split_files, write_json_atomic
from src.trajectory_benchmark import iter_karrierewege_prefixes
from src.transition_embedding import SmoothingConfig, compute_neighbour_index, rank_hybrid_destinations, transition_distributions


def _transition_counts(prefixes):
    first, totals, second, contexts = Counter(), Counter(), Counter(), Counter()
    for prefix in prefixes:
        history, target = prefix.history_role_ids, prefix.target_role_id
        source = history[-1]
        first[source, target] += 1
        totals[source] += 1
        if len(history) >= 2:
            context = history[-2], source
            second[context, target] += 1
            contexts[context] += 1
    direct = defaultdict(dict)
    for (source, target), count in first.items():
        if count >= 5:
            direct[source][target] = count / totals[source]
    grouped_second = defaultdict(dict)
    for (context, target), count in second.items():
        grouped_second[context][target] = count
    return direct, grouped_second, contexts


def _second_order_ids(history, direct, second, contexts):
    """Return train-only empirical-Bayes scores using frozen validation choice."""
    base = direct.get(history[-1], {})
    if len(history) < 2:
        return base
    context = history[-2], history[-1]
    total = contexts.get(context, 0)
    observed = second.get(context, {})
    if total < 5 or not observed:
        return base
    alpha, weight = 10.0, 0.25
    denominator = total + alpha
    scores = {
        candidate: (1.0 - weight) * probability + weight * alpha * probability / denominator
        for candidate, probability in base.items()
    }
    for candidate, count in observed.items():
        scores[candidate] = scores.get(candidate, 0.0) + weight * count / denominator
    return scores


def _rank_vector(ranked_ids, candidate_ids, row_by_id, current_id):
    """Convert a sparse deterministic ranking to ranks for every live candidate."""
    seen = set()
    ordered = []
    for role_id in ranked_ids:
        role_id = str(role_id)
        if role_id != current_id and role_id in row_by_id and role_id not in seen:
            ordered.append(role_id)
            seen.add(role_id)
    ordered.extend(role_id for role_id in candidate_ids if role_id != current_id and role_id not in seen)
    ranks = np.empty(len(candidate_ids), dtype=np.int32)
    ranks[[row_by_id[role_id] for role_id in ordered]] = np.arange(1, len(ordered) + 1, dtype=np.int32)
    ranks[row_by_id[current_id]] = len(candidate_ids) + 1
    return ranks


def _metric_sums(ranks):
    return {
        "mrr": float(np.sum(1.0 / ranks)),
        **{f"hits_at_{cutoff}": float(np.sum(ranks <= cutoff)) for cutoff in (1, 3, 5, 10)},
    }


def _mean_metrics(sums, count):
    return {key: value / count for key, value in sums.items()}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=512)
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error("--batch-size must be positive")

    metadata, candidate_ids, score, device = _load_artifact(args.artifact)
    if metadata.get("model_type") not in {"causal_transformer", "causal_transformer_id_residual"}:
        raise ValueError("Fusion currently requires a causal Transformer artifact")
    row_by_id = {role_id: index for index, role_id in enumerate(candidate_ids)}
    if len(row_by_id) != len(candidate_ids):
        raise ValueError("Artifact candidate IDs must be unique")

    settings = Settings.from_env()
    graph, graph_source = load_research_graph(settings)
    title_index = build_esco_title_index(graph)
    files = discover_split_files(settings.karrierewege_dir)
    direct, second, contexts = _transition_counts(
        iter_karrierewege_prefixes(files["train"], title_index, "train", settings.transition_chunk_size)
    )
    validation = [
        (item.person_id, item.history_role_ids[-int(metadata.get("maximum_history_length", 10)):], item.target_role_id)
        for item in iter_karrierewege_prefixes(files["validation"], title_index, "validation", settings.transition_chunk_size)
        if item.target_role_id in row_by_id and all(role in row_by_id for role in item.history_role_ids)
    ]
    if not validation:
        raise RuntimeError("No validation prefixes align with the artifact candidates")

    embeddings, _ = load_research_embeddings(settings, graph)
    distributions = transition_distributions(graph)
    smoother_config = SmoothingConfig(neighbours=25, direct_weight=0.91, temperature=0.06)
    sources = sorted({history[-1] for _, history, _ in validation})
    neighbours = compute_neighbour_index(embeddings, distributions, max_neighbours=25, query_ids=sources)
    smoother_cache = {
        source: [item.role_id for item in rank_hybrid_destinations(source, distributions, neighbours, smoother_config, graph)]
        for source in sources
    }

    grid = np.arange(0.0, 1.01, 0.25)
    single_configs = [(float(neural), float(1.0 - neural), 0.0) for neural in grid]
    multi_configs = [
        (float(neural), float(smoother), float(1.0 - neural - smoother))
        for neural in grid for smoother in grid if neural + smoother <= 1.0 + 1e-9
    ]
    totals = {"single": {config: Counter() for config in single_configs}, "multi": {config: Counter() for config in multi_configs}}
    counts = Counter()

    import torch
    candidate_rows = torch.arange(len(candidate_ids), device=device)
    with torch.no_grad():
        for start in range(0, len(validation), args.batch_size):
            batch = validation[start:start + args.batch_size]
            _, histories, targets = zip(*batch)
            neural_logits = score([[row_by_id[role] for role in history] for history in histories])
            order = torch.argsort(neural_logits, dim=1, descending=True)
            neural_ranks = torch.empty_like(order)
            neural_ranks.scatter_(1, order, candidate_rows.expand(len(batch), -1) + 1)
            smooth_ranks = np.stack([
                _rank_vector(smoother_cache.get(history[-1], ()), candidate_ids, row_by_id, history[-1])
                for history in histories
            ])
            second_ranks = np.stack([
                _rank_vector(
                    [role for role, _ in sorted(_second_order_ids(history, direct, second, contexts).items(), key=lambda item: (-item[1], item[0]))],
                    candidate_ids, row_by_id, history[-1],
                ) for history in histories
            ])
            smooth_tensor = torch.as_tensor(smooth_ranks, device=device)
            second_tensor = torch.as_tensor(second_ranks, device=device)
            target_rows = torch.as_tensor([row_by_id[target] for target in targets], device=device)
            multi_mask = torch.as_tensor([len(history) >= 2 for history in histories], device=device)
            for group, configs, mask in (("single", single_configs, ~multi_mask), ("multi", multi_configs, multi_mask)):
                if not bool(mask.any()):
                    continue
                counts[group] += int(mask.sum().item())
                for neural_weight, smoother_weight, second_weight in configs:
                    fused = (
                        neural_weight / (60.0 + neural_ranks.float())
                        + smoother_weight / (60.0 + smooth_tensor.float())
                        + second_weight / (60.0 + second_tensor.float())
                    )
                    target_scores = fused[torch.arange(len(batch), device=device), target_rows]
                    ranks = 1 + (fused > target_scores[:, None]).sum(dim=1)
                    selected = ranks[mask].detach().cpu().numpy()
                    totals[group][(neural_weight, smoother_weight, second_weight)].update(_metric_sums(selected))

    selected = {}
    for group, configs in (("single", single_configs), ("multi", multi_configs)):
        selected[group] = max(
            configs,
            key=lambda config: (
                _mean_metrics(totals[group][config], counts[group])["mrr"],
                _mean_metrics(totals[group][config], counts[group])["hits_at_5"],
                _mean_metrics(totals[group][config], counts[group])["hits_at_10"],
            ),
        )
    combined = Counter()
    for group in ("single", "multi"):
        combined.update(totals[group][selected[group]])
    payload = {
        "split": "validation_only",
        "artifact": str(args.artifact),
        "candidate_roles": len(candidate_ids),
        "examples": len(validation),
        "groups": {group: int(counts[group]) for group in ("single", "multi")},
        "selected_weights": {
            group: {"neural": value[0], "smoother": value[1], "second_order": value[2]}
            for group, value in selected.items()
        },
        "metrics": _mean_metrics(combined, len(validation)),
        "candidate_policy": "all artifact live ESCO roles; current role excluded",
        "training_evidence": "Karrierewege train only; read-only ESCO embeddings",
        "graph_source": graph_source,
    }
    write_json_atomic(payload, args.output)
    print(json.dumps({"output": str(args.output), **payload["metrics"], "selected_weights": payload["selected_weights"]}, indent=2))


if __name__ == "__main__":
    main()
