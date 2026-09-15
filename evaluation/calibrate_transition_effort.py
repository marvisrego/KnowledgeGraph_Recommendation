"""Fit a train-only, trajectory-proxy Transition Effort Score artifact.

The generated JSON is deliberately small and contains no person-level data.
Validation examples choose the effort-band thresholds; test data is not read.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from config import Settings
from evaluation.research_resources import load_research_graph
from src.karrierewege_preprocessing import build_esco_title_index, discover_split_files
from src.skill_gap import role_requirements
from src.tes_calibration import (
    TESCalibration,
    fit_nonnegative_pairwise_weights,
    thresholds_from_scores,
    write_calibration,
)
from src.trajectory_benchmark import iter_karrierewege_prefixes
from src.transition_effort import (
    domain_distance,
    empirical_support,
    get_idf_map,
    skill_gap_magnitude,
    transferability,
)


def _features(source_id, target_id, owned, graph, idf_map, threshold) -> list[float] | None:
    required, _, _ = role_requirements(target_id, graph, threshold)
    gap = skill_gap_magnitude(owned, required, idf_map)
    domain = domain_distance(source_id, target_id, graph)
    support = empirical_support(source_id, target_id, graph)
    transferable = transferability(source_id, target_id, graph, idf_map, threshold)
    if gap is None or domain is None or transferable is None:
        return None
    return [float(gap), float(domain), 1.0 - float(support), 1.0 - float(transferable)]


def _owned_history_skills(history, graph, threshold) -> set[str]:
    owned: set[str] = set()
    for role_id in history:
        required, _, _ = role_requirements(role_id, graph, threshold)
        owned.update(required)
    return owned


def _sample_alternatives(source_id, target_id, candidate_ids, graph, popular, rng) -> list[tuple[str, float]]:
    """Return 63 alternatives and inverse-probability proxy weights by stratum."""
    source_isco = str(graph.nodes[source_id].get("isco_group") or graph.nodes[source_id].get("isco_2digit") or "")
    hard = [
        role_id for role_id in candidate_ids
        if role_id not in {source_id, target_id}
        and source_isco
        and str(graph.nodes[role_id].get("isco_group") or graph.nodes[role_id].get("isco_2digit") or "").startswith(source_isco[:2])
    ]
    all_other = [role_id for role_id in candidate_ids if role_id not in {source_id, target_id}]
    selected: list[tuple[str, float]] = []
    for pool, count in ((hard or all_other, 32), (popular, 16), (all_other, 15)):
        choices = [role_id for role_id in pool if role_id not in {role for role, _ in selected}]
        if not choices:
            continue
        take = min(count, len(choices))
        for role_id in rng.choice(choices, size=take, replace=False).tolist():
            # Approximate inverse sampling correction. Exact probabilities are
            # retained per stratum rather than pretending alternatives are iid.
            selected.append((str(role_id), len(choices) / take))
    return selected


def _score(features: list[float], weights: dict[str, float]) -> float:
    return float(sum(weights[key] * features[index] for index, key in enumerate(("skill_gap", "domain", "empirical", "transferability"))))


def main() -> None:
    parser = argparse.ArgumentParser(description="Calibrate TES from train-only trajectory choice proxies")
    parser.add_argument("--output", type=Path, default=Path("artifacts/transition_effort/tes_calibration.json"))
    parser.add_argument("--max-examples", type=int, default=50_000)
    parser.add_argument("--seed", type=int, default=17)
    args = parser.parse_args()
    if args.max_examples < 1:
        parser.error("--max-examples must be positive")

    settings = Settings.from_env()
    graph, graph_source = load_research_graph(settings)
    title_index = build_esco_title_index(graph)
    split_files = discover_split_files(settings.karrierewege_dir)
    candidate_ids = sorted(
        str(node_id) for node_id, data in graph.nodes(data=True)
        if data.get("type") == "role" and data.get("source") == "esco"
    )
    idf_map = get_idf_map(graph)
    popularity = Counter()
    for example in iter_karrierewege_prefixes(split_files["train"], title_index, "train", settings.transition_chunk_size):
        popularity[example.target_role_id] += 1
    popular = [role_id for role_id, _ in popularity.most_common()]
    rng = np.random.default_rng(args.seed)
    positives: list[list[float]] = []
    negatives: list[list[float]] = []
    pair_weights: list[float] = []

    for example in iter_karrierewege_prefixes(split_files["train"], title_index, "train", settings.transition_chunk_size):
        if len(positives) >= args.max_examples * 63:
            break
        owned = _owned_history_skills(example.history_role_ids, graph, settings.onet_importance_threshold)
        positive = _features(example.history_role_ids[-1], example.target_role_id, owned, graph, idf_map, settings.onet_importance_threshold)
        if positive is None:
            continue
        for alternative, correction in _sample_alternatives(
            example.history_role_ids[-1], example.target_role_id, candidate_ids, graph, popular, rng
        ):
            negative = _features(example.history_role_ids[-1], alternative, owned, graph, idf_map, settings.onet_importance_threshold)
            if negative is not None:
                positives.append(positive)
                negatives.append(negative)
                pair_weights.append(correction)

    learned = fit_nonnegative_pairwise_weights(positives, negatives, pair_weights)
    validation_scores: list[float] = []
    for example in iter_karrierewege_prefixes(split_files["validation"], title_index, "validation", settings.transition_chunk_size):
        owned = _owned_history_skills(example.history_role_ids, graph, settings.onet_importance_threshold)
        values = _features(example.history_role_ids[-1], example.target_role_id, owned, graph, idf_map, settings.onet_importance_threshold)
        if values is not None:
            validation_scores.append(_score(values, learned))
    low, moderate = thresholds_from_scores(validation_scores)
    digest = hashlib.sha256(json.dumps({"weights": learned, "pairs": len(positives), "seed": args.seed}, sort_keys=True).encode()).hexdigest()
    artifact = TESCalibration(
        weights=learned,
        low_max=low,
        moderate_max=moderate,
        training_hash=digest,
    )
    write_calibration(artifact, args.output)
    print(json.dumps({
        "output": str(args.output),
        "graph_source": graph_source,
        "paired_examples": len(positives),
        "validation_observed_transitions": len(validation_scores),
        **artifact.to_dict(),
    }, indent=2))


if __name__ == "__main__":
    main()
