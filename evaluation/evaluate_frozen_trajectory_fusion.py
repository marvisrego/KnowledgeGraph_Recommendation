"""Test-only evaluation of immutable validation-selected trajectory fusion."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from config import Settings
from evaluation.evaluate_sequential_ranker import _load_artifact
from evaluation.evaluate_trajectory_fusion import _rank_vector, _second_order_ids, _transition_counts
from evaluation.research_resources import load_research_embeddings, load_research_graph
from src.karrierewege_preprocessing import build_esco_title_index, discover_split_files, write_json_atomic
from src.trajectory_benchmark import iter_karrierewege_prefixes
from src.transition_embedding import SmoothingConfig, compute_neighbour_index, rank_hybrid_destinations, transition_distributions


METRICS = ("mrr", "hits_at_1", "hits_at_3", "hits_at_5", "hits_at_10", "ndcg_at_10")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def load_frozen_weights(path: Path, artifact: Path) -> dict[str, tuple[float, float, float]]:
    manifest = json.loads(Path(path).read_text(encoding="utf-8"))
    if manifest.get("artifact_sha256") != sha256(artifact):
        raise ValueError("Fusion manifest artifact checksum does not match")
    if manifest.get("selection_split") != "validation":
        raise ValueError("Fusion manifest was not selected on validation")
    raw = manifest.get("selected_weights")
    if not isinstance(raw, dict) or set(raw) != {"single", "multi"}:
        raise ValueError("Fusion manifest needs single and multi selected weights")
    weights: dict[str, tuple[float, float, float]] = {}
    for group in ("single", "multi"):
        value = raw[group]
        if not isinstance(value, dict) or set(value) != {"neural", "smoother", "second_order"}:
            raise ValueError(f"Fusion manifest has invalid {group} weights")
        parsed = tuple(float(value[name]) for name in ("neural", "smoother", "second_order"))
        if any(item < 0.0 or item > 1.0 for item in parsed) or not math.isclose(sum(parsed), 1.0, abs_tol=1e-9):
            raise ValueError(f"Fusion manifest has invalid {group} simplex")
        weights[group] = parsed
    return weights


def _bootstrap_delta(values: np.ndarray, counts: np.ndarray, samples: int, seed: int) -> dict[str, list[float]]:
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(values), size=(samples, len(values)))
    estimates = values[draws].sum(axis=1) / counts[draws].sum(axis=1, keepdims=True)
    return {
        metric: [float(np.quantile(estimates[:, index], 0.025)), float(np.quantile(estimates[:, index], 0.975))]
        for index, metric in enumerate(METRICS)
    }


def _event_values(rank: int) -> np.ndarray:
    return np.asarray([
        1.0 / rank, float(rank <= 1), float(rank <= 3), float(rank <= 5), float(rank <= 10),
        1.0 / math.log2(rank + 1) if rank <= 10 else 0.0,
    ])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--fusion-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--bootstrap-samples", type=int, default=1000)
    args = parser.parse_args()
    if args.batch_size < 1 or args.bootstrap_samples < 1:
        parser.error("--batch-size and --bootstrap-samples must be positive")
    weights = load_frozen_weights(args.fusion_manifest, args.artifact)
    metadata, candidate_ids, score, device = _load_artifact(args.artifact)
    if metadata.get("model_type") not in {"causal_transformer", "causal_transformer_id_residual"}:
        raise ValueError("Frozen fusion requires a causal Transformer artifact")
    row_by_id = {role_id: index for index, role_id in enumerate(candidate_ids)}
    if len(row_by_id) != len(candidate_ids):
        raise ValueError("Artifact candidate IDs must be unique")
    settings = Settings.from_env()
    graph, graph_source = load_research_graph(settings)
    title_index = build_esco_title_index(graph)
    files = discover_split_files(settings.karrierewege_dir)
    direct, second, contexts = _transition_counts(iter_karrierewege_prefixes(
        files["train"], title_index, "train", settings.transition_chunk_size,
    ))
    examples = [
        (item.person_id, item.history_role_ids[-int(metadata.get("maximum_history_length", 10)):], item.target_role_id)
        for item in iter_karrierewege_prefixes(files["test"], title_index, "test", settings.transition_chunk_size)
        if item.target_role_id in row_by_id and all(role in row_by_id for role in item.history_role_ids)
    ]
    if not examples:
        raise RuntimeError("No test prefixes align with artifact candidates")
    embeddings, _ = load_research_embeddings(settings, graph)
    distributions = transition_distributions(graph)
    smoother_config = SmoothingConfig(neighbours=25, direct_weight=0.91, temperature=0.06)
    sources = sorted({history[-1] for _, history, _ in examples})
    neighbours = compute_neighbour_index(embeddings, distributions, max_neighbours=25, query_ids=sources)
    smoother = {
        source: [item.role_id for item in rank_hybrid_destinations(source, distributions, neighbours, smoother_config, graph)]
        for source in sources
    }
    import torch
    per_person: dict[str, np.ndarray] = defaultdict(lambda: np.zeros((3, len(METRICS)), dtype=np.float64))
    candidate_rows = torch.arange(len(candidate_ids), device=device)
    with torch.no_grad():
        for start in range(0, len(examples), args.batch_size):
            batch = examples[start:start + args.batch_size]
            people, histories, targets = zip(*batch)
            logits = score([[row_by_id[role] for role in history] for history in histories])
            order = torch.argsort(logits, dim=1, descending=True)
            neural_ranks = torch.empty_like(order)
            neural_ranks.scatter_(1, order, candidate_rows.expand(len(batch), -1) + 1)
            smooth_ranks = np.stack([_rank_vector(smoother.get(history[-1], ()), candidate_ids, row_by_id, history[-1]) for history in histories])
            second_ranks = np.stack([_rank_vector(
                [role for role, _ in sorted(_second_order_ids(history, direct, second, contexts).items(), key=lambda item: (-item[1], item[0]))],
                candidate_ids, row_by_id, history[-1],
            ) for history in histories])
            smooth_tensor = torch.as_tensor(smooth_ranks, device=device)
            second_tensor = torch.as_tensor(second_ranks, device=device)
            targets_tensor = torch.as_tensor([row_by_id[target] for target in targets], device=device)
            multi = [len(history) >= 2 for history in histories]
            for index, person in enumerate(people):
                neural_weight, smoother_weight, second_weight = weights["multi" if multi[index] else "single"]
                fused = neural_weight / (60.0 + neural_ranks[index].float()) + smoother_weight / (60.0 + smooth_tensor[index].float()) + second_weight / (60.0 + second_tensor[index].float())
                target_score = fused[targets_tensor[index]]
                fused_rank = int(1 + (fused > target_score).sum().item())
                baseline_rank = int(smooth_ranks[index, row_by_id[targets[index]]])
                per_person[person][0, 0] += 1
                per_person[person][1] += _event_values(fused_rank)
                per_person[person][2] += _event_values(baseline_rank)
    people = sorted(per_person)
    counts = np.asarray([per_person[person][0, 0] for person in people], dtype=np.int64)
    fused_sums = np.asarray([per_person[person][1] for person in people])
    baseline_sums = np.asarray([per_person[person][2] for person in people])
    total = int(counts.sum())
    metrics = {metric: float(fused_sums[:, index].sum() / total) for index, metric in enumerate(METRICS)}
    baseline = {metric: float(baseline_sums[:, index].sum() / total) for index, metric in enumerate(METRICS)}
    delta = fused_sums - baseline_sums
    payload = {
        "split": "test_only_frozen_fusion", "artifact": str(args.artifact), "artifact_sha256": sha256(args.artifact),
        "fusion_manifest": str(args.fusion_manifest), "fusion_manifest_sha256": sha256(args.fusion_manifest),
        "selected_weights": {group: dict(zip(("neural", "smoother", "second_order"), value)) for group, value in weights.items()},
        "candidate_roles": len(candidate_ids), "examples": total, "people": len(people), "coverage": 1.0,
        "metrics": metrics, "baseline_semantic_smoother": baseline,
        "delta_vs_semantic_smoother": {metric: metrics[metric] - baseline[metric] for metric in METRICS},
        "paired_person_bootstrap_delta_95": _bootstrap_delta(delta, counts, args.bootstrap_samples, int(metadata.get("seed", 17))),
        "protocol": {"candidate_set": "all_artifact_live_esco_roles", "current_role": "excluded", "weights": "validation_manifest_only", "bootstrap_unit": "person"},
        "graph_source": graph_source,
    }
    write_json_atomic(payload, args.output)
    print(json.dumps({"output": str(args.output), **metrics, "delta": payload["delta_vs_semantic_smoother"]}, indent=2))


if __name__ == "__main__":
    main()
