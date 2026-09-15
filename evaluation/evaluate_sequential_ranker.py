"""Evaluate an offline sequential ranker on leakage-safe Karrierewege prefixes.

The command intentionally evaluates one named split at a time.  Run it on
validation while selecting an experiment; run it on test only after that
selection is frozen.  It never writes to AuraDB or Qdrant.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from config import Settings
from evaluation.causal_transformer import CausalTransformerRanker
from evaluation.research_resources import load_research_graph
from src.karrierewege_preprocessing import build_esco_title_index, discover_split_files, write_json_atomic
from src.trajectory_benchmark import iter_karrierewege_prefixes


def _load_artifact(path: Path):
    try:
        import torch
        from torch.nn import functional as F
    except ImportError as exc:  # pragma: no cover - environment diagnostic
        raise SystemExit("PyTorch is required; install requirements-training.txt") from exc

    with np.load(path, allow_pickle=False) as archive:
        metadata = json.loads(str(archive["metadata"].item()))
        candidate_ids = tuple(str(value) for value in archive["candidate_ids"].tolist())
        candidate = np.asarray(archive["candidate_embeddings"], dtype=np.float32)
        params = {name: np.asarray(archive[name], dtype=np.float32) for name in archive.files
                  if name not in {"metadata", "candidate_ids", "candidate_embeddings"}}
    if metadata.get("model_type") not in {
        "recency_mlp", "step_gru", "causal_transformer", "causal_transformer_id_residual",
    }:
        raise ValueError("Artifact is not a supported sequential ranker")
    if candidate.ndim != 2 or len(candidate_ids) != len(candidate):
        raise ValueError("Artifact candidates are invalid")
    candidate /= np.linalg.norm(candidate, axis=1, keepdims=True)
    if not np.isfinite(candidate).all():
        raise ValueError("Artifact candidate embeddings are invalid")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    candidate_tensor = torch.tensor(candidate, device=device)
    parameter_tensors = {name: torch.tensor(value, device=device) for name, value in params.items()}
    dimension = candidate_tensor.shape[1]
    hidden = int(metadata.get("hidden_size", 64))

    if metadata["model_type"] in {"causal_transformer", "causal_transformer_id_residual"}:
        config = metadata.get("model_config", {})
        model = CausalTransformerRanker(
            candidate_tensor,
            maximum_history_length=int(config.get("maximum_history_length", metadata.get("maximum_history_length", 10))),
            hidden_size=int(config.get("hidden_size", hidden)),
            attention_heads=int(config.get("attention_heads", 4)),
            layers=int(config.get("layers", 2)),
            dropout=0.0,
            id_residual=bool(config.get("id_residual", metadata["model_type"] == "causal_transformer_id_residual")),
        ).to(device)
        missing, unexpected = model.load_state_dict(parameter_tensors, strict=False)
        if set(missing) != {"candidate"} or unexpected:
            raise ValueError(f"Causal Transformer artifact parameters are invalid: missing={missing}, unexpected={unexpected}")
        model.eval()

        def score(histories: list[list[int]]):
            return model(histories)

        return metadata, candidate_ids, score, device

    def score(histories: list[list[int]]):
        max_length = max(len(history) for history in histories)
        indices = torch.zeros((len(histories), max_length), dtype=torch.long, device=device)
        mask = torch.zeros((len(histories), max_length), dtype=torch.bool, device=device)
        for row, history in enumerate(histories):
            indices[row, :len(history)] = torch.tensor(history, device=device)
            mask[row, :len(history)] = True
        vectors = candidate_tensor[indices]
        if metadata["model_type"] == "recency_mlp":
            weights = torch.arange(1, max_length + 1, dtype=torch.float32, device=device)[None, :] * mask
            pooled = (vectors * weights[:, :, None]).sum(1) / weights.sum(1, keepdim=True)
            representation = torch.tanh(pooled @ parameter_tensors["w1"] + parameter_tensors["b1"])
        else:
            state = torch.zeros((len(histories), hidden), device=device)
            states = []
            for step in range(max_length):
                vector = vectors[:, step]
                update = torch.sigmoid(vector @ parameter_tensors["w_z"] + state @ parameter_tensors["u_z"] + parameter_tensors["b_z"])
                reset = torch.sigmoid(vector @ parameter_tensors["w_r"] + state @ parameter_tensors["u_r"] + parameter_tensors["b_r"])
                proposal = torch.tanh(vector @ parameter_tensors["w_h"] + (reset * state) @ parameter_tensors["u_h"] + parameter_tensors["b_h"])
                next_state = (1.0 - update) * state + update * proposal
                state = torch.where(mask[:, step, None], next_state, state)
                states.append(state)
            all_states = torch.stack(states, dim=1)
            attention = all_states @ parameter_tensors["attention_w"]
            attention = attention.masked_fill(~mask, float("-inf"))
            representation = (torch.softmax(attention, dim=1)[:, :, None] * all_states).sum(1)
        predicted = F.normalize(
            representation @ parameter_tensors["w_out"] + parameter_tensors["b_out"], dim=1,
        )
        temperature = min(1.0, max(1e-3, float(metadata.get("temperature", 0.07))))
        logits = predicted @ candidate_tensor.T / temperature
        current = indices[torch.arange(len(histories), device=device), mask.sum(1) - 1]
        logits[torch.arange(len(histories), device=device), current] = float("-inf")
        return logits

    return metadata, candidate_ids, score, device


def _bootstrap(records: dict[str, list[dict[str, float]]], samples: int, seed: int) -> dict[str, list[float]]:
    people = sorted(records)
    keys = tuple(next(iter(records.values()))[0])
    # The headline metrics weight every prefix equally.  Cluster resampling
    # must therefore re-aggregate each sampled person's prefix sums and
    # counts, rather than averaging person means (a different macro-person
    # estimand that can yield intervals not centred on the reported metric).
    values = np.asarray([
        [sum(record[key] for record in records[person]) for key in keys]
        for person in people
    ])
    counts = np.asarray([len(records[person]) for person in people], dtype=np.int64)
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(people), size=(samples, len(people)))
    estimates = values[draws].sum(axis=1) / counts[draws].sum(axis=1, keepdims=True)
    return {key: [float(np.quantile(estimates[:, index], 0.025)), float(np.quantile(estimates[:, index], 0.975))]
            for index, key in enumerate(keys)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--split", choices=("validation", "test"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--bootstrap-samples", type=int, default=500)
    args = parser.parse_args()
    if args.batch_size < 1 or args.bootstrap_samples < 1:
        parser.error("--batch-size and --bootstrap-samples must be positive")

    metadata, candidate_ids, score, device = _load_artifact(args.artifact)
    row_by_id = {role_id: index for index, role_id in enumerate(candidate_ids)}
    if len(row_by_id) != len(candidate_ids):
        raise ValueError("Artifact candidate IDs must be unique")
    settings = Settings.from_env()
    graph, graph_source = load_research_graph(settings)
    title_index = build_esco_title_index(graph)
    split_file = discover_split_files(settings.karrierewege_dir)[args.split]
    maximum_history = int(metadata.get("maximum_history_length", 10))
    examples = [
        (example.person_id, [row_by_id[role] for role in example.history_role_ids[-maximum_history:]], row_by_id[example.target_role_id])
        for example in iter_karrierewege_prefixes(split_file, title_index, args.split, settings.transition_chunk_size)
        if example.target_role_id in row_by_id and all(role in row_by_id for role in example.history_role_ids[-maximum_history:])
    ]
    if not examples:
        raise RuntimeError("No prefixes aligned with artifact candidates")

    import torch
    per_person: dict[str, list[dict[str, float]]] = defaultdict(list)
    with torch.no_grad():
        for start in range(0, len(examples), args.batch_size):
            batch = examples[start:start + args.batch_size]
            people, histories, targets = zip(*batch)
            logits = score(list(histories))
            target_tensor = torch.tensor(targets, device=device)
            target_scores = logits[torch.arange(len(batch), device=device), target_tensor]
            ranks = (1 + (logits > target_scores[:, None]).sum(1)).detach().cpu().tolist()
            for person, rank in zip(people, ranks):
                record = {"mrr": 1.0 / rank}
                for cutoff in (1, 3, 5, 10):
                    record[f"hits_at_{cutoff}"] = float(rank <= cutoff)
                    record[f"ndcg_at_{cutoff}"] = 1.0 / math.log2(rank + 1) if rank <= cutoff else 0.0
                per_person[person].append(record)

    flattened = [record for rows in per_person.values() for record in rows]
    metrics = {key: float(np.mean([record[key] for record in flattened])) for key in flattened[0]}
    payload = {
        "model": "sequential_full_candidate",
        "artifact": str(args.artifact),
        "artifact_metadata": metadata,
        "graph_source": graph_source,
        "split": args.split,
        "candidate_roles": len(candidate_ids),
        "examples": len(flattened),
        "people": len(per_person),
        "coverage": 1.0,
        **metrics,
        "confidence_intervals_95": _bootstrap(per_person, args.bootstrap_samples, int(metadata.get("seed", 17))),
        "protocol": {"candidate_set": "all_artifact_live_esco_roles", "current_role": "excluded", "bootstrap_unit": "person"},
    }
    write_json_atomic(payload, args.output)
    print(json.dumps({"output": str(args.output), **{key: payload[key] for key in ("mrr", "hits_at_5", "hits_at_10", "coverage")}}, indent=2))


if __name__ == "__main__":
    main()
