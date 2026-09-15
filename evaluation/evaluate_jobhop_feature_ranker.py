"""Evaluate a frozen JobHop feature-ranker against all live ESCO candidates."""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evaluation.causal_transformer import CausalTransformerRanker
from src.jobhop_feature_benchmark import FEATURE_VOCABULARIES, iter_jobhop_feature_prefixes
from src.karrierewege_preprocessing import write_json_atomic


def _load_artifact(path: Path):
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - environment diagnostic
        raise SystemExit("PyTorch is required offline; install requirements-training.txt") from exc
    with np.load(path, allow_pickle=False) as archive:
        metadata = json.loads(str(archive["metadata"].item()))
        candidate_ids = tuple(str(value) for value in archive["candidate_ids"].tolist())
        candidate = np.asarray(archive["candidate_embeddings"], dtype=np.float32)
        parameters = {
            name: np.asarray(archive[name], dtype=np.float32)
            for name in archive.files
            if name not in {"metadata", "candidate_ids", "candidate_embeddings"}
        }
    if metadata.get("model_type") != "jobhop_feature_causal_transformer":
        raise ValueError("Artifact is not a JobHop feature-aware causal Transformer")
    if metadata.get("selection_split") != "validation":
        raise ValueError("JobHop artifact lacks a validation-only frozen selection")
    if candidate.ndim != 2 or len(candidate_ids) != len(candidate) or not np.isfinite(candidate).all():
        raise ValueError("Artifact candidates are invalid")
    norms = np.linalg.norm(candidate, axis=1, keepdims=True)
    if np.any(norms <= 1e-12):
        raise ValueError("Artifact contains zero candidate vectors")
    candidate /= norms
    if len(set(candidate_ids)) != len(candidate_ids):
        raise ValueError("Artifact candidate IDs must be unique")
    config = metadata.get("model_config", {})
    feature_policy = metadata.get("feature_policy")
    context_sizes = FEATURE_VOCABULARIES if feature_policy != "role_only" else None
    if config.get("context_feature_sizes") != (list(context_sizes) if context_sizes else None):
        raise ValueError("Artifact feature configuration does not match its declared policy")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CausalTransformerRanker(
        torch.tensor(candidate, device=device),
        maximum_history_length=int(config["maximum_history_length"]),
        hidden_size=int(config["hidden_size"]),
        attention_heads=int(config["attention_heads"]),
        layers=int(config["layers"]),
        dropout=0.0,
        id_residual=bool(config.get("id_residual", False)),
        context_feature_sizes=context_sizes,
    ).to(device)
    parameter_tensors = {name: torch.tensor(value, device=device) for name, value in parameters.items()}
    missing, unexpected = model.load_state_dict(parameter_tensors, strict=False)
    if set(missing) != {"candidate"} or unexpected:
        raise ValueError(f"Artifact parameters are invalid: missing={missing}, unexpected={unexpected}")
    model.eval()
    return metadata, candidate_ids, model, device, context_sizes is not None


def _bootstrap(sums: np.ndarray, counts: np.ndarray, samples: int, seed: int) -> dict[str, list[float]]:
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(sums), size=(samples, len(sums)))
    estimates = sums[draws].sum(axis=1) / counts[draws].sum(axis=1, keepdims=True)
    labels = ("mrr", "hits_at_1", "hits_at_3", "hits_at_5", "hits_at_10", "ndcg_at_10")
    return {
        label: [float(np.quantile(estimates[:, index], 0.025)), float(np.quantile(estimates[:, index], 0.975))]
        for index, label in enumerate(labels)
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--split", choices=("validation", "test"), required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--bootstrap-samples", type=int, default=500)
    args = parser.parse_args()
    if args.batch_size < 1 or args.bootstrap_samples < 1:
        parser.error("--batch-size and --bootstrap-samples must be positive")
    metadata, candidate_ids, model, device, has_features = _load_artifact(args.artifact)
    include_education = metadata["feature_policy"] == "time_and_static_resume_education"
    row_by_id = {role_id: index for index, role_id in enumerate(candidate_ids)}
    prefixes = list(iter_jobhop_feature_prefixes(
        args.data,
        maximum_history_length=int(metadata["maximum_history_length"]),
        include_static_resume_education=include_education,
    ))
    unknown = {
        role
        for prefix in prefixes
        for role in (*prefix.history_role_ids, prefix.target_role_id)
        if role not in row_by_id
    }
    if unknown:
        raise ValueError(f"Evaluation split contains roles absent from artifact candidates: {len(unknown)}")
    examples = [
        (
            prefix.person_id,
            [row_by_id[role] for role in prefix.history_role_ids],
            row_by_id[prefix.target_role_id],
            prefix.context_features,
        )
        for prefix in prefixes
    ]
    if not examples:
        raise RuntimeError("No usable evaluation prefixes")

    import torch
    per_person: dict[str, list[float]] = defaultdict(lambda: [0.0] * 7)
    with torch.no_grad():
        for start in range(0, len(examples), args.batch_size):
            batch = examples[start:start + args.batch_size]
            people, histories, targets, features = zip(*batch)
            logits = model(list(histories), list(features) if has_features else None)
            target_tensor = torch.tensor(targets, device=device)
            target_scores = logits[torch.arange(len(batch), device=device), target_tensor]
            ranks = (1 + (logits > target_scores[:, None]).sum(1)).detach().cpu().tolist()
            for person, rank in zip(people, ranks):
                values = per_person[person]
                values[0] += 1
                values[1] += 1.0 / rank
                values[2] += float(rank <= 1)
                values[3] += float(rank <= 3)
                values[4] += float(rank <= 5)
                values[5] += float(rank <= 10)
                values[6] += 1.0 / math.log2(rank + 1) if rank <= 10 else 0.0
    people = sorted(per_person)
    counts = np.asarray([per_person[person][0] for person in people], dtype=np.int64)
    sums = np.asarray([per_person[person][1:] for person in people], dtype=np.float64)
    total = int(counts.sum())
    labels = ("mrr", "hits_at_1", "hits_at_3", "hits_at_5", "hits_at_10", "ndcg_at_10")
    metrics = {label: float(sums[:, index].sum() / total) for index, label in enumerate(labels)}
    payload = {
        "model": "jobhop_feature_aware_full_candidate",
        "artifact": str(args.artifact),
        "artifact_metadata": metadata,
        "split": args.split,
        "candidate_roles": len(candidate_ids),
        "examples": total,
        "people": len(people),
        "coverage": 1.0,
        **metrics,
        "confidence_intervals_95": _bootstrap(sums, counts, args.bootstrap_samples, int(metadata.get("seed", 17))),
        "protocol": {
            "candidate_set": "all_artifact_live_esco_roles",
            "current_role": "excluded",
            "bootstrap_unit": "person",
            "feature_policy": metadata["feature_policy"],
            "target_derived_features": False,
        },
    }
    write_json_atomic(payload, args.output)
    print(json.dumps({"output": str(args.output), **{key: payload[key] for key in ("mrr", "hits_at_5", "hits_at_10", "coverage")}}, indent=2))


if __name__ == "__main__":
    main()
