"""Train an unpromoted, full-candidate JobHop next-occupation ranker."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evaluation.causal_transformer import CausalTransformerRanker
from src.jobhop_feature_benchmark import (
    FEATURE_VOCABULARIES,
    assert_disjoint_splits,
    iter_jobhop_feature_prefixes,
)


def _load_embeddings(path: Path) -> tuple[list[str], np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        role_ids = [str(value) for value in archive["candidate_ids"].tolist()]
        vectors = np.asarray(archive["candidate_embeddings"], dtype=np.float32)
    if vectors.ndim != 2 or len(role_ids) != len(vectors) or not np.isfinite(vectors).all():
        raise ValueError("Embedding artifact must contain aligned finite candidate vectors")
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    if np.any(norms <= 1e-12):
        raise ValueError("Embedding artifact contains zero candidate vectors")
    return role_ids, vectors / norms


def _examples(
    path: Path,
    row_by_id: dict[str, int],
    maximum_history_length: int,
    include_static_resume_education: bool,
) -> list[tuple[str, list[int], int, tuple[int, int, int]]]:
    prefixes = list(iter_jobhop_feature_prefixes(
        path,
        maximum_history_length=maximum_history_length,
        include_static_resume_education=include_static_resume_education,
    ))
    unknown = {
        role
        for prefix in prefixes
        for role in (*prefix.history_role_ids, prefix.target_role_id)
        if role not in row_by_id
    }
    if unknown:
        raise ValueError(f"JobHop prefixes contain roles absent from the candidate artifact: {len(unknown)}")
    return [
        (
            prefix.person_id,
            [row_by_id[role] for role in prefix.history_role_ids],
            row_by_id[prefix.target_role_id],
            prefix.context_features,
        )
        for prefix in prefixes
    ]


def _feature_availability(rows: list[tuple[str, list[int], int, tuple[int, int, int]]]) -> dict[str, int]:
    return {
        "prefixes": len(rows),
        "missing_tenure": sum(features[0] == 0 for _, _, _, features in rows),
        "missing_gap": sum(features[1] == 0 for _, _, _, features in rows),
        "static_resume_education": sum(features[2] != 0 for _, _, _, features in rows),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--embedding-artifact", type=Path, required=True)
    parser.add_argument("--train", type=Path, default=Path("Data/JobHop_v2/processed/train.parquet"))
    parser.add_argument("--validation", type=Path, default=Path("Data/JobHop_v2/processed/val.parquet"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--features", choices=("role_only", "time_only", "time_and_static_resume_education"), default="time_only")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--maximum-history-length", type=int, default=10)
    parser.add_argument("--seed", type=int, default=17)
    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 1 or args.maximum_history_length < 1:
        parser.error("--epochs, --batch-size, and --maximum-history-length must be positive")
    try:
        import torch
        from torch.nn import functional as F
    except ImportError as exc:  # pragma: no cover - environment diagnostic
        raise SystemExit("PyTorch is required offline; install requirements-training.txt") from exc

    assert_disjoint_splits({"train": args.train, "validation": args.validation})
    candidate_ids, candidate_embeddings = _load_embeddings(args.embedding_artifact)
    row_by_id = {role_id: index for index, role_id in enumerate(candidate_ids)}
    if len(row_by_id) != len(candidate_ids):
        raise ValueError("Candidate artifact role IDs must be unique")
    include_education = args.features == "time_and_static_resume_education"
    train = _examples(args.train, row_by_id, args.maximum_history_length, include_education)
    validation = _examples(args.validation, row_by_id, args.maximum_history_length, include_education)
    if not train or not validation:
        raise RuntimeError("No usable JobHop train/validation prefixes")

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    candidate = torch.tensor(candidate_embeddings, device=device)
    context_sizes = FEATURE_VOCABULARIES if args.features != "role_only" else None
    model = CausalTransformerRanker(
        candidate,
        maximum_history_length=args.maximum_history_length,
        hidden_size=128,
        attention_heads=4,
        layers=2,
        context_feature_sizes=context_sizes,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    rng = np.random.default_rng(args.seed)

    def _score(histories: list[list[int]], features: list[tuple[int, int, int]]):
        return model(histories, features if context_sizes else None)

    def train_epoch(rows: list[tuple[str, list[int], int, tuple[int, int, int]]]) -> None:
        model.train()
        order = rng.permutation(len(rows))
        for start in range(0, len(order), args.batch_size):
            batch = [rows[index] for index in order[start:start + args.batch_size]]
            _, histories, targets, features = zip(*batch)
            logits = _score(list(histories), list(features))
            loss = F.cross_entropy(logits, torch.tensor(targets, device=device), label_smoothing=0.1)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

    def validation_metrics() -> dict[str, float]:
        model.eval()
        totals = {"mrr": 0.0, "hits_at_1": 0.0, "hits_at_3": 0.0, "hits_at_5": 0.0, "hits_at_10": 0.0}
        with torch.no_grad():
            for start in range(0, len(validation), args.batch_size):
                batch = validation[start:start + args.batch_size]
                _, histories, targets, features = zip(*batch)
                logits = _score(list(histories), list(features))
                target_tensor = torch.tensor(targets, device=device)
                target_scores = logits[torch.arange(len(batch), device=device), target_tensor]
                ranks = 1 + (logits > target_scores[:, None]).sum(1)
                totals["mrr"] += float((1.0 / ranks.float()).sum().item())
                for cutoff in (1, 3, 5, 10):
                    totals[f"hits_at_{cutoff}"] += float((ranks <= cutoff).sum().item())
        return {name: value / len(validation) for name, value in totals.items()}

    best_state: dict[str, object] | None = None
    best_metrics: dict[str, float] | None = None
    patience = 0
    for epoch in range(1, args.epochs + 1):
        train_epoch(train)
        metrics = validation_metrics()
        print(json.dumps({"phase": "jobhop_train", "epoch": epoch, "validation": metrics}), flush=True)
        if best_metrics is None or (
            metrics["mrr"], metrics["hits_at_5"], metrics["hits_at_10"],
        ) > (
            best_metrics["mrr"], best_metrics["hits_at_5"], best_metrics["hits_at_10"],
        ):
            best_metrics, patience = metrics, 0
            best_state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
        else:
            patience += 1
            if patience >= 2:
                break
    assert best_state is not None and best_metrics is not None
    model.load_state_dict(best_state)
    metadata = {
        "promoted": False,
        "model_type": "jobhop_feature_causal_transformer",
        "feature_policy": args.features,
        "selection_split": "validation",
        "selection_metric": {"validation": best_metrics},
        "seed": args.seed,
        "dataset": "jobhop_v2_official_person_disjoint",
        "maximum_history_length": args.maximum_history_length,
        "candidate_roles": len(candidate_ids),
        "model_config": model.configuration(),
    }
    metadata["feature_availability"] = {
        "train": _feature_availability(train),
        "validation": _feature_availability(validation),
        "static_resume_education_enabled": include_education,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        metadata=np.asarray(json.dumps(metadata, sort_keys=True)),
        candidate_ids=np.asarray(candidate_ids),
        candidate_embeddings=candidate_embeddings,
        **{name: value.detach().cpu().numpy() for name, value in model.export().items()},
    )
    print(json.dumps({"output": str(args.output), "validation": best_metrics, "promoted": False}, indent=2))


if __name__ == "__main__":
    main()
