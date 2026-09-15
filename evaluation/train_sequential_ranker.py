"""Offline full-candidate trainer for the reduced MLP/STEP trajectory rankers.

The script exports an *unpromoted* NumPy artifact. A separate held-out
evaluation must satisfy the documented promotion gate before metadata can be
changed to ``promoted=true`` and deployed.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from config import Settings
from evaluation.causal_transformer import CausalTransformerRanker
from evaluation.research_resources import load_research_graph
from src.karrierewege_preprocessing import build_esco_title_index, discover_split_files
from src.trajectory_benchmark import iter_karrierewege_prefixes


def _load_embeddings(path: Path) -> tuple[list[str], np.ndarray]:
    with np.load(path, allow_pickle=False) as data:
        role_ids = [str(item) for item in data["candidate_ids"].tolist()]
        vectors = np.asarray(data["candidate_embeddings"], dtype=np.float32)
    if vectors.ndim != 2 or len(role_ids) != len(vectors) or not np.isfinite(vectors).all():
        raise ValueError("Embedding artifact must have aligned finite candidate_ids and candidate_embeddings")
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    if np.any(norms <= 1e-12):
        raise ValueError("Embedding artifact contains zero vectors")
    return role_ids, vectors / norms


def _examples(path, title_index, split, chunk_size, row_by_id, maximum_history_length):
    return [
        (
            [row_by_id[role] for role in example.history_role_ids[-maximum_history_length:]],
            row_by_id[example.target_role_id],
        )
        for example in iter_karrierewege_prefixes(path, title_index, split, chunk_size)
        if example.target_role_id in row_by_id and all(role in row_by_id for role in example.history_role_ids)
    ]


def _jobhop_examples(path: Path, row_by_id: dict[str, int]):
    """Read already cleaned JobHop rows for reduced-mode pretraining only."""
    import pandas as pd

    frame = pd.read_parquet(path, columns=["resume_id", "role_id", "order_quarter"])
    frame = frame.sort_values(["resume_id", "order_quarter", "role_id"], kind="mergesort")
    output = []
    for _, rows in frame.groupby("resume_id", sort=False):
        roles = [str(role) for role in rows["role_id"] if str(role) in row_by_id]
        history: list[int] = []
        for role in roles:
            row = row_by_id[role]
            if history and history[-1] != row:
                output.append((list(history), row))
            if not history or history[-1] != row:
                history.append(row)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description="Train reduced full-candidate MLP or STEP ranker")
    parser.add_argument("--embedding-artifact", type=Path, required=True, help="NPZ with candidate_ids and candidate_embeddings")
    parser.add_argument(
        "--model",
        choices=("recency_mlp", "step_gru", "causal_transformer", "causal_transformer_id_residual"),
        default="step_gru",
    )
    parser.add_argument("--output", type=Path, default=Path("artifacts/sequential_ranking/candidate_ranker.npz"))
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--maximum-history-length", type=int, default=10)
    parser.add_argument("--jobhop-train", type=Path, help="Optional cleaned JobHop train.parquet for reduced-mode pretraining")
    parser.add_argument("--jobhop-pretrain-epochs", type=int, default=3)
    parser.add_argument("--seed", type=int, default=17)
    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 1 or args.maximum_history_length < 1:
        parser.error("--epochs, --batch-size, and --maximum-history-length must be positive")
    try:
        import torch
        from torch import nn
        from torch.nn import functional as F
    except ImportError as exc:
        raise SystemExit("PyTorch is required offline; install requirements-training.txt") from exc

    settings = Settings.from_env()
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    graph, graph_source = load_research_graph(settings)
    title_index = build_esco_title_index(graph)
    split_files = discover_split_files(settings.karrierewege_dir)
    candidate_ids, candidate_embeddings = _load_embeddings(args.embedding_artifact)
    # The PCA is outcome-independent: it sees only role vectors, never labels.
    if candidate_embeddings.shape[1] > 768:
        if device.type == "cuda":
            # Full SVD is needlessly expensive here.  Low-rank GPU PCA sees
            # only frozen role vectors and is therefore outcome-independent.
            vector_tensor = torch.tensor(candidate_embeddings, device=device)
            vector_tensor -= vector_tensor.mean(dim=0, keepdim=True)
            _, _, components = torch.pca_lowrank(vector_tensor, q=768, center=False, niter=5)
            candidate_embeddings = (vector_tensor @ components[:, :768]).cpu().numpy().astype(np.float32)
            del vector_tensor, components
            torch.cuda.empty_cache()
        else:
            from sklearn.decomposition import PCA
            candidate_embeddings = PCA(
                n_components=768,
                random_state=args.seed,
                svd_solver="randomized",
                iterated_power=5,
            ).fit_transform(candidate_embeddings).astype(np.float32)
        candidate_embeddings /= np.linalg.norm(candidate_embeddings, axis=1, keepdims=True)
    row_by_id = {role_id: index for index, role_id in enumerate(candidate_ids)}
    train = _examples(
        split_files["train"], title_index, "train", settings.transition_chunk_size,
        row_by_id, args.maximum_history_length,
    )
    validation = _examples(
        split_files["validation"], title_index, "validation", settings.transition_chunk_size,
        row_by_id, args.maximum_history_length,
    )
    if not train or not validation:
        raise RuntimeError("No aligned train/validation prefixes available")

    candidate = torch.tensor(candidate_embeddings, device=device)
    dimension = candidate.shape[1]
    hidden = 64

    class Model(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.register_buffer("candidate", candidate)
            if args.model == "recency_mlp":
                self.w1 = nn.Parameter(torch.empty(dimension, hidden))
                self.b1 = nn.Parameter(torch.zeros(hidden))
            else:
                for gate in ("z", "r", "h"):
                    setattr(self, f"w_{gate}", nn.Parameter(torch.empty(dimension, hidden)))
                    setattr(self, f"u_{gate}", nn.Parameter(torch.empty(hidden, hidden)))
                    setattr(self, f"b_{gate}", nn.Parameter(torch.zeros(hidden)))
                self.attention_w = nn.Parameter(torch.empty(hidden))
            self.w_out = nn.Parameter(torch.empty(hidden, dimension))
            self.b_out = nn.Parameter(torch.zeros(dimension))
            self.log_temperature = nn.Parameter(torch.tensor(np.log(0.07), dtype=torch.float32))
            for parameter in self.parameters():
                if parameter.ndim >= 2:
                    nn.init.xavier_uniform_(parameter)

        def forward(self, history: list[list[int]]) -> torch.Tensor:
            max_length = max(len(item) for item in history)
            indices = torch.zeros((len(history), max_length), dtype=torch.long, device=device)
            mask = torch.zeros((len(history), max_length), dtype=torch.bool, device=device)
            for row, items in enumerate(history):
                indices[row, :len(items)] = torch.tensor(items, device=device)
                mask[row, :len(items)] = True
            vectors = self.candidate[indices]
            if args.model == "recency_mlp":
                recency = torch.arange(1, max_length + 1, device=device, dtype=torch.float32)[None, :] * mask
                pooled = (vectors * recency[:, :, None]).sum(1) / recency.sum(1, keepdim=True)
                representation = torch.tanh(pooled @ self.w1 + self.b1)
            else:
                state = torch.zeros((len(history), hidden), device=device)
                states = []
                for step in range(max_length):
                    vector = vectors[:, step]
                    update = torch.sigmoid(vector @ self.w_z + state @ self.u_z + self.b_z)
                    reset = torch.sigmoid(vector @ self.w_r + state @ self.u_r + self.b_r)
                    proposal = torch.tanh(vector @ self.w_h + (reset * state) @ self.u_h + self.b_h)
                    new_state = (1 - update) * state + update * proposal
                    state = torch.where(mask[:, step, None], new_state, state)
                    states.append(state)
                states = torch.stack(states, dim=1)
                attention = states @ self.attention_w
                attention = attention.masked_fill(~mask, float("-inf"))
                attention = torch.softmax(attention, dim=1)
                representation = (attention[:, :, None] * states).sum(1)
            predicted = F.normalize(representation @ self.w_out + self.b_out, dim=1)
            temperature = torch.clamp(torch.exp(self.log_temperature), 1e-3, 1.0)
            logits = predicted @ self.candidate.T / temperature
            current_rows = indices[torch.arange(len(history), device=device), mask.sum(1) - 1]
            logits[torch.arange(len(history), device=device), current_rows] = float("-inf")
            return logits

        def export(self) -> dict[str, np.ndarray]:
            names = ["w_out", "b_out"]
            if args.model == "recency_mlp":
                names += ["w1", "b1"]
            else:
                names += [f"{part}_{gate}" for gate in ("z", "r", "h") for part in ("w", "u", "b")] + ["attention_w"]
            return {name: getattr(self, name).detach().cpu().numpy() for name in names}

    if args.model in {"causal_transformer", "causal_transformer_id_residual"}:
        hidden = 128
        model = CausalTransformerRanker(
            candidate,
            maximum_history_length=args.maximum_history_length,
            hidden_size=hidden,
            attention_heads=4,
            layers=2,
            id_residual=args.model == "causal_transformer_id_residual",
        ).to(device)
    else:
        model = Model().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    rng = np.random.default_rng(args.seed)

    def train_epoch(rows) -> None:
        model.train()
        order = rng.permutation(len(rows))
        for start in range(0, len(order), args.batch_size):
            batch = [rows[index] for index in order[start:start + args.batch_size]]
            histories, targets = zip(*batch)
            logits = model(list(histories))
            loss = F.cross_entropy(logits, torch.tensor(targets, device=device), label_smoothing=0.1)
            optimizer.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); optimizer.step()

    jobhop_pretrained = False
    if args.jobhop_train:
        jobhop_train = _jobhop_examples(args.jobhop_train, row_by_id)
        if jobhop_train:
            print(json.dumps({"phase": "jobhop_pretrain", "examples": len(jobhop_train)}), flush=True)
            for epoch in range(1, args.jobhop_pretrain_epochs + 1):
                train_epoch(jobhop_train)
                print(json.dumps({"phase": "jobhop_pretrain", "epoch": epoch}), flush=True)
            jobhop_pretrained = True
    def evaluate_validation() -> dict[str, float]:
        model.eval()
        totals = {"mrr": 0.0, "hits_at_1": 0.0, "hits_at_3": 0.0, "hits_at_5": 0.0, "hits_at_10": 0.0}
        with torch.no_grad():
            for start in range(0, len(validation), args.batch_size):
                batch = validation[start:start + args.batch_size]
                histories, targets = zip(*batch)
                logits = model(list(histories))
                target_tensor = torch.tensor(targets, device=device)
                target_scores = logits[torch.arange(len(batch), device=device), target_tensor]
                ranks = 1 + (logits > target_scores[:, None]).sum(1)
                totals["mrr"] += float((1.0 / ranks.float()).sum().item())
                for cutoff in (1, 3, 5, 10):
                    totals[f"hits_at_{cutoff}"] += float((ranks <= cutoff).sum().item())
        return {key: value / len(validation) for key, value in totals.items()}

    best_state, best_metrics, patience = None, None, 0
    for epoch in range(1, args.epochs + 1):
        train_epoch(train)
        metrics = evaluate_validation()
        print(json.dumps({"phase": "karrierewege_finetune", "epoch": epoch, "validation": metrics}), flush=True)
        if best_metrics is None or (
            metrics["mrr"], metrics["hits_at_5"], metrics["hits_at_10"]
        ) > (
            best_metrics["mrr"], best_metrics["hits_at_5"], best_metrics["hits_at_10"]
        ):
            best_metrics, patience = metrics, 0
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        else:
            patience += 1
            if patience >= 2:
                break
    model.load_state_dict(best_state)
    metadata = {
        "promoted": False,
        "model_type": args.model,
        "hidden_size": hidden,
        "temperature": float(torch.clamp(torch.exp(model.log_temperature), 1e-3, 1.0).item()),
        "fusion_weight_single": 0.0,
        "fusion_weight_multi": 0.0,
        "promotion_run": None,
        "selection_metric": {"validation": best_metrics},
        "dataset": "karrierewege_train_validation_only",
        "graph_source": graph_source,
        "jobhop_pretrained": jobhop_pretrained,
        "seed": args.seed,
        "maximum_history_length": args.maximum_history_length,
    }
    if args.model in {"causal_transformer", "causal_transformer_id_residual"}:
        metadata["model_config"] = model.configuration()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        metadata=np.asarray(json.dumps(metadata, sort_keys=True)),
        candidate_ids=np.asarray(candidate_ids),
        candidate_embeddings=candidate_embeddings,
        **{
            name: value.detach().cpu().numpy()
            for name, value in model.export().items()
        },
    )
    print(json.dumps({"output": str(args.output), "validation": best_metrics, "promoted": False}, indent=2))


if __name__ == "__main__":
    main()
