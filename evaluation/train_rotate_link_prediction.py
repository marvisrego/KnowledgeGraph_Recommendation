"""Research-only directed RotatE benchmark for train-only career transitions.

It reports full-candidate, unfiltered tail ranks. The experiment never writes
predicted edges to the graph and does not produce a deployable artifact.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from config import Settings
from evaluation.research_resources import load_research_graph
from src.karrierewege_preprocessing import aggregate_split_transitions, build_esco_title_index, discover_split_files, write_json_atomic
from src.transition_embedding import transition_distributions


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a research-only RotatE career-transition benchmark")
    parser.add_argument("--output", type=Path, default=Path("artifacts/link_prediction/rotate_evaluation.json"))
    parser.add_argument("--dimension", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--seed", type=int, default=17)
    args = parser.parse_args()
    try:
        import torch
        from torch.nn import functional as F
    except ImportError as exc:
        raise SystemExit("PyTorch is required offline; install requirements-training.txt") from exc

    settings = Settings.from_env()
    graph, graph_source = load_research_graph(settings)
    title_index = build_esco_title_index(graph)
    split_files = discover_split_files(settings.karrierewege_dir)
    roles = sorted(str(node_id) for node_id, data in graph.nodes(data=True) if data.get("type") == "role" and data.get("source") == "esco")
    row = {role_id: index for index, role_id in enumerate(roles)}
    role_groups = {
        row[str(role_id)]: str(data.get("isco_group") or data.get("isco_2digit") or "")
        for role_id, data in graph.nodes(data=True)
        if str(role_id) in row
    }
    by_group: dict[str, list[int]] = {}
    for role_index, group in role_groups.items():
        if group:
            by_group.setdefault(group[:2], []).append(role_index)
    distributions = transition_distributions(graph)
    triples = [(row[source], row[target]) for source, values in distributions.items() for target in values if source in row and target in row]
    if not triples:
        raise RuntimeError("No train-only transition triples were available")
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    entity = torch.nn.Parameter(torch.empty((len(roles), args.dimension * 2), device=device))
    relation = torch.nn.Parameter(torch.empty(args.dimension, device=device))
    torch.nn.init.uniform_(entity, -0.01, 0.01); torch.nn.init.uniform_(relation, -np.pi, np.pi)
    optimizer = torch.optim.Adam([entity, relation], lr=5e-4)
    rng = np.random.default_rng(args.seed)

    def score(head, tail):
        head_real, head_imag = head[..., :args.dimension], head[..., args.dimension:]
        tail_real, tail_imag = tail[..., :args.dimension], tail[..., args.dimension:]
        cosine, sine = torch.cos(relation), torch.sin(relation)
        rotated_real = head_real * cosine - head_imag * sine
        rotated_imag = head_real * sine + head_imag * cosine
        return -torch.sqrt(((rotated_real - tail_real) ** 2 + (rotated_imag - tail_imag) ** 2).sum(-1) + 1e-9)

    triples_np = np.asarray(triples, dtype=np.int64)
    for _ in range(args.epochs):
        for start in range(0, len(triples_np), 512):
            batch = triples_np[rng.permutation(len(triples_np))[start:start + 512]]
            if not len(batch):
                continue
            heads = torch.tensor(batch[:, 0], device=device)
            tails = torch.tensor(batch[:, 1], device=device)
            positive = score(entity[heads], entity[tails])
            # Self-adversarial negative sampling: high-scoring corrupt tails
            # receive greater loss weight. Half the pool is same-ISCO where
            # possible, making negatives structurally harder than uniform only.
            negative_rows = rng.integers(0, len(roles), size=(len(batch), 16))
            for index, (source_row, _) in enumerate(batch):
                hard_pool = by_group.get(role_groups.get(int(source_row), "")[:2], [])
                if hard_pool:
                    negative_rows[index, :8] = rng.choice(hard_pool, size=8, replace=len(hard_pool) < 8)
            negatives = torch.tensor(negative_rows, device=device)
            negative_scores = score(entity[heads, None], entity[negatives])
            adversarial = torch.softmax(negative_scores.detach(), dim=1)
            loss = -F.logsigmoid(positive - 6.0).mean() - (adversarial * F.logsigmoid(-negative_scores - 6.0)).sum(1).mean()
            optimizer.zero_grad(); loss.backward(); optimizer.step()

    def evaluate(split):
        held_out = aggregate_split_transitions(split_files[split], split, title_index, settings.transition_chunk_size, True, settings.karrierewege_max_invalid_row_ratio)
        ranks = []
        with torch.no_grad():
            for (source_title, target_title), count in held_out.pair_counts.items():
                if source_title not in title_index or target_title not in title_index:
                    continue
                source_id, target_id = title_index[source_title], title_index[target_title]
                if source_id not in row or target_id not in row:
                    continue
                scores = score(entity[row[source_id]].unsqueeze(0), entity).detach().cpu().numpy()
                scores[row[source_id]] = -np.inf
                rank = int(np.sum(scores > scores[row[target_id]]) + 1)
                ranks.extend([rank] * int(count))
        values = np.asarray(ranks, dtype=np.float64)
        return {
            "observations": int(len(values)),
            "mrr": float(np.mean(1.0 / values)) if len(values) else 0.0,
            **{f"hits_at_{k}": float(np.mean(values <= k)) if len(values) else 0.0 for k in (1, 3, 5, 10)},
            "protocol": "unfiltered_full_live_esco_tail_ranking",
        }

    payload = {
        "model": "RotatE",
        "dimension": args.dimension,
        "epochs": args.epochs,
        "seed": args.seed,
        "deployable": False,
        "train_triples": len(triples),
        "candidate_roles": len(roles),
        "graph_source": graph_source,
        "validation": evaluate("validation"),
        "test": evaluate("test"),
    }
    write_json_atomic(payload, args.output)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
