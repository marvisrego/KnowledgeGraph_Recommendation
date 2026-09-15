"""Export read-only live ESCO vectors for offline sequential training."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from config import Settings
from evaluation.research_resources import load_research_embeddings, load_research_graph


def main() -> None:
    parser = argparse.ArgumentParser(description="Export live ESCO embeddings for offline ranker training")
    parser.add_argument("--output", type=Path, default=Path("artifacts/sequential_ranking/role_embeddings.npz"))
    args = parser.parse_args()
    settings = Settings.from_env()
    graph, graph_source = load_research_graph(settings)
    embeddings, report = load_research_embeddings(settings, graph)
    role_ids = sorted(embeddings)
    if not role_ids:
        raise RuntimeError("No live ESCO embeddings were available")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        candidate_ids=np.asarray(role_ids),
        candidate_embeddings=np.stack([embeddings[role_id] for role_id in role_ids]).astype(np.float32),
    )
    print({"output": str(args.output), "graph_source": graph_source, **report})


if __name__ == "__main__":
    main()
