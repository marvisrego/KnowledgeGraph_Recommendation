"""Run a fixed set of offline trajectory-training seeds sequentially.

Each subprocess is isolated and writes its own log. Existing artifacts are
never overwritten unless ``--overwrite`` is supplied, so a machine restart can
resume only the missing seeds without changing completed evidence.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, nargs="+", default=[17, 29, 41, 53, 71])
    parser.add_argument("--embedding-artifact", type=Path, default=Path("artifacts/sequential_ranking/role_embeddings.npz"))
    parser.add_argument("--jobhop-train", type=Path, default=Path("Data/JobHop_v2/processed/train.parquet"))
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/sequential_ranking"))
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if len(set(args.seeds)) != len(args.seeds):
        parser.error("--seeds must be unique")
    if not args.embedding_artifact.is_file() or not args.jobhop_train.is_file():
        parser.error("Required embedding or JobHop training artifact is missing")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    trainer = Path(__file__).with_name("train_sequential_ranker.py")
    for seed in args.seeds:
        output = args.output_dir / f"causal_transformer_id_residual_jobhop_seed{seed}.npz"
        log = args.output_dir / f"seed{seed}_training.log"
        if output.is_file() and not args.overwrite:
            print(f"seed {seed}: retaining existing {output}", flush=True)
            continue
        command = [
            sys.executable, str(trainer), "--model", "causal_transformer_id_residual",
            "--embedding-artifact", str(args.embedding_artifact), "--jobhop-train", str(args.jobhop_train),
            "--jobhop-pretrain-epochs", "3", "--epochs", "10", "--batch-size", "1024",
            "--seed", str(seed), "--output", str(output),
        ]
        print(f"seed {seed}: starting", flush=True)
        with log.open("w", encoding="utf-8") as handle:
            completed = subprocess.run(command, stdout=handle, stderr=subprocess.STDOUT, check=False)
        if completed.returncode:
            raise SystemExit(f"seed {seed} failed with exit code {completed.returncode}; see {log}")
        if not output.is_file():
            raise SystemExit(f"seed {seed} completed without artifact {output}")
        print(f"seed {seed}: completed", flush=True)


if __name__ == "__main__":
    main()
