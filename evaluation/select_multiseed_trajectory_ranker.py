"""Select one sequential artifact using validation metrics from a fixed seed set.

The command deliberately never reads Karrierewege validation/test data.  It
only inspects validation metrics embedded by the offline trainer in each
unpromoted artifact, records checksums, and deterministically selects by MRR,
Hits@5, then Hits@10.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.karrierewege_preprocessing import write_json_atomic


METRICS = ("mrr", "hits_at_1", "hits_at_3", "hits_at_5", "hits_at_10")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def inspect_artifact(path: Path) -> dict:
    with np.load(path, allow_pickle=False) as archive:
        metadata = json.loads(str(archive["metadata"].item()))
    if metadata.get("promoted") is not False:
        raise ValueError(f"{path} must be an unpromoted offline artifact")
    if metadata.get("model_type") != "causal_transformer_id_residual":
        raise ValueError(f"{path} is not an ID-residual causal Transformer")
    if metadata.get("dataset") != "karrierewege_train_validation_only":
        raise ValueError(f"{path} has an invalid training-evidence declaration")
    metrics = metadata.get("selection_metric", {}).get("validation")
    if not isinstance(metrics, dict) or any(name not in metrics for name in METRICS):
        raise ValueError(f"{path} lacks complete validation metrics")
    parsed = {name: float(metrics[name]) for name in METRICS}
    if not all(np.isfinite(value) and 0.0 <= value <= 1.0 for value in parsed.values()):
        raise ValueError(f"{path} has invalid validation metrics")
    seed = metadata.get("seed")
    if not isinstance(seed, int):
        raise ValueError(f"{path} lacks an integer seed")
    return {
        "artifact": str(path), "artifact_sha256": sha256(path), "seed": seed,
        "metrics": parsed, "model_config": metadata.get("model_config"),
        "jobhop_pretrained": metadata.get("jobhop_pretrained") is True,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts", type=Path, nargs="+", required=True)
    parser.add_argument("--expected-seeds", type=int, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if len(args.artifacts) != len(args.expected_seeds):
        parser.error("--artifacts and --expected-seeds must have the same length")
    if len(set(args.expected_seeds)) != len(args.expected_seeds):
        parser.error("--expected-seeds must be unique")
    runs = [inspect_artifact(path) for path in args.artifacts]
    actual = {run["seed"] for run in runs}
    expected = set(args.expected_seeds)
    if actual != expected:
        raise ValueError(f"Artifact seeds {sorted(actual)} do not match fixed seed set {sorted(expected)}")
    if not all(run["jobhop_pretrained"] for run in runs):
        raise ValueError("Every artifact must use the same declared JobHop pretraining regime")
    winner = max(runs, key=lambda run: tuple(run["metrics"][name] for name in ("mrr", "hits_at_5", "hits_at_10")))
    spread = {
        metric: {
            "mean": float(np.mean([run["metrics"][metric] for run in runs])),
            "std": float(np.std([run["metrics"][metric] for run in runs], ddof=1)),
            "minimum": float(min(run["metrics"][metric] for run in runs)),
            "maximum": float(max(run["metrics"][metric] for run in runs)),
        }
        for metric in METRICS
    }
    payload = {
        "selection_split": "validation_only_artifact_metadata",
        "fixed_seed_set": sorted(expected),
        "selection_order": ["mrr", "hits_at_5", "hits_at_10"],
        "runs": sorted(runs, key=lambda run: run["seed"]),
        "selected_seed": winner["seed"],
        "selected_artifact": winner["artifact"],
        "selected_artifact_sha256": winner["artifact_sha256"],
        "validation_spread": spread,
        "test_data_read": False,
        "promotion": "not granted by selection; run the checksum-bound frozen test and all gates separately",
    }
    write_json_atomic(payload, args.output)
    print(json.dumps({
        "output": str(args.output), "selected_seed": winner["seed"],
        "selected_metrics": winner["metrics"], "validation_spread": spread,
    }, indent=2))


if __name__ == "__main__":
    main()
