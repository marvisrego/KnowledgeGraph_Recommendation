from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from evaluation.select_multiseed_trajectory_ranker import inspect_artifact


class MultiSeedSelectionTests(unittest.TestCase):
    def test_inspect_rejects_non_validation_artifact(self) -> None:
        with tempfile.TemporaryDirectory() as root:
            artifact = Path(root) / "artifact.npz"
            np.savez_compressed(artifact, metadata=np.asarray(json.dumps({
                "promoted": False, "model_type": "causal_transformer_id_residual",
                "dataset": "other", "seed": 17,
                "selection_metric": {"validation": {name: 0.2 for name in (
                    "mrr", "hits_at_1", "hits_at_3", "hits_at_5", "hits_at_10",
                )}},
            })))
            with self.assertRaises(ValueError):
                inspect_artifact(artifact)


if __name__ == "__main__":
    unittest.main()
