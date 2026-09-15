from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from src.sequential_ranking import SequentialRankerRuntime


class SequentialRankingTests(unittest.TestCase):
    def _write_artifact(self, path: Path, promoted: bool = True) -> None:
        np.savez(
            path,
            metadata=np.asarray(json.dumps({
                "promoted": promoted,
                "model_type": "recency_mlp",
                "hidden_size": 2,
                "fusion_weight_single": 0.0,
                "fusion_weight_multi": 0.5,
            })),
            candidate_ids=np.asarray(["a", "b", "c"]),
            candidate_embeddings=np.asarray([[1.0, 0.0], [0.8, 0.2], [0.0, 1.0]]),
            w1=np.eye(2),
            b1=np.zeros(2),
            w_out=np.eye(2),
            b_out=np.zeros(2),
        )

    def test_rank_excludes_current_role_and_reports_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / "ranker.npz"
            self._write_artifact(path)
            runtime = SequentialRankerRuntime(path)
            ranked = runtime.rank(["a", "b"], 3)
            self.assertNotIn("b", [item.role_id for item in ranked])
            self.assertEqual(ranked[0].history_length, 2)
            self.assertEqual(runtime.fusion_weight(2), 0.5)

    def test_unpromoted_artifact_is_refused(self) -> None:
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / "ranker.npz"
            self._write_artifact(path, promoted=False)
            with self.assertRaises(ValueError):
                SequentialRankerRuntime(path)

    def test_causal_runtime_matches_pytorch_id_residual_scores(self) -> None:
        import torch

        from evaluation.causal_transformer import CausalTransformerRanker

        torch.manual_seed(23)
        candidates = torch.nn.functional.normalize(torch.randn(6, 8), dim=1)
        model = CausalTransformerRanker(
            candidates,
            maximum_history_length=4,
            hidden_size=8,
            attention_heads=2,
            layers=1,
            dropout=0.0,
            id_residual=True,
        ).eval()
        with torch.no_grad():
            model.input_residual.uniform_(-0.1, 0.1)
            model.output_residual.uniform_(-0.1, 0.1)
            model.output_bias.uniform_(-0.1, 0.1)
        ids = [f"role:{index}" for index in range(6)]
        metadata = {
            "promoted": True,
            "model_type": "causal_transformer_id_residual",
            "model_config": model.configuration(),
            "maximum_history_length": 4,
            "fusion_weight_single": 0.0,
            "fusion_weight_multi": 0.0,
        }
        exported = {name: value.detach().cpu().numpy() for name, value in model.export().items()}
        history = [0, 2, 4]
        with torch.no_grad():
            expected = model([history])[0].cpu().numpy()
        expected_ranked = sorted(
            ((ids[index], float(expected[index])) for index in range(len(ids)) if index != history[-1]),
            key=lambda item: (-item[1], item[0]),
        )
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / "causal.npz"
            np.savez_compressed(
                path,
                metadata=np.asarray(json.dumps(metadata)),
                candidate_ids=np.asarray(ids),
                candidate_embeddings=candidates.cpu().numpy(),
                **exported,
            )
            runtime = SequentialRankerRuntime(path)
            actual = runtime.rank([ids[index] for index in history], len(ids))
        self.assertEqual([item.role_id for item in actual], [item[0] for item in expected_ranked])
        for prediction, (_, score) in zip(actual, expected_ranked):
            self.assertAlmostEqual(prediction.score, score, places=4)

    def test_causal_runtime_refuses_context_feature_artifact(self) -> None:
        import torch

        from evaluation.causal_transformer import CausalTransformerRanker

        model = CausalTransformerRanker(
            torch.nn.functional.normalize(torch.randn(3, 4), dim=1),
            maximum_history_length=2,
            hidden_size=4,
            attention_heads=2,
            layers=1,
            dropout=0.0,
            context_feature_sizes=(9, 8, 6),
        ).eval()
        metadata = {"promoted": True, "model_type": "causal_transformer", "model_config": model.configuration()}
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / "context.npz"
            np.savez_compressed(
                path,
                metadata=np.asarray(json.dumps(metadata)),
                candidate_ids=np.asarray(["a", "b", "c"]),
                candidate_embeddings=model.candidate.cpu().numpy(),
                **{name: value.detach().cpu().numpy() for name, value in model.export().items()},
            )
            with self.assertRaises(ValueError):
                SequentialRankerRuntime(path)


if __name__ == "__main__":
    unittest.main()
