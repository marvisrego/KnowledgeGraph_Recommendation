from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path


class CausalTransformerTests(unittest.TestCase):
    def test_output_is_full_candidate_and_excludes_current_role(self) -> None:
        import torch

        from evaluation.causal_transformer import CausalTransformerRanker

        model = CausalTransformerRanker(
            torch.nn.functional.normalize(torch.randn(5, 8), dim=1),
            maximum_history_length=3,
            hidden_size=8,
            attention_heads=2,
            layers=1,
            dropout=0.0,
        ).eval()
        with torch.no_grad():
            scores = model([[0, 3], [1]])
        self.assertEqual(tuple(scores.shape), (2, 5))
        self.assertTrue(torch.isneginf(scores[0, 3]))
        self.assertTrue(torch.isneginf(scores[1, 1]))

    def test_id_residuals_preserve_full_candidate_contract_and_change_scores(self) -> None:
        import torch

        from evaluation.causal_transformer import CausalTransformerRanker

        torch.manual_seed(17)
        candidates = torch.nn.functional.normalize(torch.randn(5, 8), dim=1)
        model = CausalTransformerRanker(
            candidates,
            maximum_history_length=3,
            hidden_size=8,
            attention_heads=2,
            layers=1,
            dropout=0.0,
            id_residual=True,
        ).eval()
        with torch.no_grad():
            baseline = model([[0, 3]])
            model.output_bias[4] = 3.0
            residual_scores = model([[0, 3]])
        self.assertEqual(tuple(residual_scores.shape), (1, 5))
        self.assertTrue(torch.isneginf(residual_scores[0, 3]))
        self.assertFalse(torch.allclose(baseline, residual_scores))
        self.assertTrue(model.configuration()["id_residual"])

    def test_id_residual_artifact_round_trip_requires_all_residual_parameters(self) -> None:
        import numpy as np
        import torch

        from evaluation.causal_transformer import CausalTransformerRanker
        from evaluation.evaluate_sequential_ranker import _load_artifact

        candidates = torch.nn.functional.normalize(torch.randn(5, 8), dim=1)
        model = CausalTransformerRanker(
            candidates,
            maximum_history_length=3,
            hidden_size=8,
            attention_heads=2,
            layers=1,
            dropout=0.0,
            id_residual=True,
        ).eval()
        metadata = {
            "model_type": "causal_transformer_id_residual",
            "hidden_size": 8,
            "maximum_history_length": 3,
            "temperature": 0.07,
            "model_config": model.configuration(),
        }
        exported = {name: value.detach().cpu().numpy() for name, value in model.export().items()}
        with tempfile.TemporaryDirectory() as directory:
            artifact = Path(directory) / "ranker.npz"
            np.savez_compressed(
                artifact,
                metadata=np.asarray(json.dumps(metadata)),
                candidate_ids=np.asarray(["r0", "r1", "r2", "r3", "r4"]),
                candidate_embeddings=candidates.numpy(),
                **exported,
            )
            loaded_metadata, _, score, _ = _load_artifact(artifact)
            self.assertEqual(loaded_metadata["model_type"], metadata["model_type"])
            self.assertEqual(tuple(score([[0, 3]]).shape), (1, 5))

            exported.pop("output_bias")
            malformed = Path(directory) / "malformed.npz"
            np.savez_compressed(
                malformed,
                metadata=np.asarray(json.dumps(metadata)),
                candidate_ids=np.asarray(["r0", "r1", "r2", "r3", "r4"]),
                candidate_embeddings=candidates.numpy(),
                **exported,
            )
            with self.assertRaises(ValueError):
                _load_artifact(malformed)

    def test_context_features_change_logits_and_preserve_current_exclusion(self) -> None:
        import torch

        from evaluation.causal_transformer import CausalTransformerRanker

        torch.manual_seed(19)
        model = CausalTransformerRanker(
            torch.nn.functional.normalize(torch.randn(5, 8), dim=1),
            maximum_history_length=3,
            hidden_size=8,
            attention_heads=2,
            layers=1,
            dropout=0.0,
            context_feature_sizes=(9, 8, 6),
        ).eval()
        with torch.no_grad():
            model.tenure_embeddings.weight[4].fill_(2.0)
            baseline = model([[0, 3]], [(0, 0, 0)])
            featured = model([[0, 3]], [(4, 0, 0)])
        self.assertEqual(tuple(featured.shape), (1, 5))
        self.assertTrue(torch.isneginf(featured[0, 3]))
        self.assertFalse(torch.allclose(baseline, featured))
        with self.assertRaises(ValueError):
            model([[0, 3]], [(9, 0, 0)])


if __name__ == "__main__":
    unittest.main()
