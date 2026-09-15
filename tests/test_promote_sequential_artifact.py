from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from evaluation.promote_sequential_artifact import promote, sha256
from evaluation.evaluate_frozen_trajectory_fusion import load_frozen_weights
from evaluation.freeze_trajectory_fusion import sha256 as fusion_sha256
from src.sequential_ranking import SequentialRankerRuntime


class PromotionArtifactTests(unittest.TestCase):
    def test_fusion_checksum_helper_matches_promotion_helper(self) -> None:
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / "payload.bin"
            path.write_bytes(b"checksum")
            self.assertEqual(fusion_sha256(path), sha256(path))

    def _source(self, path: Path) -> None:
        np.savez_compressed(
            path,
            metadata=np.asarray(json.dumps({"promoted": False, "model_type": "recency_mlp", "hidden_size": 2})),
            candidate_ids=np.asarray(["a", "b", "c"]),
            candidate_embeddings=np.asarray([[1.0, 0.0], [0.8, 0.2], [0.0, 1.0]]),
            w1=np.eye(2), b1=np.zeros(2), w_out=np.eye(2), b_out=np.zeros(2),
        )

    def _report(self, path: Path) -> None:
        path.write_text(json.dumps({
            "candidate_roles": 3039, "coverage": 1.0, "mrr": 0.4, "hits_at_5": 0.5, "hits_at_10": 0.6,
            "protocol": {"candidate_set": "all_artifact_live_esco_roles", "current_role": "excluded"},
        }), encoding="utf-8")

    def _manifest(self, source: Path, report: Path, *, gates: dict | None = None) -> dict:
        return {
            "promotion_run": "test-run",
            "source_artifact_sha256": sha256(source),
            "frozen_test_report_sha256": sha256(report),
            "candidate_roles": 3039,
            "baseline": {"mrr": 0.3, "hits_at_5": 0.4, "hits_at_10": 0.5},
            "gates": gates or {
                "multi_seed_stable": True, "paired_bootstrap_improvement": True, "fusion_policy_frozen": True,
                "api_integration_passed": True, "test_blind_for_selected_artifact": True,
                "latency_ms_p95": 10.0, "latency_limit_ms": 20.0,
                "memory_mb": 100.0, "memory_limit_mb": 200.0,
            },
        }

    def test_promotion_creates_distinct_loadable_artifact(self) -> None:
        with tempfile.TemporaryDirectory() as root:
            root = Path(root)
            source, report, manifest, output = root / "source.npz", root / "test.json", root / "manifest.json", root / "promoted.npz"
            self._source(source); self._report(report)
            manifest.write_text(json.dumps(self._manifest(source, report)), encoding="utf-8")
            promote(source, report, manifest, output)
            self.assertTrue(output.is_file())
            self.assertFalse(json.loads(str(np.load(source, allow_pickle=False)["metadata"].item()))["promoted"])
            self.assertTrue(SequentialRankerRuntime(output).metadata["promoted"])

    def test_promotion_fails_closed_for_missing_gate(self) -> None:
        with tempfile.TemporaryDirectory() as root:
            root = Path(root)
            source, report, manifest, output = root / "source.npz", root / "test.json", root / "manifest.json", root / "promoted.npz"
            self._source(source); self._report(report)
            gates = self._manifest(source, report)["gates"]
            gates["multi_seed_stable"] = False
            manifest.write_text(json.dumps(self._manifest(source, report, gates=gates)), encoding="utf-8")
            with self.assertRaises(ValueError):
                promote(source, report, manifest, output)
            self.assertFalse(output.exists())

    def test_promotion_accepts_nested_frozen_fusion_metrics(self) -> None:
        with tempfile.TemporaryDirectory() as root:
            root = Path(root)
            source, report, manifest, output = root / "source.npz", root / "test.json", root / "manifest.json", root / "promoted.npz"
            self._source(source)
            self._report(report)
            payload = json.loads(report.read_text(encoding="utf-8"))
            report.write_text(json.dumps({
                "candidate_roles": payload["candidate_roles"], "coverage": payload["coverage"],
                "protocol": payload["protocol"], "metrics": {key: payload[key] for key in ("mrr", "hits_at_5", "hits_at_10")},
            }), encoding="utf-8")
            manifest.write_text(json.dumps(self._manifest(source, report)), encoding="utf-8")
            promote(source, report, manifest, output)
            self.assertTrue(output.is_file())

    def test_frozen_fusion_manifest_rejects_non_simplex_weights(self) -> None:
        with tempfile.TemporaryDirectory() as root:
            root = Path(root)
            artifact, manifest = root / "artifact.npz", root / "fusion.json"
            artifact.write_bytes(b"artifact")
            manifest.write_text(json.dumps({
                "artifact_sha256": sha256(artifact), "selection_split": "validation",
                "selected_weights": {
                    "single": {"neural": 0.5, "smoother": 0.5, "second_order": 0.0},
                    "multi": {"neural": 0.8, "smoother": 0.8, "second_order": 0.0},
                },
            }), encoding="utf-8")
            with self.assertRaises(ValueError):
                load_frozen_weights(manifest, artifact)


if __name__ == "__main__":
    unittest.main()
