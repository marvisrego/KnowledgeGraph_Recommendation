from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from evaluation.evaluate_sequential_ranker import _bootstrap
from src.trajectory_benchmark import evaluate_prefix_rankings, iter_karrierewege_prefixes


class TrajectoryBenchmarkTests(unittest.TestCase):
    def test_prefixes_preserve_order_and_restart_after_order_gap(self) -> None:
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / "karrierewege_test.csv"
            path.write_text(
                "_id,experience_order,preferredLabel_en\n"
                "p1,1,Role A\n"
                "p1,2,Role B\n"
                "p1,3,Role C\n"
                "p2,1,Role A\n"
                "p2,3,Role C\n",
                encoding="utf-8",
            )
            examples = list(iter_karrierewege_prefixes(path, {"role a": "a", "role b": "b", "role c": "c"}, "test", 2))
        self.assertEqual([(item.person_id, item.history_role_ids, item.target_role_id) for item in examples], [
            ("p1", ("a",), "b"),
            ("p1", ("a", "b"), "c"),
        ])

    def test_full_ranking_metrics_and_person_bootstrap(self) -> None:
        examples = [
            type("E", (), {"person_id": "p1", "history_role_ids": ("a",), "target_role_id": "b"})(),
            type("E", (), {"person_id": "p2", "history_role_ids": ("a",), "target_role_id": "c"})(),
        ]
        metrics = evaluate_prefix_rankings(examples, lambda _: ["b", "c", "a"], bootstrap_samples=10)
        self.assertEqual(metrics["examples"], 2)
        self.assertAlmostEqual(metrics["mrr"], 0.75)
        self.assertEqual(metrics["hits_at_1"], 0.5)
        self.assertIn("hits_at_10", metrics["confidence_intervals_95"])

    def test_cluster_bootstrap_preserves_event_weighted_estimand(self) -> None:
        records = {
            "p1": [{"mrr": 1.0}] * 3,
            "p2": [{"mrr": 0.0}],
        }
        seed = 17
        draws = __import__("numpy").random.default_rng(seed).integers(0, 2, size=(1, 2))
        people = ("p1", "p2")
        expected = sum(sum(item["mrr"] for item in records[people[index]]) for index in draws[0])
        expected /= sum(len(records[people[index]]) for index in draws[0])
        interval = _bootstrap(records, samples=1, seed=seed)["mrr"]
        self.assertEqual(interval, [expected, expected])


if __name__ == "__main__":
    unittest.main()
