from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from src.tes_calibration import (
    TESCalibration,
    fit_nonnegative_pairwise_weights,
    load_calibration,
    thresholds_from_scores,
    write_calibration,
)


class TESCalibrationTests(unittest.TestCase):
    def test_artifact_round_trip_and_invalid_fallback(self) -> None:
        calibration = TESCalibration(
            weights={"skill_gap": 0.4, "domain": 0.2, "empirical": 0.2, "transferability": 0.2},
            low_max=0.25,
            moderate_max=0.65,
            training_hash="abc",
        )
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / "tes.json"
            write_calibration(calibration, path)
            loaded = load_calibration(path)
            self.assertEqual(loaded, calibration)
            path.write_text("not json", encoding="utf-8")
            self.assertEqual(load_calibration(path).method, "heuristic_fallback")

    def test_pairwise_fit_prefers_the_discriminating_component(self) -> None:
        # Observed transitions have lower skill-gap effort than alternatives.
        weights = fit_nonnegative_pairwise_weights(
            [[0.1, 0.5, 0.5, 0.5]] * 20,
            [[0.9, 0.5, 0.5, 0.5]] * 20,
            iterations=200,
        )
        self.assertGreater(weights["skill_gap"], weights["domain"])
        self.assertAlmostEqual(sum(weights.values()), 1.0)

    def test_thresholds_are_ordered(self) -> None:
        low, moderate = thresholds_from_scores([0.1, 0.2, 0.7, 0.8])
        self.assertLess(low, moderate)


if __name__ == "__main__":
    unittest.main()
