import math
import unittest

import numpy as np

from src.portable_lightgbm import PortableLightGBMBooster


class PortableLightGBMTests(unittest.TestCase):
    def setUp(self):
        self.model = PortableLightGBMBooster(
            {
                "num_class": 1,
                "objective": "binary sigmoid:1",
                "feature_names": ["score"],
                "tree_info": [
                    {
                        "tree_structure": {
                            "split_feature": 0,
                            "threshold": 0.5,
                            "decision_type": "<=",
                            "default_left": True,
                            "missing_type": "None",
                            "left_child": {"leaf_value": -2.0},
                            "right_child": {"leaf_value": 2.0},
                        }
                    }
                ],
            }
        )

    def test_predicts_binary_probabilities(self):
        result = self.model.predict([[0.2], [0.8]])
        self.assertAlmostEqual(result[0], 1.0 / (1.0 + math.exp(2.0)))
        self.assertAlmostEqual(result[1], 1.0 / (1.0 + math.exp(-2.0)))

    def test_missing_value_uses_default_branch(self):
        result = self.model.predict([[np.nan]])
        self.assertLess(result[0], 0.5)

    def test_rejects_wrong_feature_width(self):
        with self.assertRaisesRegex(ValueError, "1 columns"):
            self.model.predict([[0.1, 0.2]])


if __name__ == "__main__":
    unittest.main()
