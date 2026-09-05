from __future__ import annotations

import unittest

from src.transition_walk import (
    blend_walk_components,
    transition_walk_components,
)


class TransitionWalkTests(unittest.TestCase):
    def test_walk_propagates_probability_and_excludes_source_cycles(self) -> None:
        transitions = {
            "a": {"b": 0.75, "c": 0.25},
            "b": {"a": 0.20, "d": 0.80},
            "c": {"d": 1.0},
        }

        one, two, three = transition_walk_components("a", transitions, max_steps=3)

        self.assertEqual(one, {"b": 0.75, "c": 0.25})
        self.assertNotIn("a", two)
        self.assertAlmostEqual(two["d"], 0.85)
        self.assertEqual(three, {})

    def test_blend_validates_weights_and_combines_components(self) -> None:
        blended = blend_walk_components(
            ({"b": 0.5}, {"b": 0.2, "c": 0.8}),
            (0.8, 0.2),
        )
        self.assertAlmostEqual(blended["b"], 0.44)
        self.assertAlmostEqual(blended["c"], 0.16)
        with self.assertRaises(ValueError):
            blend_walk_components(({},), (0.0,))


if __name__ == "__main__":
    unittest.main()
