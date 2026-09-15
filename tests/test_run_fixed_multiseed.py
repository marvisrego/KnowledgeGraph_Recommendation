from __future__ import annotations

import unittest

from evaluation import run_fixed_multiseed


class FixedMultiSeedRunnerTests(unittest.TestCase):
    def test_module_exposes_entrypoint(self) -> None:
        self.assertTrue(callable(run_fixed_multiseed.main))


if __name__ == "__main__":
    unittest.main()
