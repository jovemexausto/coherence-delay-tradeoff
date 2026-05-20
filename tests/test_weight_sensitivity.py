from __future__ import annotations

import math
import unittest

from useful_memory_horizon.weight_sensitivity import (
    lag_weight_w1,
    lag_weights,
    run_weight_sensitivity_experiment,
)


class WeightSensitivityTest(unittest.TestCase):
    def test_uniform_weights_have_zero_w1(self) -> None:
        self.assertAlmostEqual(lag_weight_w1(lag_weights(8, "uniform")), 0.0, places=12)

    def test_tapered_weights_have_positive_w1(self) -> None:
        self.assertGreater(lag_weight_w1(lag_weights(8, "triangular")), 0.0)
        self.assertGreater(lag_weight_w1(lag_weights(8, "geometric")), 0.0)

    def test_experiment_returns_three_schemes(self) -> None:
        result = run_weight_sensitivity_experiment()
        self.assertEqual(len(result.rows), 3)
        self.assertTrue(
            all(math.isfinite(row.mean_absolute_error) for row in result.rows)
        )
        self.assertTrue(all(row.best_window > 0 for row in result.rows))


if __name__ == "__main__":
    unittest.main()
