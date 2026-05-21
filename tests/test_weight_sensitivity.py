from __future__ import annotations

import math
import unittest

import numpy as np

from useful_memory_horizon.weight_sensitivity import (
    effective_lag_mean,
    effective_sample_size,
    horizon_proxy,
    lag_weight_w1,
    lag_weights,
    weighted_moving_average,
    run_weight_sensitivity_experiment,
)


class WeightSensitivityTest(unittest.TestCase):
    def test_uniform_weights_have_zero_w1(self) -> None:
        self.assertAlmostEqual(lag_weight_w1(lag_weights(8, "uniform")), 0.0, places=12)

    def test_tapered_weights_have_positive_w1(self) -> None:
        self.assertGreater(lag_weight_w1(lag_weights(8, "triangular")), 0.0)
        self.assertGreater(lag_weight_w1(lag_weights(8, "geometric")), 0.0)

    def test_uniform_weights_maximize_effective_sample_size_on_fixed_window(
        self,
    ) -> None:
        window = 12
        self.assertAlmostEqual(
            effective_sample_size(lag_weights(window, "uniform")), window
        )
        self.assertLess(
            effective_sample_size(lag_weights(window, "triangular")), window
        )
        self.assertLess(effective_sample_size(lag_weights(window, "geometric")), window)

    def test_linear_ramp_bias_equals_effective_lag_mean(self) -> None:
        window = 10
        values = np.arange(50, dtype=float)
        for scheme in ("uniform", "triangular", "geometric"):
            weights = lag_weights(window, scheme)
            bias = values[-1] - weighted_moving_average(
                values, values.size - 1, weights
            )
            self.assertAlmostEqual(bias, effective_lag_mean(weights), places=12)

    def test_horizon_proxy_is_finite_for_non_uniform_weights(self) -> None:
        for scheme in ("uniform", "triangular", "geometric"):
            proxy = horizon_proxy(lag_weights(10, scheme))
            self.assertTrue(math.isfinite(proxy))
            self.assertGreater(proxy, 0.0)

    def test_experiment_returns_three_schemes(self) -> None:
        result = run_weight_sensitivity_experiment()
        self.assertEqual(len(result.rows), 3)
        self.assertTrue(
            all(math.isfinite(row.mean_absolute_error) for row in result.rows)
        )
        self.assertTrue(all(row.best_window > 0 for row in result.rows))
        self.assertTrue(all(row.effective_sample_size > 0.0 for row in result.rows))
        self.assertTrue(all(row.effective_lag_mean >= 0.0 for row in result.rows))


if __name__ == "__main__":
    unittest.main()
