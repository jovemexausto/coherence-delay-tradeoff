from __future__ import annotations

import unittest

import numpy as np

from scale_consistency.estimation import (
    feasible_wls,
    oracle_precision_weights,
    oracle_wls,
    pilot_ols,
    residual_statistic,
)
from scale_consistency.model import log_scale_signal, simulate_observed_discrepancies


class ScaleConsistencyEstimationTest(unittest.TestCase):
    def test_pilot_ols_recovers_exact_signal_without_noise(self) -> None:
        lags = np.arange(1, 8, dtype=float)
        y = log_scale_signal(lags, zeta=1.7, H=0.45)
        estimate = pilot_ols(y, lags)
        self.assertAlmostEqual(estimate.alpha, np.log(1.7), places=12)
        self.assertAlmostEqual(estimate.H, 0.45, places=12)
        self.assertAlmostEqual(
            residual_statistic(estimate.residuals, estimate.weights), 0.0, places=12
        )

    def test_oracle_weights_increase_with_lag(self) -> None:
        weights = oracle_precision_weights(
            np.arange(1, 8), zeta=1.0, H=0.6, sigma0=1.0, n=500
        )
        self.assertTrue(np.all(np.diff(weights) > 0.0))

    def test_feasible_and_oracle_wls_are_close_for_large_n(self) -> None:
        rng = np.random.default_rng(7)
        lags = np.arange(1, 21, dtype=float)
        obs = simulate_observed_discrepancies(lags, 1.0, 0.6, 1.0, 20000, rng=rng)
        y = np.log(obs)
        fwls = feasible_wls(y, lags, sigma0=1.0, n=20000)
        oracle = oracle_wls(y, lags, zeta=1.0, H=0.6, sigma0=1.0, n=20000)
        self.assertLess(abs(fwls.H - oracle.H), 0.02)


if __name__ == "__main__":
    unittest.main()
