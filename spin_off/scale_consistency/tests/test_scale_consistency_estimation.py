from __future__ import annotations

import unittest

import numpy as np

from scale_consistency.estimation import (
    feasible_wls,
    estimate_sigma0_squared_from_pilot,
    oracle_precision_weights,
    oracle_wls,
    pilot_ols,
    residual_statistic,
    run_split_scale_consistency_test,
    run_scale_consistency_test,
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

    def test_sigma0_plugin_estimator_and_test_df(self) -> None:
        rng = np.random.default_rng(11)
        lags = np.arange(1, 12, dtype=float)
        obs = simulate_observed_discrepancies(lags, 1.0, 0.6, 1.0, 5000, rng=rng)
        y = np.log(obs)
        pilot = pilot_ols(y, lags)
        sigma0_sq_hat = estimate_sigma0_squared_from_pilot(pilot, 5000)
        self.assertGreater(sigma0_sq_hat, 0.0)
        result = run_scale_consistency_test(obs, lags, None, 5000, alpha_level=0.05)
        self.assertEqual(result.degrees_of_freedom, len(lags) - 3)
        self.assertEqual(result.calibration, "chi2")
        self.assertIsNotNone(result.estimate.sigma0_hat)
        self.assertGreater(result.estimate.sigma0_hat, 0.0)

    def test_sigma0_bootstrap_calibration_runs(self) -> None:
        rng = np.random.default_rng(12)
        lags = np.arange(1, 12, dtype=float)
        obs = simulate_observed_discrepancies(lags, 1.0, 0.6, 1.0, 500, rng=rng)
        result = run_scale_consistency_test(
            obs,
            lags,
            None,
            500,
            alpha_level=0.05,
            calibration="bootstrap",
            bootstrap_repetitions=50,
            rng=rng,
        )
        self.assertEqual(result.calibration, "bootstrap")
        self.assertGreater(result.critical_value, 0.0)

    def test_split_scale_consistency_test_runs(self) -> None:
        rng = np.random.default_rng(13)
        lags = np.arange(1, 12, dtype=float)
        scale_obs = simulate_observed_discrepancies(lags, 1.0, 0.6, 1.0, 300, rng=rng)
        test_obs = simulate_observed_discrepancies(lags, 1.0, 0.6, 1.0, 300, rng=rng)
        result = run_split_scale_consistency_test(
            scale_obs,
            test_obs,
            lags,
            300,
            300,
            alpha_level=0.05,
        )
        self.assertEqual(result.calibration, "f")
        self.assertEqual(result.numerator_degrees_of_freedom, len(lags) - 2)
        self.assertEqual(result.denominator_degrees_of_freedom, len(lags) - 2)
        self.assertGreater(result.scale_estimate.sigma0_hat, 0.0)


if __name__ == "__main__":
    unittest.main()
