from __future__ import annotations

import unittest

import numpy as np

from scale_consistency.horizon_bridge import (
    BridgeMisspecificationConfig,
    BridgeRecoveryConfig,
    bootstrap_lag_power_law,
    continuous_optimal_horizon,
    fit_lag_power_law,
    run_bridge_misspecification_experiment,
    run_bridge_recovery_experiment,
)
from scale_consistency.model import (
    exact_scale_profile,
    misspecified_scale_profile,
    simulate_observed_discrepancies,
)


class HorizonBridgeTest(unittest.TestCase):
    def test_fit_lag_power_law_recovers_exact_profile(self) -> None:
        lags = np.arange(1, 9, dtype=float)
        observed = exact_scale_profile(lags, zeta=1.7, H=0.45)
        estimate = fit_lag_power_law(observed, lags, sigma0=1.0, n=1000)

        self.assertAlmostEqual(estimate.zeta, 1.7, places=12)
        self.assertAlmostEqual(estimate.H, 0.45, places=12)
        self.assertAlmostEqual(
            continuous_optimal_horizon(1.0, 0.5, 1.0, estimate.zeta, estimate.H),
            continuous_optimal_horizon(1.0, 0.5, 1.0, 1.7, 0.45),
            places=12,
        )

    def test_bridge_recovery_experiment_returns_rows(self) -> None:
        rows = run_bridge_recovery_experiment(
            BridgeRecoveryConfig(
                lags=20,
                n_values=(2000,),
                H_values=(0.6,),
                zeta_values=(1.0,),
                sigma0_values=(0.4,),
                repetitions=40,
                bootstrap_repetitions=20,
            )
        )

        self.assertEqual(len(rows), 1)
        row = rows[0]
        self.assertLess(abs(row.bias_H), 0.15)
        self.assertLess(abs(row.bias_zeta), 0.25)
        self.assertGreater(row.true_n_star, 0.0)
        self.assertGreater(row.rmse_n_star, 0.0)
        self.assertGreaterEqual(row.coverage_H, 0.0)
        self.assertLessEqual(row.coverage_H, 1.0)
        self.assertGreater(row.mean_interval_width_H, 0.0)
        self.assertGreaterEqual(row.coverage_n_star, 0.0)
        self.assertLessEqual(row.coverage_n_star, 1.0)
        self.assertGreater(row.mean_interval_width_n_star, 0.0)
        self.assertTrue(np.isfinite(row.mean_residual_slope))

    def test_bootstrap_lag_power_law_returns_nontrivial_intervals(self) -> None:
        rng = np.random.default_rng(123)
        lags = np.arange(1, 16, dtype=float)
        observed = exact_scale_profile(lags, zeta=1.2, H=0.55)
        H_interval, n_star_interval = bootstrap_lag_power_law(
            observed,
            lags,
            sigma0=0.5,
            n=500,
            bootstrap_repetitions=20,
            interval_level=0.9,
            C_K=1.0,
            a=0.5,
            C_S=1.0,
            rng=rng,
        )
        self.assertLess(H_interval.lower, H_interval.upper)
        self.assertLess(n_star_interval.lower, n_star_interval.upper)
        self.assertTrue(H_interval.lower <= 0.55 <= H_interval.upper)

    def test_bootstrap_lag_power_law_supports_robust_methods(self) -> None:
        rng = np.random.default_rng(123)
        lags = np.arange(1, 24, dtype=float)
        observed = simulate_observed_discrepancies(
            lags,
            zeta=1.2,
            H=0.55,
            sigma0=0.5,
            n=500,
            noise="heteroskedastic_power",
            heteroskedastic_alpha=2.0,
            heteroskedastic_beta=1.5,
            rng=rng,
        )
        for method in ("wild", "moving_block"):
            H_interval, n_star_interval = bootstrap_lag_power_law(
                observed,
                lags,
                sigma0=0.5,
                n=500,
                bootstrap_repetitions=20,
                interval_level=0.9,
                C_K=1.0,
                a=0.5,
                C_S=1.0,
                method=method,
                block_length=5,
                rng=rng,
            )
            self.assertLess(H_interval.lower, H_interval.upper)
            self.assertLess(n_star_interval.lower, n_star_interval.upper)

    def test_misspecified_profile_supports_new_kinds(self) -> None:
        lags = np.arange(1, 17, dtype=float)
        for kind in ("piecewise", "mixed"):
            profile = misspecified_scale_profile(
                lags,
                zeta=1.0,
                H=0.6,
                amplitude=0.2,
                kind=kind,
            )
            self.assertTrue(np.all(profile > 0.0))

        observed = simulate_observed_discrepancies(
            lags,
            zeta=1.0,
            H=0.6,
            sigma0=0.5,
            n=500,
            noise="heteroskedastic_power",
            heteroskedastic_alpha=1.0,
            heteroskedastic_beta=2.0,
            profile=exact_scale_profile(lags, zeta=1.0, H=0.6),
        )
        self.assertTrue(np.all(np.isfinite(observed)))

        observed_ar = simulate_observed_discrepancies(
            lags,
            zeta=1.0,
            H=0.6,
            sigma0=0.5,
            n=500,
            noise="heteroskedastic_ar",
            heteroskedastic_alpha=0.3,
            heteroskedastic_rho=0.8,
            profile=exact_scale_profile(lags, zeta=1.0, H=0.6),
        )
        self.assertTrue(np.all(np.isfinite(observed_ar)))

    def test_misspecification_experiment_is_not_flat(self) -> None:
        rows = run_bridge_misspecification_experiment(
            BridgeMisspecificationConfig(
                lags=20,
                n=2000,
                H=0.6,
                zeta=1.0,
                sigma0=0.4,
                amplitudes=(0.0, 0.2),
                kinds=("piecewise",),
                repetitions=40,
            )
        )

        self.assertEqual(len(rows), 2)
        base = next(row for row in rows if row.amplitude == 0.0)
        perturbed = next(row for row in rows if row.amplitude == 0.2)
        self.assertLess(abs(base.bias_H), 0.2)
        self.assertGreaterEqual(abs(perturbed.bias_H), abs(base.bias_H))
        self.assertGreater(abs(perturbed.mean_residual_slope), 0.0)


if __name__ == "__main__":
    unittest.main()
