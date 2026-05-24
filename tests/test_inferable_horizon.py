from __future__ import annotations

import math
import unittest

from useful_memory_horizon.inferable_horizon import (
    InferableHorizonConfig,
    continuous_optimal_horizon,
    horizon_gradient_log_map,
    horizon_log_map,
    lag_energy,
    operational_identifiability_score,
    plug_in_horizon_tau_squared,
    quadratic_relative_regret,
    relative_useful_memory_regret,
    run_inferable_horizon_suite,
    useful_region_log_radius,
)


class InferableHorizonTest(unittest.TestCase):
    def test_horizon_log_map_matches_continuous_optimizer(self) -> None:
        C_K = 100.0
        C_S = 1.3
        a = 0.5
        H = 0.6
        zeta = 0.8
        n_star = continuous_optimal_horizon(C_K, a, C_S, zeta, H)
        self.assertAlmostEqual(
            math.exp(horizon_log_map(math.log(zeta), H, C_K, a, C_S)),
            n_star,
            places=12,
        )

    def test_useful_region_log_radius_matches_small_delta_quadratic_scale(self) -> None:
        a = 0.5
        H = 0.8
        delta = 1.0e-4
        radius = useful_region_log_radius(a, H, delta)
        predicted = math.sqrt(2.0 * delta / (a * H))
        self.assertAlmostEqual(radius, predicted, delta=2.5e-3)

    def test_identifiability_score_increases_with_sample_size(self) -> None:
        lags = [1.0, 2.0, 3.0, 4.0, 5.0]
        score_small = operational_identifiability_score(
            200,
            lags,
            H=0.6,
            zeta=1.0,
            sigma0=1.0,
            C_K=100.0,
            a=0.5,
            C_S=1.0,
            delta=0.1,
        )
        score_large = operational_identifiability_score(
            1000,
            lags,
            H=0.6,
            zeta=1.0,
            sigma0=1.0,
            C_K=100.0,
            a=0.5,
            C_S=1.0,
            delta=0.1,
        )
        self.assertGreater(score_large, score_small)

    def test_tau_squared_and_gradient_are_finite(self) -> None:
        gradient = horizon_gradient_log_map(0.0, 0.6, 100.0, 0.5, 1.0)
        tau2 = plug_in_horizon_tau_squared(
            [1.0, 2.0, 3.0, 4.0, 5.0],
            H=0.6,
            zeta=1.0,
            sigma0=1.0,
            C_K=100.0,
            a=0.5,
            C_S=1.0,
        )
        self.assertEqual(gradient.shape, (2,))
        self.assertTrue(all(math.isfinite(value) for value in gradient))
        self.assertGreater(tau2, 0.0)

    def test_relative_regret_matches_quadratic_approximation_near_optimum(self) -> None:
        a = 0.5
        H = 0.6
        C_K = 100.0
        C_S = 1.0
        zeta = 1.0
        n_star = continuous_optimal_horizon(C_K, a, C_S, zeta, H)
        log_ratio = 0.01
        regret = relative_useful_memory_regret(
            n_star * math.exp(log_ratio),
            C_K,
            a,
            C_S,
            zeta,
            H,
        )
        approx = float(quadratic_relative_regret(log_ratio, a, H))
        self.assertAlmostEqual(regret, approx, delta=1.0e-4)

    def test_suite_runner_returns_bridge_rows(self) -> None:
        suite = run_inferable_horizon_suite(
            InferableHorizonConfig(
                H_values=(0.6,),
                lag_counts=(10,),
                sample_sizes=(500,),
                zeta_values=(1.0,),
                delta_values=(0.1,),
                repetitions=200,
                seed=7,
            )
        )
        self.assertEqual(len(suite.joint_clt_rows), 1)
        self.assertEqual(len(suite.plugin_clt_rows), 1)
        self.assertEqual(len(suite.coverage_rows), 1)
        self.assertEqual(len(suite.regret_rows), 1)
        plugin_row = suite.plugin_clt_rows[0]
        coverage_row = suite.coverage_rows[0]
        regret_row = suite.regret_rows[0]
        self.assertGreater(plugin_row["tau2"], 0.0)
        self.assertGreaterEqual(plugin_row["ks_pvalue"], 0.0)
        self.assertLessEqual(plugin_row["ks_pvalue"], 1.0)
        self.assertGreaterEqual(coverage_row["empirical_hit_rate"], 0.0)
        self.assertLessEqual(coverage_row["empirical_hit_rate"], 1.0)
        self.assertGreater(coverage_row["identifiability_score"], 0.0)
        self.assertGreaterEqual(regret_row["empirical_relative_regret"], 0.0)
        self.assertGreater(regret_row["theoretical_relative_regret"], 0.0)


if __name__ == "__main__":
    unittest.main()
