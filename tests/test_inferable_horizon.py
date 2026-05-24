from __future__ import annotations

import math
import unittest

from useful_memory_horizon.inferable_horizon import (
    deterministic_relative_regret_bound,
    exact_relative_regret_from_log_ratio,
    InferableHorizonConfig,
    continuous_optimal_horizon,
    horizon_log_lipschitz_constant,
    horizon_gradient_log_map,
    horizon_log_map,
    lag_energy,
    operational_identifiability_score,
    plug_in_horizon_tau_squared,
    quadratic_relative_regret,
    relative_useful_memory_regret,
    run_inferable_horizon_suite,
    useful_region_parameter_radius,
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

    def test_exact_log_ratio_regret_matches_envelope_regret(self) -> None:
        a = 0.5
        H = 0.6
        C_K = 100.0
        C_S = 1.0
        zeta = 1.0
        n_star = continuous_optimal_horizon(C_K, a, C_S, zeta, H)
        log_ratio = 0.2
        regret = relative_useful_memory_regret(
            n_star * math.exp(log_ratio),
            C_K,
            a,
            C_S,
            zeta,
            H,
        )
        exact = float(exact_relative_regret_from_log_ratio(log_ratio, a, H))
        self.assertAlmostEqual(regret, exact, places=12)

    def test_deterministic_regret_bound_controls_exact_regret(self) -> None:
        radius = 0.15
        bound = deterministic_relative_regret_bound(radius, a=0.5, H=0.7)
        for log_ratio in (-radius, -0.5 * radius, 0.5 * radius, radius):
            self.assertLessEqual(
                float(exact_relative_regret_from_log_ratio(log_ratio, 0.5, 0.7)),
                bound + 1.0e-12,
            )

    def test_lipschitz_constant_controls_log_horizon_map(self) -> None:
        alpha_bounds = (-1.0, 0.5)
        H_bounds = (0.4, 0.9)
        L = horizon_log_lipschitz_constant(alpha_bounds, H_bounds, 100.0, 0.5, 1.0)
        alpha_1, H_1 = -0.8, 0.45
        alpha_2, H_2 = 0.2, 0.8
        diff = abs(
            float(horizon_log_map(alpha_1, H_1, 100.0, 0.5, 1.0))
            - float(horizon_log_map(alpha_2, H_2, 100.0, 0.5, 1.0))
        )
        self.assertLessEqual(diff, L * (abs(alpha_1 - alpha_2) + abs(H_1 - H_2)))

    def test_useful_region_parameter_radius_is_positive(self) -> None:
        radius = useful_region_parameter_radius(
            alpha_bounds=(-0.5, 0.5),
            H_bounds=(0.4, 0.8),
            C_K=100.0,
            a=0.5,
            C_S=1.0,
            H=0.6,
            delta=0.1,
        )
        self.assertGreater(radius, 0.0)

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
        self.assertEqual(len(suite.scenario_rows), 1)
        self.assertEqual(len(suite.joint_clt_rows), 1)
        self.assertEqual(len(suite.plugin_clt_rows), 1)
        self.assertEqual(len(suite.coverage_rows), 1)
        self.assertEqual(len(suite.regret_rows), 1)
        scenario_row = suite.scenario_rows[0]
        joint_row = suite.joint_clt_rows[0]
        plugin_row = suite.plugin_clt_rows[0]
        coverage_row = suite.coverage_rows[0]
        regret_row = suite.regret_rows[0]
        self.assertIn("scenario_id", scenario_row)
        self.assertIn("scenario_seed", scenario_row)
        self.assertEqual(joint_row["scenario_id"], scenario_row["scenario_id"])
        self.assertEqual(plugin_row["scenario_id"], scenario_row["scenario_id"])
        self.assertGreater(plugin_row["tau2"], 0.0)
        self.assertGreaterEqual(plugin_row["ks_pvalue"], 0.0)
        self.assertLessEqual(plugin_row["ks_pvalue"], 1.0)
        self.assertGreaterEqual(plugin_row["std_log_horizon_empirical_se"], 0.0)
        self.assertGreaterEqual(plugin_row["ci95_coverage_se"], 0.0)
        self.assertGreaterEqual(coverage_row["empirical_hit_rate"], 0.0)
        self.assertLessEqual(coverage_row["empirical_hit_rate"], 1.0)
        self.assertGreaterEqual(coverage_row["empirical_hit_rate_se"], 0.0)
        self.assertGreater(coverage_row["identifiability_score"], 0.0)
        self.assertGreaterEqual(regret_row["empirical_relative_regret"], 0.0)
        self.assertGreaterEqual(regret_row["empirical_relative_regret_se"], 0.0)
        self.assertGreater(regret_row["theoretical_relative_regret"], 0.0)


if __name__ == "__main__":
    unittest.main()
