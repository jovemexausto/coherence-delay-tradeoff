from __future__ import annotations

import math
import unittest

import numpy as np

from useful_memory_horizon.regular_family_frontier import (
    estimate_log_slope,
    gaussian_scale_fisher_information,
    gaussian_scale_mle_asymptotic_constant,
    gaussian_scale_mle_asymptotic_shape_parameter,
    gaussian_scale_minimal_lower_asymptotic_constant,
    gaussian_scale_minimal_lower_shape_parameter,
    gaussian_scale_path_staleness_upper,
    gaussian_scale_profile,
    gaussian_scale_upper_bound,
    gaussian_scale_w2,
    regular_family_horizon_exponent,
    regular_family_local_metric_scale,
    regular_family_metric_carrier_exponent,
    regular_family_metric_holder_exponent,
    regular_family_minimal_lower_asymptotic_constant,
    regular_family_parameter_to_metric_roughness,
    regular_family_parametric_first_moment_constant,
    regular_family_rate_exponent,
    simulate_gaussian_scale_tracking_risk,
)


class RegularFamilyFrontierTest(unittest.TestCase):
    def test_regular_family_metric_exponents_scale_with_alpha(self) -> None:
        self.assertAlmostEqual(regular_family_metric_carrier_exponent(1.0), 0.5)
        self.assertAlmostEqual(regular_family_metric_carrier_exponent(2.0), 1.0)
        self.assertAlmostEqual(regular_family_metric_holder_exponent(0.5, 0.75), 0.375)
        self.assertAlmostEqual(regular_family_metric_holder_exponent(2.0, 0.75), 1.5)

    def test_regular_family_horizon_law_uses_alpha_half_and_alpha_h(self) -> None:
        self.assertAlmostEqual(regular_family_horizon_exponent(1.0, 1.0), 2.0 / 3.0)
        self.assertAlmostEqual(regular_family_horizon_exponent(0.5, 1.0), 4.0 / 3.0)
        self.assertAlmostEqual(regular_family_horizon_exponent(2.0, 1.0), 1.0 / 3.0)
        self.assertAlmostEqual(regular_family_rate_exponent(1.0, 1.0), 1.0 / 3.0)

    def test_regular_family_metric_roughness_raises_parameter_budget_to_alpha(
        self,
    ) -> None:
        self.assertAlmostEqual(
            regular_family_parameter_to_metric_roughness(1.0, 0.2), 0.2
        )
        self.assertAlmostEqual(
            regular_family_parameter_to_metric_roughness(2.0, 0.2), 0.04
        )

    def test_gaussian_scale_w2_is_linear(self) -> None:
        self.assertAlmostEqual(gaussian_scale_w2(1.0, 1.25), 0.25, places=12)

    def test_gaussian_scale_path_staleness_upper_matches_rms_gap(self) -> None:
        scales = (1.0, 1.1, 1.2, 1.3)
        expected = math.sqrt(sum((scale - 1.0) ** 2 for scale in scales) / len(scales))
        self.assertAlmostEqual(
            gaussian_scale_path_staleness_upper(1.0, scales), expected, places=12
        )

    def test_gaussian_scale_profile_is_holder(self) -> None:
        profile = gaussian_scale_profile(1.0, 0.1, 0.5, 6)
        for j, scale in enumerate(profile):
            self.assertAlmostEqual(scale - 1.0, 0.1 * math.sqrt(j), places=12)

    def test_gaussian_scale_mle_asymptotic_constant_is_one_over_sqrt_pi(self) -> None:
        self.assertAlmostEqual(
            gaussian_scale_mle_asymptotic_constant(),
            1.0 / math.sqrt(math.pi),
            places=12,
        )

    def test_regular_family_parametric_first_moment_constant_matches_local_scale(
        self,
    ) -> None:
        fisher_information = 2.0
        local_scale = regular_family_local_metric_scale(fisher_information)
        self.assertAlmostEqual(local_scale, 1.0 / math.sqrt(2.0), places=12)
        self.assertAlmostEqual(
            regular_family_parametric_first_moment_constant(fisher_information),
            1.0 / math.sqrt(math.pi),
            places=12,
        )

    def test_gaussian_scale_fisher_information_is_two_over_sigma_squared(self) -> None:
        self.assertAlmostEqual(gaussian_scale_fisher_information(2.0), 0.5, places=12)

    def test_regular_family_minimal_lower_constant_scales_with_local_metric_noise(
        self,
    ) -> None:
        H = 0.75
        value_1 = regular_family_minimal_lower_asymptotic_constant(
            H, fisher_information=2.0
        )
        value_2 = regular_family_minimal_lower_asymptotic_constant(
            H, fisher_information=0.5
        )
        expected_ratio = (2.0) ** (2.0 * H / (2.0 * H + 1.0))
        self.assertAlmostEqual(value_2 / value_1, expected_ratio, delta=1e-10)

    def test_gaussian_scale_minimal_lower_constant_matches_regular_family_proxy(
        self,
    ) -> None:
        H = 0.5
        sigma = 2.0
        self.assertAlmostEqual(
            gaussian_scale_minimal_lower_asymptotic_constant(H, sigma),
            regular_family_minimal_lower_asymptotic_constant(
                H, gaussian_scale_fisher_information(sigma)
            ),
            places=12,
        )

    def test_gaussian_scale_minimal_lower_shape_parameter_scales_with_sigma(
        self,
    ) -> None:
        H = 1.0
        shape_1 = gaussian_scale_minimal_lower_shape_parameter(H, 1.0)
        shape_2 = gaussian_scale_minimal_lower_shape_parameter(H, 2.0)
        self.assertAlmostEqual(shape_2 / shape_1, 2.0 ** (2.0 / 3.0), delta=1e-10)

    def test_gaussian_scale_upper_shape_matches_balance_law(self) -> None:
        for H in (0.5, 0.75, 1.0):
            A = gaussian_scale_mle_asymptotic_shape_parameter(H)
            residual = 0.5 * (A ** (-0.5)) - H * (A**H)
            self.assertAlmostEqual(residual, 0.0, delta=5e-12)

    def test_gaussian_scale_tracking_risk_recovers_carrier_rate_when_staleness_is_small(
        self,
    ) -> None:
        sigma = 1.0
        H = 0.5
        zeta = 1e-4
        sample_sizes = np.asarray((64, 128, 256, 512), dtype=float)
        values = tuple(
            simulate_gaussian_scale_tracking_risk(
                sigma=sigma,
                zeta=zeta,
                H=H,
                n=int(n),
                replications=3000,
                seed=10 + int(n),
            )
            for n in sample_sizes
        )
        slope = estimate_log_slope(sample_sizes, values)
        self.assertAlmostEqual(slope, -0.5, delta=0.08)

    def test_gaussian_scale_normalized_risk_matches_asymptotic_constant(self) -> None:
        sigma = 1.0
        zeta = 1e-6
        H = 1.0
        n = 2048
        risk = simulate_gaussian_scale_tracking_risk(
            sigma=sigma,
            zeta=zeta,
            H=H,
            n=n,
            replications=6000,
            seed=123,
        )
        normalized = risk * math.sqrt(n) / sigma
        self.assertAlmostEqual(
            normalized,
            gaussian_scale_mle_asymptotic_constant(),
            delta=0.035,
        )

    def test_gaussian_scale_upper_bound_has_correct_horizon_scaling(self) -> None:
        H = 0.75
        zeta = 1.0
        sigma_values = (1_000.0, 10_000.0)
        sigma_power = 2.0 * H / (2.0 * H + 1.0)
        zeta_power = 1.0 / (2.0 * H + 1.0)
        normalized_risks = []
        normalized_horizons = []
        for sigma in sigma_values:
            predicted_h = gaussian_scale_mle_asymptotic_shape_parameter(H) * (
                (gaussian_scale_mle_asymptotic_constant() * sigma) / zeta
            ) ** (1.0 / (H + 0.5))
            h_min = max(1, int(predicted_h / 3.0))
            h_max = int(3.0 * predicted_h) + 50
            best_h = h_min
            best_value = gaussian_scale_upper_bound(sigma, zeta, H, h_min)
            for h in range(h_min + 1, h_max + 1):
                value = gaussian_scale_upper_bound(sigma, zeta, H, h)
                if value < best_value:
                    best_h = h
                    best_value = value
            normalized_risks.append(
                best_value / (sigma**sigma_power * zeta**zeta_power)
            )
            normalized_horizons.append(best_h / ((sigma / zeta) ** (1.0 / (H + 0.5))))
        self.assertAlmostEqual(normalized_risks[0], normalized_risks[1], delta=0.02)
        self.assertAlmostEqual(
            normalized_horizons[0], normalized_horizons[1], delta=0.06
        )


if __name__ == "__main__":
    unittest.main()
