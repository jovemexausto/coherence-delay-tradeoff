from __future__ import annotations

import unittest

from useful_memory_horizon.gaussian_witness_frontier import (
    GaussianWitnessFrontierConfig,
    asymptotic_shape_root,
    endpoint_minimal_profile,
    exact_minimal_profile_asymptotic_constant,
    exact_minimal_profile_shape_parameter,
    exact_profile_witness_bound,
    exact_gaussian_asymptotic_constant,
    exact_gaussian_shape_parameter,
    fixed_h_beta_root,
    minimal_endpoint_energy_constant,
    normal_cdf,
    normal_pdf,
    profile_energy,
    ramp_energy_constant,
    ramp_profile,
    run_gaussian_witness_frontier,
)
from useful_memory_horizon.holder_lower_bound_research import Hölder_asymptotic_constant


class GaussianWitnessFrontierTest(unittest.TestCase):
    def test_fixed_h_beta_root_solves_scalar_equation(self) -> None:
        x_star = fixed_h_beta_root()
        residual = normal_cdf(-x_star) - x_star * normal_pdf(x_star)
        self.assertAlmostEqual(residual, 0.0, places=12)

    def test_asymptotic_shape_root_solves_h_dependent_equation(self) -> None:
        H = 0.75
        power = 2.0 * H / (2.0 * H + 1.0)
        x_star = asymptotic_shape_root(H)
        residual = power * normal_cdf(-x_star) - x_star * normal_pdf(x_star)
        self.assertAlmostEqual(residual, 0.0, places=12)

    def test_exact_constant_matches_discrete_large_ratio_optimum(self) -> None:
        result = run_gaussian_witness_frontier(
            GaussianWitnessFrontierConfig(
                H_values=(0.5, 0.75, 1.0),
                sigma_zeta_ratios=(10_000.0,),
                max_multiplier=4.0,
            )
        )
        for row in result.summary_rows:
            H = float(row["H"])
            self.assertAlmostEqual(
                float(row["normalized_best"]),
                exact_gaussian_asymptotic_constant(H),
                delta=2e-3,
            )

    def test_exact_shape_parameter_matches_discrete_horizon(self) -> None:
        result = run_gaussian_witness_frontier(
            GaussianWitnessFrontierConfig(
                H_values=(0.5, 1.0),
                sigma_zeta_ratios=(10_000.0,),
                max_multiplier=4.0,
            )
        )
        for row in result.summary_rows:
            H = float(row["H"])
            best_h = float(row["best_h"])
            ratio = float(row["sigma_zeta_ratio"])
            normalized_h = best_h / (ratio ** (2.0 / (2.0 * H + 1.0)))
            self.assertAlmostEqual(
                normalized_h,
                exact_gaussian_shape_parameter(H),
                delta=0.02,
            )

    def test_exact_gaussian_constant_improves_on_pinsker_constant(self) -> None:
        for H in (0.35, 0.5, 0.75, 1.0):
            self.assertGreater(
                exact_gaussian_asymptotic_constant(H),
                Hölder_asymptotic_constant(H),
            )

    def test_endpoint_minimal_energy_constant_matches_closed_form(self) -> None:
        for H in (0.35, 0.5, 0.75, 1.0):
            for h in (128, 512, 2048):
                profile = endpoint_minimal_profile(H, h)
                normalized_energy = profile_energy(profile) / (h ** (2.0 * H + 1.0))
                self.assertAlmostEqual(
                    normalized_energy,
                    minimal_endpoint_energy_constant(H),
                    delta=5e-3,
                )

    def test_endpoint_minimal_profile_has_no_more_energy_than_ramp(self) -> None:
        for H in (0.35, 0.5, 0.75):
            ramp = ramp_profile(H, 256)
            minimal = endpoint_minimal_profile(H, 256)
            self.assertLess(profile_energy(minimal), profile_energy(ramp))
        self.assertAlmostEqual(
            profile_energy(endpoint_minimal_profile(1.0, 256)),
            profile_energy(ramp_profile(1.0, 256)),
            places=8,
        )

    def test_minimal_profile_constant_improves_on_ramp_for_H_below_one(self) -> None:
        for H in (0.35, 0.5, 0.75):
            self.assertGreater(
                exact_minimal_profile_asymptotic_constant(H),
                exact_gaussian_asymptotic_constant(H),
            )
        self.assertAlmostEqual(
            exact_minimal_profile_asymptotic_constant(1.0),
            exact_gaussian_asymptotic_constant(1.0),
            places=12,
        )

    def test_discrete_minimal_profile_matches_asymptotic_constant(self) -> None:
        for H in (0.5, 0.75, 1.0):
            ratio = 10_000.0
            sigma = ratio
            zeta = 1.0
            predicted_h = exact_minimal_profile_shape_parameter(H) * ratio ** (
                2.0 / (2.0 * H + 1.0)
            )
            h_max = int(4.0 * predicted_h) + 20
            best = 0.0
            best_h = 1
            for h in range(1, h_max + 1):
                value = exact_profile_witness_bound(
                    sigma,
                    zeta,
                    endpoint_minimal_profile(H, h),
                )
                if value > best:
                    best = value
                    best_h = h
            normalized = best / (
                sigma ** (2.0 * H / (2.0 * H + 1.0)) * zeta ** (1.0 / (2.0 * H + 1.0))
            )
            self.assertAlmostEqual(
                normalized,
                exact_minimal_profile_asymptotic_constant(H),
                delta=2.5e-3,
            )
            normalized_h = best_h / (ratio ** (2.0 / (2.0 * H + 1.0)))
            self.assertAlmostEqual(
                normalized_h,
                exact_minimal_profile_shape_parameter(H),
                delta=0.03,
            )


if __name__ == "__main__":
    unittest.main()
