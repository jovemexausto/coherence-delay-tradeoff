from __future__ import annotations

import math
import unittest

from useful_memory_horizon.holder_lower_bound_research import (
    HolderLowerBoundResearchConfig,
    holder_asymptotic_constant,
    holder_optimal_shape_parameter,
    holder_scaling_exponents,
    holder_witness_bound,
    run_holder_lower_bound_research,
)


class HolderLowerBoundResearchTest(unittest.TestCase):
    def test_h_equals_one_recovers_existing_constant(self) -> None:
        expected = (3.0 / 10.0) * (2.0 * math.sqrt(3.0) / 5.0) ** (2.0 / 3.0)
        self.assertAlmostEqual(holder_asymptotic_constant(1.0), expected, places=12)

    def test_h_equals_one_shape_parameter_recovers_existing_scale(self) -> None:
        expected = (2.0 * math.sqrt(3.0) / 5.0) ** (2.0 / 3.0)
        self.assertAlmostEqual(holder_optimal_shape_parameter(1.0), expected, places=12)

    def test_holder_scaling_exponents_match_formula(self) -> None:
        sigma_power, zeta_power = holder_scaling_exponents(0.5)
        self.assertAlmostEqual(sigma_power, 0.5)
        self.assertAlmostEqual(zeta_power, 0.5)

    def test_large_ratio_numeric_optimum_matches_asymptotic_constant(self) -> None:
        H = 0.5
        ratio = 10_000.0
        sigma = ratio
        zeta = 1.0
        predicted_h = holder_optimal_shape_parameter(H) * ratio ** (
            2.0 / (2.0 * H + 1.0)
        )
        h_max = int(4.0 * predicted_h) + 20
        best = 0.0
        for h in range(1, h_max + 1):
            best = max(best, holder_witness_bound(sigma, zeta, H, h))
        sigma_power, zeta_power = holder_scaling_exponents(H)
        normalized = best / (sigma**sigma_power * zeta**zeta_power)
        self.assertAlmostEqual(normalized, holder_asymptotic_constant(H), delta=5e-4)

    def test_small_research_run_returns_rows(self) -> None:
        result = run_holder_lower_bound_research(
            HolderLowerBoundResearchConfig(
                H_values=(0.5, 1.0),
                sigma_zeta_ratios=(200.0,),
                max_multiplier=2.0,
            )
        )
        self.assertEqual(len(result.summary_rows), 2)
        self.assertTrue(result.curve_rows)


if __name__ == "__main__":
    unittest.main()
