from __future__ import annotations

import math
import unittest

from useful_memory_horizon.operational_dual_complexity import (
    dudley_entropy_integral_factor,
    lca_dual_entropy_bound,
    lca_dual_entropy_prefactor,
    lca_entropy_exponent,
    lca_epsilon_prefactor_exponent,
    lca_has_log_correction,
    lca_is_parametric_region,
    lca_noniid_dual_empirical_process_bound,
    lca_operational_horizon_exponent,
    lca_rate_exponent,
    lca_statistical_error_bound,
    lca_triangular_inheritance_bound,
)


class OperationalDualComplexityTest(unittest.TestCase):
    def test_lca_entropy_and_epsilon_exponents_match_formula(self) -> None:
        self.assertAlmostEqual(lca_entropy_exponent(8, 5.0), 1.6, places=12)
        self.assertAlmostEqual(
            lca_epsilon_prefactor_exponent(8, 5.0),
            0.5 * 8.0 * 4.0 / 5.0,
            places=12,
        )

    def test_lca_rate_is_parametric_below_entropy_threshold(self) -> None:
        self.assertTrue(lca_is_parametric_region(8, 5.0))
        self.assertAlmostEqual(lca_rate_exponent(8, 5.0), 0.5, places=12)

    def test_lca_rate_has_log_correction_at_entropy_threshold(self) -> None:
        self.assertTrue(lca_has_log_correction(4, 2.0))
        self.assertAlmostEqual(lca_rate_exponent(4, 2.0), 0.5, places=12)

    def test_lca_rate_degrades_above_entropy_threshold(self) -> None:
        self.assertFalse(lca_is_parametric_region(12, 5.0))
        self.assertAlmostEqual(lca_rate_exponent(12, 5.0), 5.0 / 12.0, places=12)

    def test_dual_entropy_prefactor_worsens_as_epsilon_decreases(self) -> None:
        coarse = lca_dual_entropy_prefactor(8, 5.0, 0.5)
        fine = lca_dual_entropy_prefactor(8, 5.0, 0.1)
        self.assertGreater(fine, coarse)

    def test_dual_entropy_bound_has_expected_delta_exponent(self) -> None:
        value_delta = lca_dual_entropy_bound(0.2, 8, 5.0, 0.2)
        value_half_delta = lca_dual_entropy_bound(0.1, 8, 5.0, 0.2)
        beta = lca_entropy_exponent(8, 5.0)
        self.assertAlmostEqual(value_half_delta / value_delta, 2.0**beta, places=10)

    def test_dudley_integral_factor_is_finite_only_below_threshold(self) -> None:
        self.assertAlmostEqual(dudley_entropy_integral_factor(1.6), 5.0, places=12)
        with self.assertRaises(ValueError):
            dudley_entropy_integral_factor(2.0)

    def test_noniid_dual_process_is_root_n_in_parametric_region(self) -> None:
        value_n = lca_noniid_dual_empirical_process_bound(100, 8, 5.0, 0.2)
        value_4n = lca_noniid_dual_empirical_process_bound(400, 8, 5.0, 0.2)
        self.assertAlmostEqual(value_4n, 0.5 * value_n, places=12)

    def test_lca_triangular_inheritance_preserves_iid_exponent(self) -> None:
        value_n = lca_triangular_inheritance_bound(
            sample_size=100,
            intrinsic_dim=8,
            smoothness_alpha=5.0,
            epsilon=0.2,
            inheritance_factor=1.3,
        )
        value_4n = lca_triangular_inheritance_bound(
            sample_size=400,
            intrinsic_dim=8,
            smoothness_alpha=5.0,
            epsilon=0.2,
            inheritance_factor=1.3,
        )
        self.assertAlmostEqual(value_4n, 0.5 * value_n, places=12)

    def test_lca_horizon_uses_family_specific_carrier_exponent(self) -> None:
        self.assertAlmostEqual(
            lca_operational_horizon_exponent(8, 5.0, 1.0),
            2.0 / 3.0,
            places=12,
        )
        self.assertAlmostEqual(
            lca_operational_horizon_exponent(12, 5.0, 1.0),
            1.0 / (1.0 + 5.0 / 12.0),
            places=12,
        )

    def test_lca_statistical_error_worsens_as_epsilon_decreases(self) -> None:
        coarse = lca_statistical_error_bound(100, 8, 5.0, 0.5)
        fine = lca_statistical_error_bound(100, 8, 5.0, 0.1)
        self.assertGreater(fine, coarse)


if __name__ == "__main__":
    unittest.main()
