from __future__ import annotations

import unittest

from useful_memory_horizon.operational_dual_inheritance_kernel import (
    critical_dual_smoothness_for_parametric_region,
    embedded_fixed_span_chart_complexity,
    embedded_fixed_span_dual_noniid_bound,
    embedded_fixed_span_is_parametric_region,
)


class OperationalDualInheritanceKernelTest(unittest.TestCase):
    def test_critical_dual_smoothness_threshold_is_k_over_two(self) -> None:
        self.assertAlmostEqual(critical_dual_smoothness_for_parametric_region(1), 0.5)
        self.assertAlmostEqual(critical_dual_smoothness_for_parametric_region(2), 1.0)
        self.assertAlmostEqual(critical_dual_smoothness_for_parametric_region(4), 2.0)

    def test_embedded_fixed_span_chart_complexity_is_n_invariant(self) -> None:
        value_1 = embedded_fixed_span_chart_complexity(1, 0.25, 2.0)
        value_2 = embedded_fixed_span_chart_complexity(1, 0.25, 2.0)
        self.assertAlmostEqual(value_1, value_2, places=12)

    def test_embedded_fixed_span_is_parametric_region_uses_alpha_threshold(
        self,
    ) -> None:
        self.assertTrue(embedded_fixed_span_is_parametric_region(1, 0.75))
        self.assertFalse(embedded_fixed_span_is_parametric_region(2, 0.75))
        self.assertTrue(embedded_fixed_span_is_parametric_region(2, 1.25))

    def test_embedded_fixed_span_dual_bound_is_root_n_in_parametric_region(
        self,
    ) -> None:
        value_n = embedded_fixed_span_dual_noniid_bound(100, 1, 1.0, 0.2, 0.25)
        value_4n = embedded_fixed_span_dual_noniid_bound(400, 1, 1.0, 0.2, 0.25)
        self.assertAlmostEqual(value_4n, 0.5 * value_n, places=12)

    def test_embedded_fixed_span_dual_bound_worsens_below_parametric_threshold(
        self,
    ) -> None:
        parametric = embedded_fixed_span_dual_noniid_bound(400, 2, 2.0, 0.2, 0.25)
        nonparametric = embedded_fixed_span_dual_noniid_bound(400, 2, 0.75, 0.2, 0.25)
        self.assertGreater(nonparametric, parametric)


if __name__ == "__main__":
    unittest.main()
