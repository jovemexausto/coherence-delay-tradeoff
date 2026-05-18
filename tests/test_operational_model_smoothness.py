from __future__ import annotations

import unittest

from useful_memory_horizon.operational_model_smoothness import (
    embedded_fixed_span_chart_radius,
    recommended_parametric_dual_smoothness,
    squared_euclidean_chart_derivative_bound,
    squared_euclidean_parametric_region_holds,
    squared_euclidean_supports_holder_smoothness,
)


class OperationalModelSmoothnessTest(unittest.TestCase):
    def test_embedded_fixed_span_chart_radius_matches_span_plus_template(self) -> None:
        self.assertAlmostEqual(embedded_fixed_span_chart_radius(0.25), 1.125, places=12)

    def test_squared_euclidean_derivative_bounds_have_polynomial_structure(
        self,
    ) -> None:
        self.assertGreater(squared_euclidean_chart_derivative_bound(0, 0.25), 0.0)
        self.assertGreater(squared_euclidean_chart_derivative_bound(1, 0.25), 0.0)
        self.assertEqual(squared_euclidean_chart_derivative_bound(2, 0.25), 2.0)
        self.assertEqual(squared_euclidean_chart_derivative_bound(3, 0.25), 0.0)
        self.assertEqual(squared_euclidean_chart_derivative_bound(5, 0.25), 0.0)

    def test_squared_euclidean_supports_arbitrary_holder_smoothness(self) -> None:
        self.assertTrue(squared_euclidean_supports_holder_smoothness(1.0))
        self.assertTrue(squared_euclidean_supports_holder_smoothness(3.5))
        self.assertTrue(squared_euclidean_supports_holder_smoothness(10.0))

    def test_recommended_parametric_dual_smoothness_sits_above_threshold(self) -> None:
        self.assertAlmostEqual(
            recommended_parametric_dual_smoothness(1), 1.5, places=12
        )
        self.assertAlmostEqual(
            recommended_parametric_dual_smoothness(2), 2.0, places=12
        )
        self.assertAlmostEqual(
            recommended_parametric_dual_smoothness(4), 3.0, places=12
        )

    def test_squared_euclidean_parametric_region_holds_above_threshold(self) -> None:
        self.assertTrue(squared_euclidean_parametric_region_holds(1, 1.0))
        self.assertTrue(squared_euclidean_parametric_region_holds(2, 1.25))
        self.assertFalse(squared_euclidean_parametric_region_holds(2, 0.75))


if __name__ == "__main__":
    unittest.main()
