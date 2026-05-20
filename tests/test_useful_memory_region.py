from __future__ import annotations

import math
import unittest

from useful_memory_horizon.useful_memory_region import (
    continuous_optimal_horizon,
    horizon_envelope,
    normalized_envelope_ratio,
    useful_memory_bounds,
    useful_memory_interval,
)


class UsefulMemoryRegionTest(unittest.TestCase):
    def test_continuous_optimizer_annihilates_derivative(self) -> None:
        C_K = 1.7
        C_S = 0.9
        a = 0.5
        H = 0.75
        zeta = 0.08
        n_star = continuous_optimal_horizon(C_K, a, C_S, zeta, H)
        derivative = -a * C_K * n_star ** (-a - 1.0) + H * C_S * zeta * n_star ** (
            H - 1.0
        )
        self.assertAlmostEqual(derivative, 0.0, places=12)

    def test_normalized_ratio_collapses_constants(self) -> None:
        x_values = (0.45, 0.8, 1.0, 1.25, 1.8)
        a = 0.5
        H = 0.75
        reference = [normalized_envelope_ratio(x, a, H) for x in x_values]

        for C_K, C_S, zeta in ((1.0, 1.0, 0.05), (2.3, 0.8, 0.12), (5.0, 3.1, 0.01)):
            n_star = continuous_optimal_horizon(C_K, a, C_S, zeta, H)
            observed = [
                horizon_envelope(n_star * x, C_K, a, C_S, zeta, H)
                / horizon_envelope(n_star, C_K, a, C_S, zeta, H)
                for x in x_values
            ]
            for expected, got in zip(reference, observed, strict=True):
                self.assertAlmostEqual(expected, got, places=12)

    def test_useful_memory_interval_is_interval_and_scales(self) -> None:
        a = 0.5
        H = 0.75
        delta = 0.12
        lower_x, upper_x = useful_memory_interval(a, H, delta)
        self.assertLess(lower_x, 1.0)
        self.assertGreater(upper_x, 1.0)

        interior = [0.5 * (lower_x + 1.0), 1.0, 0.5 * (1.0 + upper_x)]
        exterior = [0.95 * lower_x, 1.05 * upper_x]
        for x in interior:
            self.assertLessEqual(
                normalized_envelope_ratio(x, a, H), 1.0 + delta + 1e-12
            )
        for x in exterior:
            self.assertGreater(normalized_envelope_ratio(x, a, H), 1.0 + delta)

        region_a = useful_memory_bounds(1.4, a, 0.9, 0.07, H, delta)
        region_b = useful_memory_bounds(3.1, a, 2.8, 0.11, H, delta)
        self.assertAlmostEqual(region_a.lower / region_a.n_star, lower_x, places=12)
        self.assertAlmostEqual(region_a.upper / region_a.n_star, upper_x, places=12)
        self.assertAlmostEqual(region_b.lower / region_b.n_star, lower_x, places=12)
        self.assertAlmostEqual(region_b.upper / region_b.n_star, upper_x, places=12)

    def test_small_delta_local_width_matches_quadratic_prediction(self) -> None:
        a = 0.5
        H = 1.0
        delta = 1.0e-4
        lower_x, upper_x = useful_memory_interval(a, H, delta)
        predicted = math.sqrt(2.0 * delta / (a * H))
        self.assertAlmostEqual(1.0 - lower_x, predicted, delta=8.0e-4)
        self.assertAlmostEqual(upper_x - 1.0, predicted, delta=8.0e-4)


if __name__ == "__main__":
    unittest.main()
