from __future__ import annotations

import math
import unittest

from useful_memory_horizon.distributional_family_frontier import (
    gaussian_scale_local_diagnostic,
    gaussian_scale_w2,
    uniform_scale_local_diagnostic,
    uniform_scale_w2,
)


class DistributionalFamilyFrontierTest(unittest.TestCase):
    def test_gaussian_scale_w2_is_linear_in_scale_difference(self) -> None:
        self.assertAlmostEqual(gaussian_scale_w2(1.0, 1.2), 0.2, places=12)

    def test_uniform_scale_w2_is_linear_in_scale_difference(self) -> None:
        self.assertAlmostEqual(
            uniform_scale_w2(1.0, 1.2),
            0.2 / math.sqrt(3.0),
            places=12,
        )

    def test_gaussian_scale_has_quadratic_local_kl_and_hellinger(self) -> None:
        for delta in (1e-1, 5e-2, 1e-2, 5e-3, 1e-3):
            diagnostic = gaussian_scale_local_diagnostic(delta)
            self.assertAlmostEqual(diagnostic.w2_over_delta, 1.0, delta=1e-10)
            self.assertAlmostEqual(diagnostic.kl_over_delta_squared, 1.0, delta=0.2)
            self.assertAlmostEqual(
                diagnostic.hellinger_over_delta_squared, 0.25, delta=0.03
            )

    def test_uniform_scale_is_not_quadratically_regular_in_kl(self) -> None:
        diagnostics = [
            uniform_scale_local_diagnostic(delta) for delta in (1e-2, 5e-3, 1e-3)
        ]
        ratios = [d.kl_over_delta_squared for d in diagnostics]
        self.assertGreater(ratios[1], ratios[0])
        self.assertGreater(ratios[2], ratios[1])
        self.assertGreater(ratios[2], 500.0)

    def test_uniform_scale_w2_remains_linear_despite_nonregular_kl(self) -> None:
        for delta in (1e-1, 1e-2, 1e-3):
            diagnostic = uniform_scale_local_diagnostic(delta)
            self.assertAlmostEqual(
                diagnostic.w2_over_delta,
                1.0 / math.sqrt(3.0),
                delta=1e-10,
            )


if __name__ == "__main__":
    unittest.main()
