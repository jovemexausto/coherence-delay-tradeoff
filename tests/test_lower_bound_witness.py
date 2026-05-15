from __future__ import annotations

import math
import unittest


def sum_sq(h: int) -> float:
    return h * (h + 1) * (2 * h + 1) / 6


def ramp_bound(sigma: float, zeta: float, h: int) -> float:
    beta = min(zeta, sigma / (2.0 * math.sqrt(sum_sq(h))))
    kl = 2.0 * beta * beta * sum_sq(h) / (sigma * sigma)
    return beta * h * (1.0 - math.sqrt(kl / 2.0)) / 2.0


class LowerBoundWitnessTest(unittest.TestCase):
    def test_free_branch_optimum_has_kl_one_half(self) -> None:
        sigma = 1.0
        for h in (10, 100, 1000):
            beta = sigma / (2.0 * math.sqrt(sum_sq(h)))
            kl = 2.0 * beta * beta * sum_sq(h) / (sigma * sigma)
            self.assertAlmostEqual(kl, 0.5, places=10)

    def test_active_branch_constant_matches_asymptotic_prediction(self) -> None:
        constant = (3.0 / 10.0) * (2.0 * math.sqrt(3.0) / 5.0) ** (2.0 / 3.0)
        ratio = 1_000_000.0
        sigma = ratio
        zeta = 1.0
        h_max = int(4.0 * ratio ** (2.0 / 3.0)) + 20

        best = 0.0
        for h in range(1, h_max):
            best = max(best, ramp_bound(sigma, zeta, h))

        normalized = best / (sigma ** (2.0 / 3.0) * zeta ** (1.0 / 3.0))
        self.assertAlmostEqual(normalized, constant, delta=2e-5)


if __name__ == "__main__":
    unittest.main()
