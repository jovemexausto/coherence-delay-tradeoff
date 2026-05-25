from __future__ import annotations

import math
import unittest

import numpy as np

from scale_consistency.theory_diagnostics import (
    chi_square_degrees_of_freedom,
    chi_square_null_mean,
    chi_square_null_variance,
    information_scale,
    kappa_boundary,
    lag_energy,
    oracle_h_variance,
    scaled_rmse_constant,
)


class ScaleConsistencyTheoryDiagnosticsTest(unittest.TestCase):
    def test_chi_square_moments_match_degrees_of_freedom(self) -> None:
        self.assertEqual(chi_square_degrees_of_freedom(10), 8)
        self.assertAlmostEqual(chi_square_null_mean(10), 8.0)
        self.assertAlmostEqual(chi_square_null_variance(10), 16.0)

    def test_information_scale_matches_lag_energy(self) -> None:
        lags = np.arange(1, 6, dtype=float)
        energy = lag_energy(lags, 0.5)
        self.assertAlmostEqual(energy, float(np.sum(lags)))
        self.assertAlmostEqual(information_scale(100, lags, 0.5), 100.0 * energy)

    def test_kappa_boundary_has_inverse_square_root_scaling(self) -> None:
        lags = np.arange(1, 21, dtype=float)
        base = kappa_boundary(100, lags, 0.6)
        doubled = kappa_boundary(400, lags, 0.6)
        self.assertAlmostEqual(base / doubled, 2.0, places=12)

    def test_oracle_h_variance_decreases_with_n(self) -> None:
        lags = np.arange(1, 21, dtype=float)
        var_small = oracle_h_variance(lags, 1.0, 0.6, 1.0, 200)
        var_large = oracle_h_variance(lags, 1.0, 0.6, 1.0, 1000)
        self.assertLess(var_large, var_small)

    def test_scaled_rmse_constant_is_positive(self) -> None:
        constant = scaled_rmse_constant(0.01, 1000, 20, 0.6)
        self.assertTrue(math.isfinite(constant))
        self.assertGreater(constant, 0.0)


if __name__ == "__main__":
    unittest.main()
