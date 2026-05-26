from __future__ import annotations

import unittest

import numpy as np

from scale_consistency.bridge_diagnostics import (
    cusum_squared_statistic,
    dominant_periodogram_frequency,
    dominant_periodogram_power,
    durbin_watson,
    log_variance_trend,
    rolling_window_variances,
    standardized_residuals,
    sliding_window_kl_scores,
    quadratic_curvature_p_value,
    window_centers,
    window_scale_test_p_values,
    variance_window_kl_scores,
)


class BridgeDiagnosticsTest(unittest.TestCase):
    def test_durbin_watson_matches_closed_form(self) -> None:
        residuals = np.array([1.0, -1.0, 1.0, -1.0, 1.0, -1.0])
        expected = float(np.sum(np.diff(residuals) ** 2) / np.sum(residuals**2))
        self.assertAlmostEqual(durbin_watson(residuals), expected)

    def test_quadratic_curvature_p_value_detects_curvature(self) -> None:
        lags = np.arange(1, 11, dtype=float)
        x = np.log(lags)
        linear = 0.4 + 0.7 * x
        curved = linear + 0.5 * x**2

        self.assertAlmostEqual(quadratic_curvature_p_value(linear, lags), 1.0)
        self.assertLess(quadratic_curvature_p_value(curved, lags), 1e-6)

    def test_periodogram_identifies_dominant_frequency(self) -> None:
        sample_count = 16
        target_frequency = 2.0 / sample_count
        grid = np.arange(sample_count, dtype=float)
        residuals = np.sin(2.0 * np.pi * target_frequency * grid)

        self.assertAlmostEqual(
            dominant_periodogram_frequency(residuals),
            target_frequency,
        )
        self.assertGreater(dominant_periodogram_power(residuals), 0.0)

    def test_sliding_window_kl_scores_are_finite(self) -> None:
        values = np.concatenate([np.zeros(10), np.ones(10)])
        scores = sliding_window_kl_scores(values, window_size=6, step=2)
        self.assertGreater(scores.size, 0)
        self.assertTrue(np.all(np.isfinite(scores)))
        self.assertTrue(np.all(scores >= 0.0))

    def test_standardized_and_variance_scores_are_finite(self) -> None:
        lags = np.arange(1, 21, dtype=float)
        residuals = np.linspace(0.1, 2.0, lags.size) * np.sin(lags / 3.0)
        standardized = standardized_residuals(residuals, lags)
        variances = rolling_window_variances(residuals, window_size=5, step=2)
        centers = window_centers(lags, window_size=5, step=2)
        slope, p_value = log_variance_trend(residuals, lags, window_size=5, step=2)
        cusum = cusum_squared_statistic(residuals)
        levene_p_values, fligner_p_values = window_scale_test_p_values(
            residuals, window_size=5, step=2
        )
        self.assertEqual(standardized.shape, residuals.shape)
        self.assertGreater(variances.size, 0)
        self.assertGreater(centers.size, 0)
        self.assertTrue(np.all(np.isfinite(standardized)))
        self.assertTrue(np.all(np.isfinite(variances)))
        self.assertTrue(np.isfinite(slope))
        self.assertTrue(np.isfinite(p_value))
        self.assertTrue(np.isfinite(cusum))
        self.assertTrue(np.all(np.isfinite(levene_p_values)))
        self.assertTrue(np.all(np.isfinite(fligner_p_values)))
        self.assertTrue(
            np.all(
                np.isfinite(variance_window_kl_scores(residuals, window_size=5, step=2))
            )
        )


if __name__ == "__main__":
    unittest.main()
