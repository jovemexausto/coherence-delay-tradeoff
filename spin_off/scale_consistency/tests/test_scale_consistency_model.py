from __future__ import annotations

import unittest

import numpy as np

from scale_consistency.model import (
    exact_scale_profile,
    log_scale_signal,
    log_variance_profile,
    misspecified_scale_profile,
    simulate_observed_discrepancies,
)


class ScaleConsistencyModelTest(unittest.TestCase):
    def test_exact_profile_matches_power_law(self) -> None:
        lags = np.arange(1, 6, dtype=float)
        profile = exact_scale_profile(lags, zeta=2.0, H=0.5)
        expected = 2.0 * np.sqrt(lags)
        np.testing.assert_allclose(profile, expected)

    def test_log_signal_is_log_of_exact_profile(self) -> None:
        lags = np.arange(1, 8, dtype=float)
        np.testing.assert_allclose(
            log_scale_signal(lags, 1.3, 0.4),
            np.log(exact_scale_profile(lags, 1.3, 0.4)),
        )

    def test_log_variance_decreases_with_lag(self) -> None:
        variance = log_variance_profile(np.arange(1, 8), 1.0, 0.6, 1.0, 500)
        self.assertTrue(np.all(np.diff(variance) < 0.0))

    def test_misspecified_profile_stays_positive(self) -> None:
        profile = misspecified_scale_profile(
            np.arange(1, 20), 1.0, 0.5, 0.2, kind="sinusoid"
        )
        self.assertTrue(np.all(profile > 0.0))

    def test_simulated_observations_are_positive(self) -> None:
        rng = np.random.default_rng(123)
        obs = simulate_observed_discrepancies(
            np.arange(1, 10), 1.0, 0.6, 1.0, 300, rng=rng
        )
        self.assertTrue(np.all(obs > 0.0))


if __name__ == "__main__":
    unittest.main()
