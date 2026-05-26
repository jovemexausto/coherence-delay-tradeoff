from __future__ import annotations

import unittest

import numpy as np

from scale_consistency.model import simulate_observed_discrepancies
from scale_consistency.variance_bridge import (
    fit_best_variance_model,
    fit_variance_model,
)


class VarianceBridgeTest(unittest.TestCase):
    def test_variance_bridge_returns_finite_refit(self) -> None:
        lags = np.arange(1, 81, dtype=float)
        rng = np.random.default_rng(42)
        obs = simulate_observed_discrepancies(
            lags,
            zeta=1.0,
            H=0.6,
            sigma0=0.5,
            n=500,
            noise="heteroskedastic_power",
            heteroskedastic_alpha=2.0,
            heteroskedastic_beta=2.0,
            rng=rng,
        )
        fit = fit_best_variance_model(obs, lags, sigma0=0.5, n=500)
        self.assertTrue(np.isfinite(fit.H_pre))
        self.assertTrue(np.isfinite(fit.H_post))
        self.assertTrue(np.all(np.isfinite(fit.sigma_hat)))
        self.assertGreater(fit.n_star_pre, 0.0)
        self.assertGreater(fit.n_star_post, 0.0)

    def test_piecewise_model_fits(self) -> None:
        lags = np.arange(1, 81, dtype=float)
        rng = np.random.default_rng(7)
        obs = simulate_observed_discrepancies(
            lags,
            zeta=1.0,
            H=0.6,
            sigma0=0.5,
            n=500,
            noise="heteroskedastic_jump",
            heteroskedastic_alpha=2.0,
            heteroskedastic_jump_lag=40.0,
            rng=rng,
        )
        fit = fit_variance_model(obs, lags, sigma0=0.5, n=500, model_kind="piecewise")
        self.assertTrue(np.isfinite(fit.variance_r2))
        self.assertIsNotNone(fit.change_point)


if __name__ == "__main__":
    unittest.main()
