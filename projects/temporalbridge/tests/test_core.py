from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "code"))
sys.path.insert(0, str(PROJECT_ROOT.parent / "scale-consistency" / "code"))

from temporalbridge.core import bootstrap_horizon, fit_horizon  # noqa: E402
from temporalbridge.utils import compute_profile_diagnostics  # noqa: E402
from scale_consistency.model import exact_scale_profile, simulate_observed_discrepancies  # noqa: E402


class TemporalBridgeCoreTest(unittest.TestCase):
    def test_fit_horizon_recovers_exact_profile(self) -> None:
        lags = np.arange(1, 16, dtype=float)
        discrepancies = exact_scale_profile(lags, zeta=1.4, H=0.45)
        result = fit_horizon(lags, discrepancies, fit_options={"sigma0": 0.5, "n": 500})
        self.assertAlmostEqual(result["H"], 0.45, places=10)
        self.assertAlmostEqual(result["zeta"], 1.4, places=10)
        self.assertGreater(result["n_star"], 0.0)

    def test_bootstrap_horizon_supports_methods(self) -> None:
        lags = np.arange(1, 24, dtype=float)
        discrepancies = simulate_observed_discrepancies(
            lags,
            zeta=1.0,
            H=0.6,
            sigma0=0.5,
            n=500,
            noise="heteroskedastic_power",
            heteroskedastic_alpha=2.0,
            heteroskedastic_beta=1.5,
        )
        profile = fit_horizon(
            lags, discrepancies, fit_options={"sigma0": 0.5, "n": 500}
        )
        for method in ("parametric", "wild", "moving_block"):
            result = bootstrap_horizon(
                profile,
                method=method,
                n_boot=20,
                block_length=5,
                rng_seed=123,
            )
            self.assertEqual(result["boot_dist_H"].shape, (20,))
            self.assertEqual(result["boot_dist_n_star"].shape, (20,))
            self.assertLess(result["ci_H"][0], result["ci_H"][1])
            self.assertLess(result["ci_n_star"][0], result["ci_n_star"][1])

    def test_compute_profile_diagnostics_returns_expected_keys(self) -> None:
        lags = np.arange(1, 20, dtype=float)
        discrepancies = simulate_observed_discrepancies(
            lags,
            zeta=1.0,
            H=0.6,
            sigma0=0.5,
            n=500,
        )
        profile = fit_horizon(
            lags, discrepancies, fit_options={"sigma0": 0.5, "n": 500}
        )
        diagnostics = compute_profile_diagnostics(profile)
        self.assertIn("KL_residual", diagnostics)
        self.assertIn("KL_standardized", diagnostics)
        self.assertIn("DW", diagnostics)
        self.assertIn("curvature_p", diagnostics)


if __name__ == "__main__":
    unittest.main()
