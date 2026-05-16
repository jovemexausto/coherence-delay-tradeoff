from __future__ import annotations

import unittest

import numpy as np

from useful_memory_horizon.glue_theorem_research import (
    GlueTheoremResearchConfig,
    asymptotic_quantile_constants,
    fixed_span_means,
    kappa_identity_value,
    run_glue_theorem_research,
)


class GlueTheoremResearchTest(unittest.TestCase):
    def test_kappa_identity_is_stable_numerically(self) -> None:
        self.assertAlmostEqual(kappa_identity_value(), 4.75139339, delta=5e-4)

    def test_homogeneous_window_recovers_equal_tri_and_iid_constants(self) -> None:
        means = np.zeros(40, dtype=float)
        _, _, _, c2_tri, c2_iid = asymptotic_quantile_constants(means, sigma=1.0)
        self.assertAlmostEqual(c2_tri, c2_iid, delta=1e-5)

    def test_fixed_span_means_have_requested_span(self) -> None:
        means = fixed_span_means(64, span=0.5, H=1.0)
        self.assertAlmostEqual(float(means.max() - means.min()), 0.5, places=10)

    def test_small_glue_research_run_returns_rows(self) -> None:
        result = run_glue_theorem_research(
            GlueTheoremResearchConfig(
                fixed_span=0.25,
                n_values=(25, 50),
                fixed_span_replications=12,
                reference_size=4000,
                kappa_delta_n=20,
                kappa_delta_values=(0.0, 0.5),
                growth_betas=(0.0, 0.5),
            )
        )
        experiments = {row["experiment"] for row in result.summary_rows}
        self.assertIn("identity-check", experiments)
        self.assertIn("fixed-span-rate", experiments)
        self.assertIn("kappa-vs-span", experiments)
        self.assertIn("span-growth", experiments)
        self.assertTrue(result.curve_rows)


if __name__ == "__main__":
    unittest.main()
