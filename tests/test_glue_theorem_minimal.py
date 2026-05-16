from __future__ import annotations

import unittest

import numpy as np

from useful_memory_horizon.glue_theorem_minimal import (
    MinimalGlueConfig,
    run_minimal_glue_research,
    uniform_mixture_cdf,
    uniform_mixture_quantiles,
)


class MinimalGlueResearchTest(unittest.TestCase):
    def test_uniform_mixture_cdf_is_monotone(self) -> None:
        means = np.array([-0.5, 0.0, 0.5], dtype=float)
        xs = np.linspace(-2.0, 2.0, 25)
        vals = [uniform_mixture_cdf(float(x), means, 1.0) for x in xs]
        self.assertTrue(all(a <= b for a, b in zip(vals, vals[1:])))

    def test_quantiles_invert_cdf(self) -> None:
        means = np.array([-0.5, 0.0, 0.5], dtype=float)
        grid = (np.arange(64, dtype=float) + 0.5) / 64.0
        q = uniform_mixture_quantiles(grid, means, 1.0)
        self.assertTrue(np.all(np.diff(q) > 0))

    def test_small_research_run_returns_rows(self) -> None:
        result = run_minimal_glue_research(
            MinimalGlueConfig(n_values=(20, 40), replications=8, quantile_grid_size=128)
        )
        self.assertEqual(len(result.summary_rows), 1)
        self.assertEqual(len(result.curve_rows), 4)


if __name__ == "__main__":
    unittest.main()
