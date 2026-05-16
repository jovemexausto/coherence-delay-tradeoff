from __future__ import annotations

import unittest

from useful_memory_horizon.glue_theorem_bahadur import (
    BahadurConfig,
    run_bahadur_research,
)


class BahadurResearchTest(unittest.TestCase):
    def test_bahadur_decomposition_residual_is_small(self) -> None:
        result = run_bahadur_research(
            BahadurConfig(n_values=(25, 50), replications=24, quantile_grid_size=128)
        )
        tri_row = next(
            row
            for row in result.curve_rows
            if row["setting"] == "triangular" and row["n"] == 50
        )
        iid_row = next(
            row
            for row in result.curve_rows
            if row["setting"] == "iid-mixture" and row["n"] == 50
        )
        self.assertLess(abs(float(tri_row["residual"])), 0.05 * float(tri_row["mse"]))
        self.assertLess(abs(float(iid_row["residual"])), 0.05 * float(iid_row["mse"]))

    def test_bahadur_rates_are_root_n_like(self) -> None:
        result = run_bahadur_research(
            BahadurConfig(
                n_values=(25, 50, 100, 200), replications=16, quantile_grid_size=128
            )
        )
        row = result.summary_rows[0]
        self.assertGreater(float(row["tri_rate_a"]), 0.35)
        self.assertGreater(float(row["iid_rate_a"]), 0.35)


if __name__ == "__main__":
    unittest.main()
