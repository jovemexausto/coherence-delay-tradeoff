from __future__ import annotations

import unittest

from useful_memory_horizon.glue_theorem_bahadur_residual import (
    BahadurResidualConfig,
    run_bahadur_residual_research,
)


class BahadurResidualResearchTest(unittest.TestCase):
    def test_residual_is_smaller_than_mse_fractionally(self) -> None:
        result = run_bahadur_residual_research(
            BahadurResidualConfig(
                n_values=(50, 100), replications=24, quantile_grid_size=128
            )
        )
        row = result.summary_rows[0]
        self.assertLess(float(row["tri_residual_over_mse"]), 0.4)
        self.assertLess(float(row["iid_residual_over_mse"]), 0.4)

    def test_residual_rate_is_faster_than_root_n_like(self) -> None:
        result = run_bahadur_residual_research(
            BahadurResidualConfig(
                n_values=(25, 50, 100, 200), replications=16, quantile_grid_size=128
            )
        )
        row = result.summary_rows[-1]
        self.assertGreater(float(row["tri_residual_rate"]), 0.5)
        self.assertGreater(float(row["iid_residual_rate"]), 0.5)
        self.assertGreater(float(row["tri_sup_residual_rate"]), 0.25)
        self.assertGreater(float(row["iid_sup_residual_rate"]), 0.25)
        self.assertGreater(float(row["tri_interior_residual_rate"]), 0.5)
        self.assertGreater(float(row["iid_interior_residual_rate"]), 0.5)
        self.assertGreater(float(row["tri_interior_sup_residual_rate"]), 0.25)
        self.assertGreater(float(row["iid_interior_sup_residual_rate"]), 0.25)

    def test_interior_residual_fraction_stays_small(self) -> None:
        result = run_bahadur_residual_research(
            BahadurResidualConfig(
                n_values=(50, 100), replications=24, quantile_grid_size=128
            )
        )
        row = result.summary_rows[0]
        self.assertLess(float(row["tri_interior_residual_over_mse"]), 0.4)
        self.assertLess(float(row["iid_interior_residual_over_mse"]), 0.4)

    def test_boundary_band_is_lower_order_than_total_residual(self) -> None:
        result = run_bahadur_residual_research(
            BahadurResidualConfig(
                n_values=(50,), replications=8, quantile_grid_size=128
            )
        )
        row = result.summary_rows[0]
        tri_boundary_fraction = (
            float(row["tri_residual"]) - float(row["tri_interior_residual"])
        ) / float(row["tri_residual"])
        iid_boundary_fraction = (
            float(row["iid_residual"]) - float(row["iid_interior_residual"])
        ) / float(row["iid_residual"])
        self.assertGreaterEqual(
            float(row["tri_residual"]), float(row["tri_interior_residual"])
        )
        self.assertGreaterEqual(
            float(row["iid_residual"]), float(row["iid_interior_residual"])
        )
        self.assertLess(tri_boundary_fraction, 0.4)
        self.assertLess(iid_boundary_fraction, 0.4)

    def test_split_terms_sanity_check(self) -> None:
        result = run_bahadur_residual_research(
            BahadurResidualConfig(
                n_values=(25, 50, 100, 200, 400),
                replications=16,
                quantile_grid_size=128,
            )
        )
        row = result.summary_rows[-2]
        self.assertGreater(
            float(row["tri_empirical_term"]), 100.0 * float(row["tri_taylor_term"])
        )
        self.assertGreater(
            float(row["iid_empirical_term"]), 100.0 * float(row["iid_taylor_term"])
        )
        self.assertLess(
            float(row["tri_reconstruction_error"]),
            0.1 * float(row["tri_interior_residual"]),
        )
        self.assertLess(
            float(row["iid_reconstruction_error"]),
            0.1 * float(row["iid_interior_residual"]),
        )


if __name__ == "__main__":
    unittest.main()
