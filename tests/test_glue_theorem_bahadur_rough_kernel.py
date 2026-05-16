from __future__ import annotations

import unittest

from useful_memory_horizon.glue_theorem_bahadur_rough_kernel import (
    BahadurRoughKernelConfig,
    run_bahadur_rough_kernel_research,
)


class BahadurRoughKernelResearchTest(unittest.TestCase):
    def test_rough_kernel_rates_remain_above_root_n(self) -> None:
        result = run_bahadur_rough_kernel_research(
            BahadurRoughKernelConfig(
                alpha_values=(1.0, 0.5, 0.25),
                n_values=(25, 50, 100),
                replications=12,
                quantile_grid_size=64,
            )
        )
        for row in result.summary_rows:
            self.assertGreater(float(row["residual_rate"]), 0.5)
            self.assertGreater(float(row["sup_rate"]), 0.25)
            self.assertGreater(float(row["empirical_rate"]), 0.5)
            self.assertGreater(float(row["last_emp_over_taylor"]), 10.0)


if __name__ == "__main__":
    unittest.main()
