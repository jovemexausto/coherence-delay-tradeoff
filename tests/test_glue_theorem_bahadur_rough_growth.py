from __future__ import annotations

import unittest

from useful_memory_horizon.glue_theorem_bahadur_rough_growth import (
    BahadurRoughGrowthConfig,
    run_bahadur_rough_growth_research,
)


class BahadurRoughGrowthResearchTest(unittest.TestCase):
    def test_combined_growth_rates_stay_above_root_n_in_safe_zone(self) -> None:
        result = run_bahadur_rough_growth_research(
            BahadurRoughGrowthConfig(
                alpha_values=(1.0, 0.5),
                growth_betas=(0.0, 0.25, 0.5),
                n_values=(32, 64, 128),
                replications=10,
                quantile_grid_size=64,
            )
        )
        for row in result.summary_rows:
            self.assertGreater(float(row["residual_rate"]), 0.5)
            self.assertGreater(float(row["sup_rate"]), 0.25)
            self.assertGreater(float(row["empirical_rate"]), 0.5)


if __name__ == "__main__":
    unittest.main()
