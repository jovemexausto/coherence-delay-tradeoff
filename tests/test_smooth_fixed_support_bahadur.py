from __future__ import annotations

import unittest

from useful_memory_horizon.smooth_fixed_support_bahadur import (
    SmoothBahadurConfig,
    run_smooth_fixed_support_bahadur,
)


class SmoothFixedSupportBahadurTest(unittest.TestCase):
    def test_integrated_residual_is_faster_than_n_inverse_on_small_grid(self) -> None:
        result = run_smooth_fixed_support_bahadur(
            SmoothBahadurConfig(
                n_values=(40, 80, 160),
                replications=8,
                span_values=(0.2,),
                H_values=(0.5,),
                quantile_grid_size=128,
            )
        )
        row = result.summary_rows[0]
        self.assertGreater(float(row["residual_rate"]), 1.0)
        self.assertGreater(float(row["scaled_n_residual_rate"]), 0.0)


if __name__ == "__main__":
    unittest.main()
