from __future__ import annotations

import math
import unittest

from useful_memory_horizon.smooth_fixed_support_sinkhorn import (
    SmoothSinkhornConfig,
    run_smooth_fixed_support_sinkhorn,
)


class SmoothFixedSupportSinkhornTest(unittest.TestCase):
    def test_sinkhorn_diagnostic_returns_finite_slopes(self) -> None:
        result = run_smooth_fixed_support_sinkhorn(
            SmoothSinkhornConfig(
                n_values=(40, 80, 160),
                replications=4,
                span_values=(0.25,),
                H_values=(0.5,),
                epsilons=(0.5,),
            )
        )
        row = result.summary_rows[0]
        self.assertTrue(math.isfinite(float(row["tri_slope"])))
        self.assertTrue(math.isfinite(float(row["iid_slope"])))
        self.assertGreaterEqual(float(row["slope_gap"]), 0.0)


if __name__ == "__main__":
    unittest.main()
