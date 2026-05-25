from __future__ import annotations

import unittest

from useful_memory_horizon.sinkhorn_support_growth_hessian import (
    SupportGrowthHessianConfig,
    run_support_growth_hessian_probe,
)


class SinkhornSupportGrowthHessianTest(unittest.TestCase):
    def test_hessian_curvature_proxy_stays_bounded_on_calibrated_grid(self) -> None:
        result = run_support_growth_hessian_probe(
            SupportGrowthHessianConfig(
                epsilons=(0.2, 0.5),
                sample_sizes=(24, 48),
            )
        )
        summary = {
            (
                int(row["ambient_dim"]),
                int(row["intrinsic_dim"]),
                float(row["epsilon"]),
                str(row["direction_family"]),
            ): row
            for row in result.summary_rows
            if row["experiment"] == "support_growth_hessian"
        }

        self.assertLess(float(summary[(8, 2, 0.2, "local")]["max_curvature"]), 0.2)
        self.assertLess(float(summary[(12, 2, 0.5, "local")]["max_curvature"]), 0.2)
        self.assertLess(float(summary[(8, 2, 0.2, "collective")]["max_curvature"]), 6.5)
        self.assertLess(
            float(summary[(12, 2, 0.5, "collective")]["max_curvature"]), 6.5
        )
        self.assertLess(
            float(summary[(8, 2, 0.2, "collective")]["max_curvature_per_n"]), 0.14
        )
        self.assertGreater(
            float(summary[(8, 2, 0.2, "collective")]["min_curvature_per_n"]), 0.03
        )

        trace_rows = {
            (
                int(row["ambient_dim"]),
                int(row["intrinsic_dim"]),
                float(row["epsilon"]),
            ): row
            for row in result.summary_rows
            if row["experiment"] == "support_growth_trace_proxy"
        }
        self.assertGreater(float(trace_rows[(8, 2, 0.2)]["mean_trace_per_n"]), 0.3)
        self.assertLess(float(trace_rows[(8, 2, 0.2)]["mean_cov_weighted_proxy"]), 0.01)


if __name__ == "__main__":
    unittest.main()
