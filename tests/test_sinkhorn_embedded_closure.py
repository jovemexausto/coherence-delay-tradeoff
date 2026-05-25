from __future__ import annotations

import unittest

from useful_memory_horizon.sinkhorn_embedded_closure import (
    EmbeddedSinkhornClosureConfig,
    run_embedded_sinkhorn_closure,
)


class SinkhornEmbeddedClosureTest(unittest.TestCase):
    def test_calibrated_closure_diagnostics_support_moderate_band(self) -> None:
        result = run_embedded_sinkhorn_closure(
            EmbeddedSinkhornClosureConfig(
                epsilons=(0.2, 0.5),
                self_sample_sizes=(24, 48, 96),
                remainder_sample_sizes=(24, 48, 96, 160),
                influence_sample_sizes=(24, 48, 96),
                remainder_seed_count=12,
                influence_seed_count=6,
            )
        )

        summary = result.summary_rows
        self_rows = {
            (int(row["ambient_dim"]), int(row["intrinsic_dim"])): row
            for row in summary
            if row["experiment"] == "self_coupling"
        }
        remainder_rows = {
            (
                int(row["ambient_dim"]),
                int(row["intrinsic_dim"]),
                float(row["epsilon"]),
                str(row["sample_role"]),
            ): row
            for row in summary
            if row["experiment"] == "linearization_remainder"
        }
        influence_rows = {
            (
                int(row["ambient_dim"]),
                int(row["intrinsic_dim"]),
                float(row["epsilon"]),
                str(row["sample_role"]),
            ): row
            for row in summary
            if row["experiment"] == "influence_proxy"
        }
        quadratic_rows = {
            (
                int(row["ambient_dim"]),
                int(row["intrinsic_dim"]),
                float(row["epsilon"]),
                str(row["sample_role"]),
            ): row
            for row in summary
            if row["experiment"] == "quadratic_proxy"
        }

        self.assertLess(float(self_rows[(8, 2)]["worst_squared_centered_radius"]), 0.1)
        self.assertLess(float(self_rows[(12, 2)]["worst_squared_centered_radius"]), 0.1)

        self.assertLess(
            float(remainder_rows[(8, 2, 0.2, "triangular_window")]["slope"]), -0.75
        )
        self.assertLess(
            float(remainder_rows[(12, 2, 0.5, "iid_mixture")]["slope"]), -0.7
        )

        self.assertLess(
            float(influence_rows[(8, 2, 0.2, "triangular_window")]["rootn_ratio"]), 2.5
        )
        self.assertLess(
            float(influence_rows[(12, 2, 0.5, "iid_mixture")]["rootn_ratio"]), 2.5
        )

        self.assertLess(
            float(
                quadratic_rows[(8, 2, 0.2, "triangular_window")][
                    "max_mean_cost_to_l2sq"
                ]
            ),
            0.1,
        )
        self.assertLess(
            float(
                quadratic_rows[(12, 2, 0.5, "iid_mixture")][
                    "max_pointwise_cost_to_l2sq"
                ]
            ),
            0.25,
        )


if __name__ == "__main__":
    unittest.main()
