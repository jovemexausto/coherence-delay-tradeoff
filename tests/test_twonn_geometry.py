from __future__ import annotations

import unittest

from useful_memory_horizon.twonn_geometry import (
    TwonnGeometryConfig,
    run_twonn_geometry_experiment,
)


class TwonnGeometryTest(unittest.TestCase):
    def test_twonn_recovers_intrinsic_dimension_and_aggregated_holder_beats_naive(
        self,
    ) -> None:
        result = run_twonn_geometry_experiment(
            TwonnGeometryConfig(
                holder_exponents=(0.5, 0.75),
                time_steps=300,
                sample_size_per_time=256,
                path_seed_count=4,
                history=220,
            )
        )
        for row in result.rows:
            self.assertAlmostEqual(row.median_k_hat, 1.0, delta=0.15)
            self.assertLess(row.aggregated_holder_mae, row.naive_holder_mae)


if __name__ == "__main__":
    unittest.main()
