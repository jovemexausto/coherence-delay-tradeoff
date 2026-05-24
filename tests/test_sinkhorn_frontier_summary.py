from __future__ import annotations

import unittest

from useful_memory_horizon.sinkhorn_frontier_summary import (
    build_sinkhorn_frontier_summary,
)


class SinkhornFrontierSummaryTest(unittest.TestCase):
    def test_combined_summary_matches_calibrated_pairs(self) -> None:
        rows = build_sinkhorn_frontier_summary()
        lookup = {
            (int(row["ambient_dim"]), int(row["intrinsic_dim"])): row for row in rows
        }
        self.assertGreaterEqual(float(lookup[(8, 2)]["theorem_ready_epsilon_max"]), 0.5)
        self.assertGreaterEqual(
            float(lookup[(12, 2)]["theorem_ready_epsilon_max"]), 0.5
        )
        self.assertTrue(bool(lookup[(8, 2)]["self_coupling_positive"]))
        self.assertTrue(bool(lookup[(12, 2)]["self_coupling_positive"]))
        self.assertLessEqual(float(lookup[(12, 1)]["theorem_ready_epsilon_max"]), 0.2)


if __name__ == "__main__":
    unittest.main()
