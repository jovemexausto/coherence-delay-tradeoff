from __future__ import annotations

import unittest

from useful_memory_horizon.sinkhorn_theorem_ready_band import (
    run_sinkhorn_theorem_ready_band_report,
)


class SinkhornTheoremReadyBandTest(unittest.TestCase):
    def test_report_finds_positive_bands_for_calibrated_pairs(self) -> None:
        rows = run_sinkhorn_theorem_ready_band_report()
        lookup = {(row["ambient_dim"], row["intrinsic_dim"]): row for row in rows}
        self.assertGreaterEqual(float(lookup[(8, 1)]["theorem_ready_epsilon_max"]), 0.5)
        self.assertGreaterEqual(float(lookup[(8, 2)]["theorem_ready_epsilon_max"]), 0.5)
        self.assertLessEqual(float(lookup[(12, 1)]["theorem_ready_epsilon_max"]), 0.2)
        self.assertGreaterEqual(
            float(lookup[(12, 2)]["theorem_ready_epsilon_max"]), 0.5
        )


if __name__ == "__main__":
    unittest.main()
