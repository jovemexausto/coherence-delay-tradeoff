from __future__ import annotations

import unittest

from useful_memory_horizon.bandwise_sinkhorn_frontier import (
    derive_bandwise_sinkhorn_frontier,
)


class BandwiseSinkhornFrontierTest(unittest.TestCase):
    def test_band_summary_matches_current_calibrated_frontier(self) -> None:
        result = derive_bandwise_sinkhorn_frontier()
        self.assertEqual(len(result.band_summary), 4)
        lookup = {
            (row["ambient_dim"], row["intrinsic_dim"]): row["epsilon_max"]
            for row in result.band_summary
        }
        self.assertAlmostEqual(float(lookup[(8, 1)]), 0.5, places=12)
        self.assertAlmostEqual(float(lookup[(8, 2)]), 0.5, places=12)
        self.assertAlmostEqual(float(lookup[(12, 1)]), 0.2, places=12)
        self.assertAlmostEqual(float(lookup[(12, 2)]), 0.5, places=12)


if __name__ == "__main__":
    unittest.main()
