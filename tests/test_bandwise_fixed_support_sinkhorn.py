from __future__ import annotations

import unittest

from useful_memory_horizon.bandwise_fixed_support_sinkhorn import (
    derive_bandwise_fixed_support_sinkhorn,
)


class BandwiseFixedSupportSinkhornTest(unittest.TestCase):
    def test_compact_band_has_bounded_slope_gaps(self) -> None:
        result = derive_bandwise_fixed_support_sinkhorn()
        gaps = [float(row["slope_gap"]) for row in result.summary_rows]
        self.assertGreater(len(gaps), 0)
        self.assertLess(max(gaps), 0.3)


if __name__ == "__main__":
    unittest.main()
