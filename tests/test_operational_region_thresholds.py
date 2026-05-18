from __future__ import annotations

import unittest

from useful_memory_horizon.operational_regime_frontier import map_operational_regime
from useful_memory_horizon.operational_region_thresholds import (
    certify_operational_epsilon_band,
    maximal_stable_epsilon_band,
)


class OperationalRegionThresholdsTest(unittest.TestCase):
    def test_operational_epsilon_band_certificates_capture_small_epsilon_stability(
        self,
    ) -> None:
        rows = map_operational_regime(
            ambient_intrinsic_pairs=((8, 2), (12, 1)),
            epsilons=(0.5, 0.2, 0.1, 0.05),
            sample_sizes=(24, 48, 96, 160),
            seed_count=24,
        )
        cert_82_small = certify_operational_epsilon_band(rows, 8, 2, 0.2)
        cert_82_all = certify_operational_epsilon_band(rows, 8, 2, 0.5)
        cert_121_small = certify_operational_epsilon_band(rows, 12, 1, 0.2)
        cert_121_all = certify_operational_epsilon_band(rows, 12, 1, 0.5)
        self.assertTrue(cert_82_small.all_useful)
        self.assertTrue(cert_82_all.all_useful)
        self.assertTrue(cert_121_small.all_useful)
        self.assertFalse(cert_121_all.all_useful)

    def test_maximal_stable_epsilon_band_detects_pairwise_thresholds(self) -> None:
        rows = map_operational_regime(
            ambient_intrinsic_pairs=((8, 1), (8, 2), (12, 1), (12, 2)),
            epsilons=(0.5, 0.2, 0.1, 0.05),
            sample_sizes=(24, 48, 96, 160),
            seed_count=24,
        )
        band_81 = maximal_stable_epsilon_band(rows, 8, 1)
        band_82 = maximal_stable_epsilon_band(rows, 8, 2)
        band_121 = maximal_stable_epsilon_band(rows, 12, 1)
        band_122 = maximal_stable_epsilon_band(rows, 12, 2)
        self.assertGreaterEqual(band_81, 0.5)
        self.assertGreaterEqual(band_82, 0.2)
        self.assertLessEqual(band_121, 0.2)
        self.assertGreaterEqual(band_122, 0.5)


if __name__ == "__main__":
    unittest.main()
