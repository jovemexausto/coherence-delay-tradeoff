from __future__ import annotations

import unittest

from useful_memory_horizon.operational_regime_frontier import map_operational_regime


class OperationalRegimeFrontierTest(unittest.TestCase):
    def test_operational_regime_map_marks_useful_pairs(self) -> None:
        rows = map_operational_regime(
            ambient_intrinsic_pairs=((8, 1), (8, 2), (12, 2)),
            epsilons=(0.5, 0.2, 0.1, 0.05),
            sample_sizes=(24, 48, 96, 160),
            seed_count=8,
        )
        self.assertEqual(len(rows), 12)
        useful_rows = [row for row in rows if row.useful]
        self.assertGreaterEqual(len(useful_rows), 10)

    def test_operational_regime_map_detects_stronger_and_weaker_settings(self) -> None:
        rows = map_operational_regime(
            ambient_intrinsic_pairs=((8, 1), (12, 1)),
            epsilons=(0.5, 0.2, 0.1, 0.05),
            sample_sizes=(24, 48, 96, 160),
            seed_count=8,
        )
        pair_81 = [
            row for row in rows if (row.ambient_dim, row.intrinsic_dim) == (8, 1)
        ]
        pair_121 = [
            row for row in rows if (row.ambient_dim, row.intrinsic_dim) == (12, 1)
        ]
        self.assertTrue(all(row.useful for row in pair_81))
        self.assertGreaterEqual(sum(row.useful for row in pair_121), 2)


if __name__ == "__main__":
    unittest.main()
