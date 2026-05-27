from __future__ import annotations

import unittest

from useful_memory_horizon.partition_ratio import (
    optimal_horizon_from_ratio,
    ratio_at_optimizer,
    ratio_sweep_rows,
    validity_partition_ratio,
)


class PartitionRatioTest(unittest.TestCase):
    def test_ratio_is_exact_at_optimizer(self) -> None:
        a = 0.5
        H = 0.75
        C_K = 1.0
        C_S = 1.0
        zeta = 0.01
        n_star = ((a * C_K) / (H * C_S * zeta)) ** (1.0 / (a + H))
        rho = validity_partition_ratio(n_star, C_K=C_K, a=a, C_S=C_S, zeta=zeta, H=H)
        self.assertAlmostEqual(rho, ratio_at_optimizer(a, H), places=12)

    def test_inversion_recovers_horizon(self) -> None:
        a = 0.5
        H = 0.75
        C_K = 1.0
        C_S = 1.0
        zeta = 0.01
        n_pi = 7.0
        rho = validity_partition_ratio(n_pi, C_K=C_K, a=a, C_S=C_S, zeta=zeta, H=H)
        n_star = optimal_horizon_from_ratio(n_pi, rho, a=a, H=H)
        direct = ((a * C_K) / (H * C_S * zeta)) ** (1.0 / (a + H))
        self.assertAlmostEqual(n_star, direct, places=12)

    def test_ratio_increases_with_window_size(self) -> None:
        a = 0.5
        H = 0.75
        C_K = 1.0
        C_S = 1.0
        zeta = 0.01
        values = [
            validity_partition_ratio(n, C_K=C_K, a=a, C_S=C_S, zeta=zeta, H=H)
            for n in (4.0, 8.0, 16.0, 32.0)
        ]
        self.assertTrue(all(values[i] < values[i + 1] for i in range(len(values) - 1)))

    def test_ratio_sweep_rows_identifies_side_of_horizon(self) -> None:
        rows = ratio_sweep_rows(
            n_grid=(4.0, 8.0, 16.0, 32.0),
            C_K=1.0,
            a=0.5,
            C_S=1.0,
            zeta=0.01,
            H=0.75,
        )
        self.assertEqual(len(rows), 4)
        self.assertTrue(all(row.ratio > 0.0 for row in rows))
        self.assertEqual({row.side_of_horizon for row in rows}, {"below", "above"})
        self.assertTrue(
            all(
                abs(row.implied_n_star - rows[0].implied_n_star) < 1e-12 for row in rows
            )
        )


if __name__ == "__main__":
    unittest.main()
