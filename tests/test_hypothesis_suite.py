from __future__ import annotations

import unittest

from useful_memory_horizon.hypothesis_suite import (
    pre_detection_validity_loss,
    run_hypothesis_suite,
)
from useful_memory_horizon.partition_ratio import (
    optimal_horizon_from_ratio,
    validity_partition_ratio,
)


class HypothesisSuiteTest(unittest.TestCase):
    def test_partition_ratio_is_exact_at_the_optimizer(self) -> None:
        a = 0.5
        H = 0.75
        C_K = 1.0
        C_S = 1.0
        zeta = 0.01
        n_star = ((a * C_K) / (H * C_S * zeta)) ** (1.0 / (a + H))
        rho = validity_partition_ratio(n_star, C_K=C_K, a=a, C_S=C_S, zeta=zeta, H=H)
        self.assertAlmostEqual(rho, a / H, places=12)

    def test_partition_ratio_inversion_recovers_optimal_horizon(self) -> None:
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

    def test_pre_detection_validity_loss_is_positive(self) -> None:
        loss = pre_detection_validity_loss(
            C_K=1.0,
            a=0.5,
            C_S=1.0,
            zeta_v=0.01,
            r=0.01,
            delta=0.1,
            n_pi=7.0,
        )
        self.assertGreaterEqual(loss, 0.0)

    def test_hypothesis_suite_smoke_contains_all_records(self) -> None:
        report = run_hypothesis_suite(fast=True)
        ids = {record.hypothesis for record in report.records}
        self.assertEqual(
            ids, {f"H{i}" for i in range(1, 30)} | {"H-CP1", "H-CP3", "H-CP5"}
        )
        self.assertTrue(
            all(
                record.status in {"supported", "mixed", "not_supported", "open"}
                for record in report.records
            )
        )


if __name__ == "__main__":
    unittest.main()
