from __future__ import annotations

import math
import unittest

from useful_memory_horizon.operational_carrier_kernel import (
    mid_covering_number_upper,
    sinkhorn_mid_iid_benchmark,
    sinkhorn_mid_iid_constant,
    sinkhorn_operational_horizon_exponent,
    sinkhorn_operational_horizon_scale,
)


class OperationalCarrierKernelTest(unittest.TestCase):
    def test_mid_covering_number_scales_with_intrinsic_dimension(self) -> None:
        value_k1 = mid_covering_number_upper(1, 0.1)
        value_k2 = mid_covering_number_upper(2, 0.1)
        self.assertGreater(value_k2, value_k1)
        self.assertAlmostEqual(value_k2, value_k1 * value_k1, places=10)

    def test_sinkhorn_mid_iid_benchmark_is_parametric_in_sample_size(self) -> None:
        value_n = sinkhorn_mid_iid_benchmark(100, intrinsic_dim=1, epsilon=0.2)
        value_4n = sinkhorn_mid_iid_benchmark(400, intrinsic_dim=1, epsilon=0.2)
        self.assertAlmostEqual(value_4n, 0.5 * value_n, places=12)

    def test_sinkhorn_mid_iid_constant_worsens_with_smaller_epsilon(self) -> None:
        coarse = sinkhorn_mid_iid_constant(intrinsic_dim=2, epsilon=0.5)
        fine = sinkhorn_mid_iid_constant(intrinsic_dim=2, epsilon=0.1)
        self.assertGreater(fine, coarse)

    def test_sinkhorn_mid_iid_constant_worsens_with_intrinsic_dimension(self) -> None:
        k1 = sinkhorn_mid_iid_constant(intrinsic_dim=1, epsilon=0.1)
        k2 = sinkhorn_mid_iid_constant(intrinsic_dim=2, epsilon=0.1)
        self.assertGreater(k2, k1)

    def test_sinkhorn_operational_horizon_exponent_matches_canonical_a_half(
        self,
    ) -> None:
        self.assertAlmostEqual(sinkhorn_operational_horizon_exponent(1.0), 2.0 / 3.0)
        self.assertAlmostEqual(sinkhorn_operational_horizon_exponent(0.5), 1.0)

    def test_sinkhorn_operational_horizon_scale_tracks_constant_ratio(self) -> None:
        H = 1.0
        roughness_budget = 0.01
        n_eps_large = sinkhorn_operational_horizon_scale(H, roughness_budget, 1, 0.5)
        n_eps_small = sinkhorn_operational_horizon_scale(H, roughness_budget, 1, 0.1)
        predicted_ratio = (
            sinkhorn_mid_iid_constant(1, 0.1) / sinkhorn_mid_iid_constant(1, 0.5)
        ) ** (2.0 / 3.0)
        self.assertAlmostEqual(n_eps_small / n_eps_large, predicted_ratio, places=12)


if __name__ == "__main__":
    unittest.main()
