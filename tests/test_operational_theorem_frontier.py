from __future__ import annotations

import math
import unittest

from useful_memory_horizon.operational_regime_frontier import map_operational_regime
from useful_memory_horizon.operational_theorem_frontier import (
    certify_operational_region,
    certify_structural_operational_region,
    finite_class_noniid_empirical_process_bound,
    finite_class_noniid_empirical_process_constant,
    maximal_theorem_ready_epsilon_band,
    triangular_operational_inheritance_bound,
)


class OperationalTheoremFrontierTest(unittest.TestCase):
    def test_finite_class_noniid_process_preserves_root_n(self) -> None:
        bound_n = finite_class_noniid_empirical_process_bound(100, class_size=32)
        bound_4n = finite_class_noniid_empirical_process_bound(400, class_size=32)
        self.assertAlmostEqual(bound_4n, 0.5 * bound_n, places=12)

    def test_finite_class_noniid_constant_grows_with_class_size(self) -> None:
        small = finite_class_noniid_empirical_process_constant(8)
        large = finite_class_noniid_empirical_process_constant(64)
        self.assertGreater(large, small)

    def test_triangular_operational_inheritance_bound_preserves_iid_exponent(
        self,
    ) -> None:
        bound_n = triangular_operational_inheritance_bound(
            sample_size=100,
            intrinsic_dim=1,
            epsilon=0.2,
            span=0.25,
            inheritance_factor=1.2,
        )
        bound_4n = triangular_operational_inheritance_bound(
            sample_size=400,
            intrinsic_dim=1,
            epsilon=0.2,
            span=0.25,
            inheritance_factor=1.2,
        )
        self.assertAlmostEqual(bound_4n, 0.5 * bound_n, places=12)

    def test_operational_region_certificates_detect_stable_and_mixed_pairs(
        self,
    ) -> None:
        rows = map_operational_regime(
            ambient_intrinsic_pairs=((8, 1), (12, 1), (12, 2)),
            epsilons=(0.5, 0.2, 0.1, 0.05),
            sample_sizes=(24, 48, 96, 160),
            seed_count=8,
        )
        stable_81 = certify_operational_region(rows, ambient_dim=8, intrinsic_dim=1)
        mixed_121 = certify_operational_region(rows, ambient_dim=12, intrinsic_dim=1)
        stable_122 = certify_operational_region(rows, ambient_dim=12, intrinsic_dim=2)
        self.assertTrue(stable_81.stable)
        self.assertFalse(mixed_121.stable)
        self.assertTrue(stable_122.stable)
        self.assertAlmostEqual(stable_81.useful_fraction, 1.0, places=12)
        self.assertGreaterEqual(stable_122.useful_fraction, 0.75)

    def test_structural_operational_region_certificate_combines_all_frontier_checks(
        self,
    ) -> None:
        rows = map_operational_regime(
            ambient_intrinsic_pairs=((8, 2), (12, 1)),
            epsilons=(0.5, 0.2, 0.1, 0.05),
            sample_sizes=(24, 48, 96, 160),
            seed_count=24,
        )
        ready = certify_structural_operational_region(
            rows=rows,
            ambient_dim=8,
            intrinsic_dim=2,
            epsilon_max=0.5,
            holder_smoothness_alpha=2.0,
            span=0.25,
        )
        nonready = certify_structural_operational_region(
            rows=rows,
            ambient_dim=12,
            intrinsic_dim=1,
            epsilon_max=0.5,
            holder_smoothness_alpha=2.0,
            span=0.25,
        )
        self.assertTrue(ready.exact_complexity_inheritance)
        self.assertTrue(ready.parametric_region_holds)
        self.assertTrue(ready.empirical_band_holds)
        self.assertTrue(ready.theorem_ready)
        self.assertGreater(ready.carrier_lower_bound, 0.4)
        self.assertFalse(nonready.empirical_band_holds)
        self.assertFalse(nonready.theorem_ready)

    def test_maximal_theorem_ready_band_recovers_structural_frontier(self) -> None:
        rows = map_operational_regime(
            ambient_intrinsic_pairs=((8, 1), (8, 2), (12, 1), (12, 2)),
            epsilons=(0.5, 0.2, 0.1, 0.05),
            sample_sizes=(24, 48, 96, 160),
            seed_count=24,
        )
        self.assertGreaterEqual(
            maximal_theorem_ready_epsilon_band(
                rows,
                ambient_dim=8,
                intrinsic_dim=1,
                holder_smoothness_alpha=2.0,
                span=0.25,
            ),
            0.5,
        )
        self.assertGreaterEqual(
            maximal_theorem_ready_epsilon_band(
                rows,
                ambient_dim=8,
                intrinsic_dim=2,
                holder_smoothness_alpha=2.0,
                span=0.25,
            ),
            0.5,
        )
        self.assertLessEqual(
            maximal_theorem_ready_epsilon_band(
                rows,
                ambient_dim=12,
                intrinsic_dim=1,
                holder_smoothness_alpha=2.0,
                span=0.25,
            ),
            0.2,
        )
        self.assertGreaterEqual(
            maximal_theorem_ready_epsilon_band(
                rows,
                ambient_dim=12,
                intrinsic_dim=2,
                holder_smoothness_alpha=2.0,
                span=0.25,
            ),
            0.5,
        )

    def test_theorem_ready_band_requires_parametric_dual_smoothness(self) -> None:
        rows = map_operational_regime(
            ambient_intrinsic_pairs=((8, 2),),
            epsilons=(0.5, 0.2, 0.1, 0.05),
            sample_sizes=(24, 48, 96, 160),
            seed_count=24,
        )
        self.assertIsNone(
            maximal_theorem_ready_epsilon_band(
                rows,
                ambient_dim=8,
                intrinsic_dim=2,
                holder_smoothness_alpha=1.0,
                span=0.25,
            )
        )


if __name__ == "__main__":
    unittest.main()
