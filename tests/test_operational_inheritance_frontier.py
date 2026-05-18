from __future__ import annotations

import unittest

from useful_memory_horizon.operational_inheritance_frontier import (
    embedded_fixed_span_operational_iid_constant,
    embedded_fixed_span_support_covering_upper,
    embedded_fixed_span_support_side_lengths,
    operational_inheritance_holds,
)
from useful_memory_horizon.operational_regime_frontier import map_operational_regime


class OperationalInheritanceFrontierTest(unittest.TestCase):
    def test_embedded_fixed_span_support_is_n_invariant(self) -> None:
        self.assertEqual(
            embedded_fixed_span_support_side_lengths(intrinsic_dim=1, span=0.25),
            (2.25,),
        )
        self.assertEqual(
            embedded_fixed_span_support_side_lengths(intrinsic_dim=2, span=0.25),
            (2.25, 2.0),
        )

    def test_operational_iid_constant_depends_on_intrinsic_not_ambient_dimension(
        self,
    ) -> None:
        constant_k1 = embedded_fixed_span_operational_iid_constant(
            intrinsic_dim=1,
            epsilon=0.1,
            span=0.25,
        )
        constant_k2 = embedded_fixed_span_operational_iid_constant(
            intrinsic_dim=2,
            epsilon=0.1,
            span=0.25,
        )
        self.assertGreater(constant_k2, constant_k1)

    def test_operational_iid_constant_worsens_as_epsilon_decreases(self) -> None:
        coarse = embedded_fixed_span_operational_iid_constant(1, 0.5, 0.25)
        fine = embedded_fixed_span_operational_iid_constant(1, 0.05, 0.25)
        self.assertGreater(fine, coarse)

    def test_support_covering_upper_matches_product_structure(self) -> None:
        value = embedded_fixed_span_support_covering_upper(
            intrinsic_dim=2,
            epsilon=0.1,
            span=0.25,
        )
        expected = (1.0 + 2.25 / 0.1) * (1.0 + 2.0 / 0.1)
        self.assertAlmostEqual(value, expected, places=12)

    def test_operational_inheritance_kernel_matches_empirical_stable_and_unstable_pockets(
        self,
    ) -> None:
        rows = map_operational_regime(
            ambient_intrinsic_pairs=((8, 1), (12, 1)),
            epsilons=(0.5, 0.2, 0.1, 0.05),
            sample_sizes=(24, 48, 96, 160),
            seed_count=8,
        )
        stable = [row for row in rows if (row.ambient_dim, row.intrinsic_dim) == (8, 1)]
        mixed = [row for row in rows if (row.ambient_dim, row.intrinsic_dim) == (12, 1)]
        self.assertTrue(
            all(
                operational_inheritance_holds(
                    intrinsic_dim=row.intrinsic_dim,
                    epsilon=row.epsilon,
                    span=0.25,
                    empirical_gap=row.gap,
                )
                for row in stable
            )
        )
        self.assertGreaterEqual(
            sum(
                operational_inheritance_holds(
                    intrinsic_dim=row.intrinsic_dim,
                    epsilon=row.epsilon,
                    span=0.25,
                    empirical_gap=row.gap,
                )
                for row in mixed
            ),
            2,
        )


if __name__ == "__main__":
    unittest.main()
