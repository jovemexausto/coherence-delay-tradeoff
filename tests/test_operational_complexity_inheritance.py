from __future__ import annotations

import unittest

from useful_memory_horizon.operational_complexity_inheritance import (
    certify_operational_theorem_candidate,
    embedded_fixed_span_complexity_inheritance_ratio,
    embedded_fixed_span_support_axis_bounds,
    iid_mixture_support_axis_bounds,
    iid_mixture_support_covering_upper,
    triangular_window_support_axis_bounds,
    triangular_window_support_covering_upper,
)
from useful_memory_horizon.operational_regime_frontier import map_operational_regime


class OperationalComplexityInheritanceTest(unittest.TestCase):
    def test_embedded_fixed_span_support_bounds_are_identical_for_triangular_and_iid(
        self,
    ) -> None:
        expected = embedded_fixed_span_support_axis_bounds(2, 0.25)
        self.assertEqual(triangular_window_support_axis_bounds(2, 0.25), expected)
        self.assertEqual(iid_mixture_support_axis_bounds(2, 0.25), expected)

    def test_embedded_fixed_span_covering_numbers_match_exactly(self) -> None:
        tri = triangular_window_support_covering_upper(2, 0.25, 0.1)
        iid = iid_mixture_support_covering_upper(2, 0.25, 0.1)
        self.assertAlmostEqual(tri, iid, places=12)
        self.assertAlmostEqual(
            embedded_fixed_span_complexity_inheritance_ratio(2, 0.25, 0.1),
            1.0,
            places=12,
        )

    def test_operational_theorem_candidate_detects_ready_and_nonready_rows(
        self,
    ) -> None:
        rows = map_operational_regime(
            ambient_intrinsic_pairs=((8, 1), (12, 1)),
            epsilons=(0.5, 0.2, 0.1, 0.05),
            sample_sizes=(24, 48, 96, 160),
            seed_count=8,
        )
        ready_rows = []
        nonready_rows = []
        for row in rows:
            candidate = certify_operational_theorem_candidate(
                row=row,
                holder_smoothness_alpha=2.0,
                span=0.25,
            )
            if candidate.theorem_ready:
                ready_rows.append(candidate)
            else:
                nonready_rows.append(candidate)
        self.assertTrue(
            any(c.ambient_dim == 8 and c.intrinsic_dim == 1 for c in ready_rows)
        )
        self.assertTrue(
            any(c.ambient_dim == 12 and c.intrinsic_dim == 1 for c in nonready_rows)
        )


if __name__ == "__main__":
    unittest.main()
