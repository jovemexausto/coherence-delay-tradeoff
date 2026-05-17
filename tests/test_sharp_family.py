from __future__ import annotations

import unittest

from useful_memory_horizon.holder_lower_bound_research import Hölder_asymptotic_constant
from useful_memory_horizon.sharp_family import (
    SharpFamilyAuditConfig,
    asymptotic_staleness_constant,
    dirac_uniform_window_staleness,
    run_sharp_family_audit,
    supplement_candidate_constant,
    uniform_window_staleness_constant,
)


class SharpFamilyAuditTest(unittest.TestCase):
    def test_finite_n_staleness_constant_converges_to_closed_form_limit(self) -> None:
        for H in (0.25, 0.5, 0.75, 1.0):
            finite_n = uniform_window_staleness_constant(H, 4096)
            self.assertAlmostEqual(
                finite_n,
                asymptotic_staleness_constant(H),
                delta=8e-4,
            )

    def test_dirac_staleness_matches_constant_formula_exactly(self) -> None:
        zeta = 0.7
        H = 0.75
        n = 64
        expected = zeta * uniform_window_staleness_constant(H, n) * (n**H)
        self.assertAlmostEqual(
            dirac_uniform_window_staleness(zeta, H, n),
            expected,
            places=12,
        )

    def test_numeric_lower_bound_tracks_current_asymptotic_constant(self) -> None:
        result = run_sharp_family_audit(
            SharpFamilyAuditConfig(
                H_values=(0.5, 0.75, 1.0),
                sigma_zeta_ratios=(10_000.0,),
                n_values=(64, 256, 1024),
                max_multiplier=4.0,
            )
        )

        for row in result.lower_bound_rows:
            H = float(row["H"])
            self.assertAlmostEqual(
                float(row["normalized_best"]),
                Hölder_asymptotic_constant(H),
                delta=1.5e-3,
            )

    def test_supplement_candidate_constant_is_not_the_current_witness_constant(self) -> None:
        for H in (0.5, 0.75, 1.0):
            self.assertGreater(
                supplement_candidate_constant(H),
                2.5 * Hölder_asymptotic_constant(H),
            )


if __name__ == "__main__":
    unittest.main()
