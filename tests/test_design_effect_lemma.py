from __future__ import annotations

import unittest

from useful_memory_horizon.glue_theorem_minimal import (
    MinimalGlueConfig,
    run_minimal_glue_research,
)


class DesignEffectLemmaTest(unittest.TestCase):
    def test_fixed_span_bounded_support_has_root_n_in_both_designs(self) -> None:
        result = run_minimal_glue_research(
            MinimalGlueConfig(
                support_radius=1.0,
                fixed_span=0.5,
                n_values=(25, 50, 100, 200),
                replications=24,
                quantile_grid_size=256,
            )
        )
        row = result.summary_rows[0]
        tri_rate = float(row["tri_rate_a"])
        iid_rate = float(row["iid_rate_a"])
        rate_gap = float(row["rate_gap"])
        tri_over_iid_last = float(row["tri_over_iid_last"])
        self.assertGreater(tri_rate, 0.45)
        self.assertGreater(iid_rate, 0.40)
        self.assertLess(rate_gap, 0.10)
        self.assertLess(abs(tri_over_iid_last - 1.0), 0.15)

    def test_support_radius_sweep_keeps_constant_level_gap(self) -> None:
        ratios = []
        for radius in (0.5, 1.0, 2.0):
            result = run_minimal_glue_research(
                MinimalGlueConfig(
                    support_radius=radius,
                    fixed_span=0.5,
                    n_values=(25, 50, 100, 200),
                    replications=16,
                    quantile_grid_size=256,
                )
            )
            ratios.append(float(result.summary_rows[0]["tri_over_iid_last"]))
        self.assertTrue(all(0.8 < ratio < 1.2 for ratio in ratios))


if __name__ == "__main__":
    unittest.main()
