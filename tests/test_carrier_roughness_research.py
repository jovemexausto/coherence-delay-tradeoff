from __future__ import annotations

import unittest

import numpy as np

from useful_memory_horizon.carrier_roughness_research import (
    CarrierRoughnessResearchConfig,
    estimate_log_slope,
    joint_horizon_exponent,
    joint_minimum_error_exponents,
    run_carrier_roughness_research,
)


class CarrierRoughnessResearchTest(unittest.TestCase):
    def test_joint_horizon_exponent_matches_balance_law(self) -> None:
        self.assertAlmostEqual(joint_horizon_exponent(0.5, 1.0), 2.0 / 3.0)
        self.assertAlmostEqual(joint_horizon_exponent(0.25, 0.5), 4.0 / 3.0)

    def test_joint_minimum_error_exponents_sum_to_one(self) -> None:
        c_power, zeta_power = joint_minimum_error_exponents(0.5, 1.0)
        self.assertAlmostEqual(c_power, 2.0 / 3.0)
        self.assertAlmostEqual(zeta_power, 1.0 / 3.0)
        self.assertAlmostEqual(c_power + zeta_power, 1.0)

    def test_estimate_log_slope_recovers_power_law(self) -> None:
        sample_sizes = [16, 32, 64, 128]
        values = [n ** (-0.4) for n in sample_sizes]
        slope = estimate_log_slope(sample_sizes=np.asarray(sample_sizes), values=values)
        self.assertAlmostEqual(slope, -0.4, places=8)

    def test_small_research_run_returns_expected_experiments(self) -> None:
        result = run_carrier_roughness_research(
            CarrierRoughnessResearchConfig(
                raw_dims=(1,),
                ambient_intrinsic_pairs=((4, 1),),
                raw_sample_sizes=(8, 16),
                raw_seed_count=2,
                triangular_dims=(1,),
                H_values=(0.5,),
                fixed_spans=(0.25,),
                span_growth_fractions=(0.5,),
                span_growth_base=0.25,
                fixed_zeta=0.04,
                triangular_sample_sizes=(16, 32),
                triangular_seed_count=2,
                sinkhorn_epsilons=(0.2,),
                sinkhorn_sample_sizes=(12, 24),
                sinkhorn_seed_count=2,
            )
        )
        experiments = {row["experiment"] for row in result.summary_rows}
        self.assertEqual(
            experiments,
            {
                "raw-iid",
                "intrinsic-iid",
                "triangular-fixed-span",
                "triangular-growing-span",
                "triangular-span-growth",
                "sinkhorn-fixed-span",
            },
        )
        self.assertTrue(result.curve_rows)


if __name__ == "__main__":
    unittest.main()
