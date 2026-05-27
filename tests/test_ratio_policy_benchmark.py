from __future__ import annotations

import unittest

from useful_memory_horizon.ratio_policy_benchmark import (
    RatioPolicyBenchmarkConfig,
    run_ratio_policy_benchmark,
)


class RatioPolicyBenchmarkTest(unittest.TestCase):
    def test_benchmark_emits_all_three_policies(self) -> None:
        config = RatioPolicyBenchmarkConfig(
            H_values=(0.75,),
            zeta0_values=(0.01,),
            ramp_values=(2.0e-5,),
            ratio_noise_values=(0.05, 0.3),
            lag_noise_values=(0.05, 0.3),
            steps=48,
            lag_reps=8,
        )
        rows = run_ratio_policy_benchmark(config=config, rng_seed=0)
        self.assertEqual(len(rows), 6)
        self.assertEqual(
            {row.policy for row in rows},
            {"instant_ratio", "persistent_ratio", "lag_geometry"},
        )
        self.assertEqual(sum(row.policy == "lag_geometry" for row in rows), 2)
        self.assertEqual(sum(row.policy == "instant_ratio" for row in rows), 2)
        self.assertEqual(sum(row.policy == "persistent_ratio" for row in rows), 2)
        self.assertTrue(all(row.mean_relative_error >= 0.0 for row in rows))

    def test_ratio_policies_cross_over_with_noise(self) -> None:
        config = RatioPolicyBenchmarkConfig(
            H_values=(0.75,),
            zeta0_values=(0.01,),
            ramp_values=(2.0e-5,),
            ratio_noise_values=(0.05, 0.3),
            lag_noise_values=(0.05,),
            steps=64,
            lag_reps=6,
        )
        rows = run_ratio_policy_benchmark(config=config, rng_seed=0)
        low_noise_instant = next(
            row
            for row in rows
            if row.policy == "instant_ratio" and row.ratio_noise == 0.05
        )
        low_noise_persistent = next(
            row
            for row in rows
            if row.policy == "persistent_ratio" and row.ratio_noise == 0.05
        )
        high_noise_instant = next(
            row
            for row in rows
            if row.policy == "instant_ratio" and row.ratio_noise == 0.3
        )
        high_noise_persistent = next(
            row
            for row in rows
            if row.policy == "persistent_ratio" and row.ratio_noise == 0.3
        )
        self.assertLess(
            low_noise_instant.mean_relative_error,
            low_noise_persistent.mean_relative_error,
        )
        self.assertLess(
            high_noise_persistent.mean_relative_error,
            high_noise_instant.mean_relative_error,
        )


if __name__ == "__main__":
    unittest.main()
