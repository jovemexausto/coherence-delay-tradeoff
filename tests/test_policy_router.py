from __future__ import annotations

import unittest

from useful_memory_horizon.policy_router import (
    PolicyRouterConfig,
    run_policy_router_benchmark,
)


class PolicyRouterTest(unittest.TestCase):
    def test_policy_router_benchmark_emits_rows(self) -> None:
        config = PolicyRouterConfig(
            H_values=(0.75,),
            zeta0_values=(0.01,),
            ramp_values=(2.0e-5,),
            noise_values=(0.05, 0.3),
            steps=16,
            lag_reps=4,
            lag_count=24,
        )
        rows = run_policy_router_benchmark(config=config, rng_seed=0)
        self.assertEqual(len(rows), 8)
        self.assertTrue(all(row.mean_relative_error >= 0.0 for row in rows))
        self.assertEqual(
            {row.policy for row in rows},
            {"instant_ratio", "persistent_ratio", "lag_geometry", "regime_router"},
        )


if __name__ == "__main__":
    unittest.main()
