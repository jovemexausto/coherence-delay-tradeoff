from __future__ import annotations

import unittest

from useful_memory_horizon.policy_frontier_theorem import (
    PolicyFrontierConfig,
    run_policy_frontier_benchmark,
)


class PolicyFrontierTheoremTest(unittest.TestCase):
    def test_policy_frontier_has_positive_structural_delay(self) -> None:
        rows = run_policy_frontier_benchmark(
            config=PolicyFrontierConfig(steps=48, switch_step=24, trials=8),
            rng_seed=0,
        )
        oracle_row = next(
            row
            for row in rows
            if row.sensor_mode == "oracle" and row.sensor_noise == 0.0
        )
        self.assertGreater(oracle_row.mean_route_delay, 0.0)

    def test_multiscale_beats_single_under_moderate_noise(self) -> None:
        rows = run_policy_frontier_benchmark(
            config=PolicyFrontierConfig(steps=48, switch_step=24, trials=8),
            rng_seed=0,
        )
        single = next(
            row
            for row in rows
            if row.sensor_mode == "single" and row.sensor_noise == 0.1
        )
        multi = next(
            row
            for row in rows
            if row.sensor_mode == "multiscale" and row.sensor_noise == 0.1
        )
        self.assertLess(multi.mean_route_delay, single.mean_route_delay)
        self.assertLess(multi.mean_pre_route_cost, single.mean_pre_route_cost)


if __name__ == "__main__":
    unittest.main()
