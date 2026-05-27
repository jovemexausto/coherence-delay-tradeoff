from __future__ import annotations

import unittest

from useful_memory_horizon.meta_sensing_benchmark import (
    MetaSensingConfig,
    run_meta_sensing_benchmark,
)


class MetaSensingBenchmarkTest(unittest.TestCase):
    def test_multiscale_sensing_improves_moderate_noise(self) -> None:
        rows = run_meta_sensing_benchmark(
            config=MetaSensingConfig(
                steps=48, switch_step=24, trials=16, lag_count=40, lag_reps=8
            ),
            rng_seed=0,
        )
        single_mid = next(
            row
            for row in rows
            if row.sensor_mode == "single" and row.sensor_noise == 0.1
        )
        multi_mid = next(
            row
            for row in rows
            if row.sensor_mode == "multiscale" and row.sensor_noise == 0.1
        )
        self.assertLess(multi_mid.mean_route_delay, single_mid.mean_route_delay)
        self.assertLess(multi_mid.mean_pre_route_cost, single_mid.mean_pre_route_cost)


if __name__ == "__main__":
    unittest.main()
