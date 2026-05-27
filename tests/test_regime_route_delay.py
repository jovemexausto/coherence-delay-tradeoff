from __future__ import annotations

import unittest

from useful_memory_horizon.regime_route_delay import (
    RegimeRouteDelayConfig,
    run_regime_route_delay_benchmark,
)


class RegimeRouteDelayTest(unittest.TestCase):
    def test_route_delay_is_positive_under_perfect_sensing(self) -> None:
        rows = run_regime_route_delay_benchmark(
            config=RegimeRouteDelayConfig(trials=16, steps=80, switch_step=40),
            sensor_noise_levels=(0.0,),
            rng_seed=0,
        )
        self.assertGreater(rows[0].mean_route_delay, 0.0)

    def test_route_delay_and_cost_increase_with_sensor_noise(self) -> None:
        rows = run_regime_route_delay_benchmark(
            config=RegimeRouteDelayConfig(trials=16, steps=80, switch_step=40),
            sensor_noise_levels=(0.0, 0.3),
            rng_seed=0,
        )
        low, high = rows
        self.assertLess(low.mean_route_delay, high.mean_route_delay)
        self.assertLess(low.mean_pre_route_cost, high.mean_pre_route_cost)


if __name__ == "__main__":
    unittest.main()
