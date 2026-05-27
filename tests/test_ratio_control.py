from __future__ import annotations

import unittest

from useful_memory_horizon.ratio_control import (
    RatioControlConfig,
    run_ratio_control_benchmark,
)


class RatioControlTest(unittest.TestCase):
    def test_persistent_controller_beats_instant_under_high_noise(self) -> None:
        config = RatioControlConfig(steps=256, noise_sigma=0.3, ramp=2.0e-5, zeta0=0.01)
        result = run_ratio_control_benchmark(config=config, rng_seed=0)
        self.assertLess(
            result["persistent"]["mean_relative_error"],
            result["instant"]["mean_relative_error"],
        )
        self.assertLess(
            result["persistent"]["mean_abs_log_update"],
            result["instant"]["mean_abs_log_update"],
        )

    def test_instant_controller_beats_persistent_under_low_noise(self) -> None:
        config = RatioControlConfig(
            steps=256, noise_sigma=0.05, ramp=2.0e-5, zeta0=0.01
        )
        result = run_ratio_control_benchmark(config=config, rng_seed=0)
        self.assertLess(
            result["instant"]["mean_relative_error"],
            result["persistent"]["mean_relative_error"],
        )
        self.assertLess(
            result["instant"]["mean_relative_error"],
            result["hold"]["mean_relative_error"],
        )


if __name__ == "__main__":
    unittest.main()
