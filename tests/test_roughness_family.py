from __future__ import annotations

import unittest

from experiments.roughness_family import (
    RoughnessScalingConfig,
    run_roughness_scaling_experiment,
)


class RoughnessFamilyTest(unittest.TestCase):
    def test_empirical_slopes_follow_theory_direction(self) -> None:
        result = run_roughness_scaling_experiment(
            RoughnessScalingConfig(
                H_values=(0.5, 1.0),
                zeta_values=(0.004, 0.007, 0.012, 0.02),
                window_sizes=tuple(range(10, 181, 10)),
                seeds=(0, 1, 2),
                replicas=1500,
            )
        )

        for empirical, theory in zip(
            result.fitted_slopes, result.theory_slopes, strict=True
        ):
            self.assertLess(empirical, 0.0)
            self.assertLess(abs(empirical - theory), 0.35)


if __name__ == "__main__":
    unittest.main()
