from __future__ import annotations

import math
import unittest


def dirac_holder_staleness(zeta: float, H: float, n: int) -> float:
    return math.sqrt(sum((zeta * (j**H)) ** 2 for j in range(n)) / n)


class HolderStalenessBoundTest(unittest.TestCase):
    def test_dirac_path_saturates_averaged_coupling_bound(self) -> None:
        zeta = 0.7
        for H in (0.35, 0.5, 0.75, 1.0):
            for n in (8, 16, 32, 64):
                exact_w2 = dirac_holder_staleness(zeta, H, n)
                coupling_bound = math.sqrt(
                    sum((zeta * (j**H)) ** 2 for j in range(n)) / n
                )
                self.assertAlmostEqual(exact_w2, coupling_bound, places=12)

    def test_holder_staleness_has_n_to_the_h_scaling(self) -> None:
        zeta = 1.0
        for H in (0.35, 0.5, 0.75, 1.0):
            values = [dirac_holder_staleness(zeta, H, n) for n in (32, 64, 128, 256)]
            logs_n = [math.log(n) for n in (32, 64, 128, 256)]
            logs_v = [math.log(v) for v in values]
            mean_x = sum(logs_n) / len(logs_n)
            mean_y = sum(logs_v) / len(logs_v)
            numerator = sum(
                (x - mean_x) * (y - mean_y) for x, y in zip(logs_n, logs_v, strict=True)
            )
            denominator = sum((x - mean_x) ** 2 for x in logs_n)
            slope = numerator / denominator
            self.assertAlmostEqual(slope, H, delta=0.03)


if __name__ == "__main__":
    unittest.main()
