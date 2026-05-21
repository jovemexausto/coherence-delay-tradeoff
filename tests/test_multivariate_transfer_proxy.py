from __future__ import annotations

import unittest

from useful_memory_horizon.multivariate_transfer_proxy import (
    quadratic_remainder_bound,
    linearization_residual,
    run_multivariate_transfer_proxy_experiment,
    sample_path,
    window_average,
)


class MultivariateTransferProxyTest(unittest.TestCase):
    def test_proxy_has_interior_u_curve(self) -> None:
        result = run_multivariate_transfer_proxy_experiment()
        windows = [row.window for row in result.rows]
        totals = [row.mean_total_error for row in result.rows]
        best_index = min(range(len(totals)), key=totals.__getitem__)
        self.assertGreater(best_index, 0)
        self.assertLess(best_index, len(totals) - 1)

    def test_drift_grows_with_window_and_residual_is_lower_order(self) -> None:
        result = run_multivariate_transfer_proxy_experiment()
        drift = [row.mean_drift_error for row in result.rows]
        finite = [row.mean_finite_sample_error for row in result.rows]
        residual = [row.mean_linearization_residual for row in result.rows]
        self.assertTrue(all(x < y for x, y in zip(drift, drift[1:])))
        self.assertTrue(all(x > y for x, y in zip(finite, finite[1:])))
        self.assertTrue(all(r < f for r, f in zip(residual, finite)))

    def test_quadratic_remainder_bound_dominates_residual(self) -> None:
        means, samples = sample_path(total_steps=192, H=0.75, zeta=1.6, seed=0)
        target = means[-1]
        window_target = window_average(means, 191, 32)
        sample_target = window_average(samples, 191, 32)
        residual = linearization_residual(sample_target, window_target)
        bound = quadratic_remainder_bound(sample_target, window_target)
        self.assertLessEqual(residual, bound)


if __name__ == "__main__":
    unittest.main()
