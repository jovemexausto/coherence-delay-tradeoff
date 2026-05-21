from __future__ import annotations

import unittest

from scale_consistency.experiments import (
    BoundaryPowerConfig,
    FWLSOracleConfig,
    NullCalibrationConfig,
    RateConstantConfig,
    run_boundary_power_experiment,
    run_fwls_oracle_experiment,
    run_null_calibration_experiment,
    run_rate_constant_experiment,
)


class ScaleConsistencyExperimentsTest(unittest.TestCase):
    def test_null_calibration_experiment_returns_rows(self) -> None:
        rows = run_null_calibration_experiment(
            NullCalibrationConfig(
                L_values=(10,), n_values=(200,), H_values=(0.6,), repetitions=20
            )
        )
        self.assertEqual(len(rows), 1)
        self.assertGreaterEqual(rows[0].empirical_size, 0.0)
        self.assertLessEqual(rows[0].empirical_size, 1.0)

    def test_fwls_oracle_experiment_returns_rows(self) -> None:
        rows = run_fwls_oracle_experiment(
            FWLSOracleConfig(L_values=(10,), n_values=(200,), repetitions=20)
        )
        self.assertEqual(len(rows), 1)
        self.assertGreaterEqual(rows[0].rmse_h_gap, 0.0)

    def test_boundary_power_experiment_returns_full_grid(self) -> None:
        rows = run_boundary_power_experiment(
            BoundaryPowerConfig(n_values=(200,), c_values=(0.5, 1.0), repetitions=20)
        )
        self.assertEqual(len(rows), 2)
        self.assertTrue(all(0.0 <= row.empirical_power <= 1.0 for row in rows))

    def test_rate_constant_experiment_returns_rows(self) -> None:
        rows = run_rate_constant_experiment(
            RateConstantConfig(n_values=(200, 500), repetitions=20)
        )
        self.assertEqual(len(rows), 2)
        self.assertTrue(all(row.scaled_constant > 0.0 for row in rows))


if __name__ == "__main__":
    unittest.main()
