from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from scale_consistency.experiments import (
    BoundaryPowerConfig,
    FWLSOracleConfig,
    MisspecificationConfig,
    NoiseRobustnessConfig,
    NullCalibrationConfig,
    RateConstantConfig,
    Sigma0PluginConfig,
)
from scale_consistency.plots import generate_v1_figures
from scale_consistency.report import generate_v1_reports


class ScaleConsistencyReportingTest(unittest.TestCase):
    def test_reports_and_figures_are_generated(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            output_root = Path(tmp_dir)
            kwargs = {
                "null_config": NullCalibrationConfig(
                    L_values=(10,), n_values=(200,), H_values=(0.6,), repetitions=10
                ),
                "fwls_config": FWLSOracleConfig(
                    L_values=(10,), n_values=(200,), repetitions=10
                ),
                "boundary_config": BoundaryPowerConfig(
                    n_values=(200,), c_values=(0.5, 1.0), repetitions=10
                ),
                "rate_config": RateConstantConfig(n_values=(200,), repetitions=10),
                "misspec_config": MisspecificationConfig(
                    amplitudes=(0.0, 0.1), repetitions=10
                ),
                "noise_config": NoiseRobustnessConfig(
                    noise_models=("gaussian", "bounded"), repetitions=10
                ),
                "sigma0_plugin_config": Sigma0PluginConfig(
                    L_values=(10,), n_values=(200,), repetitions=10
                ),
            }
            report_rows = generate_v1_reports(output_root, **kwargs)
            figure_paths = generate_v1_figures(output_root, **kwargs)
            self.assertEqual(
                set(report_rows.keys()),
                {
                    "null",
                    "fwls_oracle",
                    "boundary_power",
                    "rate_constant",
                    "misspecification",
                    "noise_robustness",
                    "sigma0_plugin",
                },
            )
            self.assertTrue(
                (
                    output_root / "csv" / "scale_consistency" / "null_calibration.csv"
                ).exists()
            )
            self.assertTrue(
                (
                    output_root / "tables" / "scale_consistency" / "tab_null_size.tex"
                ).exists()
            )
            for path in figure_paths.values():
                self.assertTrue(path.exists())


if __name__ == "__main__":
    unittest.main()
