from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from scale_consistency.bridge_runner import (
    build_bridge_hetero_plan,
    run_bridge_bootstrap_coverage_experiment,
    run_bridge_suite,
)
from scale_consistency.horizon_bridge import (
    BridgeMisspecificationConfig,
    BridgeRecoveryConfig,
)


class BridgeRunnerTest(unittest.TestCase):
    def test_build_bridge_hetero_plan_includes_strong_modes(self) -> None:
        recovery_configs, misspec_configs = build_bridge_hetero_plan()
        self.assertTrue(recovery_configs)
        self.assertTrue(misspec_configs)
        self.assertTrue(
            any(cfg.heteroskedastic_mode == "power" for cfg in misspec_configs)
        )
        self.assertTrue(
            any(cfg.heteroskedastic_mode == "jump" for cfg in misspec_configs)
        )

    def test_run_bridge_suite_writes_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            output_root = Path(tmp_dir)
            summary = run_bridge_suite(
                output_root,
                recovery_configs=[
                    BridgeRecoveryConfig(
                        lags=20,
                        n_values=(200,),
                        H_values=(0.6,),
                        zeta_values=(1.0,),
                        sigma0_values=(0.5,),
                        repetitions=5,
                        bootstrap_repetitions=5,
                        seed=123,
                    )
                ],
                misspec_configs=[
                    BridgeMisspecificationConfig(
                        lags=20,
                        n=200,
                        H=0.6,
                        zeta=1.0,
                        sigma0=0.5,
                        amplitudes=(0.0, 0.1),
                        kinds=("sinusoid",),
                        repetitions=5,
                        seed=321,
                    )
                ],
                label="smoke",
            )

            self.assertTrue(summary["recovery_rows"])
            self.assertTrue(summary["misspecification_rows"])
            self.assertEqual(summary["recovery_rows"][0]["lag_count"], 20)
            self.assertEqual(summary["misspecification_rows"][0]["lag_count"], 20)
            self.assertTrue(
                (
                    output_root / "csv" / "horizon_bridge" / "bridge_recovery_smoke.csv"
                ).exists()
            )
            self.assertTrue(
                (
                    output_root
                    / "tables"
                    / "horizon_bridge"
                    / "bridge_suite_smoke.json"
                ).exists()
            )
            self.assertTrue(
                (
                    output_root
                    / "figures"
                    / "horizon_bridge"
                    / "fig_bridge_recovery_smoke.pdf"
                ).exists()
            )

    def test_run_bridge_bootstrap_coverage_experiment_writes_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            output_root = Path(tmp_dir)
            summary = run_bridge_bootstrap_coverage_experiment(
                output_root,
                repetitions=3,
                bootstrap_repetitions=10,
            )
            self.assertTrue(summary["rows"])
            self.assertTrue(summary["aggregated"])
            self.assertTrue(
                (
                    output_root
                    / "csv"
                    / "horizon_bridge"
                    / "bridge_bootstrap_coverage.csv"
                ).exists()
            )
            self.assertTrue(
                (
                    output_root
                    / "tables"
                    / "horizon_bridge"
                    / "bridge_bootstrap_coverage.json"
                ).exists()
            )


if __name__ == "__main__":
    unittest.main()
