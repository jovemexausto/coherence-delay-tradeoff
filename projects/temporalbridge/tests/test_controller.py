from __future__ import annotations

import sys
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "code"))
sys.path.insert(0, str(PROJECT_ROOT.parent / "scale-consistency" / "code"))

from temporalbridge.core import (  # noqa: E402
    ControllerParams,
    ValidityState,
    calibrate_alarms,
    detect_alarms,
    validity_controller,
)
from temporalbridge.benchmarks import (  # noqa: E402
    run_controller_benchmark,
    simulate_grid,
    run_controller_grid_benchmark,
    run_controller_monte_carlo,
    run_controller_sequential_benchmark,
)


class TemporalBridgeControllerTest(unittest.TestCase):
    def test_controller_benchmark_runs(self) -> None:
        result = run_controller_benchmark(rng_seed=123, bootstrap_method="wild")
        self.assertEqual(len(result["rows"]), 3)
        self.assertGreaterEqual(result["accuracy"], 0.0)
        self.assertLessEqual(result["accuracy"], 1.0)

    def test_controller_grid_benchmark_runs(self) -> None:
        result = run_controller_grid_benchmark(rng_seed=123, bootstrap_method="wild")
        self.assertEqual(len(result["rows"]), 7)
        self.assertGreaterEqual(result["accuracy"], 0.0)
        self.assertLessEqual(result["accuracy"], 1.0)

    def test_controller_monte_carlo_runs(self) -> None:
        result = run_controller_monte_carlo(
            repetitions=2,
            bootstrap_method="wild",
            bootstrap_repetitions=10,
            rng_seed=123,
        )
        self.assertTrue(result["rows"])
        self.assertTrue(result["aggregated"])

    def test_controller_sequential_benchmark_runs(self) -> None:
        result = run_controller_sequential_benchmark(
            repetitions=2,
            bootstrap_method="wild",
            bootstrap_repetitions=10,
            rng_seed=123,
        )
        self.assertTrue(result["rows"])
        self.assertIn("mean_tau_valid", result["rows"][0])
        self.assertIn("mean_delay_gap", result["rows"][0])
        self.assertIn("mean_masking_index", result["rows"][0])
        self.assertTrue(result["trajectory_rows"])
        self.assertIn("tau_valid", result["trajectory_rows"][0])
        self.assertIn("delay_gap", result["trajectory_rows"][0])
        self.assertIn("masking_index", result["trajectory_rows"][0])

    def test_controller_cost_grid_runs(self) -> None:
        result = simulate_grid(
            lambda0_values=(0.0, 0.5),
            lambda1_values=(0.0, 1.0),
            repetitions=1,
            bootstrap_method="wild",
            bootstrap_repetitions=5,
            rng_seed=123,
        )
        self.assertEqual(len(result["rows"]), 20)
        self.assertEqual(len(result["cell_summary"]), 4)
        self.assertIn("best_policy", result["cell_summary"][0])

    def test_controller_cost_grid_supports_truth_h_sweep(self) -> None:
        result = simulate_grid(
            lambda0_values=(0.0,),
            lambda1_values=(0.0,),
            truth_h_values=(0.3, 0.7),
            repetitions=1,
            bootstrap_method="wild",
            bootstrap_repetitions=5,
            rng_seed=123,
        )
        self.assertEqual(len(result["cell_summary"]), 2)
        self.assertEqual({row["truth_h"] for row in result["cell_summary"]}, {0.3, 0.7})

    def test_calibrate_and_detect_alarms(self) -> None:
        diagnostics = {
            "KL_residual": [0.1, 0.2, 0.7, 0.8],
            "KL_standardized": [0.1, 0.1, 0.2, 0.3],
        }
        thresholds = calibrate_alarms({}, {"method": "wild"}, diagnostics)
        detected = detect_alarms(
            diagnostics, thresholds["thresholds"], persistence_windows=1
        )
        self.assertIn("KL_residual", detected["alarms"])
        self.assertIn("alarm_summary", detected)

    def test_validity_controller_prefers_n_star_when_precise(self) -> None:
        state = ValidityState(
            n_star=100.0,
            ci_n_star=(99.3, 100.7),
            H=0.6,
            ci_H=(0.595, 0.605),
            identifiability_score=0.8,
            diagnostics={"KL_residual": 0.1},
            diagnostic_thresholds={"KL_residual": 0.2},
            alarm_persistence=0,
        )
        decision = validity_controller(state, ControllerParams())
        self.assertEqual(decision.action, "use_n_star")

    def test_validity_controller_holds_when_ci_is_wide(self) -> None:
        state = ValidityState(
            n_star=100.0,
            ci_n_star=(96.0, 104.0),
            H=0.6,
            ci_H=(0.57, 0.63),
            identifiability_score=0.8,
            diagnostics={"KL_residual": 0.1, "KL_standardized": 0.1},
            diagnostic_thresholds={"KL_residual": 0.2, "KL_standardized": 0.2},
            alarm_persistence=0,
        )
        decision = validity_controller(state, ControllerParams())
        self.assertEqual(decision.action, "hold")

    def test_validity_controller_alarms_when_threshold_is_exceeded(self) -> None:
        state = ValidityState(
            n_star=100.0,
            ci_n_star=(90.0, 120.0),
            H=0.6,
            ci_H=(0.5, 0.7),
            identifiability_score=0.8,
            diagnostics={"KL_residual": 0.4, "KL_standardized": 0.5},
            diagnostic_thresholds={"KL_residual": 0.2, "KL_standardized": 0.2},
            alarm_persistence=1,
        )
        decision = validity_controller(state, ControllerParams(persistence_required=1))
        self.assertEqual(decision.action, "alarm")


if __name__ == "__main__":
    unittest.main()
