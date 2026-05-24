from __future__ import annotations

import unittest

from useful_memory_horizon.invalidity_gap import (
    InvalidityGapConfig,
    build_invalidity_gap_sweep_configs,
    run_detector_comparison,
    run_invalidity_gap_experiment,
    run_invalidity_gap_sweep,
)


class InvalidityGapTest(unittest.TestCase):
    def test_invalidity_gap_is_positive_in_ramp_regime(self) -> None:
        result = run_invalidity_gap_experiment(
            InvalidityGapConfig(
                seeds=(0, 1, 2),
                steps=2400,
                warmup=250,
                phase_lengths=(700, 900, 800),
                low_drift=0.003,
                high_drift=0.03,
                operating_window=180,
                detector_delta=0.002,
                detector_deltas=(0.002,),
                persistence=25,
            )
        )

        summary = result.summaries[0]
        self.assertGreater(summary.mean_gap, 0.0)
        self.assertGreater(summary.positive_gap_rate, 0.5)

    def test_invalidity_gap_persists_across_detector_families(self) -> None:
        summaries = run_detector_comparison()
        for summary in summaries:
            self.assertGreater(summary.mean_gap, 0.0)
            self.assertGreater(summary.positive_gap_rate, 0.5)

    def test_invalidity_gap_supports_residual_detector_input(self) -> None:
        result = run_invalidity_gap_experiment(
            InvalidityGapConfig(
                seeds=(0, 1),
                steps=1800,
                warmup=200,
                phase_lengths=(500, 700, 600),
                low_drift=0.002,
                high_drift=0.02,
                operating_window=160,
                detector_input="absolute_residual",
                detector_deltas=(0.002,),
                persistence=20,
            )
        )
        summary = result.summaries[0]
        self.assertEqual(summary.detector_input, "absolute_residual")
        self.assertGreaterEqual(summary.gap_q90, summary.gap_q10)
        self.assertGreaterEqual(summary.detection_rate, 0.0)
        self.assertLessEqual(summary.detection_rate, 1.0)
        self.assertGreaterEqual(summary.mean_pre_detection_excess_area, 0.0)

    def test_invalidity_gap_builds_and_runs_sweep_grid(self) -> None:
        configs = build_invalidity_gap_sweep_configs(
            detector_names=("adwin",),
            detector_inputs=("observation", "absolute_residual"),
            operating_windows=(160, 220),
            detector_deltas=(0.002,),
            seeds=(0, 1),
            steps=1200,
            phase_lengths=(400, 400, 400),
            low_drift=0.002,
            high_drift=0.02,
            persistence=20,
        )
        self.assertEqual(len(configs), 4)
        results = run_invalidity_gap_sweep(configs[:1])
        self.assertEqual(len(results), 1)
        self.assertGreaterEqual(
            results[0].summaries[0].mean_pre_detection_excess_area, 0.0
        )


if __name__ == "__main__":
    unittest.main()
