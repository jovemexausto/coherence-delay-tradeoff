from __future__ import annotations

import unittest

from useful_memory_horizon.invalidity_gap import (
    InvalidityGapConfig,
    build_invalidity_gap_sweep_configs,
    calibrate_detector_delta,
    estimate_null_alarm_rate,
    run_detector_comparison,
    run_calibrated_delay_frontier,
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

    def test_invalidity_gap_supports_holder_exponent_below_one(self) -> None:
        result = run_invalidity_gap_experiment(
            InvalidityGapConfig(
                seeds=(0, 1, 2),
                steps=2400,
                warmup=250,
                phase_lengths=(700, 900, 800),
                low_drift=0.003,
                high_drift=0.03,
                holder_exponent=0.6,
                operating_window=180,
                detector_name="adwin",
                detector_input="observation",
                detector_delta=0.002,
                detector_deltas=(0.002,),
                persistence=25,
            )
        )
        summary = result.summaries[0]
        self.assertGreater(summary.mean_gap, 0.0)
        self.assertGreater(summary.positive_gap_rate, 0.5)

    def test_invalidity_gap_supports_kswin_and_cusum(self) -> None:
        detector_configs = {
            "kswin": {
                "delta": 0.008,
                "kwargs": {"kswin_window_size": 80, "kswin_stat_size": 30},
            },
            "cusum": {"delta": 0.02, "kwargs": {"cusum_threshold": 20.0}},
        }
        for detector_name, detector_config in detector_configs.items():
            result = run_invalidity_gap_experiment(
                InvalidityGapConfig(
                    seeds=(0, 1, 2),
                    steps=2200,
                    warmup=250,
                    phase_lengths=(700, 700, 800),
                    low_drift=0.002,
                    high_drift=0.02,
                    holder_exponent=0.75,
                    operating_window=180,
                    detector_name=detector_name,  # type: ignore[arg-type]
                    detector_input="observation",
                    detector_delta=detector_config["delta"],
                    detector_deltas=(detector_config["delta"],),
                    persistence=20,
                    **detector_config["kwargs"],
                )
            )
            summary = result.summaries[0]
            self.assertGreater(summary.mean_gap, 0.0)
            self.assertGreater(summary.positive_gap_rate, 0.5)
            self.assertGreaterEqual(summary.detection_rate, 0.0)
            self.assertLessEqual(summary.detection_rate, 1.0)

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

    def test_null_alarm_rate_is_a_probability(self) -> None:
        rate = estimate_null_alarm_rate(
            InvalidityGapConfig(
                seeds=(0, 1),
                steps=800,
                warmup=100,
                phase_lengths=(250, 250, 300),
                operating_window=80,
                detector_input="observation",
            ),
            detector_delta=0.002,
            calibration_seeds=(20, 21, 22),
        )
        self.assertGreaterEqual(rate, 0.0)
        self.assertLessEqual(rate, 1.0)

    def test_detector_calibration_selects_candidate_delta(self) -> None:
        summary = calibrate_detector_delta(
            InvalidityGapConfig(
                seeds=(0, 1),
                steps=800,
                warmup=100,
                phase_lengths=(250, 250, 300),
                operating_window=80,
            ),
            false_alarm_target=0.2,
            candidate_deltas=(0.0005, 0.002, 0.008),
            calibration_seeds=(20, 21, 22),
        )
        self.assertIn(summary.selected_delta, (0.0005, 0.002, 0.008))
        self.assertGreaterEqual(summary.selected_threshold, 0.0)
        self.assertGreaterEqual(summary.selected_null_alarm_rate, 0.0)
        self.assertLessEqual(summary.selected_null_alarm_rate, 1.0)

    def test_calibrated_delay_frontier_returns_positive_gap_regime(self) -> None:
        summaries = run_calibrated_delay_frontier(
            detector_names=("adwin",),
            detector_inputs=("observation",),
            holder_exponents=(0.6,),
            false_alarm_targets=(0.5,),
            high_drifts=(0.02,),
            operating_windows=(180,),
            candidate_deltas=(0.0005, 0.002, 0.008),
            page_hinkley_thresholds=(50.0, 100.0),
            cusum_thresholds=(4.0, 8.0),
            calibration_seeds=(20, 21, 22),
            base_config=InvalidityGapConfig(
                seeds=(0, 1, 2),
                steps=2200,
                warmup=250,
                phase_lengths=(700, 700, 800),
                low_drift=0.002,
                high_drift=0.02,
                operating_window=180,
                persistence=20,
            ),
        )
        self.assertEqual(len(summaries), 1)
        summary = summaries[0]
        self.assertAlmostEqual(summary.holder_exponent, 0.6)
        self.assertGreater(summary.mean_gap, 0.0)
        self.assertGreater(summary.positive_gap_rate, 0.5)


if __name__ == "__main__":
    unittest.main()
