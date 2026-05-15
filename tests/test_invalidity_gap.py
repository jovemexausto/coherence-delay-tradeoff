from __future__ import annotations

import unittest

from useful_memory_horizon.invalidity_gap import (
    InvalidityGapConfig,
    run_detector_comparison,
    run_invalidity_gap_experiment,
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


if __name__ == "__main__":
    unittest.main()
