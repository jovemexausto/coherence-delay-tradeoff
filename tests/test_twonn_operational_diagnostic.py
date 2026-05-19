from __future__ import annotations

import unittest

from useful_memory_horizon.twonn_operational_diagnostic import (
    TwoNNOperationalDiagnosticConfig,
    compare_twonn_to_ambient_on_operational_frontier,
)


class TwoNNOperationalDiagnosticTest(unittest.TestCase):
    def test_default_operational_diagnostic_prefers_twonn(self) -> None:
        result = compare_twonn_to_ambient_on_operational_frontier()
        lookup = {row["feature"]: row for row in result.comparison_rows}
        ambient = lookup["ambient_dim"]
        twonn = lookup["twonn_k_hat"]
        self.assertLessEqual(
            float(twonn["loo_mae_epsilon_max"]),
            float(ambient["loo_mae_epsilon_max"]),
        )
        self.assertGreaterEqual(
            float(twonn["stability_cut_accuracy"]),
            float(ambient["stability_cut_accuracy"]),
        )

    def test_twonn_beats_ambient_on_epsilon_max_and_stability_cut(self) -> None:
        config = TwoNNOperationalDiagnosticConfig(
            pairs=(
                (8, 1),
                (8, 2),
                (8, 3),
                (12, 1),
                (12, 2),
                (12, 3),
                (16, 1),
                (16, 2),
                (16, 3),
            ),
            epsilons=(0.8, 0.5, 0.3, 0.2, 0.1, 0.05),
            frontier_sample_sizes=(24, 48, 96, 160),
            frontier_seed_count=12,
            twonn_sample_size=256,
            twonn_seed_count=4,
            cut_epsilon=0.2,
        )
        result = compare_twonn_to_ambient_on_operational_frontier(config)
        lookup = {row["feature"]: row for row in result.comparison_rows}
        ambient = lookup["ambient_dim"]
        twonn = lookup["twonn_k_hat"]
        self.assertLessEqual(
            float(twonn["loo_mae_epsilon_max"]),
            float(ambient["loo_mae_epsilon_max"]),
        )
        self.assertGreaterEqual(
            float(twonn["stability_cut_accuracy"]),
            float(ambient["stability_cut_accuracy"]),
        )


if __name__ == "__main__":
    unittest.main()
