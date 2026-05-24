from __future__ import annotations

import unittest

from useful_memory_horizon.elec2_real import (
    Elec2DiagnosticConfig,
    build_elec2_robustness_configs,
    run_elec2_diagnostic,
)


class Elec2RealDiagnosticTest(unittest.TestCase):
    def test_real_stream_exhibits_interior_optimum(self) -> None:
        result = run_elec2_diagnostic(
            Elec2DiagnosticConfig(max_samples=4000, anchor_size=48, step=24)
        )
        self.assertGreater(result.best_window, int(result.window_sizes[0]))
        self.assertLess(result.best_window, int(result.window_sizes[-1]))
        self.assertGreaterEqual(result.useful_windows.size, 2)
        self.assertEqual(result.metadata.loaded_sample_count, 4000)

    def test_real_stream_records_start_offset_metadata(self) -> None:
        result = run_elec2_diagnostic(
            Elec2DiagnosticConfig(
                max_samples=1400, start_index=128, anchor_size=48, step=24
            )
        )
        self.assertEqual(result.metadata.start_index, 128)
        self.assertEqual(result.metadata.loaded_sample_count, 1400)

    def test_builds_robustness_config_grid(self) -> None:
        configs = build_elec2_robustness_configs(
            variables=("nswprice", "vicprice"),
            anchor_sizes=(24, 48),
            start_indices=(0, 2000),
            max_samples=4000,
            step=24,
            useful_delta=0.05,
        )
        self.assertEqual(len(configs), 8)
        self.assertEqual(configs[0].max_samples, 4000)


if __name__ == "__main__":
    unittest.main()
