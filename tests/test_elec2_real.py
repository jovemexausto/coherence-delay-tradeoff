from __future__ import annotations

import unittest

from useful_memory_horizon.elec2_real import Elec2DiagnosticConfig, run_elec2_diagnostic


class Elec2RealDiagnosticTest(unittest.TestCase):
    def test_real_stream_exhibits_interior_optimum(self) -> None:
        result = run_elec2_diagnostic(
            Elec2DiagnosticConfig(max_samples=4000, anchor_size=48, step=24)
        )
        self.assertGreater(result.best_window, int(result.window_sizes[0]))
        self.assertLess(result.best_window, int(result.window_sizes[-1]))
        self.assertGreaterEqual(result.useful_windows.size, 2)


if __name__ == "__main__":
    unittest.main()
