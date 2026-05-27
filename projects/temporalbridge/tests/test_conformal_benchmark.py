from __future__ import annotations

import sys
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "code"))
sys.path.insert(0, str(PROJECT_ROOT.parent / "scale-consistency" / "code"))

from temporalbridge.benchmarks import run_conformal_benchmark  # noqa: E402
from temporalbridge.benchmarks.conformal_benchmark import (  # noqa: E402
    ConformalBenchmarkConfig,
)


class TemporalBridgeConformalTest(unittest.TestCase):
    def test_conformal_benchmark_runs(self) -> None:
        result = run_conformal_benchmark(
            config=ConformalBenchmarkConfig(repetitions=4),
            rng_seed=0,
        )
        self.assertTrue(result["rows"])
        self.assertTrue(result["window_summary"])
        self.assertIn("best_window", result["summary"])
        self.assertIn("u_curve", result["summary"])


if __name__ == "__main__":
    unittest.main()
