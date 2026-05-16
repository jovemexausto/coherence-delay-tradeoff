from __future__ import annotations

import unittest

from useful_memory_horizon.glue_theorem_useful import (
    UsefulCarrierConfig,
    run_useful_carrier_research,
)


class UsefulCarrierResearchTest(unittest.TestCase):
    def test_moderate_theorem_bridge_stays_close_to_root_n(self) -> None:
        result = run_useful_carrier_research(
            UsefulCarrierConfig(
                ambient_intrinsic_pairs=((8, 1),),
                spans=(0.25,),
                sample_sizes=(32, 64, 128, 256),
                replications=4,
            )
        )

        rows = result.summary_rows
        tri_row = next(row for row in rows if "triangular" in str(row["setting"]))
        iid_row = next(row for row in rows if "iid mixture" in str(row["setting"]))

        tri_a = float(tri_row["carrier_a"])
        iid_a = float(iid_row["carrier_a"])

        self.assertGreater(tri_a, 0.40)
        self.assertGreater(iid_a, 0.40)
        self.assertLess(abs(tri_a - iid_a), 0.15)


if __name__ == "__main__":
    unittest.main()
