from __future__ import annotations

import unittest

from useful_memory_horizon.sinkhorn_self_coupling_certificate import (
    certify_self_coupling_stability,
)


class SinkhornSelfCouplingCertificateTest(unittest.TestCase):
    def test_self_coupling_proxy_is_positive_on_calibrated_pairs(self) -> None:
        rows = certify_self_coupling_stability(
            max_spectral_radius=0.97,
            max_largest_n_mean_inverse_norm=3.0,
        )
        lookup = {
            (
                int(row["ambient_dim"]),
                int(row["intrinsic_dim"]),
                str(row["coupling"]),
            ): row
            for row in rows
        }
        self.assertTrue(bool(lookup[(8, 2, "xx")]["stable_proxy"]))
        self.assertTrue(bool(lookup[(8, 2, "yy")]["stable_proxy"]))
        self.assertTrue(bool(lookup[(12, 2, "xx")]["stable_proxy"]))
        self.assertTrue(bool(lookup[(12, 2, "yy")]["stable_proxy"]))


if __name__ == "__main__":
    unittest.main()
