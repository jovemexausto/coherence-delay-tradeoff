from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from temporalbridge.core.bootstrap import bootstrap_horizon
from temporalbridge.core.fit import fit_horizon


@dataclass
class HorizonBridgeCallback:
    fit_options: dict[str, Any] = field(default_factory=dict)
    bootstrap_method: str = "wild"
    n_boot: int = 200

    def on_validation_epoch_end(
        self,
        *,
        lags: np.ndarray,
        discrepancies: np.ndarray,
    ) -> dict[str, Any]:
        profile = fit_horizon(lags, discrepancies, fit_options=self.fit_options)
        bootstrap = bootstrap_horizon(
            profile,
            method=self.bootstrap_method,
            n_boot=self.n_boot,
        )
        return {
            "H": profile["H"],
            "n_star": profile["n_star"],
            "ci_H": bootstrap["ci_H"],
            "ci_n_star": bootstrap["ci_n_star"],
        }
