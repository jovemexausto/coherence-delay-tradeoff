from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from temporalbridge.core.bootstrap import bootstrap_horizon
from temporalbridge.core.fit import fit_horizon


@dataclass
class HorizonBridgeEstimator:
    fit_options: dict[str, Any] = field(default_factory=dict)
    bootstrap_method: str = "wild"
    n_boot: int = 500
    fitted_profile_: dict[str, Any] | None = None
    bootstrap_: dict[str, Any] | None = None

    def fit(
        self, lags: np.ndarray, discrepancies: np.ndarray
    ) -> "HorizonBridgeEstimator":
        self.fitted_profile_ = fit_horizon(
            lags, discrepancies, fit_options=self.fit_options
        )
        self.bootstrap_ = bootstrap_horizon(
            self.fitted_profile_,
            method=self.bootstrap_method,
            n_boot=self.n_boot,
        )
        return self

    def predict_horizon(self) -> float:
        if self.fitted_profile_ is None:
            raise ValueError("fit must be called before predict_horizon")
        return float(self.fitted_profile_["n_star"])

    def score(self) -> float:
        if self.bootstrap_ is None:
            raise ValueError("fit must be called before score")
        lower, upper = self.bootstrap_["ci_n_star"]
        return float(upper - lower)
