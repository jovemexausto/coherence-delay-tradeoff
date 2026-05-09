from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.stats import norm


@dataclass(slots=True)
class RFFBayesRegressor:
    input_dim: int
    num_features: int = 64
    ridge: float = 1.0
    noise_floor: float = 1e-6
    seed: int = 0
    _weight_matrix: np.ndarray = field(init=False, repr=False)
    _bias: np.ndarray = field(init=False, repr=False)
    _coef: np.ndarray = field(init=False, repr=False)
    _cov_inv: np.ndarray = field(init=False, repr=False)
    _noise_var: float = field(init=False, repr=False)

    def __post_init__(self) -> None:
        rng = np.random.default_rng(self.seed)
        self._weight_matrix = rng.normal(
            scale=1.0, size=(self.input_dim, self.num_features)
        )
        self._bias = rng.uniform(0.0, 2.0 * np.pi, size=self.num_features)
        self._coef = np.zeros(self.num_features, dtype=float)
        self._cov_inv = np.eye(self.num_features, dtype=float)
        self._noise_var = 1.0

    def _features(self, x: np.ndarray) -> np.ndarray:
        x_2d = np.atleast_2d(np.asarray(x, dtype=float))
        proj = x_2d @ self._weight_matrix + self._bias
        return np.sqrt(2.0 / self.num_features) * np.cos(proj)

    def fit(self, x: np.ndarray, y: np.ndarray) -> None:
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        if x.ndim != 2:
            raise ValueError("x must be 2-D")
        if x.shape[0] != y.shape[0]:
            raise ValueError("x and y must have matching rows")
        if x.shape[0] == 0:
            raise ValueError("cannot fit on an empty window")

        phi = self._features(x)
        precision = phi.T @ phi + self.ridge * np.eye(self.num_features)
        precision += 1e-9 * np.eye(self.num_features)
        self._cov_inv = np.linalg.inv(precision)
        self._coef = self._cov_inv @ (phi.T @ y)
        residuals = y - phi @ self._coef
        self._noise_var = max(float(np.mean(residuals * residuals)), self.noise_floor)

    def predict(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        phi = self._features(x)
        mean = phi @ self._coef
        variance = self._noise_var + np.einsum("ij,jk,ik->i", phi, self._cov_inv, phi)
        return mean, np.maximum(variance, self.noise_floor)


def scale_for_miscalibration(
    y_true: np.ndarray,
    y_mean: np.ndarray,
    y_std: np.ndarray,
    *,
    grid: np.ndarray | None = None,
    levels: np.ndarray | None = None,
) -> tuple[float, float]:
    y_true = np.asarray(y_true, dtype=float)
    y_mean = np.asarray(y_mean, dtype=float)
    y_std = np.asarray(y_std, dtype=float)
    valid = np.isfinite(y_true) & np.isfinite(y_mean) & np.isfinite(y_std)
    if not np.any(valid):
        return 1.0, float("inf")

    err = np.abs(y_true[valid] - y_mean[valid])
    sigma = np.maximum(y_std[valid], 1e-9)
    if grid is None:
        grid = np.geomspace(0.25, 4.0, 25)
    if levels is None:
        levels = np.asarray([0.1, 0.2, 0.3, 0.5, 0.7, 0.8, 0.9], dtype=float)

    best_scale = 1.0
    best_loss = float("inf")
    for alpha in grid:
        coverage = []
        scaled = alpha * sigma
        for p in levels:
            z = norm.ppf((1.0 + p) / 2.0)
            coverage.append(float(np.mean(err <= z * scaled)))
        loss = float(np.mean(np.abs(np.asarray(coverage) - levels)))
        if loss < best_loss:
            best_loss = loss
            best_scale = float(alpha)

    return best_scale, best_loss


def prequential_rff_predictions(
    x: np.ndarray,
    y: np.ndarray,
    *,
    buffer_size: int,
    num_features: int,
    ridge: float,
    noise_floor: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.ndim != 2:
        raise ValueError("x must be 2-D")
    if x.shape[0] != y.shape[0]:
        raise ValueError("x and y must have matching rows")

    preds = np.full(y.shape, np.nan, dtype=float)
    stds = np.full(y.shape, np.nan, dtype=float)
    model = RFFBayesRegressor(
        input_dim=x.shape[1],
        num_features=num_features,
        ridge=ridge,
        noise_floor=noise_floor,
        seed=seed,
    )

    for index in range(1, y.size):
        start = max(0, index - buffer_size)
        if index - start < 1:
            continue
        model.fit(x[start:index], y[start:index])
        mean, variance = model.predict(x[index : index + 1])
        preds[index] = float(mean[0])
        stds[index] = float(np.sqrt(variance[0]))

    return preds, stds
