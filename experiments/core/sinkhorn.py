from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(slots=True)
class SinkhornResult:
    cost: float
    iterations: int


def _pairwise_squared_distances(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    x_norm = np.sum(x * x, axis=1, keepdims=True)
    y_norm = np.sum(y * y, axis=1, keepdims=True).T
    return np.maximum(x_norm + y_norm - 2.0 * x @ y.T, 0.0)


def sinkhorn_cost(
    x: np.ndarray,
    y: np.ndarray,
    epsilon: float,
    *,
    max_iters: int = 120,
    tol: float = 1e-6,
) -> SinkhornResult:
    if epsilon <= 0:
        raise ValueError("epsilon must be positive")
    a = np.full(x.shape[0], 1.0 / x.shape[0])
    b = np.full(y.shape[0], 1.0 / y.shape[0])
    cost = _pairwise_squared_distances(x, y)
    cost_shift = float(np.min(cost))
    kernel = np.exp(-(cost - cost_shift) / epsilon)
    u = np.ones_like(a)
    v = np.ones_like(b)

    for iteration in range(1, max_iters + 1):
        u_prev = u.copy()
        Kv = kernel @ v
        Kv = np.maximum(Kv, 1e-300)
        u = a / Kv
        Ku = kernel.T @ u
        Ku = np.maximum(Ku, 1e-300)
        v = b / Ku
        if np.max(np.abs(u - u_prev) / np.maximum(np.abs(u_prev), 1e-12)) < tol:
            break

    transport = (u[:, None] * kernel) * v[None, :]
    transport_cost = float(np.sum(transport * cost))
    return SinkhornResult(cost=transport_cost, iterations=iteration)


def debiased_sinkhorn_divergence(
    x: np.ndarray,
    y: np.ndarray,
    epsilon: float,
    *,
    max_iters: int = 250,
    tol: float = 1e-9,
) -> SinkhornResult:
    xy = sinkhorn_cost(x, y, epsilon, max_iters=max_iters, tol=tol)
    xx = sinkhorn_cost(x, x, epsilon, max_iters=max_iters, tol=tol)
    yy = sinkhorn_cost(y, y, epsilon, max_iters=max_iters, tol=tol)
    return SinkhornResult(
        cost=xy.cost - 0.5 * xx.cost - 0.5 * yy.cost,
        iterations=max(xy.iterations, xx.iterations, yy.iterations),
    )
